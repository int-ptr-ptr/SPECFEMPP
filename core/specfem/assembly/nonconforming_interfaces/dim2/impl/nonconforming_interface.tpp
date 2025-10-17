#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/macros.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/coupled_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.tpp"
#include "specfem/data_access.hpp"

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::coupled_interfaces_impl::interface_container<
    specfem::dimension::type::dim2, InterfaceTag, BoundaryTag,
    specfem::connections::type::nonconforming>::
    interface_container(
        const int ngllz, const int ngllx,
        const specfem::assembly::edge_types<specfem::dimension::type::dim2>
            &edge_types,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  // TODO: make this a parameter for now, use same gll quadrature
  Kokkos::View<type_real *, Kokkos::HostSpace> interface_quadrature(
      "interface_quadrature", ngllx);
  Kokkos::View<type_real *, Kokkos::HostSpace> interface_weights(
      "interface_weights", ngllx);
  Kokkos::View<type_real **, Kokkos::HostSpace> interface_deriv(
      "interface_deriv", ngllx, ngllx);
  for (int i = 0; i < mesh.h_xi.extent(0); i++) {
    interface_quadrature(i) = mesh.h_xi(i);
    interface_weights(i) = mesh.h_weights(i);
    for (int j = 0; j < mesh.h_xi.extent(0); j++) {
      interface_deriv(i, j) = mesh.h_hprime(i, j);
    }
  }

  const int nquad_intersection = interface_quadrature.extent(0);

  // transfer function setting is not symmetric, so we need to make sure we know
  // what side 1 is.
  bool is_side1 =
      (InterfaceTag == specfem::interface::interface_tag::elastic_acoustic);

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto connection_mapping =
      specfem::connections::connection_mapping(ngllx, ngllz);

  const auto [self_edges, coupled_edges] = edge_types.get_edges_on_host(
      specfem::connections::type::nonconforming, InterfaceTag, BoundaryTag);

  const auto nedges = self_edges.size();

  this->intersection_factor = EdgeFactorView(
      "specfem::assembly::nonconforming_interfaces::intersection_factor",
      nedges, nquad_intersection);

  this->intersection_normal =
      EdgeNormalView("specfem::assembly::nonconforming_interfaces::intersection_normal",
                     nedges, nquad_intersection, 2);
  this->h_intersection_normal = Kokkos::create_mirror_view(intersection_normal);

  // consider linking conjugate containers so that we don't need to do
  // set_transfer_functions twice.
  this->transfer_function = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function", nedges,
      nquad_intersection, ngllx);

  this->transfer_function_other = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function_other",
      nedges, nquad_intersection, ngllx);

  this->h_intersection_factor = Kokkos::create_mirror_view(intersection_factor);
  this->h_transfer_function = Kokkos::create_mirror_view(transfer_function);
  this->h_transfer_function_other =
      Kokkos::create_mirror_view(transfer_function_other);

  const auto weights = mesh.h_weights;

  // used when computing transfer functions
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      icoorg("icoorg", mesh.ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      jcoorg("jcoorg", mesh.ngnod);

  for (int i = 0; i < nedges; ++i) {
    const int ispec = self_edges(i).ispec;
    const auto iedge_type = self_edges(i).edge_type;
    const int jspec = coupled_edges(i).ispec;
    const auto jedge_type = coupled_edges(i).edge_type;
    for (int inod = 0; inod < mesh.ngnod; inod++) {
      icoorg(inod).x = mesh.h_control_node_coord(0, ispec, inod);
      icoorg(inod).z = mesh.h_control_node_coord(1, ispec, inod);
      jcoorg(inod).x = mesh.h_control_node_coord(0, jspec, inod);
      jcoorg(inod).z = mesh.h_control_node_coord(1, jspec, inod);
    }
    auto transfer_subview =
        Kokkos::subview(h_transfer_function, i, Kokkos::ALL, Kokkos::ALL);
    auto transfer_subview_other =
        Kokkos::subview(h_transfer_function_other, i, Kokkos::ALL, Kokkos::ALL);
    if (is_side1) {
      specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
          icoorg, jcoorg, iedge_type, jedge_type, interface_quadrature,
          mesh.h_xi, transfer_subview, transfer_subview_other);
    } else {
      specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
          jcoorg, icoorg, jedge_type, iedge_type, interface_quadrature,
          mesh.h_xi, transfer_subview_other, transfer_subview);
    }
    // compute normal on edge
    const int npoints =
        connection_mapping.number_of_points_on_orientation(iedge_type);

    // compute factor by finding first derivative of position
    // along the edge and multiplying by the quadrature weight
    const Kokkos::View<type_real **, Kokkos::HostSpace> dr_intersection(
        "dr_intersection", nquad_intersection, 2);

    for (int iquad = 0; iquad < nquad_intersection; iquad++) {
      dr_intersection(iquad, 0) = 0;
      dr_intersection(iquad, 1) = 0;
    }
    for (int iknot = 0; iknot < nquad_intersection; iknot++) {
      // get local coordinate (we can recover this from the transfer function by
      // interpolating x)
      type_real local_coord = 0;
      for (int ipoint = 0; ipoint < npoints; ipoint++) {
        local_coord += transfer_subview(iknot, ipoint) * mesh.h_xi(ipoint);
      }

      // get global coordinate -- we interpolate against shape prime
      const auto [xi, gamma] = [&]() -> std::pair<type_real, type_real> {
        if (iedge_type == specfem::mesh_entity::dim2::type::bottom) {
          return { local_coord, -1 };
        } else if (iedge_type == specfem::mesh_entity::dim2::type::right) {
          return { 1, local_coord };
        } else if (iedge_type == specfem::mesh_entity::dim2::type::top) {
          return { local_coord, 1 };
        } else {
          return { -1, local_coord };
        }
      }();
      const auto loc =
          jacobian::compute_locations(icoorg, mesh.ngnod, xi, gamma);

      // accumulate derivative at each quadrature point
      for (int iquad = 0; iquad < nquad_intersection; iquad++) {
        dr_intersection(iquad, 0) += interface_deriv(iquad, iknot) * loc.x;
        dr_intersection(iquad, 1) += interface_deriv(iquad, iknot) * loc.z;
      }
    }

    // convert dr to ds and multiply by weights
    for (int iquad = 0; iquad < nquad_intersection; iquad++) {
      this->h_intersection_factor(i, iquad) =
          interface_weights(iquad) *
          std::sqrt(dr_intersection(iquad, 0) * dr_intersection(iquad, 0) +
                    dr_intersection(iquad, 1) * dr_intersection(iquad, 1));

      type_real nx = dr_intersection(iquad, 1);
      type_real nz = -dr_intersection(iquad, 0);
      type_real mag = std::sqrt(nx * nx + nz * nz);

      // previous computation was a 90 deg clockwise rotation. it should be CCW
      // for these cases:
      if (iedge_type == specfem::mesh_entity::dim2::type::top ||
          iedge_type == specfem::mesh_entity::dim2::type::left) {
        mag *= -1;
      }
      this->h_intersection_normal(i, iquad, 0) = nx / mag;
      this->h_intersection_normal(i, iquad, 1) = nz / mag;
    }
  }

  Kokkos::deep_copy(intersection_factor, h_intersection_factor);
  Kokkos::deep_copy(intersection_normal, h_intersection_normal);

  Kokkos::deep_copy(transfer_function, h_transfer_function);
  Kokkos::deep_copy(transfer_function_other, h_transfer_function_other);
}
