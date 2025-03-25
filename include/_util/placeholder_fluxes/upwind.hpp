#pragma once
#include "interface.hpp"

namespace _util {
namespace placeholder_fluxes {
namespace upwind {

static void compute_fluxes(specfem::compute::assembly &assembly,
                           _util::placeholder_fluxes::ContainerType &container,
                           const type_real trace_relaxation_parameter,
                           const type_real crossover_relaxation_parameter) {
  const type_real one_minus_trace_relaxation = 1 - trace_relaxation_parameter;
  const type_real inverse_one_minus_half_trace_relaxation =
      1 / (1 - trace_relaxation_parameter / 2);
  const type_real one_minus_crossover_relaxation =
      1 - crossover_relaxation_parameter;
  constexpr bool UseSIMD = false;
  constexpr auto DimensionType = _util::placeholder_fluxes::DimensionType;
  using AcousticAccelType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, UseSIMD>;
  using AcousticDispType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, true, false,
                            false, false, UseSIMD>;
  using AcousticProperties = specfem::point::properties<
      DimensionType, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, UseSIMD>;
  //   for(int iedge = 0; iedge < container.num_medium1_edges; iedge++){

  //   }
  for (int iinterface = 0; iinterface < container.num_interfaces;
       iinterface++) {
    type_real vel_edge1[ContainerType::NGLL_EDGE];
    type_real vel_edge2[ContainerType::NGLL_EDGE];
    type_real rcinv_edge1[ContainerType::NGLL_EDGE];
    type_real rcinv_edge2[ContainerType::NGLL_EDGE];
    type_real sigma_edge1[ContainerType::NGLL_EDGE];
    type_real sigma_edge2[ContainerType::NGLL_EDGE];
    type_real chi1[ContainerType::NGLL_EDGE];
    type_real chi2[ContainerType::NGLL_EDGE];

    type_real outflow_edge1[ContainerType::NGLL_EDGE];
    type_real outflow_edge2[ContainerType::NGLL_EDGE];

    AcousticAccelType accel1[ContainerType::NGLL_EDGE];
    AcousticAccelType accel2[ContainerType::NGLL_EDGE];
    AcousticProperties props;

    AcousticDispType fieldread[ContainerType::NGLL_EDGE];

    const int edge1_index = container.h_interface_medium1_index(iinterface);
    const int edge2_index = container.h_interface_medium2_index(iinterface);

    const int edge1_ispec = container.h_medium1_index_mapping(edge1_index);
    const auto edge1_type = container.h_medium1_edge_type(edge1_index);
    const int edge2_ispec = container.h_medium2_index_mapping(edge2_index);
    const auto edge2_type = container.h_medium2_edge_type(edge2_index);

    specfem::point::index<specfem::dimension::type::dim2> index(edge1_ispec, 0,
                                                                0);
    container.template load_field<1, false>(edge1_index, assembly, fieldread);
    for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE; igll_edge++) {
      chi1[igll_edge] = fieldread[igll_edge].displacement(0);
      accel1[igll_edge].acceleration(0) = 0;

      vel_edge1[igll_edge] =
          container.h_medium1_field_vel(edge1_index, igll_edge, 0);

      specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
          index.iz, index.ix, edge1_type, igll_edge);
      specfem::compute::load_on_host(index, assembly.properties, props);
      rcinv_edge1[igll_edge] = props.rho_vpinverse();
      sigma_edge1[igll_edge] =
          container.h_medium1_field_nderiv(edge1_index, igll_edge, 0) *
          props.rho_inverse();

      // w^- = (-1/2kappa, nx/2c, nz/2c)
      // (sigma . n)^- = c w^- = (-c/kappa, nx, nz)/2
      outflow_edge1[igll_edge] =
          (-rcinv_edge1[igll_edge] * vel_edge1[igll_edge] +
           sigma_edge1[igll_edge] * one_minus_trace_relaxation) /
          2;
    }

    index.ispec = edge2_ispec;
    container.template load_field<2, false>(edge2_index, assembly, fieldread);
    for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE; igll_edge++) {
      chi2[igll_edge] = fieldread[igll_edge].displacement(0);
      accel2[igll_edge].acceleration(0) = 0;

      vel_edge2[igll_edge] =
          container.h_medium1_field_vel(edge2_index, igll_edge, 0);

      specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
          index.iz, index.ix, edge2_type, igll_edge);
      specfem::compute::load_on_host(index, assembly.properties, props);
      rcinv_edge2[igll_edge] = props.rho_vpinverse();
      sigma_edge2[igll_edge] =
          container.h_medium1_field_nderiv(edge2_index, igll_edge, 0) *
          props.rho_inverse();

      // w^- = (-1/2kappa, nx/2c, nz/2c)
      // (sigma . n)^- = c w^- = (-c/kappa, nx, nz)/2
      outflow_edge2[igll_edge] =
          (-rcinv_edge2[igll_edge] * vel_edge2[igll_edge] +
           sigma_edge2[igll_edge] * one_minus_trace_relaxation) /
          2;
    }

    for (int igll_interface = 0; igll_interface < ContainerType::NGLL_INTERFACE;
         igll_interface++) {
      type_real jump_penalty =
          (container.template edge_to_mortar<1, false>(iinterface,
                                                       igll_interface, chi1) -
           container.template edge_to_mortar<2, false>(iinterface,
                                                       igll_interface, chi2)) *
          container.h_interface_relaxation_parameter(iinterface) / 2;
      type_real c1 = container.template edge_to_mortar<1, false>(
          iinterface, igll_interface, rcinv_edge1);
      type_real c2 = container.template edge_to_mortar<2, false>(
          iinterface, igll_interface, rcinv_edge2);

      // w^+ = (1/2kappa, nx/2c, nz/2c)
      // recall: opposite n is in opposite direction.
      // (sigma . n)^+ = c w^+ = (c/kappa, nx, nz)/2
      type_real inflow_edge1 =
          (c1 * container.template edge_to_mortar<2, false>(
                    iinterface, igll_interface, vel_edge2) -
           container.template edge_to_mortar<2, false>(
               iinterface, igll_interface, sigma_edge2)) /
          2;
      type_real inflow_edge2 =
          (c2 * container.template edge_to_mortar<1, false>(
                    iinterface, igll_interface, vel_edge1) -
           container.template edge_to_mortar<1, false>(
               iinterface, igll_interface, sigma_edge1)) /
          2;
      // type_real inflow_edge1 = - container.template edge_to_mortar<2, false>(
      //   iinterface, igll_interface, outflow_edge2);
      // type_real inflow_edge2 = - container.template edge_to_mortar<1, false>(
      //               iinterface, igll_interface, outflow_edge1);

      const type_real Jw = container.h_interface_surface_jacobian_times_weight(
          iinterface, igll_interface);

      // positive (outward):   kappa * c * w^+(...)
      // negative (inward):  + kappa * c * w^-(...)
      type_real integrand1 =
          (container.template edge_to_mortar<1, false>(
               iinterface, igll_interface, outflow_edge1) +
           inflow_edge1 * one_minus_crossover_relaxation - jump_penalty) *
          Jw * inverse_one_minus_half_trace_relaxation;
      type_real integrand2 =
          (container.template edge_to_mortar<2, false>(
               iinterface, igll_interface, outflow_edge2) +
           inflow_edge2 * one_minus_crossover_relaxation + jump_penalty) *
          Jw * inverse_one_minus_half_trace_relaxation;

      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        accel1[igll_edge].acceleration(0) +=
            integrand1 * container.h_interface_medium1_mortar_transfer(
                             iinterface, igll_interface, igll_edge);
        accel2[igll_edge].acceleration(0) +=
            integrand2 * container.h_interface_medium2_mortar_transfer(
                             iinterface, igll_interface, igll_edge);
      }
    }

    _util::placeholder_fluxes::fix_free_surface(assembly, accel1, edge1_ispec,
                                                edge1_type);
    _util::placeholder_fluxes::fix_free_surface(assembly, accel2, edge2_ispec,
                                                edge2_type);

    container.template atomic_add_to_field<1, false>(edge1_index, assembly,
                                                     accel1);
    container.template atomic_add_to_field<2, false>(edge2_index, assembly,
                                                     accel2);
  }
}

static void store_vel(specfem::compute::assembly &assembly,
                      _util::placeholder_fluxes::ContainerType &container,
                      type_real dt_inc) {
  constexpr bool UseSIMD = false;
  constexpr auto DimensionType = _util::placeholder_fluxes::DimensionType;
  using AcousticType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, false, true,
                            true, false, UseSIMD>;

  Kokkos::parallel_for(
      container.num_medium1_edges * container.NGLL_EDGE,
      KOKKOS_LAMBDA(const int iworker) {
        int iedge = iworker / container.NGLL_EDGE;
        int ipt = iworker % container.NGLL_EDGE;
        AcousticType vel;
        specfem::point::index<DimensionType, UseSIMD> index(
            container.medium1_index_mapping(iedge), 0, 0);
        specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
            index.iz, index.ix, container.medium1_edge_type(iedge), ipt);
        specfem::compute::load_on_device(index, assembly.fields.forward, vel);

        container.medium1_field_vel(iedge, ipt, 0) =
            vel.velocity(0) + dt_inc * vel.acceleration(0);
      });
  Kokkos::parallel_for(
      container.num_medium1_edges, KOKKOS_LAMBDA(const int iedge) {
        _util::placeholder_fluxes::compute_field_nderiv<1, true>(
            iedge, assembly, container);
      });
  Kokkos::fence();
  Kokkos::deep_copy(container.h_medium1_field_vel, container.medium1_field_vel);
  // on (-) side,
  // covectors:
  //        w+:
}

} // namespace upwind
} // namespace placeholder_fluxes
} // namespace _util
