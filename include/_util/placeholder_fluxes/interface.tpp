#pragma once
#include "interface.hpp"

template <typename PointFieldType>
KOKKOS_INLINE_FUNCTION void _util::placeholder_fluxes::fix_free_surface(
    specfem::compute::assembly &assembly, PointFieldType *point_field,
    const int ispec, const specfem::enums::edge::type edgetype) {

  const auto h_is_bdry_at_pt =
      [&](const specfem::point::index<specfem::dimension::type::dim2> index,
          specfem::element::boundary_tag tag) -> bool {
    specfem::point::boundary<
        specfem::element::boundary_tag::acoustic_free_surface,
        specfem::dimension::type::dim2, false>
        point_boundary_afs;
    specfem::point::boundary<specfem::element::boundary_tag::none,
                             specfem::dimension::type::dim2, false>
        point_boundary_none;
    specfem::point::boundary<specfem::element::boundary_tag::stacey,
                             specfem::dimension::type::dim2, false>
        point_boundary_stacey;
    specfem::point::boundary<
        specfem::element::boundary_tag::composite_stacey_dirichlet,
        specfem::dimension::type::dim2, false>
        point_boundary_composite;
    switch (assembly.boundaries.boundary_tags(index.ispec)) {
    case specfem::element::boundary_tag::acoustic_free_surface:
      specfem::compute::load_on_host(index, assembly.boundaries,
                                     point_boundary_afs);
      return point_boundary_afs.tag == tag;
    case specfem::element::boundary_tag::none:
      specfem::compute::load_on_host(index, assembly.boundaries,
                                     point_boundary_none);
      return point_boundary_none.tag == tag;
    case specfem::element::boundary_tag::stacey:
      specfem::compute::load_on_host(index, assembly.boundaries,
                                     point_boundary_stacey);
      return point_boundary_stacey.tag == tag;
    case specfem::element::boundary_tag::composite_stacey_dirichlet:
      specfem::compute::load_on_host(index, assembly.boundaries,
                                     point_boundary_composite);
      return point_boundary_composite.tag == tag;
    default:
      return false;
    }
  };

  specfem::point::index<specfem::dimension::type::dim2> index(ispec, 0, 0);
  for (int igll = 0; igll < ContainerType::NGLL_EDGE; igll++) {
    specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
        index.iz, index.ix, edgetype, igll);
    if (!h_is_bdry_at_pt(index, specfem::element::boundary_tag::none)) {
      point_field[igll].acceleration(0) = 0;
    }
  }
}

template <int medium, bool on_device, typename ContainerType>
void _util::placeholder_fluxes::compute_field_nderiv(
    int index, const specfem::compute::assembly &assembly, ContainerType &container) {
  // TODO optimize this.
  constexpr bool UseSIMD = false;
  using FieldDispType =
      specfem::point::field<ContainerType::dim_type,
                            ContainerType::template medium_type<medium>, true,
                            false, false, false, UseSIMD>;
  constexpr int ncomp = specfem::element::attributes<
      ContainerType::dim_type,
      ContainerType::template medium_type<medium> >::components();
#define _ispec                                                                 \
  [&] {                                                                        \
    if constexpr (medium == 1) {                                               \
      if constexpr (on_device) {                                               \
        return container.medium1_index_mapping(index);                         \
      } else {                                                                 \
        return container.h_medium1_index_mapping(index);                       \
      }                                                                        \
    } else {                                                                   \
      if constexpr (on_device) {                                               \
        return container.medium2_index_mapping(index);                         \
      } else {                                                                 \
        return container.h_medium2_index_mapping(index);                       \
      }                                                                        \
    }                                                                          \
  }()
#define _edge                                                                  \
  [&] {                                                                        \
    if constexpr (medium == 1) {                                               \
      if constexpr (on_device) {                                               \
        return container.medium1_edge_type(index);                             \
      } else {                                                                 \
        return container.h_medium1_edge_type(index);                           \
      }                                                                        \
    } else {                                                                   \
      if constexpr (on_device) {                                               \
        return container.medium2_edge_type(index);                             \
      } else {                                                                 \
        return container.h_medium2_edge_type(index);                           \
      }                                                                        \
    }                                                                          \
  }()
#define _normal                                                                \
  [&] {                                                                        \
    if constexpr (on_device) {                                                 \
      return container.medium1_edge_normal;                                    \
    } else {                                                                   \
      return container.h_medium1_edge_normal;                                  \
    }                                                                          \
  }()
#define _field_nderiv                                                          \
  [&] {                                                                        \
    if constexpr (on_device) {                                                 \
      return container.medium1_field_nderiv_noncontra;                                   \
    } else {                                                                   \
      return container.h_medium1_field_nderiv_noncontra;                                 \
    }                                                                          \
  }()
  int ispec = _ispec;
  auto edgetype = _edge;
  FieldDispType disp;
  int ix, iz;
  type_real dfdxi[ncomp];
  type_real dfdga[ncomp];

  for (int igll = 0; igll < ContainerType::NGLL_EDGE; igll++) {
#pragma unroll
    for (int icomp = 0; icomp < ncomp; icomp++) {
      dfdxi[icomp] = 0;
      dfdga[icomp] = 0;
    }
    specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
        iz, ix, edgetype, igll);
    for (int k = 0; k < ContainerType::NGLL_EDGE; k++) {
      specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz, k);
      specfem::compute::load_on_host(index, assembly.fields.forward, disp);
#pragma unroll
      for (int icomp = 0; icomp < ncomp; icomp++) {
        dfdxi[icomp] += assembly.mesh.quadratures.gll.h_hprime(ix, k) *
                        disp.displacement(icomp);
      }
      index =
          specfem::point::index<specfem::dimension::type::dim2>(ispec, k, ix);
      specfem::compute::load_on_host(index, assembly.fields.forward, disp);
#pragma unroll
      for (int icomp = 0; icomp < ncomp; icomp++) {
        dfdga[icomp] += assembly.mesh.quadratures.gll.h_hprime(iz, k) *
                        disp.displacement(icomp);
      }
    }

#pragma unroll
    for (int icomp = 0; icomp < ncomp; icomp++) {
      _field_nderiv(index, igll, icomp) =
          dfdxi[icomp] * _normal(index, igll, 0) +
          dfdga[icomp] * _normal(index, igll, 1);
    }
  }

#undef _ispec
#undef _edge
#undef _normal
#undef _field_nderiv
}
