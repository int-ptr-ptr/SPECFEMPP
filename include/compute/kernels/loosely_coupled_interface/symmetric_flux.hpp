#pragma once

#include "compute/coupled_interfaces/loose_couplings/interface_container.hpp"
#include "compute/coupled_interfaces/loose_couplings/symmetric_flux_container.hpp"

template <int medium, bool on_device, typename ContainerType>
static void compute_field_nderiv(int index,
                                 specfem::compute::assembly &assembly,
                                 ContainerType &container) {
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
      return container.medium1_edge_contravariant_normal;                      \
    } else {                                                                   \
      return container.h_medium1_edge_contravariant_normal;                    \
    }                                                                          \
  }()
#define _field_nderiv                                                          \
  [&] {                                                                        \
    if constexpr (on_device) {                                                 \
      return container.medium1_field_nderiv;                                   \
    } else {                                                                   \
      return container.h_medium1_field_nderiv;                                 \
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

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux::kernel<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, QuadratureType> {
  using ContainerType = specfem::compute::loose::interface_container<
      DimensionType, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::acoustic, QuadratureType,
      specfem::coupled_interface::loose::flux::type::symmetric_flux>;

  // this, along with the original code is wrong.
  // the [[u]]*{{dv}} integral is nonzero for more than just the edge, so we do
  // not have a symmetric form.
  static void compute_fluxes(specfem::compute::assembly &assembly,
                             ContainerType &container) {
    constexpr bool UseSIMD = false;
    using AcousticAccelType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::acoustic, false,
                              false, true, false, UseSIMD>;
    using AcousticDispType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::acoustic, true,
                              false, false, false, UseSIMD>;

    using AcousticProperties = specfem::point::properties<
        DimensionType, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD>;
    // we already computed displacement dot normal
    container
        .template foreach_interface<false>(
            [&](const int iinterface) {
              // setters
              AcousticAccelType accel1[ContainerType::NGLL_EDGE];
              AcousticAccelType accel2[ContainerType::NGLL_EDGE];

              // accessor
              AcousticDispType chi_get[ContainerType::NGLL_EDGE];

              // compute storage
              type_real chi1[ContainerType::NGLL_EDGE];
              type_real chi2[ContainerType::NGLL_EDGE];
              type_real speed_param1[ContainerType::NGLL_EDGE]; // rho inverse
              type_real speed_param2[ContainerType::NGLL_EDGE];
              type_real dchidn1[ContainerType::NGLL_EDGE];
              type_real dchidn2[ContainerType::NGLL_EDGE];

              const int edge1_index =
                  container.h_interface_medium1_index(iinterface);
              const int edge2_index =
                  container.h_interface_medium2_index(iinterface);

              const int edge1_ispec =
                  container.h_medium1_index_mapping(edge1_index);
              const auto edge1_type =
                  container.h_medium1_edge_type(edge1_index);
              const int edge2_ispec =
                  container.h_medium2_index_mapping(edge2_index);
              const auto edge2_type =
                  container.h_medium2_edge_type(edge2_index);
              const int against_edge1_component =
                  (edge1_type == specfem::enums::edge::type::RIGHT ||
                   edge1_type == specfem::enums::edge::type::LEFT)
                      ? 0
                      : 1;
              const int against_edge2_component =
                  (edge2_type == specfem::enums::edge::type::RIGHT ||
                   edge2_type == specfem::enums::edge::type::LEFT)
                      ? 0
                      : 1;

              const int against_edge1_index =
                  (edge1_type == specfem::enums::edge::type::RIGHT ||
                   edge1_type == specfem::enums::edge::type::TOP)
                      ? (ContainerType::NGLL_EDGE - 1)
                      : 0;
              const int against_edge2_index =
                  (edge2_type == specfem::enums::edge::type::RIGHT ||
                   edge2_type == specfem::enums::edge::type::TOP)
                      ? (ContainerType::NGLL_EDGE - 1)
                      : 0;
              specfem::point::index<specfem::dimension::type::dim2> index(
                  edge1_ispec, 0, 0);

              AcousticProperties props;

              container.template load_field<1, false>(edge1_index, assembly,
                                                      chi_get);
              for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                   igll_edge++) {
                accel1[igll_edge].acceleration(0) = 0;
                chi1[igll_edge] = chi_get[igll_edge].displacement(0);
                dchidn1[igll_edge] =
                    container.h_medium1_field_nderiv(edge1_index, igll_edge, 0);

                specfem::compute::loose::point_from_edge<
                    ContainerType::NGLL_EDGE>(index.iz, index.ix, edge1_type,
                                              igll_edge);
                specfem::compute::load_on_host(index, assembly.properties,
                                               props);
                speed_param1[igll_edge] = props.rho_inverse;
              }

              index.ispec = edge2_ispec;
              container.template load_field<2, false>(edge2_index, assembly,
                                                      chi_get);
              for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                   igll_edge++) {
                accel2[igll_edge].acceleration(0) = 0;
                chi2[igll_edge] = chi_get[igll_edge].displacement(0);
                dchidn2[igll_edge] =
                    container.h_medium1_field_nderiv(edge2_index, igll_edge, 0);

                specfem::compute::loose::point_from_edge<
                    ContainerType::NGLL_EDGE>(index.iz, index.ix, edge2_type,
                                              igll_edge);
                specfem::compute::load_on_host(index, assembly.properties,
                                               props);
                speed_param2[igll_edge] = props.rho_inverse;
              }

              for (int igll_interface = 0;
                   igll_interface < ContainerType::NGLL_INTERFACE;
                   igll_interface++) {
                type_real half_c1 =
                    container.template edge_to_mortar<1, false>(
                        iinterface, igll_interface, speed_param1) /
                    2;
                type_real half_c2 =
                    container.template edge_to_mortar<2, false>(
                        iinterface, igll_interface, speed_param2) /
                    2;
                type_real u1 = container.template edge_to_mortar<1, false>(
                    iinterface, igll_interface, chi1);
                type_real u2 = container.template edge_to_mortar<2, false>(
                    iinterface, igll_interface, chi2);
                type_real du1 = container.template edge_to_mortar<1, false>(
                    iinterface, igll_interface, dchidn1);
                type_real du2 = container.template edge_to_mortar<2, false>(
                    iinterface, igll_interface, dchidn2);
                type_real ujmp = u1 - u2;
                type_real cdu_avg =
                    (half_c1 * du1 - half_c2 * du2); // subtract, since we want
                                                     // w.r.t. edge1 out-facing
                                                     // normal
                type_real Jw =
                    container.h_interface_surface_jacobian_times_weight(
                        iinterface,
                        igll_interface); // jacobian (1d) times quadrature
                                         // weight
                for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                     igll_edge++) {
                  type_real dv =
                      (container
                           .h_interface_medium1_mortar_transfer_deriv_times_n(
                               iinterface, igll_interface,
                               igll_edge) // along-edge
                                          // direction
                       + container.h_interface_medium1_mortar_transfer(
                             iinterface, igll_interface,
                             igll_edge) // against-edge
                                        // component
                             * container.h_medium1_edge_normal(
                                   edge1_index, igll_edge,
                                   against_edge1_component)
                             // hprime(i,j) = L_j'(t_i)
                             * assembly.mesh.quadratures.gll.h_hprime(
                                   against_edge1_index, against_edge1_index));
                  // dv = 0;
                  // copy down edge fluxes
                  //  flux += (
                  //        np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                  //      + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                  //      - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                  //  )
                  accel1[igll_edge].acceleration(0) +=
                      Jw *
                      (ujmp * half_c1 * dv +
                       0.5 *
                           container.h_interface_medium1_mortar_transfer(
                               iinterface, igll_interface, igll_edge) *
                           (cdu_avg -
                            ujmp * container.h_interface_relaxation_parameter(
                                       iinterface)));

                  dv = (container
                            .h_interface_medium2_mortar_transfer_deriv_times_n(
                                iinterface, igll_interface,
                                igll_edge) // along-edge
                                           // direction
                        + container.h_interface_medium2_mortar_transfer(
                              iinterface, igll_interface,
                              igll_edge) // against-edge
                                         // component
                              * container.h_medium1_edge_normal(
                                    edge2_index, igll_edge,
                                    against_edge2_component)
                              // hprime(i,j) = L_j'(t_i)
                              * assembly.mesh.quadratures.gll.h_hprime(
                                    against_edge2_index, against_edge2_index));
                  // dv = 0;
                  accel2[igll_edge].acceleration(0) +=
                      Jw *
                      (-ujmp * half_c2 * dv -
                       0.5 *
                           container.h_interface_medium2_mortar_transfer(
                               iinterface, igll_interface, igll_edge) *
                           (cdu_avg -
                            ujmp * container.h_interface_relaxation_parameter(
                                       iinterface)));
                }
              }
              for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                   igll_edge++) {
                if (std::isnan((type_real)accel1[igll_edge].acceleration(0)) ||
                    std::isnan((type_real)accel2[igll_edge].acceleration(0)) ||
                    fabs(chi1[2] - chi2[2]) > 1) {
                  _util::edge_manager::edge_intersection<5> inter = _util::edge_manager::edge_storage_instance<QuadratureType,20> -> load_intersection(iinterface);

                  type_real mt_derivs1[ContainerType::NGLL_INTERFACE]
                                      [ContainerType::NGLL_INTERFACE];
                  type_real mt_derivs2[ContainerType::NGLL_INTERFACE]
                                      [ContainerType::NGLL_INTERFACE];
                  type_real flux1[ContainerType::NGLL_EDGE]
                                 [ContainerType::NGLL_INTERFACE];
                  type_real flux2[ContainerType::NGLL_EDGE]
                                 [ContainerType::NGLL_INTERFACE];
                  type_real flux3[ContainerType::NGLL_EDGE]
                                 [ContainerType::NGLL_INTERFACE];
                  type_real flux1a[5];
                  type_real flux1b[5];
                  type_real flux2a[5];
                  type_real flux2b[5];
                  type_real flux3a[5];
                  type_real flux3b[5];
                  for (int igll_interface = 0;
                       igll_interface < ContainerType::NGLL_INTERFACE;
                       igll_interface++) {
                    type_real half_c1 =
                        container.template edge_to_mortar<1, false>(
                            iinterface, igll_interface, speed_param1) /
                        2;
                    type_real half_c2 =
                        container.template edge_to_mortar<2, false>(
                            iinterface, igll_interface, speed_param2) /
                        2;
                    type_real u1 = container.template edge_to_mortar<1, false>(
                        iinterface, igll_interface, chi1);
                    type_real u2 = container.template edge_to_mortar<2, false>(
                        iinterface, igll_interface, chi2);
                    type_real du1 = container.template edge_to_mortar<1, false>(
                        iinterface, igll_interface, dchidn1);
                    type_real du2 = container.template edge_to_mortar<2, false>(
                        iinterface, igll_interface, dchidn2);
                    type_real ujmp = u1 - u2;
                    type_real cdu_avg =
                        (half_c1 * du1 - half_c2 * du2); // subtract, since we
                                                         // want w.r.t. edge1
                                                         // out-facing normal
                    type_real Jw =
                        container.h_interface_surface_jacobian_times_weight(
                            iinterface,
                            igll_interface); // jacobian (1d) times quadrature
                                             // weight
                    for (int igll_edge = 0;
                         igll_edge < ContainerType::NGLL_EDGE; igll_edge++) {
                      mt_derivs1[igll_edge][igll_interface] =
                          container
                              .h_interface_medium1_mortar_transfer_deriv_times_n(
                                  iinterface, igll_interface, igll_edge);
                      mt_derivs2[igll_edge][igll_interface] =
                          container
                              .h_interface_medium2_mortar_transfer_deriv_times_n(
                                  iinterface, igll_interface, igll_edge);
                      type_real hprime = assembly.mesh.quadratures.gll.h_hprime(
                          against_edge1_index, against_edge1_index);
                      type_real contranorm = container.h_medium1_edge_normal(
                          edge1_index, igll_edge, against_edge1_component);
                      type_real dv =
                          (container
                               .h_interface_medium1_mortar_transfer_deriv_times_n(
                                   iinterface, igll_interface,
                                   igll_edge) // along-edge
                                              // direction
                           +
                           container.h_interface_medium1_mortar_transfer(
                               iinterface, igll_interface,
                               igll_edge) // against-edge
                                          // component
                               * container.h_medium1_edge_normal(
                                     edge1_index, igll_edge,
                                     against_edge1_component)
                               // hprime(i,j) = L_j'(t_i)
                               * assembly.mesh.quadratures.gll.h_hprime(
                                     against_edge1_index, against_edge1_index));
                      type_real dv1 = dv;
                      // copy down edge fluxes
                      //  flux += (
                      //        np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                      //      + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                      //      - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                      //  )
                      accel1[igll_edge].acceleration(0) +=
                          Jw *
                          (ujmp * half_c1 * dv +
                           0.5 *
                               container.h_interface_medium1_mortar_transfer(
                                   iinterface, igll_interface, igll_edge) *
                               (cdu_avg -
                                ujmp *
                                    container.h_interface_relaxation_parameter(
                                        iinterface)));
                      flux1[igll_edge][igll_interface] = ujmp * half_c1 * dv;
                      flux2[igll_edge][igll_interface] =
                          0.5 *
                          container.h_interface_medium1_mortar_transfer(
                              iinterface, igll_interface, igll_edge) *
                          cdu_avg;
                      flux3[igll_edge][igll_interface] =
                          -0.5 *
                          container.h_interface_medium1_mortar_transfer(
                              iinterface, igll_interface, igll_edge) *
                          ujmp *
                          container.h_interface_relaxation_parameter(
                              iinterface);
                      if (igll_edge == igll_interface) {
                        flux1a[igll_edge] = flux1[igll_edge][igll_interface];
                        flux2a[igll_edge] = flux2[igll_edge][igll_interface];
                        flux3a[igll_edge] = flux3[igll_edge][igll_interface];
                      }
                      dv =
                          (container
                               .h_interface_medium2_mortar_transfer_deriv_times_n(
                                   iinterface, igll_interface,
                                   igll_edge) // along-edge
                                              // direction
                           +
                           container.h_interface_medium2_mortar_transfer(
                               iinterface, igll_interface,
                               igll_edge) // against-edge
                                          // component
                               * container.h_medium1_edge_normal(
                                     edge2_index, igll_edge,
                                     against_edge2_component)
                               // hprime(i,j) = L_j'(t_i)
                               * assembly.mesh.quadratures.gll.h_hprime(
                                     against_edge2_index, against_edge2_index));
                      accel2[igll_edge].acceleration(0) +=
                          Jw *
                          (-ujmp * half_c2 * dv -
                           0.5 *
                               container.h_interface_medium2_mortar_transfer(
                                   iinterface, igll_interface, igll_edge) *
                               (cdu_avg -
                                ujmp *
                                    container.h_interface_relaxation_parameter(
                                        iinterface)));
                      if (igll_edge == igll_interface) {
                        flux1b[igll_edge] = -ujmp * half_c1 * dv;
                        flux2b[igll_edge] =
                            -0.5 *
                            container.h_interface_medium2_mortar_transfer(
                                iinterface, igll_interface, igll_edge) *
                            cdu_avg;
                        flux3b[igll_edge] =
                            0.5 *
                            container.h_interface_medium2_mortar_transfer(
                                iinterface, igll_interface, igll_edge) *
                            ujmp *
                            container.h_interface_relaxation_parameter(
                                iinterface);
                      }
                    }
                  }

                  container.h_interface_relaxation_parameter(iinterface);
                }
              }

              const auto h_is_bdry_at_pt =
                  [&](const specfem::point::index<
                          specfem::dimension::type::dim2>
                          index,
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
                  throw std::runtime_error(
                      "h_is_bdry_at_pt: unknown "
                      "assembly->boundaries.boundary_tags(ispec) "
                      "value!");
                  return false;
                }
              };

              index.ispec = edge1_ispec;
              for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                   igll_edge++) {
                specfem::compute::loose::point_from_edge<
                    ContainerType::NGLL_EDGE>(index.iz, index.ix, edge1_type,
                                              igll_edge);
                if (!h_is_bdry_at_pt(index,
                                     specfem::element::boundary_tag::none)) {
                  accel1[igll_edge].acceleration(0) = 0;
                }
              }
              container.template atomic_add_to_field<1, false>(
                  edge1_index, assembly, accel1);

              index.ispec = edge2_ispec;
              for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
                   igll_edge++) {
                specfem::compute::loose::point_from_edge<
                    ContainerType::NGLL_EDGE>(index.iz, index.ix, edge2_type,
                                              igll_edge);
                if (!h_is_bdry_at_pt(index,
                                     specfem::element::boundary_tag::none)) {
                  accel2[igll_edge].acceleration(0) = 0;
                }
              }
              container.template atomic_add_to_field<2, false>(
                  edge2_index, assembly, accel2);
            });
  }

  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_edge_intermediate(int index, specfem::compute::assembly &assembly,
                            ContainerType &container) {
    compute_field_nderiv<medium, on_device>(index, assembly, container);
  }

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_relaxation_parameter(int iinterface,
                               specfem::compute::assembly &assembly,
                               ContainerType &container) {
    if constexpr (on_device) {
      static_assert(on_device == false,
                    "on_device not written for compute_relaxation_paramter");
    }
    constexpr bool UseSIMD = false;
    using AcousticProperties = specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD>;
    auto spec_charlen2 = [&](int ispec) {
      type_real lx1 =
          assembly.mesh.points.coord(0, ispec, 0, 0) -
          assembly.mesh.points.coord(0, ispec, ContainerType::NGLL_EDGE - 1,
                                     ContainerType::NGLL_EDGE - 1);
      type_real lz1 =
          assembly.mesh.points.coord(1, ispec, 0, 0) -
          assembly.mesh.points.coord(1, ispec, ContainerType::NGLL_EDGE - 1,
                                     ContainerType::NGLL_EDGE - 1);
      type_real lx2 =
          assembly.mesh.points.coord(0, ispec, ContainerType::NGLL_EDGE - 1,
                                     0) -
          assembly.mesh.points.coord(0, ispec, 0, ContainerType::NGLL_EDGE - 1);
      type_real lz2 =
          assembly.mesh.points.coord(1, ispec, ContainerType::NGLL_EDGE - 1,
                                     0) -
          assembly.mesh.points.coord(1, ispec, 0, ContainerType::NGLL_EDGE - 1);
      return std::max(lx1 * lx1 + lz1 * lz1, lx2 * lx2 + lz2 * lz2);
    };

    const int edge1_index = container.h_interface_medium1_index(iinterface);
    const int edge2_index = container.h_interface_medium2_index(iinterface);

    const int edge1_ispec = container.h_medium1_index_mapping(edge1_index);
    const int edge2_ispec = container.h_medium2_index_mapping(edge2_index);
    type_real rho_inv_max = 0;
    AcousticProperties props;
    for (int iz = 0; iz < ContainerType::NGLL_EDGE; iz++) {
      for (int ix = 0; ix < ContainerType::NGLL_EDGE; ix++) {
        specfem::point::index<specfem::dimension::type::dim2> index(edge1_ispec,
                                                                    iz, ix);
        specfem::compute::load_on_host(index, assembly.properties, props);
        rho_inv_max = std::max(rho_inv_max, props.rho_inverse);
        index.ispec = edge2_ispec;
        specfem::compute::load_on_host(index, assembly.properties, props);
        rho_inv_max = std::max(rho_inv_max, props.rho_inverse);
      }
    }

    container.h_interface_relaxation_parameter(iinterface) =
        _RELAX_PARAM_COEF_ACOUSTIC_ * rho_inv_max /
        sqrt(std::max(spec_charlen2(edge1_ispec), spec_charlen2(edge2_ispec)));
  }
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_mortar_trans_deriv(int iinterface,
                             specfem::compute::assembly &assembly,
                             ContainerType &container) {
    if constexpr (on_device) {
      static_assert(on_device == false,
                    "on_device not written for compute_relaxation_paramter");
    }
    constexpr bool UseSIMD = false;
    using AcousticProperties = specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD>;
    const int edge1_index = container.h_interface_medium1_index(iinterface);
    const int edge2_index = container.h_interface_medium2_index(iinterface);
    const auto edge1_type = container.h_medium1_edge_type(edge1_index);
    const auto edge2_type = container.h_medium2_edge_type(edge2_index);

    const int along_edge1_component =
        (edge1_type == specfem::enums::edge::type::RIGHT ||
         edge1_type == specfem::enums::edge::type::LEFT)
            ? 1
            : 0;
    const int along_edge2_component =
        (edge2_type == specfem::enums::edge::type::RIGHT ||
         edge2_type == specfem::enums::edge::type::LEFT)
            ? 1
            : 0;
    type_real derivs[ContainerType::NGLL_EDGE];
    for (int iedge = 0; iedge < ContainerType::NGLL_EDGE; iedge++) {
      for (int igll = 0; igll < ContainerType::NGLL_INTERFACE; igll++) {
        container.h_interface_medium1_mortar_transfer_deriv_times_n(
            iinterface, igll, iedge) = 0;
        container.h_interface_medium2_mortar_transfer_deriv_times_n(
            iinterface, igll, iedge) = 0;
        for (int jedge = 0; jedge < ContainerType::NGLL_EDGE; jedge++) {
          container.h_interface_medium1_mortar_transfer_deriv_times_n(
              iinterface, igll, iedge) +=
              assembly.mesh.quadratures.gll.h_hprime(jedge, iedge) *
              container.h_interface_medium1_mortar_transfer(iinterface, igll,
                                                            jedge) *
              container.h_medium1_edge_contravariant_normal(
                  edge1_index, jedge, along_edge1_component);
          container.h_interface_medium2_mortar_transfer_deriv_times_n(
              iinterface, igll, iedge) +=
              assembly.mesh.quadratures.gll.h_hprime(jedge, iedge) *
              container.h_interface_medium2_mortar_transfer(iinterface, igll,
                                                            jedge) *
              container.h_medium1_edge_contravariant_normal(
                  edge2_index, jedge, along_edge2_component);
        }
      }
    }
  }
};
template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux::kernel<
    DimensionType, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic, QuadratureType> {
  using ContainerType = specfem::compute::loose::interface_container<
      DimensionType, specfem::element::medium_tag::elastic,
      specfem::element::medium_tag::elastic, QuadratureType,
      specfem::coupled_interface::loose::flux::type::symmetric_flux>;

  static void compute_fluxes(specfem::compute::assembly &assembly,
                             ContainerType &container) {}

  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_edge_intermediate(int index, specfem::compute::assembly &assembly,
                            ContainerType &container) {}
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
