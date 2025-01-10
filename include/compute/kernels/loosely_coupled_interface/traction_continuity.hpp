#pragma once

#include "compute/coupled_interfaces/loose_couplings/interface_container.hpp"
#include "compute/coupled_interfaces/loose_couplings/traction_continuity_container.hpp"

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {
template <specfem::dimension::type DimensionType, typename QuadratureType>
struct traction_continuity::kernel<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic, QuadratureType> {
  using ContainerType = specfem::compute::loose::interface_container<
      DimensionType, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::elastic, QuadratureType,
      specfem::coupled_interface::loose::flux::type::traction_continuity>;

  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_edge_intermediate(int index, specfem::compute::assembly &assembly,
                            ContainerType &container) {
    if constexpr (medium == 1) {
      return;
    } else if constexpr (medium != 2) {
      static_assert(false, "Medium can only be 1 or 2!");
    }
    constexpr bool UseSIMD = false;
    using ElasticDispType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::elastic, true,
                              false, false, false, UseSIMD>;
    ElasticDispType disp[ContainerType::NGLL_EDGE];

#define _disp_dot_normal                                                       \
  [&] {                                                                        \
    if constexpr (on_device) {                                                 \
      return container.disp_dot_normal;                                        \
    } else {                                                                   \
      return container.h_disp_dot_normal;                                      \
    }                                                                          \
  }()
#define _normal                                                                \
  [&] {                                                                        \
    if constexpr (on_device) {                                                 \
      return container.medium2_edge_normal;                                    \
    } else {                                                                   \
      return container.h_medium2_edge_normal;                                  \
    }                                                                          \
  }()

    container.template load_field<2, on_device>(index, assembly, disp);
    for (int igll = 0; igll < ContainerType::NGLL_EDGE; igll++) {
      _disp_dot_normal(index, igll) =
          _normal(index, igll, 0) * disp[igll].displacement(0) +
          _normal(index, igll, 1) * disp[igll].displacement(1);
    }

#undef _normal
#undef _disp_dot_normal
  }

  static void elastic_to_acoustic_accel(specfem::compute::assembly &assembly,
                                        ContainerType &container) {
    constexpr bool UseSIMD = false;
    using AcousticAccelType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::acoustic, false,
                              false, true, false, UseSIMD>;

    // we already computed displacement dot normal
    container.template foreach_interface<false>([&](const int iinterface) {
      AcousticAccelType acoustic[ContainerType::NGLL_EDGE];

      const int edge1_index = container.h_interface_medium1_index(iinterface);
      const int edge2_index = container.h_interface_medium2_index(iinterface);

      const int edge1_ispec = container.h_medium1_index_mapping(edge1_index);
      const auto edge1_type = container.h_medium1_edge_type(edge1_index);

      type_real sn[ContainerType::NGLL_EDGE];

      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        acoustic[igll_edge].acceleration(0) = 0;
        sn[igll_edge] = container.h_disp_dot_normal(edge2_index, igll_edge);
      }
      for (int igll_interface = 0;
           igll_interface < ContainerType::NGLL_INTERFACE; igll_interface++) {
        type_real sn_w_dS = -container.template edge_to_mortar<2, false>(
                                edge2_index, igll_interface, sn) *
                            container.h_interface_surface_jacobian_times_weight(
                                iinterface, igll_interface);
        for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
             igll_edge++) {
          acoustic[igll_edge].acceleration(0) +=
              sn_w_dS * container.h_interface_medium1_mortar_transfer(
                            igll_interface, igll_interface, igll_edge);
        }
      }

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
          throw std::runtime_error("h_is_bdry_at_pt: unknown "
                                   "assembly->boundaries.boundary_tags(ispec) "
                                   "value!");
          return false;
        }
      };

      specfem::point::index<specfem::dimension::type::dim2> index(edge1_ispec,
                                                                  0, 0);

      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
            index.iz, index.ix, edge1_type, igll_edge);
        if (!h_is_bdry_at_pt(index, specfem::element::boundary_tag::none)) {
          acoustic[igll_edge].acceleration(0) = 0;
        }
      }

      container.template atomic_add_to_field<1, false>(edge1_index, assembly,
                                                       acoustic);
    });
  }
  static void acoustic_to_elastic_accel(specfem::compute::assembly &assembly,
                                        ContainerType &container) {

    constexpr bool UseSIMD = false;
    using AcousticAccelType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::acoustic, false,
                              false, true, false, UseSIMD>;
    using ElasticAccelType =
        specfem::point::field<DimensionType,
                              specfem::element::medium_tag::elastic, false,
                              false, true, false, UseSIMD>;
    // foreach edge
    container.template foreach_interface<false>([&](const int iinterface) {
      AcousticAccelType acoustic[ContainerType::NGLL_EDGE];
      ElasticAccelType elastic[ContainerType::NGLL_INTERFACE];

      const int edge1_index = container.h_interface_medium1_index(iinterface);
      const int edge2_index = container.h_interface_medium2_index(iinterface);

      const int edge2_ispec = container.h_medium2_index_mapping(edge2_index);
      const auto edge2_type = container.h_medium2_edge_type(edge2_index);

      container.template load_field<1, false>(edge1_index, assembly, acoustic);

      type_real accel[ContainerType::NGLL_EDGE];

      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        elastic[igll_edge].acceleration(0) = 0;
        elastic[igll_edge].acceleration(1) = 0;
        accel[igll_edge] = acoustic[igll_edge].acceleration(0);
      }
      for (int igll_interface = 0;
           igll_interface < ContainerType::NGLL_INTERFACE; igll_interface++) {
        type_real chitt_w_dS =
            container.template edge_to_mortar<1, false>(edge1_index,
                                                        igll_interface, accel) *
            container.h_interface_surface_jacobian_times_weight(iinterface,
                                                                igll_interface);
        for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
             igll_edge++) {
          type_real chitt_wv_dS =
              chitt_w_dS * container.h_interface_medium2_mortar_transfer(
                               igll_interface, igll_interface, igll_edge);
          elastic[igll_edge].acceleration(0) +=
              chitt_wv_dS *
              container.h_medium2_edge_normal(edge1_index, igll_edge, 0);
          elastic[igll_edge].acceleration(1) +=
              chitt_wv_dS *
              container.h_medium2_edge_normal(edge1_index, igll_edge, 1);
        }
      }

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
          throw std::runtime_error("h_is_bdry_at_pt: unknown "
                                   "assembly->boundaries.boundary_tags(ispec) "
                                   "value!");
          return false;
        }
      };

      specfem::point::index<specfem::dimension::type::dim2> index(edge2_ispec,
                                                                  0, 0);
      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
            index.iz, index.ix, edge2_type, igll_edge);
        if (!h_is_bdry_at_pt(index, specfem::element::boundary_tag::none)) {
          elastic[igll_edge].acceleration(0) = 0;
          elastic[igll_edge].acceleration(1) = 0;
        }
      }

      container.template atomic_add_to_field<2, false>(edge2_index, assembly,
                                                       elastic);
    });
  }
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
