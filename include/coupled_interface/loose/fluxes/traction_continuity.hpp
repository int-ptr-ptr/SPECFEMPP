#pragma once

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

struct traction_continuity {

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct container;

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct kernel;
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem

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
      traction_continuity>;

  static void elastic_to_acoustic_accel(specfem::compute::assembly &assembly,
                                        ContainerType &container);
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
            container.h_interface_surface_jacobian(iinterface, igll_interface) *
            assembly.mesh.quadratures.gll.h_weights(igll_interface);
        for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
             igll_edge++) {
          type_real chitt_wv_dS =
              chitt_w_dS * container.h_interface_medium1_mortar_transfer(
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
            index.ix, index.iz, edge2_type, igll_edge);
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
