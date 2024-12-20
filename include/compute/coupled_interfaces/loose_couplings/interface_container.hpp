#pragma once

#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {
namespace loose {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType,
          typename FluxScheme>
struct interface_container : FluxScheme::container<DimensionType, MediumTag1,
                                                   MediumTag2, QuadratureType> {
private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types
  using RealView = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static specfem::element::medium_tag medium1_type =
      MediumTag1; ///< Self medium of the interface
  constexpr static specfem::element::medium_tag medium2_type =
      MediumTag2; ///< Other medium of the interface

  interface_container() = default;
  interface_container(const int num_medium1_edges, const int num_medium2_edges);

  int num_medium1_edges;
  int num_medium2_edges;
  int num_interfaces;

  IndexView medium1_index_mapping; ///< Spectral element index for every edge on
                                   ///< self medium
  IndexView medium2_index_mapping; ///< Spectral element index for every edge on
                                   ///< other medium

  EdgeTypeView medium1_edge_type; ///< Edge orientation for every edge on self
                                  ///< medium
  EdgeTypeView medium2_edge_type; ///< Edge orientation for every edge on other
                                  ///< medium

  IndexView::HostMirror h_medium1_index_mapping; ///< Host mirror for @ref
                                                 ///< medium1_index_mapping
  IndexView::HostMirror h_medium2_index_mapping; ///< Host mirror for @ref
                                                 ///< medium2_index_mapping

  EdgeTypeView::HostMirror h_medium1_edge_type; ///< Host mirror for @ref
                                                ///< medium1_edge_type
  EdgeTypeView::HostMirror h_medium2_edge_type; ///< Host mirror for @ref
                                                ///< medium2_edge_type

  /// reference indices to edges on the medium1 and medium2 sides
  IndexView interface_medium1_index;
  IndexView interface_medium2_index;
  IndexView::HostMirror h_interface_medium1_index;
  IndexView::HostMirror h_interface_medium2_index;

  /// start and end parameters for edges
  RealView interface_medium1_param_start;
  RealView interface_medium1_param_end;
  RealView interface_medium2_param_start;
  RealView interface_medium2_param_end;
  RealView::HostMirror h_interface_medium1_param_start;
  RealView::HostMirror h_interface_medium1_param_end;
  RealView::HostMirror h_interface_medium2_param_start;
  RealView::HostMirror h_interface_medium2_param_end;

  // These are temporary until we figure out where to put them all
  using EdgeScalarView = Kokkos::View<type_real * [QuadratureType::NGLL],
                                      Kokkos::DefaultExecutionSpace>;
  using EdgeVectorView =
      Kokkos::View<type_real *
                       [QuadratureType::NGLL]
                           [specfem::dimension::dimension<DimensionType>::dim],
                   Kokkos::DefaultExecutionSpace>;
  using EdgeQuadView =
      Kokkos::View<type_real * [QuadratureType::NGLL][QuadratureType::NGLL],
                   Kokkos::DefaultExecutionSpace>;

  EdgeVectorView a_POSITION;
  EdgeVectorView b_POSITION;
  typename EdgeVectorView::HostMirror h_a_POSITION;
  typename EdgeVectorView::HostMirror h_b_POSITION;
  EdgeVectorView a_NORMAL;
  EdgeVectorView b_NORMAL;
  typename EdgeVectorView::HostMirror h_a_NORMAL;
  typename EdgeVectorView::HostMirror h_b_NORMAL;
  EdgeScalarView a_DET;
  EdgeScalarView b_DET;
  typename EdgeScalarView::HostMirror h_a_DET;
  typename EdgeScalarView::HostMirror h_b_DET;
  EdgeScalarView a_DS;
  EdgeScalarView b_DS;
  typename EdgeScalarView::HostMirror h_a_DS;
  typename EdgeScalarView::HostMirror h_b_DS;
  EdgeVectorView a_FIELD;
  EdgeVectorView b_FIELD;
  typename EdgeVectorView::HostMirror h_a_FIELD;
  typename EdgeVectorView::HostMirror h_b_FIELD;
  EdgeVectorView a_FIELDNDERIV;
  EdgeVectorView b_FIELDNDERIV;
  typename EdgeVectorView::HostMirror h_a_FIELDNDERIV;
  typename EdgeVectorView::HostMirror h_b_FIELDNDERIV;
  EdgeScalarView a_SPEEDPARAM;
  EdgeScalarView b_SPEEDPARAM;
  typename EdgeScalarView::HostMirror h_a_SPEEDPARAM;
  typename EdgeScalarView::HostMirror h_b_SPEEDPARAM;
  EdgeQuadView a_SHAPENDERIV;
  EdgeQuadView b_SHAPENDERIV;
  typename EdgeQuadView::HostMirror h_a_SHAPENDERIV;
  typename EdgeQuadView::HostMirror h_b_SHAPENDERIV;
};

} // namespace loose
} // namespace compute
} // namespace specfem
