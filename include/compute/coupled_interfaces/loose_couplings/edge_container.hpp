#pragma once

#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {
namespace loose {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType>
struct interface_container {
private:
  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types
public:
  constexpr static specfem::element::medium_tag medium1_type =
      MediumTag1; ///< Self medium of the interface
  constexpr static specfem::element::medium_tag medium2_type =
      MediumTag2; ///< Other medium of the interface

  interface_container() = default;
  interface_container(const int num_medium1_edges, const int num_medium2_edges);

  int num_medium1_edges;
  int num_medium2_edges;

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
};

} // namespace loose
} // namespace compute
} // namespace specfem
