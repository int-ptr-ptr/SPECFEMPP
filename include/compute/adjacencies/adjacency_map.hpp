#pragma once

#include "Kokkos_Macros.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace specfem {
namespace compute {
namespace adjacencies {

struct nonconforming_edge {
  int edgeL, edgeR;
  type_real param_startL, param_startR;
  type_real param_endL, param_endR;

  nonconforming_edge() = default;
};

/**
 * @brief Stores the adjacencies between elements.
 *
 */
struct adjacency_map {
  static KOKKOS_INLINE_FUNCTION int
  edge_to_index(const specfem::enums::edge::type edge) {
    switch (edge) {
    case specfem::enums::edge::RIGHT:
      return 0;
    case specfem::enums::edge::TOP:
      return 1;
    case specfem::enums::edge::LEFT:
      return 2;
    case specfem::enums::edge::BOTTOM:
      return 3;
    default:
      return 0; // this should never be called
    }
  }
  static KOKKOS_INLINE_FUNCTION specfem::enums::edge::type
  edge_from_index(const int edge) {
    switch (edge) {
    case 0:
      return specfem::enums::edge::RIGHT;
    case 1:
      return specfem::enums::edge::TOP;
    case 2:
      return specfem::enums::edge::LEFT;
    case 3:
      return specfem::enums::edge::BOTTOM;
    default:
      return specfem::enums::edge::NONE;
    }
  }

  adjacency_map() = default;
  adjacency_map(int nspec)
      : adjacent_indices("specfem::compute::adjacency_map::adjacent_indices",
                         nspec),
        h_adjacent_indices(Kokkos::create_mirror_view(adjacent_indices)),
        adjacent_edges("specfem::compute::adjacency_map::adjacent_edges",
                       nspec),
        h_adjacent_edges(Kokkos::create_mirror_view(adjacent_edges)) {
    for (int i = 0; i < nspec; i++) {
      for (int j = 0; j < 4; j++) {
        h_adjacent_edges(i, j) = specfem::enums::edge::type::NONE;
      }
    }
    Kokkos::deep_copy(adjacent_edges, h_adjacent_edges);
  }

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION bool
  has_conforming_adjacency(const int ispec,
                           const specfem::enums::edge::type edge);
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION bool has_conforming_adjacency(const int ispec,
                                                       const int edge);
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION bool
  has_boundary(const int ispec, const specfem::enums::edge::type edge);
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION bool has_boundary(const int ispec, const int edge);

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec,
                           const specfem::enums::edge::type edge) const;
  template <bool on_device>
  KOKKOS_INLINE_FUNCTION std::pair<int, specfem::enums::edge::type>
  get_conforming_adjacency(const int ispec, const int edge) const;

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION void create_conforming_adjacency(
      const int ispec1, const specfem::enums::edge::type edge1,
      const int ispec2, const specfem::enums::edge::type edge2);

  template <bool on_device>
  KOKKOS_INLINE_FUNCTION void
  set_as_boundary(const int ispec, const specfem::enums::edge::type edge);

  void fill_nonconforming_adjacencies(
      const specfem::kokkos::HostView4d<double> &global_coordinates);

  static inline bool are_elements_conforming(
      const specfem::kokkos::HostView4d<double> &global_coordinates,
      const int ispec1, const specfem::enums::edge::type edge1,
      const int ispec2, const specfem::enums::edge::type edge2,
      type_real tolerance);

private:
  struct nonconforming_element_anchor {
    // element identifier
    int ispec;
    specfem::enums::edge::type edge;

    // nonconforming edge identifiers
    int edge_plus;  // edge at +1 local coordinate
    int edge_minus; // edge at -1 local coordinate

    bool side_plus;  // true if left side of edge at +1 local coordinate
    bool side_minus; // true if left side of edge at -1 local coordinate

    nonconforming_element_anchor()
        : ispec(-1), edge(specfem::enums::edge::type::NONE), edge_plus(-1),
          edge_minus(-1) {}
  };
  using IspecViewType =
      Kokkos::View<int *[4], Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using EdgeViewType =
      Kokkos::View<specfem::enums::edge::type *[4], Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;
  using NonconformingElementAnchorViewType =
      Kokkos::View<nonconforming_element_anchor *, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;
  using NonconformingEdgeViewType =
      Kokkos::View<nonconforming_edge *, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;
  /* ---- adjacency storage ----
   * Each pair `(ispec, edge)` follows the following rule:
   * - If `adjacent_edges(ispec,edge)` is an edge (not NONE), then
   *   the `(adjecent_indices(ispec,edge), adjacent_edges(ispec,edge))` pair
   * shares a conforming edge to `(ispec, edge)`. Edges are conforming iff the
   * nodes along that edge share the same coordinates. This has no awareness of
   * any material decoupling.
   *
   * - If `adjacent_edges(ispec,edge)` is NONE, then the edge does not have a
   * conforming interface. In this case, if `adjecent_indices(ispec,edge)` is
   * negative, then this is a boundary, and no other edge shares an adjacency.
   * Otherwise, `adjecent_indices(ispec,edge)` is an index to <insert data
   * structure> for an interface.
   */
  IspecViewType adjacent_indices;
  IspecViewType::HostMirror h_adjacent_indices;
  EdgeViewType adjacent_edges;
  EdgeViewType::HostMirror h_adjacent_edges;

  NonconformingElementAnchorViewType::HostMirror
      h_nonconforming_element_anchors;
  NonconformingEdgeViewType::HostMirror h_nonconforming_edges;

public:
  void full_sync_to_device() {
    Kokkos::deep_copy(adjacent_edges, h_adjacent_edges);
    Kokkos::deep_copy(adjacent_indices, h_adjacent_indices);
  }
  void full_sync_to_host() {
    Kokkos::deep_copy(h_adjacent_edges, adjacent_edges);
    Kokkos::deep_copy(h_adjacent_indices, adjacent_indices);
  }
};

} // namespace adjacencies
} // namespace compute
} // namespace specfem

#include "adjacency_map.tpp"
