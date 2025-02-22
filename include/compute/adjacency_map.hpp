#pragma once

#include "Kokkos_Macros.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace specfem {
namespace compute {
namespace adjacencies {

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
  edge_to_index(const int edge) {
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

  using IspecViewType =
      Kokkos::View<int *[4], Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using EdgeViewType =
      Kokkos::View<specfem::enums::edge::type *[4], Kokkos::LayoutLeft,
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
  KOKKOS_INLINE_FUNCTION void create_conforming_adjacency(
      const int ispec1, const specfem::enums::edge::type edge1,
      const int ispec2, const specfem::enums::edge::type edge2) {
    int edge1_ = edge_to_index(edge1);
    int edge2_ = edge_to_index(edge2);
    if constexpr (on_device) {
      adjacent_indices(ispec1, edge1_) = ispec2;
      adjacent_indices(ispec2, edge2_) = ispec1;
      adjacent_edges(ispec1, edge1_) = edge2;
      adjacent_edges(ispec2, edge2_) = edge1;
    } else {
      h_adjacent_indices(ispec1, edge1_) = ispec2;
      h_adjacent_indices(ispec2, edge2_) = ispec1;
      h_adjacent_edges(ispec1, edge1_) = edge2;
      h_adjacent_edges(ispec2, edge2_) = edge1;
    }
  }

  void fill_nonconforming_adjacencies(
      const specfem::kokkos::HostView4d<double> &global_coordinates) {
    throw std::runtime_error("Kentaro forgot to populate the "
                             "`adjacency_map::fill_nonconforming_adjacencies` "
                             "function. Call them a dingus for me.");
  }

  static inline bool are_elements_conforming(
      const specfem::kokkos::HostView4d<double> &global_coordinates,
      const int ispec1, const specfem::enums::edge::type edge1,
      const int ispec2, const specfem::enums::edge::type edge2,
      type_real tolerance) {
    const int ngllx = global_coordinates.extent(2);
    const int ngllz = global_coordinates.extent(1);
    /*     Computing parity:
     * To compare positions, we need to know which node corresponds on either
     * side. We could take the minimum among node reorderings (flip or not), but
     * there will always be one specific state of flipping or not, since only
     * rotations are allowed.
     *
     * If two edges align, then the coordinate systems on either side may be in
     * the same or opposite directions. If we denote the direction from negative
     * to positive local coordinate as positive edge orientation, two edges of
     * the same type mating has parity one, since the 180 degree rotation of one
     * element flips the system. Similarly, if two edges of opposite sides mate,
     * the parity is zero, since the two elements can be joined without any
     * rotation.
     *
     * We can break this down into the following rule. If we index the edges as
     * above, where edge1 -> i and edge2 -> j, we can follow the table (using
     * mod 4 arithmetic): i    j-i    parity
     * -------------------
     *  i     0       1
     *  i     2       0
     *  0     1       0
     *  1     1       1
     *  2     1       0
     *  3     1       1
     *
     * These can be verified easily on paper, but this is (unless I made a
     * mistake) equivalent to what we have below.
     */
    int erot1 = edge_to_index(edge1);
    int rotdif = (edge_to_index(edge2) - erot1 + 4) %
                 4; // +4 since % doesn't "like" negatives.
    const bool parity =
        (rotdif % 2 == 0) ? (1 - rotdif / 2) : ((erot1 + rotdif / 2) % 2);

    if ((rotdif % 2 == 1) && ngllx != ngllz) {
      // unequal nodes in x and z, there is no way a 90 degree rotation would be
      // conforming
      return false;
    }

    int x1, z1, x2, z2;
    // set the static coordinate; the other will be modified in the loop:
    if (edge1 == specfem::enums::edge::type::TOP ||
        edge1 == specfem::enums::edge::type::RIGHT) {
      x1 = ngllx - 1;
      z1 = ngllz - 1;
    } else {
      x1 = 0;
      z1 = 0;
    }
    if (edge2 == specfem::enums::edge::type::TOP ||
        edge2 == specfem::enums::edge::type::RIGHT) {
      x2 = ngllx - 1;
      z2 = ngllz - 1;
    } else {
      x2 = 0;
      z2 = 0;
    }

    // left/right: move along z; top/bottom: move along x
    const int ngll = (erot1 % 2 == 0) ? ngllz : ngllx;
    for (int iedge = 0; iedge < ngllz; iedge++) {
      if (edge1 == specfem::enums::edge::type::LEFT ||
          edge1 == specfem::enums::edge::type::RIGHT) {
        z1 = iedge;
      } else {
        x1 = iedge;
      }
      if (edge2 == specfem::enums::edge::type::LEFT ||
          edge2 == specfem::enums::edge::type::RIGHT) {
        z2 = iedge;
      } else {
        x2 = iedge;
      }
      // points within tolerance?
      if ((std::abs(global_coordinates(ispec1, z1, x1, 0) -
                    global_coordinates(ispec2, z2, x2, 0)) > tolerance) ||
          (std::abs(global_coordinates(ispec1, z1, x1, 1) -
                    global_coordinates(ispec2, z2, x2, 1)) > tolerance)) {
        return false;
      }
    }
    return true;
  }
};

} // namespace adjacencies
} // namespace compute
} // namespace specfem
