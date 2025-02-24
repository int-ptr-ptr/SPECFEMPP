#include "compute/adjacencies/adjacency_map.hpp"

#include <deque>
#include <list>

static bool are_elements_intersecting(
    const specfem::kokkos::HostView4d<double> &global_coordinates,
    const int ispec1, const specfem::enums::edge::type edge1, const int ispec2,
    const specfem::enums::edge::type edge2) {
  throw std::runtime_error("Kentaro forgot to populate the "
                           "`are_elements_intersecting` "
                           "function. Call them a dingus for me.");
  return false;
}

namespace specfem {
namespace compute {
namespace adjacencies {

void adjacency_map::fill_nonconforming_adjacencies(
    const specfem::kokkos::HostView4d<double> &global_coordinates) {

  const int nspec = global_coordinates.extent(0);
  const int ngllz = global_coordinates.extent(1);
  const int ngllx = global_coordinates.extent(2);

  // we want to list out the elements

  // collect nonconforming edges (these are sorted by iglob).
  using edgelist = std::list<std::pair<int, specfem::enums::edge::type> >;
  edgelist remaining_edges;
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int iedge = 0; iedge < 4; iedge++) {
      if ((!has_conforming_adjacency<false>(ispec, iedge)) &&
          (!has_boundary<false>(ispec, iedge))) {
        remaining_edges.push_back(
            std::make_pair(ispec, edge_from_index(iedge)));
      }
    }
  }

  // consume nonconforming edges, tracing things out
  while (!remaining_edges.empty()) {
    // trace involves either BFS or DFS. either way, let's just use a deque
    std::deque<edgelist::iterator> edge_search;
    edge_search.push_back(remaining_edges.begin());
    while (!edge_search.empty()) {
      const auto rem_edge = edge_search.back();
      edge_search.pop_back();
      int ispec;
      specfem::enums::edge::type edge;
      std::tie(ispec, edge) = *rem_edge;

      // this edge was completely resolved.
      remaining_edges.erase(rem_edge);
    }
  }
  throw std::runtime_error("Kentaro forgot to finish populating the "
                           "`adjacency_map::fill_nonconforming_adjacencies` "
                           "method. Call them a dingus for me.");
  h_nonconforming_element_anchors =
      NonconformingElementAnchorViewType::HostMirror(
          "specfem::compute::adjacencies::adjacency_map::nonconforming_edges",
          0);
  h_nonconforming_edges = NonconformingEdgeViewType::HostMirror(
      "specfem::compute::adjacencies::adjacency_map::nonconforming_edges", 0);
}

} // namespace adjacencies
} // namespace compute
} // namespace specfem
