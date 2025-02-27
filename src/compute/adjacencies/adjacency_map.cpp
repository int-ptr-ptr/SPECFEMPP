#include "compute/adjacencies/adjacency_map.hpp"

#include <deque>
#include <list>

#include "intersection_check.cpp"

static bool are_elements_intersecting(
    const int ngllx, const int ngllz,
    const specfem::kokkos::HostView4d<double> &global_coordinates,
    const int ispec1, const specfem::enums::edge::type type1, const int ispec2,
    const specfem::enums::edge::type type2, const int edge1, const int edge2,
    specfem::compute::adjacencies::nonconforming_edge &edge_instance) {

  const bool a_along_x = (type1 == specfem::enums::edge::type::BOTTOM ||
                          type1 == specfem::enums::edge::type::TOP);
  const bool b_along_x = (type2 == specfem::enums::edge::type::BOTTOM ||
                          type2 == specfem::enums::edge::type::TOP);
  const int nglla = a_along_x ? ngllx : ngllz;
  const int ngllb = b_along_x ? ngllx : ngllz;
  std::vector<type_real> ax(nglla);
  std::vector<type_real> az(nglla);
  std::vector<type_real> bx(ngllb);
  std::vector<type_real> bz(ngllb);
  int ix, iz;
  ix = iz = (type1 == specfem::enums::edge::type::BOTTOM ||
             type1 == specfem::enums::edge::type::LEFT)
                ? 0
                : nglla;

  for (int iedge = 0; iedge < nglla; iedge++) {
    if (a_along_x) {
      ix = iedge;
    } else {
      iz = iedge;
    }
    ax[iedge] = global_coordinates(ispec1, iz, ix, 0);
    az[iedge] = global_coordinates(ispec1, iz, ix, 1);
  }
  ix = iz = (type2 == specfem::enums::edge::type::BOTTOM ||
             type2 == specfem::enums::edge::type::LEFT)
                ? 0
                : ngllb;
  for (int iedge = 0; iedge < ngllb; iedge++) {
    if (b_along_x) {
      ix = iedge;
    } else {
      iz = iedge;
    }
    bx[iedge] = global_coordinates(ispec2, iz, ix, 0);
    bz[iedge] = global_coordinates(ispec2, iz, ix, 1);
  }
  int erot1 =
      specfem::compute::adjacencies::adjacency_map::edge_to_index(type1);
  int rotdif =
      (specfem::compute::adjacencies::adjacency_map::edge_to_index(type2) -
       erot1 + 4) %
      4; // +4 since % doesn't "like" negatives.
  const bool parity =
      (rotdif % 2 == 0) ? (1 - rotdif / 2) : ((erot1 + rotdif / 2) % 2);
  // for now, just align edge to a.
  const bool a_left_side = type1 == specfem::enums::edge::type::BOTTOM ||
                           type1 == specfem::enums::edge::type::RIGHT;
  if (intersect(ax.data(), az.data(), nglla, bx.data(), bz.data(), ngllb,
                edge_instance, a_left_side, false, parity)) {
    edge_instance.edgeL = a_left_side ? edge1 : edge2;
    edge_instance.edgeR = a_left_side ? edge2 : edge1;
    return true;
  }
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
  // |remaining_edges| is the number of anchors.
  const int num_edges = remaining_edges.size();
  h_nonconforming_element_anchors =
      NonconformingElementAnchorViewType::HostMirror(
          "specfem::compute::adjacencies::adjacency_map::nonconforming_element_"
          "anchors",
          num_edges);
  {
    int i = 0;
    for (const auto &[ispec, edge] : remaining_edges) {
      nonconforming_element_anchor &anchor = h_nonconforming_element_anchors(i);
      anchor.ispec = ispec;
      anchor.edge = edge;
      const int iedge = edge_to_index(edge);
      h_adjacent_edges(ispec, iedge) = specfem::enums::edge::type::NONE;
      h_adjacent_indices(ispec, iedge) = i;
      i++;
    }
  }

  std::vector<nonconforming_edge> intersections;

  nonconforming_edge edge_instance;
  for (int it1 = 0; it1 < num_edges; it1++) {
    for (int it2 = it1 + 1; it2 < num_edges; it2++) {
      if (are_elements_intersecting(ngllx, ngllz, global_coordinates,
                                    h_nonconforming_element_anchors(it1).ispec,
                                    h_nonconforming_element_anchors(it1).edge,
                                    h_nonconforming_element_anchors(it2).ispec,
                                    h_nonconforming_element_anchors(it2).edge,
                                    it1, it2, edge_instance)) {
        intersections.push_back(edge_instance);
      }
    }
  }
  const int num_intersections = intersections.size();
  h_nonconforming_edges = NonconformingEdgeViewType::HostMirror(
      "specfem::compute::adjacencies::adjacency_map::nonconforming_edges",
      num_intersections);
  for (int iinter = 0; iinter < num_intersections; iinter++) {
    h_nonconforming_edges(iinter) = intersections[iinter];
  }
}

} // namespace adjacencies
} // namespace compute
} // namespace specfem
