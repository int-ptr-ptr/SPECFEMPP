#pragma once

#include "control_nodes.hpp"
#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "shape_functions.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

template <> struct points<specfem::dimension::type::dim3> {
public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  using IndexMappingViewType =
      Kokkos::View<int ****, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< Index view type
  using CoordViewType =
      Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>; ///< Coordinate view type
private:
  constexpr static int ndim = 3; ///< Number of dimensions

public:
  IndexMappingViewType index_mapping; ///< Index mapping from local to global
                                      ///< indices
  IndexMappingViewType::HostMirror h_index_mapping; ///< Host mirror of index
                                                    ///< mapping

  CoordViewType coord; ///< Coordinate view for every quadrature point
  CoordViewType::HostMirror h_coord; ///< Host mirror of coordinate view

  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z dimension
  int nglly; ///< Number of quadrature points in y dimension
  int ngllx; ///< Number of quadrature points in x dimension

  int nglob; ///< Total number of global points

  points() = default;
  points(const specfem::mesh::mapping<dimension_tag> &mapping,
         const specfem::mesh::coordinates<dimension_tag> &coordinates);

  points(const int &nspec, const int &ngllz, const int &nglly, const int &ngllx,
         const specfem::mesh::meshfem3d::adjacency_graph<dimension_tag>
             &adjacency_graph,
         const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
             &control_nodes,
         const specfem::assembly::mesh_impl::shape_functions<dimension_tag>
             &shape_functions);
};

} // namespace specfem::assembly::mesh_impl
