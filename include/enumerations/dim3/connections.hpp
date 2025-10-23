/**
 * @file connections.hpp
 * @brief 3D connection mapping for spectral element mesh entity transformations
 *
 * This file contains the template specialization for 3D connection mapping
 * functionality in the SPECFEM++ spectral element framework. The connection
 * mapping class provides coordinate transformation capabilities between
 * different mesh entities (faces, edges, corners) of adjacent hexahedral
 * spectral elements.
 *
 * The implementation handles:
 * - Face-to-face coordinate mapping with affine transformations
 * - Edge-to-edge coordinate mapping with permutation handling
 * - Corner-to-corner coordinate mapping for element interfaces
 * - Geometric transformations accounting for element orientations
 *
 * @see specfem::connections::connection_mapping
 * @see specfem::mesh_entity::dim3
 */

#pragma once

#include "../mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

/**
 * @brief 3D connection mapping for spectral element mesh entity transformations
 *
 * This template specialization provides coordinate mapping functionality
 * between mesh entities (faces, edges, corners) of adjacent 3D hexahedral
 * spectral elements. The class handles geometric transformations required for
 * coupling elements with different orientations in the spectral element method.
 *
 * The connection mapping is essential for:
 * - **Interface flux calculations**: Computing numerical fluxes across element
 * boundaries
 * - **Mesh conformity**: Handling non-conforming interfaces and h-p adaptivity
 * - **Parallel domain decomposition**: Managing element connections across MPI
 * boundaries
 *
 * **Mathematical Foundation:**
 * For face-to-face mapping, the transformation uses affine geometry:
 * \f[
 * \mathbf{x}_{to} = \mathbf{A} \cdot \mathbf{x}_{from} + \mathbf{b}
 * \f]
 * where \f$\mathbf{A}\f$ is the transformation matrix and \f$\mathbf{b}\f$ is
 * the translation vector.
 *
 * **Coordinate Systems:**
 * - Reference coordinates: \f$(\xi, \eta, \zeta) \in [-1,1]^3\f$
 * - Grid coordinates: \f$(i_x, i_y, i_z) \in [0, \text{ngll}-1]^3\f$
 * - Physical coordinates: \f$(x, y, z)\f$ in problem domain
 *
 * @tparam specfem::dimension::type::dim3 Template specialization for 3D
 * elements
 *
 * @code
 * // Example: Map coordinates from left face of element1 to right face of
 * element2 using ElementView = Kokkos::View<int *, Kokkos::LayoutStride,
 * Kokkos::HostSpace>; ElementView elem1_nodes = ...; // Control node indices
 * for element 1 ElementView elem2_nodes = ...; // Control node indices for
 * element 2
 *
 * specfem::connections::connection_mapping<specfem::dimension::type::dim3>
 *     mapping(5, 5, 5, elem1_nodes, elem2_nodes);
 *
 * // Map point (iz=2, iy=1, ix=0) on left face to corresponding point on right
 * face auto [mapped_iz, mapped_iy, mapped_ix] = mapping.map_coordinates(
 *     specfem::mesh_entity::dim3::type::left,
 *     specfem::mesh_entity::dim3::type::right,
 *     2, 1, 0);
 *
 * // Use mapped coordinates for flux computation or coupling
 * @endcode
 *
 * @note This class assumes hexahedral elements with standard node numbering
 *       and supports both conforming and non-conforming element interfaces.
 *
 * @see specfem::mesh_entity::dim3::type
 * @see specfem::shape_function::shape_function
 */
template <> class connection_mapping<specfem::dimension::type::dim3> {
private:
  /**
   * @brief Type alias for Kokkos view storing element control node indices
   *
   * Uses LayoutStride to accommodate flexible memory layouts for element
   * connectivity data read from various mesh formats.
   */
  using ElementIndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>;

public:
  /** @brief Dimension tag for 3D spectral elements */
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Constructor for 3D connection mapping between two elements
   *
   * Initializes the connection mapping object with quadrature grid dimensions
   * and control node connectivity information for two adjacent hexahedral
   * elements. The constructor validates that both elements have the same
   * number of control nodes (typically 8 for hexahedral elements).
   *
   * @param ngllz Number of Gauss-Lobatto-Legendre points in z-direction
   * @param nglly Number of Gauss-Lobatto-Legendre points in y-direction
   * @param ngllx Number of Gauss-Lobatto-Legendre points in x-direction
   * @param element1 Control node indices for the first element (source element)
   * @param element2 Control node indices for the second element (target
   * element)
   *
   * @throws std::runtime_error if elements have different numbers of control
   * nodes
   *
   * @note For cubic elements, typically ngllz = nglly = ngllx = ngll
   * @note Control node indices should reference a global node numbering system
   *
   * @code
   * // Create mapping between two 5×5×5 GLL elements
   * Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>
   *     elem1_nodes("elem1", 8);
   * Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>
   *     elem2_nodes("elem2", 8);
   *
   * // Initialize with global node indices
   * for (int i = 0; i < 8; ++i) {
   *     elem1_nodes(i) = base_index1 + i;
   *     elem2_nodes(i) = base_index2 + i;
   * }
   *
   * connection_mapping mapping(5, 5, 5, elem1_nodes, elem2_nodes);
   * @endcode
   */
  connection_mapping(const int ngllz, const int nglly, const int ngllx,
                     ElementIndexView element1, ElementIndexView element2)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), element1(element1),
        element2(element2) {
    if (element1.extent(0) != element2.extent(0)) {
      throw std::runtime_error(
          "The 2 elements must have the same number of control nodes.");
    }
  }

  /**
   * @brief Map coordinates from one mesh entity to another with point
   * specification
   *
   * Transforms grid coordinates from a source mesh entity (a face, or an edge)
   * to the corresponding coordinates on a target mesh entity of an adjacent
   * element. This is the primary coordinate transformation function used for
   * implementing coupling between spectral elements with potentially different
   * orientations.
   *
   * **Transformation Process:**
   * 1. **Entity Validation**: Verifies that source and target entities are
   * compatible
   * 2. **Coordinate Normalization**: Converts grid indices to normalized
   * coordinates [0,1]
   * 3. **Geometric Transformation**: Applies affine transformation (faces) or
   * permutation (edges)
   * 4. **Index Reconstruction**: Maps transformed coordinates back to target
   * grid indices
   *
   * **Supported Mappings:**
   * - Face-to-face: Uses affine transformation with permutation matrix
   * - Edge-to-edge: Applies 1D coordinate transformation with orientation
   * handling
   *
   * @param from Source mesh entity (face, edge, or corner) on element1
   * @param to Target mesh entity (face, edge, or corner) on element2
   * @param iz Grid index in z-direction (ζ-direction) on source entity
   * @param iy Grid index in y-direction (η-direction) on source entity
   * @param ix Grid index in x-direction (ξ-direction) on source entity
   *
   * @return std::tuple<int, int, int> Mapped grid coordinates (iz', iy', ix')
   * on target entity
   *
   * @throws std::runtime_error if entities are incompatible or indices are out
   * of bounds
   *
   * @code
   * // Map point from left face of element1 to right face of element2
   * auto [mapped_iz, mapped_iy, mapped_ix] = mapping.map_coordinates(
   *     specfem::mesh_entity::dim3::type::left,    // Source: left face
   *     specfem::mesh_entity::dim3::type::right,   // Target: right face
   *     2, 1, 0);                                  // Point: (iz=2, iy=1, ix=0)
   *
   * // Use mapped coordinates for coupling calculation
   * auto flux = compute_interface_flux(
   *     element1_data(2, 1, 0),                    // Source point data
   *     element2_data(mapped_iz, mapped_iy, mapped_ix)); // Target point data
   * @endcode
   *
   * @see specfem::mesh_entity::dim3::type
   * @see affine_transform (internal implementation)
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to, const int iz,
                  const int iy, const int ix) const;

  /**
   * @brief Map corner coordinates between mesh entities without point
   * specification
   *
   * Simplified coordinate mapping for corner-to-corner transformations where
   * no specific point indices are needed. This overload is typically used for
   * corner entities where only a single point exists, or for obtaining the
   * base mapping between entity coordinate systems.
   *
   * **Use Cases:**
   * - Corner-to-corner mappings (single point entities)
   * - Obtaining transformation matrix information
   * - Base coordinate system alignment between entities
   *
   * @param from Source mesh entity on element1
   * @param to Target mesh entity on element2
   *
   * @return std::tuple<int, int, int> Base mapped coordinates for the entity
   * transformation
   *
   * @throws std::runtime_error if entities are incompatible for direct mapping
   *
   * @code
   * // Map corner entities between elements
   * auto [base_iz, base_iy, base_ix] = mapping.map_coordinates(
   *     specfem::mesh_entity::dim3::type::bottom_front_left,
   *     specfem::mesh_entity::dim3::type::top_back_right);
   * @endcode
   *
   * @note This overload is primarily used for corner entities or coordinate
   * system queries
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to) const;

private:
  /**
   * @brief Number of GLL points in z-direction (ζ-direction)
   *
   * Defines the grid resolution along the z-axis for both elements.
   * Used for coordinate transformation and bounds checking.
   */
  int ngllz;

  /**
   * @brief Number of GLL points in y-direction (η-direction)
   *
   * Defines the grid resolution along the y-axis for both elements.
   * Used for coordinate transformation and bounds checking.
   */
  int nglly;

  /**
   * @brief Number of GLL points in x-direction (ξ-direction)
   *
   * Defines the grid resolution along the x-axis for both elements.
   * Used for coordinate transformation and bounds checking.
   */
  int ngllx;

  /**
   * @brief Control node indices for the first (source) element
   *
   * Kokkos view containing the global node indices that define the
   * geometry of the first element. These indices reference nodes in
   * a global mesh connectivity structure and are used to determine
   * element orientation and face/edge node relationships.
   */
  ElementIndexView element1;

  /**
   * @brief Control node indices for the second (target) element
   *
   * Kokkos view containing the global node indices that define the
   * geometry of the second element. Used in conjunction with element1
   * to compute coordinate transformations between adjacent elements.
   */
  ElementIndexView element2;
};

} // namespace specfem::connections
