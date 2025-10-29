#pragma once

#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <list>
#include <stdexcept>
#include <string>

namespace specfem::mesh_entity {
/**
 * @namespace specfem::mesh_entity::dim3
 * @brief Defines mesh entity types and utilities for 3D spectral element method
 *
 * This namespace provides enumerations and utility functions for working with
 * mesh entities in 3D spectral element grids, including faces, edges, and
 * corners of hexahedral elements.
 */

namespace dim3 {

/**
 * @brief Enumeration of mesh entity types for 3D hexahedral elements
 *
 * Defines the different types of mesh entities that can exist in a 3D
 * hexahedral element used in spectral element meshes. The numbering follows the
 * convention:
 *
 * Hexahedral element with 6 faces, 12 edges, and 8 corners:
 * @code
 *     8 -------- 7
 *    /|         /|
 *   / |        / |
 *  5 -------- 6  |
 *  |  |       |  |
 *  |  4 ------|--3
 *  | /        | /
 *  |/         |/
 *  1 -------- 2
 * @endcode
 *
 * Face numbering (1-6): Element boundary surfaces
 * Edge numbering (7-18): Connects adjacent faces
 * Corner numbering (19-26): Element vertices
 */
enum class type : int {
  bottom = 1,              ///< Bottom face of the element
  right = 2,               ///< Right face of the element
  top = 3,                 ///< Top face of the element
  left = 4,                ///< Left face of the element
  front = 5,               ///< Front face of the element
  back = 6,                ///< Back face of the element
  bottom_left = 7,         ///< Bottom-left edge of the element
  bottom_right = 8,        ///< Bottom-right edge of the element
  top_right = 9,           ///< Top-right edge of the element
  top_left = 10,           ///< Top-left edge of the element
  front_bottom = 11,       ///< Front-bottom edge of the element
  front_top = 12,          ///< Front-top edge of the element
  front_left = 13,         ///< Front-left edge of the element
  front_right = 14,        ///< Front-right edge of the element
  back_bottom = 15,        ///< Back-bottom edge of the element
  back_top = 16,           ///< Back-top edge of the element
  back_left = 17,          ///< Back-left edge of the element
  back_right = 18,         ///< Back-right edge of the element
  bottom_front_left = 19,  ///< Bottom-front-left corner of the element
  bottom_front_right = 20, ///< Bottom-front-right corner of the element
  bottom_back_left = 21,   ///< Bottom-back-left corner of the element
  bottom_back_right = 22,  ///< Bottom-back-right corner of the element
  top_front_left = 23,     ///< Top-front-left corner of the element
  top_front_right = 24,    ///< Top-front-right corner of the element
  top_back_left = 25,      ///< Top-back-left corner of the element
  top_back_right = 26      ///< Top-back-right corner of the element
};

/**
 * @brief Recovers a human-readable string for a given mesh entity.
 *
 * @param entity The mesh entity type to convert to string
 * @return std::string Human-readable name of the entity
 */
const std::string to_string(const specfem::mesh_entity::dim3::type &entity);

/**
 * @brief List of all face types in a hexahedral element
 *
 * Contains all face mesh entities of a 3D hexahedral element. The faces are
 * organized as the 6 boundary surfaces of the hexahedron. This list is useful
 * for iterating over all faces of an element for boundary condition
 * application.
 *
 * Face orientations:
 * - bottom: z-negative boundary
 * - right: x-positive boundary
 * - top: z-positive boundary
 * - left: x-negative boundary
 * - front: y-positive boundary
 * - back: y-negative boundary
 */
const std::list<specfem::mesh_entity::dim3::type> faces = {
  type::bottom, type::right, type::top, type::left, type::front, type::back
};

/**
 * @brief List of all edge types in a hexahedral element
 *
 * Contains all edge mesh entities of a 3D hexahedral element. The 12 edges
 * connect adjacent faces and vertices. This list is useful for iterating over
 * all edges of an element for connectivity analysis and interface operations.
 *
 * Edge organization:
 * - Horizontal edges (bottom/top): bottom_left, bottom_right, top_left,
 * top_right
 * - Front vertical edges: front_bottom, front_top, front_left, front_right
 * - Back vertical edges: back_bottom, back_top, back_left, back_right
 */
const std::list<specfem::mesh_entity::dim3::type> edges = {
  type::bottom_left,  type::bottom_right, type::top_right,  type::top_left,
  type::front_bottom, type::front_top,    type::front_left, type::front_right,
  type::back_bottom,  type::back_top,     type::back_left,  type::back_right
};

/**
 * @brief List of all corner types in a hexahedral element
 *
 * Contains all corner mesh entities (vertices) of a 3D hexahedral element.
 * The 8 corners define the geometric vertices of the hexahedron. This list is
 * useful for iterating over all corners of an element for coordinate mapping
 * and geometric transformations.
 *
 * Corner organization follows right-handed coordinate system:
 * - Bottom corners (z-negative): bottom_front_left, bottom_front_right,
 *   bottom_back_left, bottom_back_right
 * - Top corners (z-positive): top_front_left, top_front_right,
 *   top_back_left, top_back_right
 */
const std::list<specfem::mesh_entity::dim3::type> corners = {
  type::bottom_front_left, type::bottom_front_right, type::bottom_back_left,
  type::bottom_back_right, type::top_front_left,     type::top_front_right,
  type::top_back_left,     type::top_back_right
};

/**
 * @brief Returns the faces that are connected by a given edge
 *
 * @param edge The edge mesh entity type
 * @return std::list<specfem::mesh_entity::dim3::type> List of face types
 * connected by the edge
 *
 * For each edge of a hexahedral element, this function returns the two faces
 * that meet at that edge. The faces are returned in a consistent order.
 *
 * @throws std::runtime_error if the input is not a valid edge type
 *
 * @code
 * auto faces = faces_of_edge(type::bottom_left);
 * // Returns faces that share the bottom_left edge
 * @endcode
 */
const std::list<specfem::mesh_entity::dim3::type>
faces_of_edge(const specfem::mesh_entity::dim3::type &edge);

/**
 * @brief Returns the edges that meet at a given corner
 *
 * @param corner The corner mesh entity type
 * @return std::list<specfem::mesh_entity::dim3::type> List of edge types that
 * meet at the corner
 *
 * For each corner of a hexahedral element, this function returns the three
 * edges that converge at that corner. The edges are returned in a consistent
 * order.
 *
 * @throws std::runtime_error if the input is not a valid corner type
 *
 * @code
 * auto edges = edges_of_corner(type::bottom_front_left);
 * // Returns the three edges meeting at bottom_front_left corner
 * @endcode
 */
const std::list<specfem::mesh_entity::dim3::type>
edges_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Returns the corners that define a given face
 *
 * @param face The face mesh entity type
 * @return std::list<specfem::mesh_entity::dim2::type> List of corner types
 * that define the face
 *
 * For each face of a hexahedral element, this function returns the four
 * corners (vertices) that outline that face. The corners are returned in a
 * consistent order (e.g., counter-clockwise when viewed from outside the
 * element).
 *
 * @throws std::runtime_error if the input is not a valid face type
 *
 * @code
 * auto corners = corners_of_face(type::bottom);
 * // Returns the four corners defining the bottom face
 * @endcode
 */
const std::list<specfem::mesh_entity::dim3::type>
corners_of_face(const specfem::mesh_entity::dim3::type &face);

} // namespace dim3

/**
 * @brief Generic utility function to check if a container contains a specific
 * mesh entity type
 *
 * @tparam T Container type that supports begin() and end() iterators
 * @param list The container to search in
 * @param value The mesh entity type to search for
 * @return bool True if the value is found in the container, false otherwise
 *
 * This template function provides a generic way to check membership in any
 * container of mesh entity types. It's commonly used with the predefined
 * faces, edges, and corners lists.
 *
 * @code
 * if (contains(faces, type::bottom)) {
 *     // Handle face case
 * }
 * if (contains(edges, type::bottom_left)) {
 *     // Handle edge case
 * }
 * @endcode
 */
template <typename T> bool contains(const T &list, const dim3::type &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}

// clang-format off
/**
 * @brief Get the nodes associated with a specific orientation of a 3D mesh entity
 *
 * This function returns a vector of node indices that correspond to the nodes
 * located on a particular orientation of the given 3D mesh entity. The orientation
 * determines which face, edge, or vertex nodes are retrieved based on the entity type.
 *
 * @code{.cpp}
  * const Kokkos::View<int *, Kokkos::HostSpace> control_nodes("control_nodes", 8); // control nodes of the element
  * std::vector<int> bottom_face_nodes = {control_nodes(0), control_nodes(1), control_nodes(2), control_nodes(3)};
  * auto nodes = nodes_on_orientation(specfem::mesh_entity::dim3::type::bottom);
  * // nodes now contains the indices of nodes on the bottom face of the element
  * assert(bottom_face_nodes == {control_nodes(nodes[0]), control_nodes(nodes[1]), control_nodes(nodes[2]), control_nodes(nodes[3])});
  * @endcode
 *
 * @param entity The 3D mesh entity (element, face, edge, etc.) for which to retrieve oriented nodes
 * @return std::vector<int> Vector containing the indices of nodes on the specified orientation
 */
// clang-format on
std::vector<int>
nodes_on_orientation(const specfem::mesh_entity::dim3::type &entity);

template <specfem::dimension::type Dimension> struct edge;

/**
 * @brief Edge structure for 3D hexahedral elements (Specialization)
 *
 * Represents an edge entity in a 3D hexahedral spectral element mesh.
 * Edges are 1D entities that connect faces and are essential for element
 * connectivity and interface operations.
 */
template <> struct edge<specfem::dimension::type::dim3> {
  specfem::mesh_entity::dim3::type edge_type; ///< Type of edge from dim3::type
                                              ///< enumeration
  int ispec;                                  ///< Spectral element index
  bool reverse_orientation;                   ///< Edge orientation flag

  /**
   * @brief Constructs an edge with specified parameters
   *
   * @param ispec Spectral element index
   * @param edge_type Type of edge from dim3::type enumeration
   * @param reverse_orientation Whether edge orientation is reversed (default:
   * false)
   */
  KOKKOS_INLINE_FUNCTION
  edge(const int ispec, const specfem::mesh_entity::dim3::type edge_type,
       const bool reverse_orientation = false)
      : edge_type(edge_type), ispec(ispec),
        reverse_orientation(reverse_orientation) {}

  /**
   * @brief Default constructor
   */
  KOKKOS_INLINE_FUNCTION
  edge() = default;
};

/**
 * @brief Mesh element structure for a specific dimension
 *
 * @tparam Dimension The dimension type (e.g., dim2, dim3)
 */
template <specfem::dimension::type Dimension> struct element;

/**
 * @brief Mesh element structure for 3D hexahedral elements (Specialization)
 *
 * Represents a 3D hexahedral spectral element with Gauss-Lobatto-Legendre
 * quadrature points. This structure defines the polynomial order and
 * discretization properties of the element.
 */
template <> struct element<specfem::dimension::type::dim3> {

public:
  int ngllz;  ///< Number of Gauss-Lobatto-Legendre points in the z-direction
  int nglly;  ///< Number of Gauss-Lobatto-Legendre points in the y-direction
  int ngllx;  ///< Number of Gauss-Lobatto-Legendre points in the x-direction
  int orderz; ///< Polynomial order of the element
  int ordery; ///< Polynomial order of the element
  int orderx; ///< Polynomial order of the element
  int size;   ///< Total number of GLL points in the element

  /**
   * @brief Default constructor for the element struct
   */
  element() = default;

  /**
   * @brief Constructs an element entity given the number of
   * Gauss-Lobatto-Legendre points
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   */
  element(const int ngll)
      : ngllx(ngll), nglly(ngll), ngllz(ngll), orderz(ngll - 1),
        ordery(nglly - 1), orderx(ngllx - 1), size(ngll * ngll * ngll) {};

  /**
   * @brief Constructs an element entity given the number of
   * Gauss-Lobatto-Legendre points in each dimension
   *
   * @param ngllz The number of Gauss-Lobatto-Legendre points in z-direction
   * @param nglly The number of Gauss-Lobatto-Legendre points in y-direction
   * @param ngllx The number of Gauss-Lobatto-Legendre points in x-direction
   *
   * @throws std::invalid_argument if GLL points differ between dimensions
   *
   * @note Currently requires identical GLL points in all dimensions for
   * stability
   */
  element(const int ngllz, const int nglly, const int ngllx);

  /**
   * @brief Checks if the element is consistent across dimensions against a
   *        specific number of GLL points.
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   * @return true If all dimensions match the specified number of GLL points
   * @return false If any dimension does not match
   */
  bool operator==(const int ngll) const {
    return ngll == ngllz && ngll == nglly && ngll == ngllx;
  }

  /**
   * @brief Checks if the element is not consistent across dimensions against a
   *        specific number of GLL points.
   *
   * @param ngll The number of Gauss-Lobatto-Legendre points
   * @return false If all dimensions match the specified number of GLL points
   * @return true If any dimension does not match
   */
  bool operator!=(const int ngll) const { return !(*this == ngll); }

  /**
   * @brief Get the total number of GLL points on a given mesh entity
   * @param entity The mesh entity type
   * @return int Total number of GLL points on the entity
   */
  int number_of_points_on_orientation(
      const specfem::mesh_entity::dim3::type &entity) const;

  /**
   * @brief Get the coordinates of a point on a given mesh entity
   *
   * @param entity The mesh entity type
   * @param point The index of the point on the entity
   * @return std::tuple<int, int, int> The coordinates of the point
   *
   * @code{.cpp}
   * const auto npoints = element.number_of_points_on_orientation(
   *     specfem::mesh_entity::dim3::type::bottom);
   * for (int ipoint = 0; ipoint < npoints; ++ipoint) {
   *     const auto [iz, iy, ix] = element.coordinates_at_face(
   *         specfem::mesh_entity::dim3::type::bottom, ipoint);
   *     assert(iz == 0); // Bottom face has iz = 0
   * }
   * @endcode
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &entity,
                  const int point) const;

  /**
   * @brief Get the coordinates for a corner mesh entity
   *
   * @param corner The corner mesh entity type
   * @return std::tuple<int, int, int> The coordinates of the corner
   *
   * @code{.cpp}
   * const auto [iz, iy, ix] = element.coordinates_at_corner(
   *     specfem::mesh_entity::dim3::type::bottom_front_left);
   * assert(iz == 0 && iy == 0 && ix == 0);
   * @endcode
   */
  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &corner) const;

private:
  int ngll2d;
  int ngll;

  std::unordered_map<specfem::mesh_entity::dim3::type,
                     std::function<std::tuple<int, int, int>(int, int)> >
      face_coordinates; ///< Maps face types to coordinate functions
  std::unordered_map<specfem::mesh_entity::dim3::type,
                     std::function<std::tuple<int, int, int>(int)> >
      edge_coordinates; ///< Maps edge types to coordinate functions
  std::unordered_map<specfem::mesh_entity::dim3::type,
                     std::tuple<int, int, int> >
      corner_coordinates; ///< Maps corner types to coordinates
};

/**
 * @brief Returns the edges that define a given face
 *
 * @param face The face mesh entity type
 * @return std::array<specfem::mesh_entity::dim3::type, 4> Array of edge types
 * that define the face
 *
 * For each face of a hexahedral element, this function returns the four edges
 * that outline that face.
 *
 * @throws std::runtime_error if the input is not a valid face type
 *
 * @code
 * auto edges = edges_of_face(type::bottom);
 * // Returns the four edges defining the bottom face
 * @endcode
 */
std::array<specfem::mesh_entity::dim3::type, 4>
edges_of_face(const specfem::mesh_entity::dim3::type &face);

/**
 * @brief Returns the corners that define a given face
 *
 * @param face The face mesh entity type
 * @return std::array<specfem::mesh_entity::dim3::type, 4> Array of corner types
 * that define the face
 *
 * For each face of a hexahedral element, this function returns the four
 * corners (vertices) that outline that face. The corners are returned in a
 * consistent order (e.g., counter-clockwise when viewed from outside the
 * element).
 *
 * @throws std::runtime_error if the input is not a valid face type
 *
 * @code
 * auto corners = corners_of_face(type::bottom);
 * // Returns the four corners defining the bottom face
 * @endcode
 */
std::array<specfem::mesh_entity::dim3::type, 4>
corners_of_face(const specfem::mesh_entity::dim3::type &face);

/**
 * @brief Returns the faces that meet at a given corner
 *
 * @param corner The corner mesh entity type
 * @return std::list<specfem::mesh_entity::dim3::type> List of face types
 * connected at the corner
 *
 * For each corner of a hexahedral element, this function returns the three
 * faces that converge at that corner. The faces are returned in a consistent
 * order.
 *
 * @throws std::runtime_error if the input is not a valid corner type
 *
 * @code
 * auto faces = faces_of_corner(type::bottom_front_left);
 * // Returns the three faces meeting at bottom_front_left corner
 * @endcode
 */
std::list<specfem::mesh_entity::dim3::type>
faces_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Returns the edges that meet at a given corner
 *
 * @param corner The corner mesh entity type
 * @return std::list<specfem::mesh_entity::dim3::type> List of edge types that
 * meet at the corner
 *
 * For each corner of a hexahedral element, this function returns the three
 * edges that converge at that corner. The edges are returned in a consistent
 * order.
 *
 * @throws std::runtime_error if the input is not a valid corner type
 *
 * @code
 * auto edges = edges_of_corner(type::bottom_front_left);
 * // Returns the three edges meeting at bottom_front_left corner
 * @endcode
 */
std::list<specfem::mesh_entity::dim3::type>
edges_of_corner(const specfem::mesh_entity::dim3::type &corner);

/**
 * @brief Returns the faces that are connected by a given edge
 *
 * @param edge The edge mesh entity type
 * @return std::list<specfem::mesh_entity::dim3::type> List of face types
 * connected by the edge
 *
 * For each edge of a hexahedral element, this function returns the two faces
 * that meet at that edge. The faces are returned in a consistent order.
 *
 * @throws std::runtime_error if the input is not a valid edge type
 *
 * @code
 * auto faces = faces_of_edge(type::bottom_left);
 * // Returns faces that share the bottom_left edge
 * @endcode
 */
std::list<specfem::mesh_entity::dim3::type>
faces_of_edge(const specfem::mesh_entity::dim3::type &edge);
} // namespace specfem::mesh_entity
