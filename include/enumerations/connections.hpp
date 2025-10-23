#pragma once

#include "dimension.hpp"

namespace specfem::connections {

/**
 * @brief Enumeration of connection types for mesh connectivity
 *
 * Defines the types of connections that can exist between mesh elements
 * in the spectral element method.
 */
enum class type : int {
  /// @brief Strongly conforming connection where nodes match exactly
  strongly_conforming = 1,
  /// @brief Weakly conforming connection where nodes match, but the shape
  /// function can be discontinuous. (example: coupling across different media,
  /// kinematic faults).
  weakly_conforming = 2,
  /// @brief Nonconforming connections have no matching nodes, but are
  /// geometrically (spatially) adjacent
  nonconforming = 3
};

/**
 * @brief Recovers a human-readable string for a given connection
 *
 */
const std::string to_string(const specfem::connections::type &conn);

template <specfem::dimension::type DimensionTag> class connection_mapping;

} // namespace specfem::connections

#include "dim2/connections.hpp"
#include "dim3/connections.hpp"

namespace specfem::connections {

// CTAD guides
connection_mapping(const int, const int)
    -> connection_mapping<specfem::dimension::type::dim2>;

connection_mapping(const int, const int, const int,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>)
    -> connection_mapping<specfem::dimension::type::dim3>;

} // namespace specfem::connections
