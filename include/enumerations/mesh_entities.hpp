#pragma once
#include "enumerations/dimension.hpp"

namespace specfem::mesh_entity {

/**
 * @brief Template structure for edge mesh entities in various dimensions
 *
 * @tparam DimensionTag The dimension type (e.g., dim2, dim3)
 */
template <specfem::dimension::type DimensionTag> struct edge;

/**
 * @brief Mesh element structure for a specific dimension
 *
 * @tparam Dimension The dimension type (e.g., dim2, dim3)
 */
template <specfem::dimension::type DimensionTag> struct element;

} // namespace specfem::mesh_entity

#include "dim2/mesh_entities.hpp"
#include "dim3/mesh_entities.hpp"

// CTAD guides
namespace specfem::mesh_entity {

element(const int, const int) -> element<specfem::dimension::type::dim2>;

element(const int, const int, const int)
    -> element<specfem::dimension::type::dim3>;
} // namespace specfem::mesh_entity
