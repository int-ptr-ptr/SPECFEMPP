#pragma once
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_coupling/tags.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_connections::type ConnectionTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename = void>
struct flux_scheme_data;
// struct flux_scheme_data {
//   flux_scheme_data(
//       const specfem::assembly::element_intersections<DimensionTag>
//           &element_intersections,
//       const specfem::assembly::jacobian_matrix<DimensionTag>
//       &jacobian_matrix, const specfem::assembly::mesh<DimensionTag> &mesh,
//       const specfem::element_coupling::flux_scheme_configuration
//           &flux_scheme_config) {};
// };

} // namespace specfem::assembly::nonconforming_interfaces_impl
