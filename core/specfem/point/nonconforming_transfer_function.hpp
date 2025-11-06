#pragma once

#include "impl/nonconforming_transfer_function.hpp"

namespace specfem::point {

/**
 * @brief Primary template for transfer functions (point-wise)
 *
 * @tparam IsSelf self side of InterfaceTag or coupled (other) side.
 * element edges.
 * @tparam NQuadIntersection Number of quadrature points on each mortar element.
 * @tparam DimensionTag Spatial dimension
 * @tparam ConnectionTag Connection type (strongly/weakly conforming)
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <bool IsSelf, int NQuadIntersection,
          specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function;

template <bool IsSelf, int NQuadIntersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function<
    IsSelf, NQuadIntersection, specfem::dimension::type::dim2,
    specfem::connections::type::nonconforming, InterfaceTag, BoundaryTag>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false>,
      public impl::nonconforming_transfer_function<
          IsSelf, NQuadIntersection, specfem::dimension::type::dim2> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;
  using impl_type =
      impl::nonconforming_transfer_function<IsSelf, NQuadIntersection,
                                            specfem::dimension::type::dim2>;

public:
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto n_quad_intersection = NQuadIntersection;

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_function(const Args &...args)
      : impl_type(args...) {}
};
} // namespace specfem::point
