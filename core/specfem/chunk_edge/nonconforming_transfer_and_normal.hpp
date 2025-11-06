#pragma once

#include "impl/nonconforming_intersection_normal.hpp"
#include "impl/nonconforming_transfer_function.hpp"

namespace specfem::chunk_edge {

/**
 * @brief Stores both the transfer function and normal. Use this when a
 * compute_coupling call needs both.
 *
 * @tparam IsSelf self side of InterfaceTag or coupled (other) side.
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NQuadElement Number of quadrature points (Gauss-Lobatto-Legendre) on
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
template <bool IsSelf, int NumberElements, int NQuadElement,
          int NQuadIntersection, specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_transfer_and_normal;

template <bool IsSelf, int NumberElements, int NQuadElement,
          int NQuadIntersection, specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_transfer_and_normal<
    IsSelf, NumberElements, NQuadElement, NQuadIntersection,
    specfem::dimension::type::dim2, specfem::connections::type::nonconforming,
    InterfaceTag, BoundaryTag, MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false>,
      public impl::nonconforming_transfer_function<
          IsSelf, NumberElements, NQuadElement, NQuadIntersection,
          specfem::dimension::type::dim2, MemorySpace, MemoryTraits>,
      public impl::nonconforming_intersection_normal<
          NumberElements, NQuadIntersection, specfem::dimension::type::dim2,
          MemorySpace, MemoryTraits> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;
  using impl_type_transfer = impl::nonconforming_transfer_function<
      IsSelf, NumberElements, NQuadElement, NQuadIntersection,
      specfem::dimension::type::dim2, MemorySpace, MemoryTraits>;
  using impl_normal_type =
      impl::nonconforming_intersection_normal<NumberElements, NQuadIntersection,
                                              specfem::dimension::type::dim2,
                                              MemorySpace, MemoryTraits>;

public:
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto chunk_size = NumberElements;
  static constexpr auto n_quad_element = NQuadElement;
  static constexpr auto n_quad_intersection = NQuadIntersection;

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_and_normal(const Args &...args)
      : impl_type_transfer(args...), impl_normal_type(args...) {}

  constexpr static int shmem_size() {
    return impl_type_transfer::shmem_size() + impl_normal_type::shmem_size();
  }
};
} // namespace specfem::chunk_edge
