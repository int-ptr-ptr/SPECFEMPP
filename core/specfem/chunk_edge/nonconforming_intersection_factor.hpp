#pragma once

#include "impl/nonconforming_intersection_factor.hpp"

namespace specfem::chunk_edge {

/**
 * @brief Stores the Jacobian (1D) times quadrature weight of each quadrature
 * point, used for boundary integration.
 *
 * @tparam IsSelf self side of InterfaceTag or coupled (other) side.
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NQuadIntersection Number of quadrature points on each mortar element.
 * @tparam DimensionTag Spatial dimension
 * @tparam ConnectionTag Connection type (strongly/weakly conforming)
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 */
template <int NumberElements, int NQuadIntersection,
          specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_factor;

template <int NumberElements, int NQuadIntersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_factor<
    NumberElements, NQuadIntersection, specfem::dimension::type::dim2,
    specfem::connections::type::nonconforming, InterfaceTag, BoundaryTag,
    MemorySpace, MemoryTraits>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false>,
      public impl::nonconforming_intersection_factor<
          NumberElements, NQuadIntersection, specfem::dimension::type::dim2,
          MemorySpace, MemoryTraits> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;
  using impl_type =
      impl::nonconforming_intersection_factor<NumberElements, NQuadIntersection,
                                              specfem::dimension::type::dim2,
                                              MemorySpace, MemoryTraits>;

public:
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto chunk_size = NumberElements;
  static constexpr auto n_quad_intersection = NQuadIntersection;

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION nonconforming_intersection_factor(const Args &...args)
      : impl_type(args...) {}

  constexpr static int shmem_size() { return impl_type::shmem_size(); }
};
} // namespace specfem::chunk_edge
