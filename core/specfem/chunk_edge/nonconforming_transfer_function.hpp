#pragma once

#include "nonconforming_coupled_interface/transfer.hpp"

namespace specfem::chunk_edge {

/**
 * @brief Primary template for transfer functions
 *
 * @tparam IsSelf self side of InterfaceTag or coupled (other) side.
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NQuadElement Number of quadrature points (Gauss-Lobatto-Legendre) on
 * element edges.
 * @tparam NQuadInterface Number of quadrature points on each mortar element.
 * @tparam DimensionTag Spatial dimension
 * @tparam ConnectionTag Connection type (strongly/weakly conforming)
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <bool IsSelf, int NumberElements, int NQuadElement, int NQuadInterface,
          specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct nonconforming_transfer_function;

/**
 * @brief 2D coupled interface point data structure
 *
 * Represents a point on a coupled interface between different physical
 * media in 2D spectral element simulations. Contains geometric data
 * (edge factor and normal vector) needed for interface computations.
 *
 * @tparam NumberElements Number of elements in the chunk.
 * @tparam NQuadElement Number of quadrature points (Gauss-Lobatto-Legendre) on
 * element edges.
 * @tparam NQuadInterface Number of quadrature points on each mortar element.
 * @tparam ConnectionTag Connection type between elements
 * @tparam InterfaceTag Type of interface (elastic-acoustic or acoustic-elastic)
 * @tparam BoundaryTag Boundary condition applied to the interface
 * @tparam MemorySpace Memory space for data storage.
 * @tparam MemoryTraits Memory traits for data storage.
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <bool IsSelf, int NumberElements, int NQuadElement, int NQuadInterface,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct nonconforming_transfer_function<
    IsSelf, NumberElements, NQuadElement, NQuadInterface,
    specfem::dimension::type::dim2, specfem::connections::type::nonconforming,
    InterfaceTag, BoundaryTag, MemorySpace, MemoryTraits, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false>,
      public impl::nonconforming_transfer_function<
          IsSelf, NumberElements, NQuadElement, NQuadInterface,
          specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;
  using impl_type = impl::nonconforming_transfer_function<
      IsSelf, NumberElements, NQuadElement, NQuadInterface,
      specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD>;

public:
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto chunk_size = NumberElements;
  static constexpr auto n_quad_element = NQuadElement;
  static constexpr auto n_quad_interface = NQuadInterface;

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_function(Args... args)
      : impl_type(args...) {}
};
} // namespace specfem::chunk_edge
