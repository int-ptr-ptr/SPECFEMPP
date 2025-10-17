#pragma once
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <type_traits>

namespace specfem::chunk_edge {

/*
 * The definition in the impl namespace is used to define the transfer functions
 * without the accessor, so that the multiple inheritance of
 * nonconforming_coupled_interface isn't ambiguous. Outside of the impl
 * namespace, we re-endow the Accessor type, so that it can be accessed
 * independently.
 */

namespace impl {
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
          specfem::dimension::type DimensionTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct nonconforming_transfer_function;

template <int NumberElements, int NQuadElement, int NQuadInterface,
          typename MemorySpace, typename MemoryTraits, bool UseSIMD>
struct nonconforming_transfer_function<
    true, NumberElements, NQuadElement, NQuadInterface,
    specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD> {
private:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

protected:
  using TransferViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      NQuadInterface, UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.

public:
  /** @brief Side polarity */
  static constexpr bool is_self = true;

  /** @brief Transfer function (edge -> mortar). Only the relevant side is
   * enabled.
   */
  TransferViewType transfer_function_self;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param transfer_function Transfer function from the edge to the mortar
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function(const TransferViewType &transfer_function)
      : transfer_function_self(transfer_function) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function() = default;

  /**
   * @brief Constructor that initializes data views in Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_function(const MemberType &team)
      : transfer_function_self(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() { return TransferViewType::shmem_size(); }
};

template <int NumberElements, int NQuadElement, int NQuadInterface,
          typename MemorySpace, typename MemoryTraits, bool UseSIMD>
struct nonconforming_transfer_function<
    false, NumberElements, NQuadElement, NQuadInterface,
    specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD> {
private:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

protected:
  using TransferViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      NQuadInterface, UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.

public:
  /** @brief Side polarity */
  static constexpr bool is_self = false;

  /** @brief Transfer function (edge -> mortar). Only the relevant side is
   * enabled.
   */
  TransferViewType transfer_function_coupled;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param transfer_function Transfer function from the edge to the mortar
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function(const TransferViewType &transfer_function)
      : transfer_function_coupled(transfer_function) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function() = default;

  /**
   * @brief Constructor that initializes data views in Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION nonconforming_transfer_function(const MemberType &team)
      : transfer_function_coupled(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() { return TransferViewType::shmem_size(); }
};

} // namespace impl

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
