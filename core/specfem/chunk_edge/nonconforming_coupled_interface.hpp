#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::chunk_edge {

/**
 * @brief Primary template for coupled interface points
 *
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
template <int NumberElements, int NQuadElement, int NQuadInterface,
          specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct nonconforming_coupled_interface;

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
template <int NumberElements, int NQuadElement, int NQuadInterface,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename MemorySpace,
          typename MemoryTraits, bool UseSIMD>
struct nonconforming_coupled_interface<
    NumberElements, NQuadElement, NQuadInterface,
    specfem::dimension::type::dim2, specfem::connections::type::nonconforming,
    InterfaceTag, BoundaryTag, MemorySpace, MemoryTraits, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

  // We're using components here to index the mortar quadrature. Should we add a
  // new index type for that?
  using TransferViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      NQuadInterface, UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.
  using EdgeNormalViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      2, UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.
  using MortarFactorViewType = specfem::datatype::ScalarChunkEdgeViewType<
      type_real, specfem::dimension::type::dim2, NumberElements, NQuadElement,
      UseSIMD, MemorySpace,
      MemoryTraits>; ///< Underlying view used to store data of the transfer
                     ///< function.

public:
  /** @brief Dimension tag for 2D specialization */
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Connection type between elements */
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  static constexpr auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  static constexpr auto boundary_tag = BoundaryTag;
  /** @brief number of quadrature points on the element (NGLL) */
  static constexpr int n_quad_element = NQuadElement;
  /** @brief number of quadrature points on the interface */
  static constexpr int n_quad_interface = NQuadInterface;

  /** @brief Edge scaling factor for interface computations */
  MortarFactorViewType mortar_factor;
  /** @brief Edge normal vector (2D) */
  EdgeNormalViewType edge_normal;
  /** @brief Transfer function (edge -> mortar) */
  TransferViewType transfer_function_self;
  /** @brief Transfer function (edge -> mortar) */
  TransferViewType transfer_function_coupled;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param mortar_factor Scaling factor for the interface edge
   * @param edge_normal_ Normal vector at the interface edge
   * @param transfer_function Transfer function from the edge to the mortar
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_coupled_interface(
      const MortarFactorViewType &mortar_factor,
      const EdgeNormalViewType &edge_normal_,
      const TransferViewType &transfer_function_self,
      const TransferViewType &transfer_function_coupled)
      : mortar_factor(mortar_factor), edge_normal(edge_normal_),
        transfer_function_self(transfer_function_self),
        transfer_function_coupled(transfer_function_coupled) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_coupled_interface() = default;

  /**
   * @brief Constructor that initializes data views in Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION nonconforming_coupled_interface(const MemberType &team)
      : mortar_factor(team.team_scratch(0)), edge_normal(team.team_scratch(0)),
        transfer_function_self(team.team_scratch(0)),
        transfer_function_coupled(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() {
    return MortarFactorViewType::shmem_size() +
           EdgeNormalViewType::shmem_size() +
           TransferViewType::shmem_size() * 2;
  }
};

} // namespace specfem::chunk_edge
