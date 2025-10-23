#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"

#include "impl/nonconforming_transfer_function.hpp"

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
          specfem::dimension::type::dim2, false>,
      public specfem::chunk_edge::impl::nonconforming_transfer_function<
          true, NumberElements, NQuadElement, NQuadInterface,
          specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD>,
      public specfem::chunk_edge::impl::nonconforming_transfer_function<
          false, NumberElements, NQuadElement, NQuadInterface,
          specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;

  template <bool is_self>
  using transfer_type =
      specfem::chunk_edge::impl::nonconforming_transfer_function<
          is_self, NumberElements, NQuadElement, NQuadInterface,
          specfem::dimension::type::dim2, MemorySpace, MemoryTraits, UseSIMD>;

  using TransferViewType = typename transfer_type<true>::TransferViewType;
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

  using EdgeNormalViewType =
      Kokkos::View<typename specfem::datatype::simd<type_real, UseSIMD>::
                       datatype[NumberElements][NQuadInterface][2],
                   MemorySpace, MemoryTraits>; ///< Underlying view used to
                                               ///< store data of the transfer
                                               ///< function.
  using MortarFactorViewType =
      Kokkos::View<typename specfem::datatype::simd<type_real, UseSIMD>::
                       datatype[NumberElements][NQuadInterface],
                   MemorySpace, MemoryTraits>; ///< Underlying view used to
                                               ///< store data of the transfer
                                               ///< function.

public:
  static constexpr int chunk_size = NumberElements;
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
  MortarFactorViewType intersection_factor;
  /** @brief Edge normal vector (2D) */
  EdgeNormalViewType intersection_normal;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param intersection_factor Scaling factor for the interface edge
   * @param intersection_normal_ Normal vector at the interface edge
   * @param transfer_function Transfer function from the edge to the mortar
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_coupled_interface(
      const MortarFactorViewType &intersection_factor,
      const EdgeNormalViewType &intersection_normal_,
      const TransferViewType &transfer_function_self,
      const TransferViewType &transfer_function_coupled)
      : transfer_type<true>(transfer_function_self),
        transfer_type<false>(transfer_function_self),
        intersection_factor(intersection_factor),
        intersection_normal(intersection_normal_) {}

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
      : transfer_type<true>(team), transfer_type<false>(team),
        intersection_factor(team.team_scratch(0)),
        intersection_normal(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() {
    return MortarFactorViewType::shmem_size() +
           EdgeNormalViewType::shmem_size() +
           transfer_type<true>::shmem_size() +
           transfer_type<false>::shmem_size();
  }
};

} // namespace specfem::chunk_edge
