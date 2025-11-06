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
 * @brief Stores the Jacobian (1D) times quadrature weight of each quadrature
 * point, used for boundary integration.
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
 */
template <int NumberElements, int NQuadIntersection,
          specfem::dimension::type DimensionTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_factor;

template <int NumberElements, int NQuadIntersection, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_factor<NumberElements, NQuadIntersection,
                                         specfem::dimension::type::dim2,
                                         MemorySpace, MemoryTraits> {
private:
  static constexpr bool UseSIMD = false;
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

protected:
  using IntersectionFactorViewType =
      Kokkos::View<typename specfem::datatype::simd<type_real, UseSIMD>::
                       datatype[NumberElements][NQuadIntersection],
                   MemorySpace, MemoryTraits>; ///< Underlying view used to
                                               ///< store data of the transfer
                                               ///< function.

public:
  IntersectionFactorViewType intersection_factor;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param intersection_factor Normal vector at intersection quadrature points
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_factor(
      const IntersectionFactorViewType &intersection_factor)
      : intersection_factor(intersection_factor) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_factor() = default;

  /**
   * @brief Constructor that initializes data views in Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_factor(const MemberType &team)
      : intersection_factor(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() {
    return IntersectionFactorViewType::shmem_size();
  }
};

} // namespace impl
} // namespace specfem::chunk_edge
