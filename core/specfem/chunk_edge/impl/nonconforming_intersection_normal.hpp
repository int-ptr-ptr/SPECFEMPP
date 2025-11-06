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
 * @brief Stores the normal vectors on the intersection, at intersection
 * quadrature points.
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
 * @tparam UseSIMD Flag to indicate if SIMD should be used.
 */
template <int NumberElements, int NQuadIntersection,
          specfem::dimension::type DimensionTag, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_normal;

template <int NumberElements, int NQuadIntersection, typename MemorySpace,
          typename MemoryTraits>
struct nonconforming_intersection_normal<NumberElements, NQuadIntersection,
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
  using IntersectionNormalViewType =
      Kokkos::View<typename specfem::datatype::simd<type_real, UseSIMD>::
                       datatype[NumberElements][NQuadIntersection][2],
                   MemorySpace, MemoryTraits>; ///< Underlying view used to
                                               ///< store data of the transfer
                                               ///< function.

public:
  IntersectionNormalViewType intersection_normal;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param intersection_normal Normal vector at intersection quadrature points
   */
  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_normal(
      const IntersectionNormalViewType &intersection_normal)
      : intersection_normal(intersection_normal) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_normal() = default;

  /**
   * @brief Constructor that initializes data views in Scratch
   * Memory.
   *
   * @tparam MemberType Kokos team member type.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION
  nonconforming_intersection_normal(const MemberType &team)
      : intersection_normal(team.team_scratch(0)) {}

  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() {
    return IntersectionNormalViewType::shmem_size();
  }
};

} // namespace impl
} // namespace specfem::chunk_edge
