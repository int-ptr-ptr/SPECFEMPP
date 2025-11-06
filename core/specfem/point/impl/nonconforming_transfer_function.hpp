#pragma once
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <type_traits>

namespace specfem::point {

/*
 * The definition in the impl namespace is used to define the transfer functions
 * without the accessor, so that the multiple inheritance of
 * nonconforming_coupled_interface isn't ambiguous. Outside of the impl
 * namespace, we re-endow the Accessor type, so that it can be accessed
 * independently.
 */

namespace impl {
/**
 * @brief The point container for the transfer function. This maps a single
 * nodal basis function (Kronecker delta property) onto the interface, which is
 * used for integration, when a thread needs to only know that basis function.
 *
 * @tparam IsSelf self side of InterfaceTag or coupled (other) side.
 * @tparam NQuadIntersection Number of quadrature points on each mortar element.
 * @tparam DimensionTag Spatial dimension
 * @tparam ConnectionTag Connection type (strongly/weakly conforming)
 * @tparam InterfaceTag Interface type (elastic-acoustic, acoustic-elastic)
 * @tparam BoundaryTag Boundary condition type
 */
template <bool IsSelf, int NQuadIntersection,
          specfem::dimension::type DimensionTag>
struct nonconforming_transfer_function;

template <int NQuadIntersection>
struct nonconforming_transfer_function<true, NQuadIntersection,
                                       specfem::dimension::type::dim2> {
private:
  static constexpr bool UseSIMD = false;
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

protected:
  using TransferViewType =
      specfem::datatype::VectorPointViewType<type_real, NQuadIntersection,
                                             UseSIMD>; ///< Underlying view used
                                                       ///< to store data of the
                                                       ///< transfer function.

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
};

template <int NQuadIntersection>
struct nonconforming_transfer_function<false, NQuadIntersection,
                                       specfem::dimension::type::dim2> {
private:
  static constexpr bool UseSIMD = false;
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type.

protected:
  using TransferViewType =
      specfem::datatype::VectorPointViewType<type_real, NQuadIntersection,
                                             UseSIMD>; ///< Underlying view used
                                                       ///< to store data of the
                                                       ///< transfer function.

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
};

} // namespace impl
} // namespace specfem::point
