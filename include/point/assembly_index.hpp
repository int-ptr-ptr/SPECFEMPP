#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Struct to store the assembled index for a quadrature point
 *
 * This struct stores a 1D index that corresponds to a global numbering of the
 * quadrature point within the mesh.
 *
 */
struct assembly_index {
  int iglob; ///< Global index number of the quadrature point

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  assembly_index() = default;

  /**
   * @brief Constructor with values
   *
   * @param iglob Global index number of the quadrature point
   */
  KOKKOS_FUNCTION
  assembly_index(const int &iglob) : iglob(iglob) {}
  ///@}
};

/**
 * @brief Struct to store the SIMD assembled indices for a quadrature point
 *
 * SIMD indices are intended to be used for loading @c load_on_device and
 * storing @c store_on_device data into SIMD vectors and operating on those data
 * using SIMD instructions.
 *
 */
struct simd_assembly_index {
  int number_points; ///< Number of points in the SIMD vector
  int iglob;         ///< Global index number of the quadrature point

  /**
   * @brief Mask function to determine if a lane is valid
   *
   * @param lane Lane index
   */
  KOKKOS_FUNCTION
  bool mask(const std::size_t &lane) const { return int(lane) < number_points; }

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  simd_assembly_index() = default;

  /**
   * @brief Constructor with values
   *
   * @param iglob Global index number of the quadrature point
   * @param number_points Number of points in the SIMD vector
   */
  KOKKOS_FUNCTION
  simd_assembly_index(const int &iglob, const int &number_points)
      : number_points(number_points), iglob(iglob) {}
  ///@}
};
} // namespace point
} // namespace specfem