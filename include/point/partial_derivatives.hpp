#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store spatial derivatives of the basis functions at a quadrature point
 *
 * @tparam DimensionType Dimension of the spectral element
 * @tparam StoreJacobian Boolean indicating whether to store the Jacobian
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <specfem::dimension::type DimensionType, bool StoreJacobian,
          bool UseSIMD>
struct partial_derivatives;

/**
 * @brief Template specialization for 2D spectral elements without storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD data type
  using value_type =
      typename simd::datatype; ///< Datatype for storing values. Is a scalar if
                               ///< UseSIMD is false, otherwise is a SIMD
                               ///< vector.
  constexpr static bool store_jacobian = false;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  ///@}

  value_type xix;    ///< @xix
  value_type gammax; ///< @gammax
  value_type xiz;    ///< @xiz
  value_type gammaz; ///< @gammaz

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiz, const value_type &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : xix(constant), gammax(constant), xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    return;
  }
};

/**
 * @brief Template specialization for 2D spectral elements with storing the
 * Jacobian
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD
 */
template <bool UseSIMD>
struct partial_derivatives<specfem::dimension::type::dim2, true, UseSIMD>
    : public partial_derivatives<specfem::dimension::type::dim2, false,
                                 UseSIMD> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD data type
  using value_type = typename simd::datatype; ///< Datatype for storing values.
                                              ///< Is a scalar if UseSIMD is
                                              ///< false, otherwise is a SIMD
                                              ///< vector.
  constexpr static bool store_jacobian = true;
  constexpr static auto dimension = specfem::dimension::type::dim2;
  ///@}

  value_type jacobian; ///< Jacobian

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
  partial_derivatives() = default;

  /**
   * @brief Constructor with values
   *
   * @param xix @xix
   * @param gammax @gammax
   * @param xiz @xiz
   * @param gammaz @gammaz
   * @param jacobian Jacobian
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type &xix, const value_type &gammax,
                      const value_type &xiz, const value_type &gammaz,
                      const value_type &jacobian)
      : partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD>(
            xix, gammax, xiz, gammaz),
        jacobian(jacobian) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  partial_derivatives(const value_type constant)
      : partial_derivatives<specfem::dimension::type::dim2, false, UseSIMD>(
            constant),
        jacobian(constant) {}
  ///@}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  /**
   * @name Member functions
   *
   */
  ///@{

  /**
   * @brief Compute the normal vector at a quadrature point
   *
   * @param type Type of edge (bottom, top, left, right)
   * @return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
   * Normal vector
   */
  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  compute_normal(const specfem::enums::edge::type &type) const;
  ///@}

private:
  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax * this->jacobian),
             static_cast<value_type>(this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->xix *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiz *
                                     this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix * this->jacobian),
             static_cast<value_type>(this->xiz * this->jacobian) };
  };
};

// operator+
template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::store_jacobian, int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType
operator+(const PointPartialDerivativesType &lhs,
          const PointPartialDerivativesType &rhs) {
  return PointPartialDerivativesType(lhs.xix + rhs.xix, lhs.gammax + rhs.gammax,
                                     lhs.xiz + rhs.xiz,
                                     lhs.gammaz + rhs.gammaz);
}

// operator+=
template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::store_jacobian, int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType &
operator+=(PointPartialDerivativesType &lhs,
           const PointPartialDerivativesType &rhs) {
  lhs.xix += rhs.xix;
  lhs.gammax += rhs.gammax;
  lhs.xiz += rhs.xiz;
  lhs.gammaz += rhs.gammaz;
  return lhs;
}

// operator*
template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::store_jacobian, int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType
operator*(const PointPartialDerivativesType &lhs, const type_real &rhs) {
  return PointPartialDerivativesType(lhs.xix * rhs, lhs.gammax * rhs,
                                     lhs.xiz * rhs, lhs.gammaz * rhs);
}

// operator*
template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::store_jacobian, int> = 0>
KOKKOS_FUNCTION PointPartialDerivativesType
operator*(const type_real &lhs, const PointPartialDerivativesType &rhs) {
  return PointPartialDerivativesType(lhs * rhs.xix, lhs * rhs.gammax,
                                     lhs * rhs.xiz, lhs * rhs.gammaz);
}
} // namespace point
} // namespace specfem
