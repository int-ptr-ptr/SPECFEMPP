#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <string>
#include <type_traits>

namespace specfem::test_fixture {

template <typename QuadraturePointsType> struct QuadratureRule {
  static_assert(std::is_base_of_v<QuadraturePoints::QuadraturePoints,
                                  QuadraturePointsType>,
                "QuadratureRule template parameter expects QuadraturePoints!");

  static constexpr int nquad = QuadraturePointsType::nquad;
  static constexpr std::array<double, nquad> quadrature_points =
      QuadraturePointsType::quadrature_points;

  static type_real evaluate_lagrange_polynomial(const int &iquad,
                                                const type_real &x) {
    double val = 1;
    for (int i = 0; i < nquad; i++) {
      if (i != iquad) {
        val *= (x - quadrature_points[i]) /
               (quadrature_points[iquad] - quadrature_points[i]);
      }
    }
    return (type_real)val;
  }

  /**
   * @brief Provides the corresponding lagrange interpolation polynomial over
   * the basis {x^k}, that is, $L_{iquad} = \sum_k
   * computelagrangepolynomialcoefficients(iquad)[k] * x^k$
   *
   * @param iquad index of the lagrange interpolating polynomial to find
   * @return std::array<double, nquad> the array of coefficients
   */
  static std::array<double, nquad>
  compute_lagrange_polynomial_coefficients(const int &iquad) {
    std::array<double, nquad> coeffs{ 0 };
    coeffs[0] = 1;

    for (int i = 0; i < nquad; ++i) {
      if (i != iquad) {
        // *= (x - quadrature_points[i])/(quadrature_points[iquad] -
        // quadrature_points[i])
        double factor = 1 / (quadrature_points[iquad] - quadrature_points[i]);
        coeffs[nquad - 1] = coeffs[nquad - 2] * factor;
        for (int j = nquad - 2; j >= 0; --j) {
          coeffs[j] =
              (coeffs[j - 1] - coeffs[j] * quadrature_points[i]) * factor;
        }
      }
    }
    return coeffs;
  }

  static double
  compute_lagrange_quadrature_weight(const int &iquad,
                                     const double &integral_start = -1,
                                     const double &integral_end = 1) {
    std::array<double, nquad> L =
        compute_lagrange_polynomial_coefficients(iquad);

    // evaluate integral at integral_end and integral_start
    double result = 0;

    double start_pow = integral_start;
    double end_pow = integral_end;
    for (int i = 0; i < nquad; i++) {
      // add L_integral[k] * integral_end^k - L_integral[k] * integral_start^k
      result += (L[i] / (i + 1)) * (end_pow - start_pow);
      start_pow *= integral_start;
      end_pow *= integral_end;
    }

    return result;
  }
};
namespace QuadraturePoints {

struct GLL1 : QuadraturePoints {
  static constexpr int nquad = 2;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 1 };

  static std::string description() { return "GLL1 (-1, 1)"; }
};
struct GLL2 : QuadraturePoints {
  static constexpr int nquad = 3;
  static constexpr std::array<double, nquad> quadrature_points = { -1, 0, 1 };
  static std::string description() { return "GLL2 (-1, 0, 1)"; }
};

struct Asymm5Point : QuadraturePoints {
  static constexpr int nquad = 5;
  static constexpr std::array<double, nquad> quadrature_points = { -1, -0.8,
                                                                   -0.5, 0.2,
                                                                   0.7 };
  static std::string description() {
    return "5 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};
struct Asymm4Point : QuadraturePoints {
  static constexpr int nquad = 4;
  static constexpr std::array<double, nquad> quadrature_points = { -0.3, 0, 0.4,
                                                                   0.6 };
  static std::string description() {
    return "4 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};

} // namespace QuadraturePoints

} // namespace specfem::test_fixture
