#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

namespace specfem::test_fixture {

namespace AnalyticalFunctionType {

/**
 * @brief Describes a function f(x) = x^k for a power k
 *
 * @tparam power the exponent.
 */
template <int power> struct Power : AnalyticalFunctionType {
  static constexpr int num_components = 1;

  static std::array<double, num_components> evaluate(const double &coord) {
    return { std::pow(coord, power) };
  }

  static std::array<double, num_components>
  evaluate_derivative(const double &coord) {
    return { (power * std::pow(coord, power - 1)) };
  }

  static std::string description() {
    return std::string("xi^") + std::to_string(power);
  }
};
} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
