#pragma once

#include "specfem_setup.hpp"
#include <type_traits>

namespace specfem::test_fixture::impl {

constexpr double numerical_eval_eps = 1e-4;
constexpr double one_over_two_numerical_eval_eps = 1.0 / numerical_eval_eps;

template <typename T, typename = void> struct path_derivative {
  static constexpr bool has_analytical = false;
  static constexpr int num_components = T::num_components;
  static std::array<double, num_components> evaluate(const double &coord) {
    std::array<double, num_components> deriv;
    for (int i = 0; i < num_components; ++i) {
      deriv[i] = ((T::evaluate(coord + numerical_eval_eps) -
                   T::evaluate(coord - numerical_eval_eps)) *
                  one_over_two_numerical_eval_eps);
    }
    return deriv;
  }
};

template <typename T>
struct path_derivative<
    T, std::enable_if_t<std::is_same_v<decltype(T::evaluate_derivative(0)),
                                       std::array<double, T::num_components> >,
                        void> > {
  static constexpr bool has_analytical = true;
  static constexpr int num_components = T::num_components;
  static std::array<double, num_components> evaluate(const double &coord) {
    return T::evaluate_derivative();
  }
};

} // namespace specfem::test_fixture::impl
