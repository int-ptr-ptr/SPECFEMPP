#include "initializers.hpp"
#include "specfem_setup.hpp"

#include "../impl/analytical_functions.hpp"
#include "../impl/descriptions.hpp"

namespace specfem::test_fixture {
template <typename Initializer> struct IntersectionFactor2D {
  static_assert(
      std::is_base_of_v<specfem::test_fixture::IntersectionFactorInitializer2D::
                            IntersectionFactorInitializer2D,
                        Initializer>,
      "IntersectionFactor2D needs an IntersectionFactorInitializer2D!");

public:
  using IntersectionFactorInitializer = Initializer;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static constexpr int num_edges = Initializer::num_edges;
  static constexpr int nquad = Initializer::nquad;

  static std::string description(const int &indent = 0) {
    return specfem::test_fixture::impl::description<Initializer>(indent);
  }

private:
  std::array<std::array<type_real, nquad>, num_edges> _intersection_factor;
  using IntersectionFactorView =
      Kokkos::View<type_real[num_edges][nquad], memory_space>;

public:
  /**
   * @brief Construct intersection factor with initializer.
   * @param initializer Initialization strategy
   */
  IntersectionFactor2D(const Initializer &initializer) {
    _intersection_factor = Initializer::init_intersection_factor();
  }

  /**
   * @brief Get Kokkos view of intersection factor data.
   * @return Kokkos view for device access
   */
  IntersectionFactorView get_view() const {
    IntersectionFactorView view("transfer_function_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad; ++j) {
        view(i, j) = (*this)(i, j);
      }
    }
    return view;
  }

  /**
   * @brief Access intersection factor values.
   * @param i Edge index
   * @param j Quadrature index
   * @return Reference to intersection factor value
   */
  type_real &operator()(const int i, const int j) {
    return _intersection_factor[i][j];
  }

  const type_real &operator()(const int i, const int j) const {
    return _intersection_factor[i][j];
  }
};

namespace IntersectionFactorInitializer2D {

/**
 * @brief Initializes the IntersectionFactor with the weights corresponding to
 * an integral over [-1,1] in intersection local coordinates.
 *
 * @tparam QuadraturePointsType Knots of the quadrature.
 * @tparam IntersectionCoordinateField Local -> global mapping of the
 * intersection.
 */
template <typename QuadraturePointsType, typename IntersectionCoordinateField>
struct FromIntersectionKnots : IntersectionFactorInitializer2D {
  using QuadraturePoints = QuadraturePointsType;
  static_assert(std::is_base_of_v<
                    specfem::test_fixture::QuadraturePoints::QuadraturePoints,
                    QuadraturePoints>,
                "FromIntersectionKnots IntersectionFactorInitializer needs a "
                "QuadraturePoints parameter in its first argument!");

  using IntersectionCoordinates = IntersectionCoordinateField;
  static_assert(
      std::is_base_of_v<
          specfem::test_fixture::AnalyticalFunctionType::AnalyticalFunctionType,
          IntersectionCoordinates>,
      "FromIntersectionKnots IntersectionFactorInitializer needs an "
      "AnalyticalFunctionType "
      "parameter in its second argument!");
  static_assert(
      IntersectionCoordinates::num_components == 2,
      "FromIntersectionKnots needs IntersectionCoordinates to map to R2!");

  static constexpr int num_edges = 1;
  static constexpr int nquad = QuadraturePoints::nquad;

private:
  using ArrayType = std::array<std::array<type_real, nquad>, num_edges>;

public:
  static ArrayType init_transfer_function() {
    ArrayType arr;
    for (int i = 0; i < num_edges; ++i) {
      for (int j = 0; j < nquad; ++j) {
        const auto &diff = specfem::test_fixture::impl::
            path_derivative<IntersectionCoordinates>::evaluate(
                QuadraturePoints::quadrature_points[j]);
        const auto &quad_weight = QuadratureRule<
            QuadraturePoints>::compute_lagrange_quadrature_weight(j);
        arr[i][j] =
            (type_real)(std::sqrt(diff[0] * diff[0] + diff[1] * diff[1]) *
                        quad_weight);
      }
    }
    return arr;
  }
};
} // namespace IntersectionFactorInitializer2D
} // namespace specfem::test_fixture
