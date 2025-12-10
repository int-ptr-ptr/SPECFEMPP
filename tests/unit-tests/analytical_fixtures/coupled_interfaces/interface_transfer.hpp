#pragma once
#include "analytical_fixtures/field.hpp"
#include "specfem/chunk_edge.hpp"
#include <memory>
#include <sstream>
#include <vector>

/**
 * @brief Evaluates the Lagrange interpolation polynomials at a given point x:
 *     $$L_j(x) = \prod_{k \ne j} \frac{x - \xi_k}{\xi_j - \xi_k}$$
 *
 * @tparam nquad - number of quadrature points, and size of the
 * `quadrature_points` array
 * @param quadrature_points - the array of $\xi_k$
 * @param poly_index - the index of the lagrange polynomial to evaluate
 * @param x - the point to evaluate at
 * @return type_real - the evaluated $L_j(x)$, where $j$ is given by
 * `poly_index`
 */
template <std::size_t nquad>
static KOKKOS_FORCEINLINE_FUNCTION type_real
eval_lagrange(const std::array<type_real, nquad> &quadrature_points,
              const int &poly_index, const type_real &x) {
  type_real val = 1;
  for (int i = 0; i < nquad; i++) {
    if (i != poly_index) {
      val *= (x - quadrature_points[i]) /
             (quadrature_points[poly_index] - quadrature_points[i]);
    }
  }
  return val;
}

namespace specfem::test::analytical::interface_transfer {

/**
 * @brief Contains the quadrature points of the intersection and both edges in a
 * globalized interface coordinate system.
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 */
template <specfem::dimension::type DimensionTag, int nquad_edge_,
          int nquad_intersection_>
class InterfaceTransfer {
public:
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr int nquad_edge = nquad_edge_;
  static constexpr int nquad_intersection = nquad_intersection_;

  std::array<type_real, nquad_edge_> edge_quadrature_points_self;
  std::array<type_real, nquad_edge_> edge_quadrature_points_coupled;
  std::array<type_real, nquad_intersection_> intersection_quadrature_points;

  // takes const array& (lvalue), converts to std::array, which requires data
  // copying. for some reason, I could not get initializer lists to work with
  // std::array.
  InterfaceTransfer(
      const type_real (&edge_quadrature_points_self)[nquad_edge_],
      const type_real (&intersection_quadrature_points)[nquad_intersection_],
      const type_real (&edge_quadrature_points_coupled)[nquad_edge_]) {
    std::copy(std::begin(edge_quadrature_points_self),
              std::end(edge_quadrature_points_self),
              std::begin(this->edge_quadrature_points_self));
    std::copy(std::begin(edge_quadrature_points_coupled),
              std::end(edge_quadrature_points_coupled),
              std::begin(this->edge_quadrature_points_coupled));
    std::copy(std::begin(intersection_quadrature_points),
              std::end(intersection_quadrature_points),
              std::begin(this->intersection_quadrature_points));
  }

  InterfaceTransfer(const InterfaceTransfer<DimensionTag, nquad_edge,
                                            nquad_intersection> &orig)
      : edge_quadrature_points_self(orig.edge_quadrature_points_self),
        intersection_quadrature_points(orig.intersection_quadrature_points),
        edge_quadrature_points_coupled(orig.edge_quadrature_points_coupled) {}

  /**
   * @brief Populates the `transfer_function` view, of shape
   * `(nquad_edge, nquad_intersection)`
   *
   * @tparam is_self - `true` if `transfer_function` corresponds to the self
   * side. `false` otherwise.
   * @tparam ViewType - type of the `transfer_function` object
   * @param transfer_function - the view to set.
   */
  template <bool is_self, typename ViewType>
  void set_transfer_function(const ViewType &transfer_function) const {
    const std::array<type_real, nquad_edge_> &edge_points =
        [&]() -> const std::array<type_real, nquad_edge_> & {
      if constexpr (is_self) {
        return edge_quadrature_points_self;
      } else {
        return edge_quadrature_points_coupled;
      }
    }();

    for (int iedge = 0; iedge < nquad_edge; iedge++) {
      for (int iintersection = 0; iintersection < nquad_intersection;
           iintersection++) {
        transfer_function(iedge, iintersection) = eval_lagrange(
            edge_points, iedge, intersection_quadrature_points[iintersection]);
      }
    }
  }

  std::string verbose_intersection_point(const int &iintersection) const {
    std::ostringstream oss;

    oss << "intersection_points[" << iintersection
        << "] == " << intersection_quadrature_points[iintersection];

    return oss.str();
  }
};

template <specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
struct Generator {
private:
  std::vector<std::shared_ptr<
      InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> > >
      interface_transfers;

public:
  Generator() {};

  Generator &add_interface_transfer(
      const InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection>
          &interface_shape) {
    interface_transfers.push_back(
        std::make_shared<
            InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> >(
            interface_shape));
    return *this;
  }

  RoundRobinIterator<
      InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> >
  iterator() const {
    return RoundRobinIterator<
        InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> >(
        &interface_transfers);
  }
};

template <specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
using Iterator = RoundRobinIterator<
    InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> >;

const Generator<dimension::type::dim2, 5, 4> interface_transfer_2d_5_4 =
    Generator<dimension::type::dim2, 5, 4>()
        .add_interface_transfer({ { -1, -0.5, 0, 0.5, 1 },
                                  { -0.8, -0.4, 0.4, 0.8 },
                                  { -1, -0.5, 0, 0.5, 1 } })
        .add_interface_transfer({ { -1, -0.5, 0, 0.3, 0.5 },
                                  { -0.38, -0.3, 0.2, 0.38 },
                                  { -0.25, -0.1, 0, 0.5, 1 } });

const Generator<dimension::type::dim2, 2, 2> interface_transfer_2d_2_2 =
    Generator<dimension::type::dim2, 2, 2>().add_interface_transfer(
        { { -1, 1 }, { 0, 1 }, { 0, 2 } });

} // namespace specfem::test::analytical::interface_transfer
