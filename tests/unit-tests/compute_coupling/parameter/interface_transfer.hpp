#pragma once
#include "specfem/chunk_edge.hpp"
#include <memory>
#include <type_traits>
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

namespace specfem::testing::interface_transfer {

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
};

/**
 * @brief Iterator for interface transfers
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 */
template <specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
struct Iterator {
public:
  using InterfaceTransferType =
      InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection>;

  // impl and unique_ptr is used to hide away the abstract nature of the
  // iterator.
  struct IteratorImpl {
    virtual const InterfaceTransferType &operator*() const = 0;
    virtual void operator++() = 0;
    virtual std::unique_ptr<IteratorImpl> copy() const = 0;
    virtual ~IteratorImpl() {}
  };

private:
  std::unique_ptr<IteratorImpl> iter;

public:
  Iterator(std::unique_ptr<IteratorImpl> &&iter) : iter(std::move(iter)) {}
  Iterator(const Iterator &other) { *this = other; }
  Iterator() = default;
  void operator=(const Iterator &other) {
    if (other.iter == nullptr) {
      iter = nullptr;
    } else {
      iter = other.iter->copy();
    }
  }
  const InterfaceTransferType &operator*() const { return **iter; };
  void operator++() { ++(*iter); };
  const InterfaceTransferType &next() {
    ++(*this);
    return **this;
  }
};

/**
 * @brief Abstract generator for interface transfers. As a type of fixture, a
 * Generator should not be modified by iteration. Instead, it spawns iterators
 * that hold the iteration state.
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 */
template <specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
class Generator {
protected:
  using IteratorType = Iterator<DimensionTag, nquad_edge, nquad_intersection>;
  using IteratorPtrType = std::unique_ptr<typename IteratorType::IteratorImpl>;
  using InterfaceTransferType =
      InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection>;

  virtual IteratorPtrType get_iterator_impl() const = 0;

public:
  IteratorType iterator() const { return IteratorType(get_iterator_impl()); };

  virtual ~Generator() {}
};

/**
 * @brief Generator based on looping a std::vector of InterfaceTransfers.
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 */
template <specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
class Vector
    : public Generator<DimensionTag, nquad_edge, nquad_intersection>,
      public std::vector<std::shared_ptr<
          InterfaceTransfer<DimensionTag, nquad_edge, nquad_intersection> > > {
private:
  using BaseGeneratorType =
      Generator<DimensionTag, nquad_edge, nquad_intersection>;

public:
  struct Iterator
      : public specfem::testing::interface_transfer::Iterator<
            DimensionTag, nquad_edge, nquad_intersection>::IteratorImpl {
    const Vector &parent;
    const int num_elems;
    int current_element;

  public:
    Iterator(const Vector &parent, const int &initial_element = 0)
        : parent(parent), num_elems(std::max((int)parent.size(), 1)),
          current_element(((initial_element % num_elems) + num_elems) %
                          num_elems) {}

    const typename BaseGeneratorType::InterfaceTransferType &
    operator*() const override {
      return *(parent[current_element]);
    }

    void operator++() override {
      current_element = (current_element + 1) % num_elems;
    }

    typename BaseGeneratorType::IteratorPtrType copy() const override {
      return std::make_unique<Iterator>(parent, current_element);
    }
  };

public:
  typename BaseGeneratorType::IteratorPtrType get_iterator_impl() const {
    return std::make_unique<Iterator>(*this);
  };

  /**
   * @brief Construct a new Vector object, whose elements are from the
   * arguments.
   */
  template <typename... Args> Vector(const Args &...args) {
    ( // fold expansion: calls this function for each arg
        [&] {
          this->push_back(
              std::make_shared<std::decay_t<decltype(args)> >(args));
        }(),
        ...);
  }
};

} // namespace specfem::testing::interface_transfer
