#include "specfem/chunk_edge.hpp"

namespace specfem::testing::interface_transfer {

template <specfem::dimension::type DimensionTag, typename Visitor>
struct InterfaceTransferBase {
public:
  virtual void accept(Visitor &) const = 0;
};

template <specfem::dimension::type DimensionTag, typename Visitor>
class Generator {
public:
  virtual const InterfaceTransferBase<DimensionTag, Visitor> &
  get_interface_transfer(const int &index) const = 0;
  virtual int get_generator_size() const = 0;
};

/**
 * @brief Specialized version of InterfaceTransferBase, containing the
 * quadrature points of both the intersection and edges. The virtual method
 * `run_test()` is specialized to the given template parameters
 *
 * @tparam nquad_edge_ - the number of quadrature points on (dimension of) the
 * edge
 * @tparam nquad_intersection_ - the number of quadrature points on (dimension
 * of) the intersections
 * @tparam Visitor - a class that accepts InterfaceTransfer instances with a
 * templated call operator.
 */
template <specfem::dimension::type DimensionTag, typename Visitor,
          int nquad_edge_, int nquad_intersection_>
class InterfaceTransfer : public InterfaceTransferBase<DimensionTag, Visitor> {
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

  virtual void accept(Visitor &visitor) const { visitor(*this); }

  InterfaceTransfer(const InterfaceTransfer<DimensionTag, Visitor, nquad_edge,
                                            nquad_intersection> &orig)
      : edge_quadrature_points_self(orig.edge_quadrature_points_self),
        intersection_quadrature_points(orig.intersection_quadrature_points),
        edge_quadrature_points_coupled(orig.edge_quadrature_points_coupled) {}
};

template <specfem::dimension::type DimensionTag, typename Visitor>
class Vector
    : public Generator<DimensionTag, Visitor>,
      public std::vector<
          std::shared_ptr<InterfaceTransferBase<DimensionTag, Visitor> > > {
public:
  virtual const InterfaceTransferBase<DimensionTag, Visitor> &
  get_interface_transfer(const int &index) const {
    return *((*this)[this->get_generator_size() - index - 1]);
  }
  virtual int get_generator_size() const { return this->size(); }

  template <typename... Args, typename tail_type>
  Vector(const tail_type &head, const Args &...args) : Vector(args...) {
    this->push_back(std::make_shared<tail_type>(head));
  }
  Vector() = default;
};

} // namespace specfem::testing::interface_transfer
