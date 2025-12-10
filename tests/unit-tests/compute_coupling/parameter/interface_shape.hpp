#pragma once
#include "analytical_fixtures/field.hpp"
#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "interface_transfer.hpp"
#include "specfem/point/coordinates.hpp"
#include <memory>

namespace specfem::testing::interface_shape {

/**
 * @brief A testing parameter that sets the shape of the interface being tested.
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 */
template <specfem::dimension::type DimensionTag> class InterfaceShapeBase {
protected:
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  using CoordinateType = specfem::point::global_coordinates<dimension_tag>;
  using VectorType =
      specfem::datatype::VectorPointViewType<type_real, ndim, false>;

public:
  virtual CoordinateType coordinate(const type_real &edge_coord) const = 0;
  virtual VectorType normal(const type_real &edge_coord) const = 0;

  /**
   * @brief Populates the `intersection_normal` view, of shape
   * `(nquad_intersection, ndim)`
   *
   * @tparam ViewType - type of the `intersection_normal` object
   * @param intersection_normal - the view to set.
   * @param interface_transfer - the InterfaceTransfer, containing reference
   * coordinates for sampling.
   */
  template <typename ViewType, typename InterfaceTransferType>
  void set_intersection_normal(
      const ViewType &intersection_normal,
      const InterfaceTransferType &interface_transfer) const;

  /**
   * @brief Populates the `edge_field` view, of shape
   * `(nquad_edge,)`
   *
   * @tparam is_self - `true` if `edge_field` corresponds to the self
   * side. `false` otherwise.
   * @tparam ViewType - type of the `edge_field` object
   * @param edge_field - the view to set.
   * @param field - the field to sample from.
   * @param interface_transfer - the InterfaceTransfer, containing reference
   * coordinates for sampling.
   */
  template <bool is_self, typename ViewType, typename InterfaceTransferType>
  void set_edge_field(
      const ViewType &edge_field,
      const specfem::test::analytical::field::FieldBase<DimensionTag> &field,
      const InterfaceTransferType &interface_transfer) const {
    const auto &edge_points = [&]() -> auto & {
      if constexpr (is_self) {
        return interface_transfer.edge_quadrature_points_self;
      } else {
        return interface_transfer.edge_quadrature_points_coupled;
      }
    }();

    const int nquad = InterfaceTransferType::nquad_edge;

    for (int iedge = 0; iedge < nquad; iedge++) {
      edge_field(iedge) = field.eval(coordinate(edge_points[iedge]));
    }
  }

  virtual ~InterfaceShapeBase() {}
};

/**
 * @brief Iterator for interface shapes
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 */
template <specfem::dimension::type DimensionTag> struct Iterator {
public:
  using InterfaceShapeType = InterfaceShapeBase<DimensionTag>;

  // impl and unique_ptr is used to hide away the abstract nature of the
  // iterator.
  struct IteratorImpl {
    virtual const InterfaceShapeType &operator*() const = 0;
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
  const InterfaceShapeType &operator*() const { return **iter; };
  void operator++() { ++(*iter); };
  const InterfaceShapeType &next() {
    ++(*this);
    return **this;
  }
};

/**
 * @brief Abstract iterator for interface shapes
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 */
template <specfem::dimension::type DimensionTag> class Generator {
protected:
  using IteratorType = Iterator<DimensionTag>;
  using IteratorPtrType = std::unique_ptr<typename IteratorType::IteratorImpl>;
  using InterfaceShapeType = InterfaceShapeBase<DimensionTag>;

  virtual IteratorPtrType get_iterator_impl() const = 0;

public:
  IteratorType iterator() const { return IteratorType(get_iterator_impl()); };

  virtual ~Generator() {}
};

/**
 * @brief interface is flat (straight line at given angle)
 */
class Flat2D : public InterfaceShapeBase<specfem::dimension::type::dim2> {
private:
  type_real tangent_x;
  type_real tangent_z;

  type_real normal_x;
  type_real normal_z;

public:
  CoordinateType coordinate(const type_real &edge_coord) const override {
    return { edge_coord * tangent_x, edge_coord * tangent_z };
  };
  VectorType normal(const type_real &edge_coord) const override {
    return { edge_coord * normal_x, edge_coord * normal_z };
  }

  Flat2D(const type_real &theta, const bool &flip_normal = false)
      : tangent_x(std::cos(theta)), tangent_z(std::sin(theta)),
        normal_x(flip_normal ? (-tangent_z) : tangent_z),
        normal_z(flip_normal ? tangent_x : (-tangent_x)) {}
  Flat2D() {}
};

/**
 * @brief section of circle, centered at (0,0).
 */
class Arc : public InterfaceShapeBase<specfem::dimension::type::dim2> {
private:
  type_real radius;
  bool outward_normal;
  type_real angle_start;
  type_real angle_scale;

public:
  CoordinateType coordinate(const type_real &edge_coord) const override {
    const type_real angle = angle_start + edge_coord * angle_scale;
    return { radius * std::cos(angle), radius * std::sin(angle) };
  };
  VectorType normal(const type_real &edge_coord) const override {
    const type_real angle = angle_start + edge_coord * angle_scale;
    return outward_normal ? VectorType(std::cos(angle), std::sin(angle))
                          : VectorType(-std::cos(angle), -std::sin(angle));
  }

  Arc(const type_real &radius, const type_real &angle_start,
      const type_real &angle_scale, const bool &outward_normal = true)
      : radius(radius), angle_scale(angle_scale), angle_start(angle_start),
        outward_normal(outward_normal) {}
};

class RandomFlat2DGenerator : public Generator<specfem::dimension::type::dim2> {
private:
  using BaseGeneratorType = Generator<specfem::dimension::type::dim2>;

public:
  struct Iterator : public specfem::testing::interface_shape::Iterator<
                        specfem::dimension::type::dim2>::IteratorImpl {
    std::shared_ptr<InterfaceShapeBase<specfem::dimension::type::dim2> >
        current_shape;
    int next_seed;
    const RandomFlat2DGenerator &parent;

  public:
    Iterator(
        const RandomFlat2DGenerator &parent, const int &next_seed,
        std::shared_ptr<InterfaceShapeBase<specfem::dimension::type::dim2> >
            current_shape = nullptr)
        : parent(parent), next_seed(next_seed), current_shape(current_shape) {
      if (current_shape == nullptr) {
        operator++();
      }
    }

    const typename BaseGeneratorType::InterfaceShapeType &
    operator*() const override {
      return *current_shape;
    }

    void operator++() override {
      std::srand(next_seed);
      current_shape =
          std::make_shared<Flat2D>(((type_real)std::rand()) / RAND_MAX, false);
      next_seed = std::rand();
    }

    std::unique_ptr<IteratorImpl> copy() const override {
      return std::make_unique<Iterator>(parent, next_seed, current_shape);
    }
  };

private:
  int seed_offset;

public:
  RandomFlat2DGenerator(int seed_offset = 0) : seed_offset(seed_offset) {}

  typename BaseGeneratorType::IteratorPtrType
  get_iterator_impl() const override {
    return std::make_unique<Iterator>(*this, seed_offset);
  };
};

template <>
template <typename ViewType, typename InterfaceTransferType>
void InterfaceShapeBase<specfem::dimension::type::dim2>::
    set_intersection_normal(
        const ViewType &intersection_normal,
        const InterfaceTransferType &interface_transfer) const {
  constexpr int nquad = InterfaceTransferType::nquad_intersection;

  for (int iintersection = 0; iintersection < nquad; iintersection++) {
    const auto normal = this->normal(
        interface_transfer.intersection_quadrature_points[iintersection]);
    for (int idim = 0; idim < ndim; idim++) {
      intersection_normal(iintersection, idim) = normal(idim);
    }
  }
}

} // namespace specfem::testing::interface_shape
