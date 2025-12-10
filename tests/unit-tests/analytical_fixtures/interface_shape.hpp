#pragma once
#include "analytical_fixtures/field.hpp"
#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "specfem/point/coordinates.hpp"
#include <memory>

namespace specfem::test::analytical::interface_shape {

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

template <specfem::dimension::type DimensionTag> struct Generator {
private:
  std::vector<std::shared_ptr<InterfaceShapeBase<DimensionTag> > >
      interface_shapes;

public:
  Generator() {};

  template <typename InterfaceShapeType,
            std::enable_if_t<std::is_base_of_v<InterfaceShapeBase<DimensionTag>,
                                               InterfaceShapeType>,
                             int> = 0>
  Generator &add_interface_shape(const InterfaceShapeType &interface_shape) {
    interface_shapes.push_back(
        std::make_shared<InterfaceShapeType>(interface_shape));
    return *this;
  }

  RoundRobinIterator<InterfaceShapeBase<DimensionTag> > iterator() const {
    return RoundRobinIterator<InterfaceShapeBase<DimensionTag> >(
        &interface_shapes);
  }
};

template <specfem::dimension::type DimensionTag>
using Iterator = RoundRobinIterator<InterfaceShapeBase<DimensionTag> >;

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

const Generator<dimension::type::dim2> interface_shapes_2d =
    Generator<dimension::type::dim2>()
        .add_interface_shape<Flat2D>(Flat2D(0.1, false))
        .add_interface_shape<Flat2D>(Flat2D(2.4, true))
        .add_interface_shape<Flat2D>(Flat2D(-2.5))
        .add_interface_shape<Arc>(Arc(5.0, -1.3, 0.3, true))
        .add_interface_shape<Arc>(Arc(8.0, 3.0, 0.3, false));

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

} // namespace specfem::test::analytical::interface_shape
