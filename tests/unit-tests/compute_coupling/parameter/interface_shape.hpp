#pragma once
#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/point/coordinates.hpp"
#include <memory>

namespace specfem::testing::interface_shape {

template <specfem::dimension::type DimensionTag> class InterfaceShapeBase {
protected:
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  using CoordinateType = specfem::point::global_coordinates<dimension_tag>;
  using VectorType =
      specfem::datatype::VectorPointViewType<type_real, ndim, false>;

public:
  virtual CoordinateType coordinate(const type_real &edge_coord) = 0;
  virtual VectorType normal(const type_real &edge_coord) = 0;
};

template <specfem::dimension::type DimensionTag> class Generator {
public:
  virtual const InterfaceShapeBase<DimensionTag> &next_shape() = 0;
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
  virtual CoordinateType coordinate(const type_real &edge_coord) {
    return { edge_coord * tangent_x, edge_coord * tangent_z };
  };
  virtual VectorType normal(const type_real &edge_coord) {
    return { edge_coord * normal_x, edge_coord * normal_z };
  }

  Flat2D(const type_real &theta, const bool &flip_normal = false)
      : tangent_x(std::cos(theta)), tangent_z(std::sin(theta)),
        normal_x(flip_normal ? (-tangent_z) : tangent_z),
        normal_z(flip_normal ? tangent_x : (-tangent_x)) {}
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
  virtual CoordinateType coordinate(const type_real &edge_coord) {
    const type_real angle = angle_start + edge_coord * angle_scale;
    return { radius * std::cos(angle), radius * std::sin(angle) };
  };
  virtual VectorType normal(const type_real &edge_coord) {
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
  std::shared_ptr<InterfaceShapeBase<specfem::dimension::type::dim2> >
      current_shape;
  int next_seed;

public:
  RandomFlat2DGenerator(int seed_offset = 0) : next_seed(seed_offset) {}

  virtual const InterfaceShapeBase<specfem::dimension::type::dim2> &
  next_shape() {
    std::srand(next_seed);
    current_shape =
        std::make_shared<Flat2D>(((type_real)std::rand()) / RAND_MAX);

    next_seed = std::rand();
    return *current_shape;
  }
};

} // namespace specfem::testing::interface_shape
