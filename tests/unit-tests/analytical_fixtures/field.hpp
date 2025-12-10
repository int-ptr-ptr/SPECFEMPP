#pragma once
#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "specfem/point/coordinates.hpp"

#include "round_robin_iterator.hpp"

#include <memory>
#include <type_traits>

namespace specfem::test::analytical::field {

/**
 * @brief A testing parameter for a field in global coordinates that can be
 * sampled continuously.
 *
 * @tparam DimensionTag - dimension of the domain
 */
template <specfem::dimension::type DimensionTag> class FieldBase {
protected:
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

public:
  using VectorType =
      specfem::datatype::VectorPointViewType<type_real, ndim, false>;
  virtual type_real
  eval(const specfem::point::global_coordinates<DimensionTag> &) const = 0;
  virtual VectorType
  gradient(const specfem::point::global_coordinates<DimensionTag> &) const = 0;

  virtual ~FieldBase() {}
};

template <specfem::dimension::type DimensionTag> struct Generator {
private:
  std::vector<std::shared_ptr<FieldBase<DimensionTag> > > fields;

public:
  Generator() {};

  template <typename FieldType,
            std::enable_if_t<
                std::is_base_of_v<FieldBase<DimensionTag>, FieldType>, int> = 0>
  Generator &add_field(const FieldType &field) {
    fields.push_back(std::make_shared<FieldType>(field));
    return *this;
  }

  RoundRobinIterator<FieldBase<DimensionTag> > iterator() const {
    return RoundRobinIterator<FieldBase<DimensionTag> >(&fields);
  }
};

template <specfem::dimension::type DimensionTag>
using Iterator = RoundRobinIterator<FieldBase<DimensionTag> >;

class Polynomial2D : public FieldBase<specfem::dimension::type::dim2> {
  std::map<std::pair<int, int>, type_real> coefficients;

public:
  Polynomial2D(const std::map<std::pair<int, int>, type_real> &coefficients)
      : coefficients(coefficients) {}

  type_real eval(const specfem::point::global_coordinates<dimension_tag> &coord)
      const override {
    type_real v = 0;
    for (const auto &[pows, coef] : coefficients) {
      const auto &[pow_x, pow_z] = pows;
      v += coef * std::pow(coord.x, pow_x) * std::pow(coord.z, pow_z);
    }
    return v;
  }
  VectorType gradient(const specfem::point::global_coordinates<dimension_tag>
                          &coord) const override {
    type_real grad_x = 0;
    type_real grad_z = 0;
    for (const auto &[pows, coef] : coefficients) {
      const auto &[pow_x, pow_z] = pows;
      if (pow_x != 0) {
        grad_x += coef * pow_x * std::pow(coord.x, pow_x - 1) *
                  std::pow(coord.z, pow_z);
      }
      if (pow_z != 0) {
        grad_z += coef * pow_z * std::pow(coord.x, pow_x) *
                  std::pow(coord.z, pow_z - 1);
      }
    }
    return { grad_x, grad_z };
  }
};

const Generator<dimension::type::dim2> sample_fields_2d =
    Generator<dimension::type::dim2>()
        .add_field<Polynomial2D>(Polynomial2D({ { { 0, 1 }, 1 } }))
        .add_field<Polynomial2D>(Polynomial2D({ { { 1, 0 }, 1 } }))
        .add_field<Polynomial2D>(Polynomial2D({ { { 1, 1 }, 1 } }))
        .add_field<Polynomial2D>(Polynomial2D({ { { 2, 1 }, 1 } }))
        .add_field<Polynomial2D>(Polynomial2D({ { { 1, 2 }, 1 } }))
        .add_field<Polynomial2D>(
            Polynomial2D({ { { 2, 2 }, 1 }, { { 0, 0 }, -2.0 } }))
        .add_field<Polynomial2D>(
            Polynomial2D({ { { 1, 2 }, 1 }, { { 0, 1 }, -2.0 } }));

} // namespace specfem::test::analytical::field
