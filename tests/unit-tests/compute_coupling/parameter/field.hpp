#include "datatypes/point_view.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/point/coordinates.hpp"
#include <unordered_map>

namespace specfem::testing::field {

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
};
template <specfem::dimension::type DimensionTag> class Generator {
public:
  virtual const FieldBase<DimensionTag> &get_field(const int &index) const = 0;
  virtual int get_generator_size() const = 0;
};

class Polynomial2D : public FieldBase<specfem::dimension::type::dim2> {
  std::map<std::pair<int, int>, type_real> coefficients;

public:
  Polynomial2D(const std::map<std::pair<int, int>, type_real> &coefficients)
      : coefficients(coefficients) {}

  virtual type_real
  eval(const specfem::point::global_coordinates<dimension_tag> &coord) const {
    type_real v = 0;
    for (const auto &[pows, coef] : coefficients) {
      const auto &[pow_x, pow_z] = pows;
      v += coef * std::pow(coord.x, pow_x) * std::pow(coord.z, pow_z);
    }
    return v;
  }
  virtual VectorType gradient(
      const specfem::point::global_coordinates<dimension_tag> &coord) const {
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

class RandomPolynomial2DGenerator
    : public Generator<specfem::dimension::type::dim2> {
  std::vector<Polynomial2D> entries;

public:
  RandomPolynomial2DGenerator(int generator_size, int max_degree,
                              int seed_offset = 0) {

    std::srand(seed_offset + generator_size + max_degree * 3723392557);
    for (int i = 0; i < generator_size; i++) {
      int degree = std::rand() % max_degree;
      int deg_x = std::rand() % (degree + 1);
      entries.push_back(Polynomial2D({ { { deg_x, degree - deg_x }, 1 } }));
    }
  }

  virtual const FieldBase<specfem::dimension::type::dim2> &
  get_field(const int &index) const {
    return entries[index];
  }
  virtual int get_generator_size() const { return entries.size(); };
};

} // namespace specfem::testing::field
