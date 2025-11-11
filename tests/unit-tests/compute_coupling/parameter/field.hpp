#pragma once
#include "datatypes/point_view.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/point/coordinates.hpp"
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace specfem::testing::field {

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

/**
 * @brief Iterator for fields
 *
 * @tparam DimensionTag - dimension of the domain - the interface has
 * codimension 1
 */
template <specfem::dimension::type DimensionTag> struct Iterator {
public:
  using FieldType = FieldBase<DimensionTag>;

  // impl and unique_ptr is used to hide away the abstract nature of the
  // iterator.
  struct IteratorImpl {
    virtual const FieldType &operator*() const = 0;
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
  const FieldType &operator*() const { return **iter; };
  void operator++() { ++(*iter); };
  const FieldType &next() {
    ++(*this);
    return **this;
  }
};

template <specfem::dimension::type DimensionTag> class Generator {
public:
  using IteratorType = Iterator<DimensionTag>;

protected:
  using IteratorPtrType = std::unique_ptr<typename IteratorType::IteratorImpl>;
  using FieldType = FieldBase<DimensionTag>;

  virtual IteratorPtrType get_iterator_impl() const = 0;

public:
  IteratorType iterator() const { return IteratorType(get_iterator_impl()); };

  virtual ~Generator() {}
};

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

class RandomPolynomial2DGenerator
    : public Generator<specfem::dimension::type::dim2> {
private:
  using BaseGeneratorType = Generator<specfem::dimension::type::dim2>;

public:
  struct Iterator : public BaseGeneratorType::IteratorType::IteratorImpl {
    std::shared_ptr<FieldBase<specfem::dimension::type::dim2> > current_field;
    int next_seed;
    const RandomPolynomial2DGenerator &parent;

  public:
    Iterator(const RandomPolynomial2DGenerator &parent, const int &next_seed,
             std::shared_ptr<FieldBase<specfem::dimension::type::dim2> >
                 current_field = nullptr)
        : parent(parent), next_seed(next_seed), current_field(current_field) {
      if (current_field == nullptr) {
        operator++();
      }
    }

    const typename BaseGeneratorType::FieldType &operator*() const override {
      return *current_field;
    }

    void operator++() override {
      std::srand(next_seed);
      int degree = std::rand() % parent.max_degree;
      int deg_x = std::rand() % (degree + 1);
      current_field = std::make_shared<Polynomial2D>(
          std::map<std::pair<int, int>, type_real>{
              { { deg_x, degree - deg_x }, 1 } });

      next_seed = std::rand();
    }

    std::unique_ptr<IteratorImpl> copy() const override {
      return std::make_unique<Iterator>(parent, next_seed, current_field);
    }
  };

private:
  int seed_offset;
  int max_degree;

public:
  RandomPolynomial2DGenerator(const int &max_degree, const int &seed_offset = 0)
      : max_degree(max_degree),
        seed_offset(seed_offset + max_degree * 3723392557) {}

  typename BaseGeneratorType::IteratorPtrType
  get_iterator_impl() const override {
    return std::make_unique<Iterator>(*this, seed_offset);
  };
};

} // namespace specfem::testing::field
