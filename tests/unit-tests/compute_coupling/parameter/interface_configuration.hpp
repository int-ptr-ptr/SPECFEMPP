#include "enumerations/medium.hpp"
#include "field.hpp"
#include "interface_shape.hpp"
#include "interface_transfer.hpp"

namespace specfem::testing::interface_configuration {

/**
 * @brief Specifies a configuration for a (nonconforming) interface. This is a
 * testing analogy to the coupled_interface container, symbolically storing data
 * of the interface and fields. Each connection, however, is independent,
 * lacking an assembly or universal coordinate system.
 *
 * @tparam InterfaceTag type of interface to model
 * @tparam DimensionTag dimension of the domain (the interface has codimension
 * 1)
 * @tparam use_full_element whether or not the interface uses a full element, or
 * a representation of the mating faces.
 */
template <bool use_full_element, specfem::interface::interface_tag InterfaceTag,
          specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
class InterfaceConfiguration;

template <specfem::interface::interface_tag InterfaceTag, int nquad_edge,
          int nquad_intersection>
class InterfaceConfiguration<false, InterfaceTag,
                             specfem::dimension::type::dim2, nquad_edge,
                             nquad_intersection> {
public:
  static constexpr specfem::dimension::type DimensionTag =
      specfem::dimension::type::dim2;
  static constexpr specfem::element::medium_tag self_medium =
      specfem::interface::attributes<DimensionTag, InterfaceTag>::self_medium();
  static constexpr specfem::element::medium_tag coupled_medium =
      specfem::interface::attributes<DimensionTag,
                                     InterfaceTag>::coupled_medium();
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  static constexpr int ncomp_self =
      specfem::element::attributes<DimensionTag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<DimensionTag, coupled_medium>::components;

  const specfem::testing::field::Generator<DimensionTag> &field_generator;
  const specfem::testing::interface_shape::Generator<DimensionTag>
      &interface_shape_generator;
  const specfem::testing::interface_transfer::Generator<
      DimensionTag, nquad_edge, nquad_intersection>
      &interface_transfer_generator;
  const int num_intersections;

private:
  /**
   * @brief A single intersection, with the generatiors handled.
   *
   */
  struct IntersectionFrame {
    const InterfaceConfiguration &parent;
    int iedge;

  private:
    std::array<specfem::testing::field::Iterator<DimensionTag>, ncomp_self>
        fields_self_;
    std::array<specfem::testing::field::Iterator<DimensionTag>, ncomp_coupled>
        fields_coupled_;
    specfem::testing::interface_shape::Iterator<DimensionTag> interface_shape_;
    specfem::testing::interface_transfer::Iterator<DimensionTag, nquad_edge,
                                                   nquad_intersection>
        interface_transfer_;

  public:
    const specfem::testing::interface_shape::InterfaceShapeBase<DimensionTag>
        &interface_shape;
    const specfem::testing::interface_transfer::InterfaceTransfer<
        DimensionTag, nquad_edge, nquad_intersection> &interface_transfer;

    const specfem::testing::field::FieldBase<DimensionTag> &
    field_self(const int &icomp) {
      return *fields_self_[icomp];
    }
    const specfem::testing::field::FieldBase<DimensionTag> &
    field_coupled(const int &icomp) {
      return *fields_coupled_[icomp];
    }

    IntersectionFrame(
        const InterfaceConfiguration &parent, const int &iedge,
        const specfem::testing::field::Iterator<DimensionTag> &field_iter,
        const specfem::testing::interface_shape::Iterator<DimensionTag>
            &interface_shape_iter,
        const specfem::testing::interface_transfer::Iterator<
            DimensionTag, nquad_edge, nquad_intersection>
            &interface_transfer_iter)
        : parent(parent), iedge(iedge),
          interface_transfer_(interface_transfer_iter),
          interface_shape_(interface_shape_iter),
          interface_transfer(*interface_transfer_),
          interface_shape(*interface_shape_) {
      specfem::testing::field::Iterator<DimensionTag> field = field_iter;
      for (int icomp = 0; icomp < ncomp_self; icomp++) {
        fields_self_[icomp] = field;
        ++field;
      }
      for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
        fields_coupled_[icomp] = field;
        ++field;
      }
    }
  };

  struct IntersectionIterable {
    const InterfaceConfiguration &parent;
    struct IntersectionIterator {
      const InterfaceConfiguration &parent;
      int iedge;
      specfem::testing::field::Iterator<DimensionTag> field_iter;
      specfem::testing::interface_shape::Iterator<DimensionTag>
          interface_shape_iter;
      specfem::testing::interface_transfer::Iterator<DimensionTag, nquad_edge,
                                                     nquad_intersection>
          interface_transfer_iter;
      IntersectionIterator(const InterfaceConfiguration &parent,
                           const bool &start)
          : parent(parent), iedge(start ? 0 : parent.num_intersections) {
        if (start) {
          field_iter = parent.field_generator.iterator();
          interface_shape_iter = parent.interface_shape_generator.iterator();
          interface_transfer_iter =
              parent.interface_transfer_generator.iterator();
        }
      }
      IntersectionFrame operator*() const {
        return IntersectionFrame(parent, iedge, field_iter,
                                 interface_shape_iter, interface_transfer_iter);
      }
      void operator++() {
        ++iedge;
        ++interface_shape_iter;
        ++interface_transfer_iter;
        for (int i = 0; i < ncomp_self + ncomp_coupled; i++) {
          ++field_iter;
        }
      }
      bool operator!=(const IntersectionIterator &other) const {
        return iedge != other.iedge;
      }
    };

    IntersectionIterator begin() { return IntersectionIterator(parent, true); }
    IntersectionIterator end() { return IntersectionIterator(parent, false); }
    const IntersectionIterator begin() const {
      return IntersectionIterator(parent, true);
    }
    const IntersectionIterator end() const {
      return IntersectionIterator(parent, false);
    }
    IntersectionIterable(const InterfaceConfiguration &parent)
        : parent(parent) {}
  };

public:
  const IntersectionIterable edges;

public:
  InterfaceConfiguration(
      const specfem::testing::field::Generator<DimensionTag> &field_generator,
      const specfem::testing::interface_shape::Generator<DimensionTag>
          &interface_shape_generator,
      const specfem::testing::interface_transfer::Generator<
          DimensionTag, nquad_edge, nquad_intersection>
          &interface_transfer_generator,
      int num_intersections)
      : field_generator(field_generator),
        interface_shape_generator(interface_shape_generator),
        interface_transfer_generator(interface_transfer_generator),
        num_intersections(num_intersections), edges(*this) {}
};

} // namespace specfem::testing::interface_configuration
