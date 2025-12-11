#pragma once
#include "analytical_fixtures/coupled_interfaces/interface_shape.hpp"
#include "analytical_fixtures/coupled_interfaces/interface_transfer.hpp"
#include "analytical_fixtures/field.hpp"
#include "enumerations/medium.hpp"
#include <iomanip>
#include <ios>
#include <sstream>
#include <vector>

namespace specfem::test::analytical::interface_configuration {

/**
 * @brief Specifies a configuration for a (nonconforming) interface. This is a
 * testing analogy to the coupled_interface container, symbolically storing data
 * of the interface and fields. Each connection, however, is independent,
 * lacking an assembly or universal coordinate system.
 *
 * @tparam use_full_element whether or not the interface uses a full element, or
 * a representation of the mating faces.
 * @tparam InterfaceTag type of interface to model
 * @tparam DimensionTag dimension of the domain (the interface has codimension
 * 1)
 * @tparam nquad_edge number of quadrature points on the edge (for kernel)
 * @tparam nquad_edge number of quadrature points on the intersection (for
 * kernel)
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

  const specfem::test::analytical::field::Generator<DimensionTag>
      &field_generator;
  const specfem::test::analytical::interface_shape::Generator<DimensionTag>
      &interface_shape_generator;
  const specfem::test::analytical::interface_transfer::Generator<
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
    std::array<specfem::test::analytical::field::Iterator<DimensionTag>,
               ncomp_self>
        fields_self_;
    std::array<specfem::test::analytical::field::Iterator<DimensionTag>,
               ncomp_coupled>
        fields_coupled_;
    specfem::test::analytical::interface_shape::Iterator<DimensionTag>
        interface_shape_;
    specfem::test::analytical::interface_transfer::Iterator<
        DimensionTag, nquad_edge, nquad_intersection>
        interface_transfer_;

  public:
    const specfem::test::analytical::interface_shape::InterfaceShapeBase<
        DimensionTag> &interface_shape;
    const specfem::test::analytical::interface_transfer::InterfaceTransfer<
        DimensionTag, nquad_edge, nquad_intersection> &interface_transfer;

    const specfem::test::analytical::field::FieldBase<DimensionTag> &
    field_self(const int &icomp) {
      return *fields_self_[icomp];
    }
    const specfem::test::analytical::field::FieldBase<DimensionTag> &
    field_coupled(const int &icomp) {
      return *fields_coupled_[icomp];
    }

    IntersectionFrame(
        const InterfaceConfiguration &parent, const int &iedge,
        const specfem::test::analytical::field::Iterator<DimensionTag>
            &field_iter,
        const specfem::test::analytical::interface_shape::Iterator<DimensionTag>
            &interface_shape_iter,
        const specfem::test::analytical::interface_transfer::Iterator<
            DimensionTag, nquad_edge, nquad_intersection>
            &interface_transfer_iter)
        : parent(parent), iedge(iedge),
          interface_transfer_(interface_transfer_iter),
          interface_shape_(interface_shape_iter),
          interface_transfer(*interface_transfer_),
          interface_shape(*interface_shape_) {
      specfem::test::analytical::field::Iterator<DimensionTag> field =
          field_iter;
      for (int icomp = 0; icomp < ncomp_self; icomp++) {
        fields_self_[icomp] = field;
        ++field;
      }
      for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
        fields_coupled_[icomp] = field;
        ++field;
      }
    }

    std::string verbose_intersection_data() {
      std::ostringstream oss;
      oss << "Verbose printing waiting on tabulator PR";

      // specfem::test::Table self_table(3, nquad_edge);
      // specfem::test::Table coupled_table(3, nquad_edge);
      // specfem::test::Table intersection_table(4, nquad_intersection);

      // self_table.set_row_format(0, specfem::test::Table::EntryType::integer);
      // self_table.set_row_label(0, "ipoint");
      // self_table.set_row_format(1, specfem::test::Table::EntryType::real);
      // self_table.set_row_label(1, "interface coord");
      // self_table.set_row_format(2, specfem::test::Table::EntryType::vector);
      // self_table.set_row_label(2, "field");

      // coupled_table.set_row_format(-1,
      //                              specfem::test::Table::EntryType::integer);
      // coupled_table.set_row_label(-1, "ipoint");
      // coupled_table.set_row_format(-2,
      // specfem::test::Table::EntryType::real); coupled_table.set_row_label(-2,
      // "interface coord"); coupled_table.set_row_format(-3,
      // specfem::test::Table::EntryType::vector);
      // coupled_table.set_row_label(-3, "field");

      // intersection_table.set_row_format(
      //     0, specfem::test::Table::EntryType::vector);
      // intersection_table.set_row_label(0, "self field");
      // intersection_table.set_row_format(1,
      //                                   specfem::test::Table::EntryType::real);
      // intersection_table.set_row_label(1, "interface coord");
      // intersection_table.set_row_format(
      //     2, specfem::test::Table::EntryType::vector);
      // intersection_table.set_row_label(2, "normal");
      // intersection_table.set_row_format(
      //     -1, specfem::test::Table::EntryType::vector);
      // intersection_table.set_row_label(-1, "coupled field");

      // for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
      //   type_real local_coord_self =
      //       interface_transfer.edge_quadrature_points_self[ipoint];
      //   self_table.set_data(0, ipoint, ipoint);
      //   self_table.set_data(1, ipoint, local_coord_self);
      //   std::vector<type_real> field_data;
      //   for (int icomp = 0; icomp < ncomp_self; icomp++) {
      //     field_data.push_back(field_self(icomp).eval(
      //         interface_shape.coordinate(local_coord_self)));
      //   }
      //   self_table.set_data(2, ipoint, field_data);

      //   type_real local_coord_coupled =
      //       interface_transfer.edge_quadrature_points_coupled[ipoint];
      //   coupled_table.set_data(-1, ipoint, ipoint);
      //   coupled_table.set_data(-2, ipoint, local_coord_coupled);
      //   field_data = std::vector<type_real>();
      //   for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
      //     field_data.push_back(field_coupled(icomp).eval(
      //         interface_shape.coordinate(local_coord_coupled)));
      //   }
      //   coupled_table.set_data(-3, ipoint, field_data);
      // }

      // for (int ipoint = 0; ipoint < nquad_intersection; ipoint++) {
      //   std::vector<type_real> field_data;
      //   type_real local_coord =
      //       interface_transfer.intersection_quadrature_points[ipoint];
      //   for (int icomp = 0; icomp < ncomp_self; icomp++) {
      //     field_data.push_back(
      //         field_self(icomp).eval(interface_shape.coordinate(local_coord)));
      //   }
      //   intersection_table.set_data(0, ipoint, field_data);

      //   intersection_table.set_data(1, ipoint, local_coord);

      //   const auto normal = interface_shape.normal(local_coord);
      //   field_data = { normal(0), normal(1) };
      //   intersection_table.set_data(2, ipoint, field_data);

      //   field_data = std::vector<type_real>();
      //   for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
      //     field_data.push_back(field_coupled(icomp).eval(
      //         interface_shape.coordinate(local_coord)));
      //   }
      //   intersection_table.set_data(-1, ipoint, field_data);
      // }

      // oss << "self\n" << self_table << "\n";

      // oss << "intersection\n" << intersection_table << "\n";

      // oss << "coupled\n" << coupled_table << "\n";
      return oss.str();
    }
  };

  // IntersectionFrame iterator boilerplate: necessary for foreach loops.
  struct IntersectionIterable {
    const InterfaceConfiguration &parent;
    struct IntersectionIterator {
      const InterfaceConfiguration &parent;
      int iedge;
      specfem::test::analytical::field::Iterator<DimensionTag> field_iter;
      specfem::test::analytical::interface_shape::Iterator<DimensionTag>
          interface_shape_iter;
      specfem::test::analytical::interface_transfer::Iterator<
          DimensionTag, nquad_edge, nquad_intersection>
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
      const specfem::test::analytical::field::Generator<DimensionTag>
          &field_generator,
      const specfem::test::analytical::interface_shape::Generator<DimensionTag>
          &interface_shape_generator,
      const specfem::test::analytical::interface_transfer::Generator<
          DimensionTag, nquad_edge, nquad_intersection>
          &interface_transfer_generator,
      int num_intersections)
      : field_generator(field_generator),
        interface_shape_generator(interface_shape_generator),
        interface_transfer_generator(interface_transfer_generator),
        num_intersections(num_intersections), edges(*this) {}
};

} // namespace specfem::test::analytical::interface_configuration
