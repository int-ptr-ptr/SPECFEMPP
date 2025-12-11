#include "nonconforming_tester.hpp"
#include <gtest/gtest.h>

template <>
std::array<
    type_real,
    PointData<specfem::dimension::type::dim2,
              specfem::interface::interface_tag::acoustic_elastic>::ncomp_self>
compute_coupling_expected<specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::acoustic_elastic>(
    const PointData<specfem::dimension::type::dim2,
                    specfem::interface::interface_tag::acoustic_elastic>
        &point_data) {
  return { point_data.field_coupled(0) * point_data.normal(0) +
           point_data.field_coupled(1) * point_data.normal(1) };
}

TEST(NonconformingComputeCoupling, AcousticElastic) {

  test_interface<specfem::interface::interface_tag::acoustic_elastic>(
      specfem::test::analytical::interface_shape::interface_shapes_2d,
      specfem::test::analytical::field::sample_fields_2d,
      specfem::test::analytical::interface_transfer::interface_transfer_2d_6_6,
      3);

  // add this back later -- I want to make test case printing more verbose:

  //   test_interface<specfem::interface::interface_tag::acoustic_elastic>(
  //       specfem::test::analytical::interface_shape::interface_shapes_2d,
  //       specfem::test::analytical::field::sample_fields_2d,
  //       specfem::test::analytical::interface_transfer::interface_transfer_2d_5_4,
  //       3);
}
