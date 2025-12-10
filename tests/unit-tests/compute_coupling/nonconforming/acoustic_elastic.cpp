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
      specfem::testing::interface_shape::RandomFlat2DGenerator(50352),
      specfem::test::analytical::field::sample_fields_2d,
      specfem::testing::interface_transfer::Vector<
          specfem::dimension::type::dim2, 5, 4>(
          specfem::testing::interface_transfer::InterfaceTransfer<
              specfem::dimension::type::dim2, 5, 4>({ -1, -0.5, 0, 0.5, 1 },
                                                    { -0.8, -0.4, 0.4, 0.8 },
                                                    { -1, -0.5, 0, 0.5, 1 })),
      3);
  test_interface<specfem::interface::interface_tag::acoustic_elastic>(
      specfem::testing::interface_shape::RandomFlat2DGenerator(50352),
      specfem::test::analytical::field::sample_fields_2d,
      specfem::testing::interface_transfer::Vector<
          specfem::dimension::type::dim2, 2, 2>(
          specfem::testing::interface_transfer::InterfaceTransfer<
              specfem::dimension::type::dim2, 2, 2>({ -1, 1 }, { 0, 1 },
                                                    { 1, 2 })),
      3);
}
