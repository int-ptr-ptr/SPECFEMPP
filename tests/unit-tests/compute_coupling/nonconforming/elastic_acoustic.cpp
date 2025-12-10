#include "nonconforming_tester.hpp"
#include <gtest/gtest.h>

template <>
std::array<
    type_real,
    PointData<specfem::dimension::type::dim2,
              specfem::interface::interface_tag::elastic_acoustic>::ncomp_self>
compute_coupling_expected<specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::elastic_acoustic>(
    const PointData<specfem::dimension::type::dim2,
                    specfem::interface::interface_tag::elastic_acoustic>
        &point_data) {
  return { point_data.field_coupled(0) * point_data.normal(0),
           point_data.field_coupled(0) * point_data.normal(1) };
}

TEST(NonconformingComputeCoupling, ElasticAcoustic) {

  test_interface<specfem::interface::interface_tag::elastic_acoustic>(
      specfem::test::analytical::interface_shape::interface_shapes_2d,
      specfem::test::analytical::field::sample_fields_2d,
      specfem::test::analytical::interface_transfer::interface_transfer_2d_5_4,
      3);
}
