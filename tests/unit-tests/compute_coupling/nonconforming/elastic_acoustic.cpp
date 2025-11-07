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
           point_data.field_coupled(1) * point_data.normal(1) };
}

TEST(NonconformingComputeCoupling, ElasticAcoustic) {
  using CKV = compute_kernel_visitor<
      specfem::dimension::type::dim2,
      specfem::interface::interface_tag::elastic_acoustic>;
  specfem::testing::interface_transfer::Vector<specfem::dimension::type::dim2,
                                               CKV>
      transfers(specfem::testing::interface_transfer::InterfaceTransfer<
                specfem::dimension::type::dim2, CKV, 5, 4>(
          { -1, -0.5, 0, 0.5, 1 }, { -0.8, -0.4, 0.4, 0.8 },
          { -1, -0.5, 0, 0.5, 1 }));
  test_interface<specfem::dimension::type::dim2,
                 specfem::interface::interface_tag::elastic_acoustic>(
      specfem::testing::interface_shape::RandomFlat2DGenerator(5, 33576),
      specfem::testing::field::RandomPolynomial2DGenerator(10, 4, 34631),
      transfers, 3);
}
