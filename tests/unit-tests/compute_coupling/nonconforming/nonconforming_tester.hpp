#include "compute_coupling/parameter/field.hpp"
#include "compute_coupling/parameter/interface_shape.hpp"
#include "compute_coupling/parameter/interface_transfer.hpp"
#include "datatypes/point_view.hpp"
#include "specfem/point/field_derivatives.hpp"

#include "compute_coupling/parameter/interface_configuration.hpp"
#include "kernel.hpp"

#include "test_macros.hpp" // for expected_got()
#include <gtest/gtest.h>

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
struct PointData {
  static constexpr specfem::element::medium_tag self_medium =
      specfem::interface::attributes<DimensionTag, InterfaceTag>::self_medium();
  static constexpr specfem::element::medium_tag coupled_medium =
      specfem::interface::attributes<DimensionTag,
                                     InterfaceTag>::coupled_medium();
  static constexpr int ncomp_self =
      specfem::element::attributes<DimensionTag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<DimensionTag, coupled_medium>::components;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  static constexpr bool use_simd = false;

  specfem::datatype::VectorPointViewType<type_real, ncomp_self, use_simd>
      field_self;
  specfem::datatype::VectorPointViewType<type_real, ncomp_coupled, use_simd>
      field_coupled;
  specfem::point::field_derivatives<DimensionTag, self_medium, use_simd>
      field_gradient_self;
  specfem::point::field_derivatives<DimensionTag, coupled_medium, use_simd>
      field_gradient_coupled;
  specfem::datatype::VectorPointViewType<type_real, ndim, use_simd> normal;
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
std::array<type_real, PointData<DimensionTag, InterfaceTag>::ncomp_self>
compute_coupling_expected(
    const PointData<DimensionTag, InterfaceTag> &point_data);

/**
 * @brief Verifies the nonconforming compute_coupling call against a pointwise
 * compute_coupling algorithm given by compute_coupling_expected.
 *
 * @tparam InterfaceTag Interface being tested
 * @tparam DimensionTag dimension of the kernel
 * @tparam nquad_edge number of quadrature points on the edge (NGLL)
 * @tparam nquad_intersection number of quadrature points on the intersection
 * @param interface_shape_generator generator for interface shapes
 * @param field_generator generator for fields
 * @param interface_transfer_generator generator for transfer functions
 * @param num_chunks league size of the test kernel
    compute_kernel<DimensionTag, InterfaceTag, parallel_config::chunk_size, */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::dimension::type DimensionTag, int nquad_edge,
          int nquad_intersection>
void test_interface(
    const specfem::testing::interface_shape::Generator<DimensionTag>
        &interface_shape_generator,
    const specfem::testing::field::Generator<DimensionTag> &field_generator,
    const specfem::testing::interface_transfer::Generator<
        DimensionTag, nquad_edge, nquad_intersection>
        &interface_transfer_generator,
    const int &num_edges) {
  using parallel_config = specfem::parallel_config::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;
  static constexpr int chunk_size = parallel_config::chunk_size;

  // smallest number of chunks that has at least this many edges
  const int num_chunks = ((num_edges - 1) / chunk_size) + 1;

  // ====================================================================
  // views for inputs / outputs
  // ====================================================================
  ComputeCouplingKernelStorage<DimensionTag, InterfaceTag, chunk_size,
                               nquad_edge, nquad_intersection>
      kernel_data(num_chunks);
  specfem::testing::interface_configuration::InterfaceConfiguration<
      false, InterfaceTag, DimensionTag, nquad_edge, nquad_intersection>
      interface_config(field_generator, interface_shape_generator,
                       interface_transfer_generator, num_chunks * chunk_size);

  // ====================================================================
  // initialize views
  // ====================================================================

  auto field_iter = field_generator.iterator();
  auto interface_transfer_iter = interface_transfer_generator.iterator();
  auto interface_shape_iter = interface_shape_generator.iterator();

  // consider putting this into some kind of function with a generalized (or
  // further templated) ComputeCouplingKernelStorage

  // we will need to load more things for different schemes. Handle that
  // later.
  for (auto intersection : interface_config.edges) {
    const int ichunk = intersection.iedge / chunk_size;
    const int iedge = intersection.iedge % chunk_size;

    intersection.interface_transfer.template set_transfer_function<true>(
        Kokkos::subview(kernel_data.h_transfer_self, ichunk, iedge, Kokkos::ALL,
                        Kokkos::ALL));
    intersection.interface_transfer.template set_transfer_function<false>(
        Kokkos::subview(kernel_data.h_transfer_coupled, ichunk, iedge,
                        Kokkos::ALL, Kokkos::ALL));
    intersection.interface_shape.set_intersection_normal(
        Kokkos::subview(kernel_data.h_normals, ichunk, iedge, Kokkos::ALL,
                        Kokkos::ALL),
        intersection.interface_transfer);
    for (int icomp = 0; icomp < interface_config.ncomp_self; icomp++) {
      intersection.interface_shape.template set_edge_field<true>(
          Kokkos::subview(kernel_data.h_field_self, ichunk, iedge, Kokkos::ALL,
                          icomp),
          intersection.field_self(icomp), intersection.interface_transfer);
    }
    for (int icomp = 0; icomp < interface_config.ncomp_coupled; icomp++) {
      intersection.interface_shape.template set_edge_field<false>(
          Kokkos::subview(kernel_data.h_field_coupled, ichunk, iedge,
                          Kokkos::ALL, icomp),
          intersection.field_coupled(icomp), intersection.interface_transfer);
    }
  }

  kernel_data.sync_to_device();

  // ====================================================================
  // run kernel
  // ===================================================================='
  compute_kernel<DimensionTag, InterfaceTag, chunk_size, nquad_edge,
                 nquad_intersection>(kernel_data);

  kernel_data.sync_to_host();

  int num_comparisons = 0;
  for (auto intersection : interface_config.edges) {
    const int ichunk = intersection.iedge / chunk_size;
    const int iedge = intersection.iedge % chunk_size;

    // verify self transfers
    for (int ipoint = 0; ipoint < nquad_intersection; ipoint++) {
      PointData<DimensionTag, InterfaceTag> point_data;

      const type_real edge_coord = intersection.interface_transfer
                                       .intersection_quadrature_points[ipoint];

      const auto point_loc =
          intersection.interface_shape.coordinate(edge_coord);
      const auto point_normal = intersection.interface_shape.normal(edge_coord);

      for (int idim = 0; idim < point_data.ndim; idim++) {
        point_data.normal(idim) = point_normal(idim);
      }
      for (int icomp = 0; icomp < interface_config.ncomp_self; icomp++) {
        point_data.field_self(icomp) =
            intersection.field_self(icomp).eval(point_loc);
        const auto grad = intersection.field_self(icomp).gradient(point_loc);
        for (int idim = 0; idim < point_data.ndim; idim++) {
          point_data.field_gradient_self.du(icomp, idim) = grad(idim);
        }
      }
      for (int icomp = 0; icomp < interface_config.ncomp_coupled; icomp++) {
        point_data.field_coupled(icomp) =
            intersection.field_coupled(icomp).eval(point_loc);
        const auto grad = intersection.field_coupled(icomp).gradient(point_loc);
        for (int idim = 0; idim < point_data.ndim; idim++) {
          point_data.field_gradient_coupled.du(icomp, idim) = grad(idim);
        }
      }
      const auto expected = compute_coupling_expected(point_data);

      for (int icomp = 0; icomp < interface_config.ncomp_self; icomp++) {
        // TODO replace with more verbose test

        type_real got =
            kernel_data.h_computed_coupling(ichunk, iedge, ipoint, icomp);

        EXPECT_TRUE(specfem::utilities::is_close(got, expected[icomp]))
            << expected_got(expected[icomp], got);
        num_comparisons++;
      }
    }

    // skip coupled side, since we don't have a way to get the conjugate
    // interface tag.

    // // verify coupled transfers
    // for (int ipoint = 0; ipoint < nquad_intersection; ipoint++) {
    //   PointData<DimensionTag, CONJUGATE_INTERFACE<InterfaceTag>> point_data;
    //   const auto expected = compute_coupling_expected(point_data);

    //   for (int icomp = 0; icomp < interface_config.ncomp_self; icomp++) {
    //     // TODO replace with more verbose test

    //     type_real got =
    //         kernel_data.h_computed_coupling(ichunk, iedge, ipoint, icomp);

    //     EXPECT_TRUE(specfem::utilities::is_close(got, expected[icomp]))
    //         << expected_got(expected[icomp], got);
    //   }
    // }
  }

  EXPECT_EQ(num_comparisons, num_chunks * chunk_size * nquad_intersection *
                                 interface_config.ncomp_self)
      << "Test failed to perform the correct number of comparisons!\n"
      << "  - " << num_chunks << " chunks of " << chunk_size
      << " edges for a total of " << num_chunks * chunk_size << " edges\n"
      << "  - each edge has " << nquad_intersection
      << " points on the intersection quadrature ("
      << num_chunks * chunk_size * nquad_intersection << " quadrature points)\n"
      << "  - " << interface_config.ncomp_self
      << " components in self_field and " << interface_config.ncomp_coupled
      << " components in coupled_field";
}
