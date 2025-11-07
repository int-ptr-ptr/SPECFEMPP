#include "compute_coupling/parameter/field.hpp"
#include "compute_coupling/parameter/interface_shape.hpp"
#include "compute_coupling/parameter/interface_transfer.hpp"
#include "datatypes/point_view.hpp"
#include "specfem/point/field_derivatives.hpp"

#include "kernel.hpp"

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
  specfem::datatype::VectorPointViewType<type_real, ncomp_self, use_simd>
      field_coupled;
  specfem::point::field_derivatives<DimensionTag, self_medium, use_simd>
      field_gradient_self;
  specfem::point::field_derivatives<DimensionTag, coupled_medium, use_simd>
      field_gradient_coupled;
  specfem::datatype::VectorPointViewType<type_real, ncomp_self, use_simd>
      normal;
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
std::array<type_real, PointData<DimensionTag, InterfaceTag>::ncomp_self>
compute_coupling_expected(
    const PointData<DimensionTag, InterfaceTag> &point_data);

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
class compute_kernel_visitor {
  int num_chunks;

public:
  compute_kernel_visitor(const int &num_chunks) : num_chunks(num_chunks) {}

  template <typename T> void operator()(const T &transfer_type) {
    constexpr int nquad_edge = T::nquad_edge;
    constexpr int nquad_intersection = T::nquad_intersection;

    using parallel_config = specfem::parallel_config::default_chunk_edge_config<
        DimensionTag, Kokkos::DefaultExecutionSpace>;
    compute_kernel<DimensionTag, InterfaceTag, parallel_config::chunk_size,
                   nquad_edge, nquad_intersection>(num_chunks);
  };
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
void test_interface(
    const specfem::testing::interface_shape::Generator<DimensionTag>
        &interface_shape_generator,
    const specfem::testing::field::Generator<DimensionTag> &field_generator,
    const specfem::testing::interface_transfer::Generator<
        DimensionTag, compute_kernel_visitor<DimensionTag, InterfaceTag> >
        &interface_transfer_generator,
    const int &num_chunks) {
  for (int i = 0; i < interface_transfer_generator.get_generator_size(); i++) {
    compute_kernel_visitor<DimensionTag, InterfaceTag> kernel_visitor(
        num_chunks);
    interface_transfer_generator.get_interface_transfer(i).accept(
        kernel_visitor);
  }
}
