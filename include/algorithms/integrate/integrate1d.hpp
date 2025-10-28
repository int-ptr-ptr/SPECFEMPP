#include <Kokkos_Core.hpp>

// template <typename ChunkIndexType, typename TensorFieldType,
//           typename WeightsType, typename QuadratureType, typename
//           CallableType, std::enable_if_t<
//               specfem::data_access::is_chunk_element<TensorFieldType>::value,
//               int> = 0>
// KOKKOS_FUNCTION void divergence(
//     const ChunkIndexType &chunk_index,
//     const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
//         &jacobian_matrix,
//     const WeightsType &weights, const QuadratureType &hprimewgll,
//     const TensorFieldType &f, const CallableType &callback) {
//     const ChunkIndexType &chunk_index,
//     const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
//         &jacobian_matrix,
//     const WeightsType &weights, const QuadratureType &hprimewgll,
//     const TensorFieldType &f, const CallableType &callback) {

//   using VectorPointViewType = specfem::datatype::VectorPointViewType<
//       type_real, TensorFieldType::components, TensorFieldType::using_simd>;

//   static_assert(
//       std::is_invocable_v<CallableType,
//                           typename ChunkIndexType::iterator_type::index_type,
//                           VectorPointViewType>,
//       "CallableType must be invocable with arguments (int, "
//       "specfem::point::index, "
//       "specfem::datatype::VectorPointViewType<type_real, components>)");

//   specfem::execution::for_each_level(
//       chunk_index.get_iterator(),
//       [&](const typename ChunkIndexType::iterator_type::index_type
//               &iterator_index) {
//         const auto local_index = iterator_index.get_local_index();
//         const auto result =
//             impl::element_divergence(f, local_index, weights, hprimewgll);
//         callback(iterator_index, result);
//       });

//   return;
// }

namespace specfem::algorithms {

template <specfem::dimension::type dimension_tag, typename IndexType,
          typename IntersectionFieldViewType,
          typename ChunkEdgeWeightJacobianType, typename CallableType>
KOKKOS_FUNCTION void integrate_fieldtilde_1d(
    const specfem::assembly::assembly<dimension_tag> &assembly,
    const IndexType &chunk_index,
    const IntersectionFieldViewType &intersection_field,
    const ChunkEdgeWeightJacobianType &weight_jacobian,
    const CallableType &callback) {

  constexpr auto self_medium_tag = specfem::interface::attributes<
      dimension_tag, ChunkEdgeWeightJacobianType::interface_tag>::self_medium();

  using PointIndexType =
      typename IndexType::iterator_type::index_type::index_type;
  using PointFieldType =
      specfem::point::acceleration<dimension_tag, self_medium_tag,
                                   IntersectionFieldViewType::using_simd>;
  using SelfTransferFunctionType =
      typename specfem::point::nonconforming_transfer_function<
          true, ChunkEdgeWeightJacobianType::n_quad_intersection, dimension_tag,
          ChunkEdgeWeightJacobianType::connection_tag,
          ChunkEdgeWeightJacobianType::interface_tag,
          ChunkEdgeWeightJacobianType::boundary_tag,
          specfem::kokkos::DevScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>,
          IntersectionFieldViewType::using_simd>;

  // an is_invocable static check here prevents autos

  constexpr int ncomp = PointFieldType::components;
  constexpr int nquad_intersection =
      ChunkEdgeWeightJacobianType::n_quad_intersection;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &index) {
        const auto self_index = index.get_index();

        auto self_index_with_global = self_index;
        self_index_with_global.iedge +=
            chunk_index.get_policy_index().league_rank() *
            ChunkEdgeWeightJacobianType::chunk_size;

        SelfTransferFunctionType transfer_function_self;
        specfem::assembly::load_on_device(self_index_with_global,
                                          assembly.coupled_interfaces,
                                          transfer_function_self);

        PointFieldType result;
#pragma unroll
        for (int icomp = 0; icomp < ncomp; icomp++) {
          result(icomp) = 0;
        }
        const int &iedge = self_index.iedge;
        const int &ipoint = self_index.ipoint;

#pragma unroll
        for (int iquad = 0; iquad < nquad_intersection; iquad++) {

#pragma unroll
          for (int icomp = 0; icomp < ncomp; icomp++) {
            result(icomp) +=
                intersection_field(iedge, iquad, icomp) *
                weight_jacobian.intersection_factor(iedge, iquad) *
                transfer_function_self.transfer_function_self(iquad);
          }
        }

        callback(self_index, result);
      });
}

} // namespace specfem::algorithms
