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

template <typename IndexType, typename ResultPointFieldType,
          typename ChunkEdgeWeightJacobianType, typename CallableType>
KOKKOS_FUNCTION void integrate_fieldtilde_1d(
    const IndexType &index, const ChunkEdgeWeightJacobianType &weight_jacobian,
    const PointTransferFunctionType &transfer_function,
    ResultPointFieldType result, const CallableType &integrand_callback) {
  static_assert(std::is_invocable_v<CallableType, int, ResultPointFieldType>,
                "CallableType must be invocable with arguments (int, "
                "ResultPointFieldType)");

  constexpr int ncomp = ResultPointFieldType::components;
  constexpr int nquad_intersection =
      ChunkEdgeWeightJacobianType::n_quad_intersection;

  ResultPointFieldType integrand;

  const int &iedge = index.iedge;
  const int &ipoint = index.ipoint;

#pragma unroll
  for (int icomp = 0; icomp < ncomp; icomp++) {
    result(icomp) = 0;
  }

#pragma unroll
  for (int iquad = 0; iquad < nquad; iquad++) {
    integrand_callback(iquad, integrand);

#pragma unroll
    for (int icomp = 0; icomp < ncomp; icomp++) {
      result(icomp) += integrand(icomp) * weight_jacobian(iedge, iquad) *
                       transfer_function(iedge, ipoint, iquad);
    }
  }
}

} // namespace specfem::algorithms
