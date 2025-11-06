#pragma once

#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

// TODO replace when this gets reworked.
#include "specfem/assembly/coupled_interfaces/dim2/data_access/impl/load_access_compatibility.hpp"
namespace specfem::algorithms {

/**
 * @brief Takes a chunk_edge::field and maps it onto the intersection, using a
 single-sided
 * transfer-function container.
 *
 * @tparam CoupledInterfaceType transfer function container type
 (specfem::assembly::coupled_interfaces_impl::stores_transfer_function_coupled<CoupledInterfaceType>::value
 must be true)
 * @tparam EdgeFieldType The chunk_edge field type
 * @tparam IntersectionFieldViewType - a view that the intersection field should
 be stored into
 * @param interface_data transfer function container
 * @param coupled_field The chunk_edge field to map from
 * @param intersection_field a view that the intersection field should be stored
 into
 */
template <typename IndexType, typename CoupledInterfaceType,
          typename EdgeFieldType, typename IntersectionReturnCallback,
          typename std::enable_if_t<
              CoupledInterfaceType::connection_tag ==
                      specfem::connections::type::nonconforming &&
                  specfem::assembly::coupled_interfaces_impl::
                      stores_transfer_function_coupled<
                          CoupledInterfaceType>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION void
transfer_coupled(const IndexType &chunk_edge_index,
         const CoupledInterfaceType &interface_data,
         const EdgeFieldType &coupled_field,
         const IntersectionReturnCallback &callback) {

  constexpr auto dimension_tag = EdgeFieldType::dimension_tag;
  constexpr auto edge_medium_tag = EdgeFieldType::medium_tag;
  constexpr auto interface_tag = CoupledInterfaceType::interface_tag;

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");
  static_assert(
      specfem::data_access::is_coupled_interface<CoupledInterfaceType>::value,
      "interface_data is not a coupled interface type");
  static_assert(specfem::data_access::is_chunk_edge<EdgeFieldType>::value &&
                    specfem::data_access::is_field<EdgeFieldType>::value,
                "coupled_field is not a point field type");

  // no medium check for intersection.
  static_assert(specfem::interface::attributes<dimension_tag,
                                            interface_tag>::coupled_medium() ==
          edge_medium_tag,
      "Inconsistent medium tag between CoupledInterfaceType's side of the "
      "interface and EdgeFieldType");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_edge_index.get_policy_index();
  const int &num_edges = chunk_edge_index.nedges();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, EdgeFieldType::components, EdgeFieldType::using_simd>;

  static_assert(std::is_invocable_v<IntersectionReturnCallback, int, int,
                                    VectorPointViewType>,
                "CallableType must be invocable with arguments (int (iedge), "
                "int (iintersection), "
                "specfem::datatype::VectorPointViewType<type_real, components> "
                "(field evaluated at intersection))");

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, edge_medium_tag>::components;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(
          team, num_edges * CoupledInterfaceType::n_quad_intersection),
      [&](const int &ichunkmortar) {
        const int ipoint_intersection =
            ichunkmortar % CoupledInterfaceType::n_quad_intersection;
        const int iedge =
            ichunkmortar / CoupledInterfaceType::n_quad_intersection;

        const int &local_slot = iedge;

        VectorPointViewType intersection_point_view;

        for (int icomp = 0; icomp < ncomp; icomp++) {
          intersection_point_view(icomp) = 0;

          for (int ipoint_edge = 0;
               ipoint_edge < CoupledInterfaceType::n_quad_element;
               ipoint_edge++) {
            intersection_point_view(icomp) +=
                coupled_field(iedge, ipoint_edge, icomp) *
                interface_data.transfer_function_coupled(iedge, ipoint_edge,
                                                 ipoint_intersection);
          }
        }

        callback(iedge, ipoint_intersection, intersection_point_view);
      });
}

/**
 * @brief Takes a chunk_edge::field and maps it onto the intersection, using a
 single-sided
 * transfer-function container.
 *
 * @tparam CoupledInterfaceType transfer function container type
 (specfem::assembly::coupled_interfaces_impl::stores_transfer_function_self<CoupledInterfaceType>::value
 must be true)
 * @tparam EdgeFieldType The chunk_edge field type
 * @tparam IntersectionFieldViewType - a view that the intersection field should
 be stored into
 * @param interface_data transfer function container
 * @param self_field The chunk_edge field to map from
 * @param intersection_field a view that the intersection field should be stored
 into
 */
template <typename IndexType, typename CoupledInterfaceType,
          typename EdgeFieldType, typename IntersectionReturnCallback,
          typename std::enable_if_t<
              CoupledInterfaceType::connection_tag ==
                      specfem::connections::type::nonconforming &&
                  specfem::assembly::coupled_interfaces_impl::
                      stores_transfer_function_self<
                          CoupledInterfaceType>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION void
transfer_self(const IndexType &chunk_edge_index,
         const CoupledInterfaceType &interface_data,
         const EdgeFieldType &self_field,
         const IntersectionReturnCallback &callback) {

  constexpr auto dimension_tag = EdgeFieldType::dimension_tag;
  constexpr auto edge_medium_tag = EdgeFieldType::medium_tag;
  constexpr auto interface_tag = CoupledInterfaceType::interface_tag;

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");
  static_assert(
      specfem::data_access::is_coupled_interface<CoupledInterfaceType>::value,
      "interface_data is not a coupled interface type");
  static_assert(specfem::data_access::is_chunk_edge<EdgeFieldType>::value &&
                    specfem::data_access::is_field<EdgeFieldType>::value,
                "self_field is not a point field type");

  // no medium check for intersection.
  static_assert(specfem::interface::attributes<dimension_tag,
                                            interface_tag>::self_medium() ==
          edge_medium_tag,
      "Inconsistent medium tag between CoupledInterfaceType's side of the "
      "interface and EdgeFieldType");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_edge_index.get_policy_index();
  const int &num_edges = chunk_edge_index.nedges();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, EdgeFieldType::components, EdgeFieldType::using_simd>;

  static_assert(std::is_invocable_v<IntersectionReturnCallback, int, int,
                                    VectorPointViewType>,
                "CallableType must be invocable with arguments (int (iedge), "
                "int (iintersection), "
                "specfem::datatype::VectorPointViewType<type_real, components> "
                "(field evaluated at intersection))");

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, edge_medium_tag>::components;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(
          team, num_edges * CoupledInterfaceType::n_quad_intersection),
      [&](const int &ichunkmortar) {
        const int ipoint_intersection =
            ichunkmortar % CoupledInterfaceType::n_quad_intersection;
        const int iedge =
            ichunkmortar / CoupledInterfaceType::n_quad_intersection;

        const int &local_slot = iedge;

        VectorPointViewType intersection_point_view;

        for (int icomp = 0; icomp < ncomp; icomp++) {
          intersection_point_view(icomp) = 0;

          for (int ipoint_edge = 0;
               ipoint_edge < CoupledInterfaceType::n_quad_element;
               ipoint_edge++) {
            intersection_point_view(icomp) +=
                self_field(iedge, ipoint_edge, icomp) *
                interface_data.transfer_function_self(iedge, ipoint_edge,
                                                 ipoint_intersection);
          }
        }

        callback(iedge, ipoint_intersection, intersection_point_view);
      });
}


} // namespace specfem::algorithms
