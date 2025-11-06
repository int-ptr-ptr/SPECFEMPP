#pragma once

#include "algorithms/transfer.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

// TODO replace when this gets reworked.
#include "specfem/assembly/coupled_interfaces/dim2/data_access/impl/load_access_compatibility.hpp"
namespace specfem::medium::impl {

template <typename CoupledInterfaceType, typename CoupledFieldType,
          typename SelfFieldType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::dimension::type,
        specfem::dimension::type::dim2> /*dimension_dispatch*/,
    const std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*connection_dispatch*/,
    const std::integral_constant<specfem::interface::interface_tag,
                                 specfem::interface::interface_tag::
                                     elastic_acoustic> /*interface_dispatch*/,
    const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_acceleration<SelfFieldType>::value,
                "SelfFieldType must be an acceleration type");
  static_assert(specfem::data_access::is_acceleration<CoupledFieldType>::value,
                "CoupledFieldType must be an acceleration type");

  self_field(0) = interface_data.edge_factor * interface_data.edge_normal(0) *
                  coupled_field(0);
  self_field(1) = interface_data.edge_factor * interface_data.edge_normal(1) *
                  coupled_field(0);
}

template <typename IndexType, typename CoupledInterfaceType,
          typename CoupledFieldType, typename IntersectionFieldViewType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::dimension::type,
        specfem::dimension::type::dim2> /*dimension_dispatch*/,
    const std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::nonconforming> /*connection_dispatch*/,
    const std::integral_constant<specfem::interface::interface_tag,
                                 specfem::interface::interface_tag::
                                     elastic_acoustic> /*interface_dispatch*/,
    const IndexType &chunk_edge_index,
    const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field,
    IntersectionFieldViewType &intersection_field) {

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");
  static_assert(
      specfem::assembly::coupled_interfaces_impl::stores_intersection_normal<
          CoupledInterfaceType>::value,
      "acoustic_elastic compute_coupling needs CoupledInterfaceType to have "
      "the normal vector.");

  static_assert(specfem::data_access::is_displacement<CoupledFieldType>::value,
                "CoupledFieldType must be a displacement type");

  specfem::algorithms::transfer_coupled(
      chunk_edge_index, interface_data, coupled_field,
      [&](const int &iedge, const int &iintersection, const auto &point) {
        intersection_field(iedge, iintersection, 0) =
            interface_data.intersection_normal(iedge, iintersection, 0) *
            point(0);
        intersection_field(iedge, iintersection, 1) =
            interface_data.intersection_normal(iedge, iintersection, 1) *
            point(0);
      });
}

} // namespace specfem::medium::impl
