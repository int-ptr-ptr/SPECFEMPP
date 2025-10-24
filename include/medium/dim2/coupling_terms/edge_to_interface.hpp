#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly/coupled_interfaces/dim2/data_access/impl/load_access_compatibility.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
namespace specfem::medium::impl {

template <typename CoupledInterfaceType, typename CoupledFieldType,
          typename IntersectionFieldViewType>
KOKKOS_INLINE_FUNCTION void compute_coupling(
    const std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::nonconforming> /*connection_dispatch*/,
    const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field,
    IntersectionFieldViewType &intersection_field) {

  constexpr auto dimension_tag = CoupledFieldType::dimension_tag;
  constexpr auto medium_tag = CoupledFieldType::medium_tag;

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  const auto &transfer_function = interface_data.get_transfer_function();

  // TODO decide how to handle TeamThreadRange passing for inner parfor here
  for (int iedge = 0; iedge < CoupledInterfaceType::chunk_size; iedge++) {
    for (int ipoint_intersection = 0;
         ipoint_intersection < CoupledInterfaceType::n_quad_intersection;
         ipoint_intersection++) {
      for (int icomp = 0; icomp < ncomp; icomp++) {
        intersection_field(iedge, ipoint_intersection, icomp) = 0;

        for (int ipoint_edge = 0;
             ipoint_edge < CoupledInterfaceType::n_quad_element;
             ipoint_edge++) {
          intersection_field(iedge, ipoint_intersection, icomp) +=
              coupled_field(iedge, ipoint_edge, icomp) *
              transfer_function(iedge, ipoint_edge, ipoint_intersection);
        }
      }
    }
  }
}

} // namespace specfem::medium::impl
