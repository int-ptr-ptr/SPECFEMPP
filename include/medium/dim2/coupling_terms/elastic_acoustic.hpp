#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/data_access/check_compatibility.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

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
          typename CoupledFieldType, typename SelfFieldType>
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
    const IndexType &index, const CoupledInterfaceType &interface_data,
    const CoupledFieldType &coupled_field, SelfFieldType &self_field) {

  static_assert(specfem::data_access::is_acceleration<SelfFieldType>::value,
                "SelfFieldType must be an acceleration type");
  static_assert(specfem::data_access::is_acceleration<CoupledFieldType>::value,
                "CoupledFieldType must be an acceleration type");

  self_field(0) = 0;
  self_field(1) = 0;
  for (int ipoint_mortar = 0;
       ipoint_mortar < CoupledInterfaceType::n_quad_interface;
       ipoint_mortar++) {
    type_real s_tilde = interface_data.transfer_function_self(
        index.iedge, index.ipoint, ipoint_mortar);

    type_real nx_at_mortar =
        interface_data.intersection_normal(index.iedge, ipoint_mortar, 0);
    type_real nz_at_mortar =
        interface_data.intersection_normal(index.iedge, ipoint_mortar, 1);

    type_real chitt_at_mortar = 0;
    for (int ipoint_coupled = 0;
         ipoint_coupled < CoupledInterfaceType::n_quad_element;
         ipoint_coupled++) {
      chitt_at_mortar += coupled_field(index.iedge, ipoint_coupled, 0) *
                         interface_data.transfer_function_coupled(
                             index.iedge, ipoint_coupled, ipoint_mortar);
    }

    type_real stilde_chitt_J =
        s_tilde * chitt_at_mortar *
        interface_data.intersection_factor(index.iedge, ipoint_mortar);

    self_field(0) += stilde_chitt_J * nx_at_mortar;
    self_field(1) += stilde_chitt_J * nz_at_mortar;
  }
}
} // namespace specfem::medium::impl
