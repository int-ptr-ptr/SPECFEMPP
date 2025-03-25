#pragma once
#include "interface.hpp"

namespace _util {
namespace placeholder_fluxes {
namespace midpoint {
static void compute_fluxes(specfem::compute::assembly &assembly,
                           _util::placeholder_fluxes::ContainerType &container,
                           const type_real trace_relaxation_parameter,
                           const type_real crossover_relaxation_parameter) {
  const type_real one_minus_tr_plus_xr =
      1 - trace_relaxation_parameter + crossover_relaxation_parameter;
  const type_real inverse_one_minus_half_trace_relaxation =
      1 / (1 - trace_relaxation_parameter / 2);
  const type_real one_minus_crossover_relaxation =
      1 - crossover_relaxation_parameter;
  constexpr bool UseSIMD = false;
  constexpr auto DimensionType = _util::placeholder_fluxes::DimensionType;
  using AcousticAccelType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, false,
                            false, true, false, UseSIMD>;
  using AcousticDispType =
      specfem::point::field<DimensionType,
                            specfem::element::medium_tag::acoustic, true, false,
                            false, false, UseSIMD>;
  using AcousticProperties = specfem::point::properties<
      DimensionType, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, UseSIMD>;

  for (int iinterface = 0; iinterface < container.num_interfaces;
       iinterface++) {
    type_real sigma_edge1[ContainerType::NGLL_EDGE];
    type_real sigma_edge2[ContainerType::NGLL_EDGE];
    type_real chi1[ContainerType::NGLL_EDGE];
    type_real chi2[ContainerType::NGLL_EDGE];

    AcousticAccelType accel1[ContainerType::NGLL_EDGE];
    AcousticAccelType accel2[ContainerType::NGLL_EDGE];

    AcousticDispType fieldread[ContainerType::NGLL_EDGE];
    AcousticProperties props;

    const int edge1_index = container.h_interface_medium1_index(iinterface);
    const int edge2_index = container.h_interface_medium2_index(iinterface);

    const int edge1_ispec = container.h_medium1_index_mapping(edge1_index);
    const auto edge1_type = container.h_medium1_edge_type(edge1_index);
    const int edge2_ispec = container.h_medium2_index_mapping(edge2_index);
    const auto edge2_type = container.h_medium2_edge_type(edge2_index);

    specfem::point::index<specfem::dimension::type::dim2> index(edge1_ispec, 0,
                                                                0);
    container.template load_field<1, false>(edge1_index, assembly, fieldread);
    for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE; igll_edge++) {
      chi1[igll_edge] = fieldread[igll_edge].displacement(0);
      accel1[igll_edge].acceleration(0) = 0;
      specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
          index.iz, index.ix, edge1_type, igll_edge);
      specfem::compute::load_on_host(index, assembly.properties, props);
      sigma_edge1[igll_edge] =
          container.h_medium1_field_nderiv(edge1_index, igll_edge, 0) *
          props.rho_inverse();
    }

    index.ispec = edge2_ispec;
    container.template load_field<2, false>(edge2_index, assembly, fieldread);
    for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE; igll_edge++) {
      chi2[igll_edge] = fieldread[igll_edge].displacement(0);
      accel2[igll_edge].acceleration(0) = 0;
      specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(
          index.iz, index.ix, edge2_type, igll_edge);
      specfem::compute::load_on_host(index, assembly.properties, props);
      sigma_edge2[igll_edge] =
          container.h_medium1_field_nderiv(edge2_index, igll_edge, 0) *
          props.rho_inverse();
    }

    for (int igll_interface = 0; igll_interface < ContainerType::NGLL_INTERFACE;
         igll_interface++) {
      type_real jump_penalty =
          (container.template edge_to_mortar<1, false>(iinterface,
                                                       igll_interface, chi1) -
           container.template edge_to_mortar<2, false>(iinterface,
                                                       igll_interface, chi2)) *
          container.h_interface_relaxation_parameter(iinterface) / 2;

      type_real mean_1 = inverse_one_minus_half_trace_relaxation *
                         (container.template edge_to_mortar<1, false>(
                              iinterface, igll_interface, sigma_edge1) *
                              one_minus_tr_plus_xr -
                          container.template edge_to_mortar<2, false>(
                              iinterface, igll_interface, sigma_edge2) *
                              one_minus_crossover_relaxation) /
                         2;
      type_real mean_2 = inverse_one_minus_half_trace_relaxation *
                         (container.template edge_to_mortar<2, false>(
                              iinterface, igll_interface, sigma_edge2) *
                              one_minus_tr_plus_xr -
                          container.template edge_to_mortar<1, false>(
                              iinterface, igll_interface, sigma_edge1) *
                              one_minus_crossover_relaxation) /
                         2;

      const type_real Jw = container.h_interface_surface_jacobian_times_weight(
          iinterface, igll_interface);

      type_real integrand1 = (mean_1 - jump_penalty) * Jw;
      type_real integrand2 = (mean_2 + jump_penalty) * Jw;
      //   integrand1 = container.template edge_to_mortar<1, false>(
      //     iinterface, igll_interface, sigma_edge1) * Jw / 2;
      //   integrand2 = container.template edge_to_mortar<2, false>(
      //     iinterface, igll_interface, sigma_edge2) * Jw / 2;

      for (int igll_edge = 0; igll_edge < ContainerType::NGLL_EDGE;
           igll_edge++) {
        accel1[igll_edge].acceleration +=
            integrand1 * container.h_interface_medium1_mortar_transfer(
                             iinterface, igll_interface, igll_edge);
        accel2[igll_edge].acceleration +=
            integrand2 * container.h_interface_medium2_mortar_transfer(
                             iinterface, igll_interface, igll_edge);
      }
    }

    _util::placeholder_fluxes::fix_free_surface(assembly, accel1, edge1_ispec,
                                                edge1_type);
    _util::placeholder_fluxes::fix_free_surface(assembly, accel2, edge2_ispec,
                                                edge2_type);

    container.template atomic_add_to_field<1, false>(edge1_index, assembly,
                                                     accel1);
    container.template atomic_add_to_field<2, false>(edge2_index, assembly,
                                                     accel2);
  }
}
} // namespace midpoint
} // namespace placeholder_fluxes
} // namespace _util
