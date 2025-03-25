#pragma once
#include "Kokkos_Macros.hpp"
#include "_util/quadrature_template_type.hpp"
#include "compute/assembly/assembly.hpp"
#include "compute/coupled_interfaces/loose_couplings/interface_container.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
namespace _util {
namespace placeholder_fluxes {

constexpr auto DimensionType = specfem::dimension::type::dim2;
using ContainerType = specfem::compute::loose::interface_container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, _util::static_quadrature_points<5>,
    specfem::coupled_interface::loose::flux::type::symmetric_flux>;

template <typename PointFieldType>
KOKKOS_INLINE_FUNCTION void
fix_free_surface(specfem::compute::assembly &assembly,
                 PointFieldType *point_field, const int ispec,
                 const specfem::enums::edge::type edgetype);

template <int medium, bool on_device, typename ContainerType>
void compute_field_nderiv(int index, const specfem::compute::assembly &assembly,
                          ContainerType &container);

} // namespace placeholder_fluxes
} // namespace _util

#include "interface.tpp"
#include "midpoint.hpp"
#include "upwind.hpp"
