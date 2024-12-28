#pragma once

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

struct symmetric_flux {

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct container;

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct kernel;
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem

#include "compute/coupled_interfaces/loose_couplings/symmetric_flux_container.hpp"

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux::kernel<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, QuadratureType> {
  using ContainerType = specfem::compute::loose::interface_container<
      DimensionType, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::acoustic, QuadratureType, symmetric_flux>;

  static void compute_fluxes(specfem::compute::assembly &assembly,
                             ContainerType &container) {}

  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_edge_intermediate(int index, specfem::compute::assembly &assembly,
                            ContainerType &container) {}
};
template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux::kernel<
    DimensionType, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic, QuadratureType> {
  using ContainerType = specfem::compute::loose::interface_container<
      DimensionType, specfem::element::medium_tag::elastic,
      specfem::element::medium_tag::elastic, QuadratureType, symmetric_flux>;

  static void compute_fluxes(specfem::compute::assembly &assembly,
                             ContainerType &container) {}

  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION static void
  compute_edge_intermediate(int index, specfem::compute::assembly &assembly,
                            ContainerType &container) {}
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
