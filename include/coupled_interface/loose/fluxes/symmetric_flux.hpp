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
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem

#include "compute/coupled_interfaces/loose_couplings/symmetric_flux_container.hpp"
