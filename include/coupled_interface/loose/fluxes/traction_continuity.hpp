#pragma once

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

struct traction_continuity {

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct container;
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem

#include "compute/coupled_interfaces/loose_couplings/traction_continuity_container.hpp"
