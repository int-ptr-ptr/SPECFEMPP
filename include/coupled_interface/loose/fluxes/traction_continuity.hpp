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

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  struct kernel;
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
