#pragma once

#include "compute/coupled_interfaces/loose_couplings/symmetric_flux_container.hpp"
#include "compute/coupled_interfaces/loose_couplings/traction_continuity_container.hpp"
#include "coupled_interface/loose/fluxes/symmetric_flux.hpp"
#include "coupled_interface/loose/fluxes/traction_continuity.hpp"
#include <string>

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace flux {

/**
 * @brief Loosely coupled interface flux type
 *
 */
enum class type { symmetric_flux, traction_continuity };

template <specfem::coupled_interface::loose::flux::type FluxType>
class FluxScheme;

/**
 * @brief symmetric flux scheme
 *
 */
template <>
class FluxScheme<
    specfem::coupled_interface::loose::flux::type::symmetric_flux> {
public:
  static constexpr auto value =
      specfem::coupled_interface::loose::flux::type::symmetric_flux;
  static std::string to_string() { return "symmetric-flux"; }

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  using ContainerType =
      specfem::compute::loosely_coupled_interface::symmetric_flux_container<
          DimensionType, MediumTag1, MediumTag2, QuadratureType>;

  // TODO remove
  using orig_flux_scheme =
      specfem::coupled_interface::loose::flux::symmetric_flux;
};

/**
 * @brief traction continuity scheme (traditional solid-fluid interface from
 * Komatitsch & Tromp)
 *
 */
template <>
class FluxScheme<
    specfem::coupled_interface::loose::flux::type::traction_continuity> {
public:
  static constexpr auto value =
      specfem::coupled_interface::loose::flux::type::traction_continuity;
  static std::string to_string() { return "traction-continuity"; }

  template <specfem::dimension::type DimensionType,
            specfem::element::medium_tag MediumTag1,
            specfem::element::medium_tag MediumTag2, typename QuadratureType>
  using ContainerType = specfem::compute::loosely_coupled_interface::
      traction_continuity_container<DimensionType, MediumTag1, MediumTag2,
                                    QuadratureType>;

  // TODO remove
  using orig_flux_scheme =
      specfem::coupled_interface::loose::flux::traction_continuity;
};

} // namespace flux
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
