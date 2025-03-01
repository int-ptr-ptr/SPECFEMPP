#pragma once

#include <string>
namespace specfem {
namespace enums {

namespace interface_resolution {

enum class type {
  UNKNOWN,           /// < null value, error out.
  CONTINUOUS,        /// < resolve by assembly (only for fully conforming)
  DOMAIN_SEPARATION, /// < resolve by completely separating the two domains (no
                     /// flux/etc)
  FLUX_TRACTION_AND_DISP /// < coupled_interface -- regular fluid-solid
                         /// resolution.
};

/**
 * @brief Returns for an interface resolution type whether or not that rule
 * requires a conforming mesh along the interface.
 *
 */
constexpr bool requires_conforming_domain(const type type) {
  // constexpr in case we want to do templates.
  switch (type) {
  case type::UNKNOWN:
    return false;
  case type::CONTINUOUS:
    return true;
  case type::DOMAIN_SEPARATION:
    return false;
  case type::FLUX_TRACTION_AND_DISP:
    return true; // for now; mark false once dG rule comes in.
  }

  return false;
}

/**
 * @brief Returns for an interface resolution type whether or not that rule
 * expects elements along it to be assembled together. This should just be the
 * CONTINUOUS rule for now.
 *
 */
constexpr bool requires_assembly(const type type) {
  // constexpr in case we want to do templates.
  return type == type::CONTINUOUS;
}

/**
 * @brief Returns the corresponding rule, or UNKNOWN if the string cannot be
 * parsed into a type.
 *
 * @param value the string to evaluate
 */
inline type from_string(const std::string value) {
  if (value == "continuous") {
    return type::CONTINUOUS;
  }
  if (value == "domain-separation") {
    return type::DOMAIN_SEPARATION;
  }
  if (value == "fluid-solid") {
    return type::FLUX_TRACTION_AND_DISP;
  }
  return type::UNKNOWN;
}

} // namespace interface_resolution
} // namespace enums
} // namespace specfem
