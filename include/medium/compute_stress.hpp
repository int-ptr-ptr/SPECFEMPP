#pragma once

#include "acoustic_isotropic2d/acoustic_isotropic2d.hpp"
#include "elastic_isotropic2d/elastic_isotropic2d.hpp"

namespace specfem {
namespace medium {

/**
 * @brief Compute the stress tensor at a quadrature point
 *
 * @tparam PointPropertiesType Material properties at the quadrature point
 * specfem::point::properties
 * @tparam PointFieldDerivativesType Field derivatives at the quadrature point
 * specfem::point::field_derivatives
 * @param properties Material properties at the quadrature point
 * @param field_derivatives Field derivatives at the quadrature point
 * @return specfem::point::stress The stress tensor at the quadrature point
 */
template <typename PointPropertiesType, typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto
compute_stress(const PointPropertiesType &properties,
               const PointFieldDerivativesType &field_derivatives)
    -> decltype(specfem::medium::impl_compute_stress(properties,
                                                     field_derivatives)) {

  static_assert(PointPropertiesType::is_point_properties,
                "properties is not a point properties type");
  static_assert(PointFieldDerivativesType::is_point_field_derivatives,
                "field_derivatives is not a point field derivatives type");

  static_assert(PointPropertiesType::dimension ==
                    PointFieldDerivativesType::dimension,
                "properties and field_derivatives have different dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointFieldDerivativesType::medium_tag,
                "properties and field_derivatives have different medium tags");

  static_assert(
      PointPropertiesType::simd::using_simd ==
          PointFieldDerivativesType::simd::using_simd,
      "properties and field_derivatives have different SIMD settings");

  return specfem::medium::impl_compute_stress(properties, field_derivatives);
}

} // namespace medium
} // namespace specfem
