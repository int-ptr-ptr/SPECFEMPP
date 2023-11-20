#ifndef _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_HPP_
#define _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_HPP_

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/quadrature.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "stacey.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace enums {
namespace boundary_conditions {
template <typename property, typename qp_type>
class stacey<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::elastic, property, qp_type> {

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  /**
   * @brief Medium type of the boundary.
   *
   */
  using medium_type = specfem::enums::element::medium::elastic;
  /**
   * @brief Dimension of the boundary.
   *
   */
  using dimension = specfem::enums::element::dimension::dim2;
  /**
   * @brief Quadrature points object to define the quadrature points either at
   * compile time or run time.
   *
   */
  using quadrature_points_type = qp_type;
  /**
   * @brief Property type of the boundary.
   *
   */
  using property_type = property;
  ///@}

  constexpr static specfem::enums::element::boundary_tag value =
      specfem::enums::element::boundary_tag::stacey; ///< boundary tag

  stacey(){};

  stacey(const specfem::compute::boundaries &boundary_conditions,
         const quadrature_points_type &quadrature_points);

  template <specfem::enums::time_scheme::type time_scheme>
  KOKKOS_INLINE_FUNCTION void mass_time_contribution(
      const int &ielement, const int &xz, const type_real &dt,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &rmass_inverse) const;

  KOKKOS_INLINE_FUNCTION
  void enforce_gradient(
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dx,
      specfem::kokkos::array_type<type_real, medium_type::components> &df_dz)
      const {};

  KOKKOS_INLINE_FUNCTION void enforce_stress(
      const int &ielement, const int &xz,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xi,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &stress_integrand_xgamma) const {};

  KOKKOS_INLINE_FUNCTION void enforce_traction(
      const int &ielement, const int &xz,
      const specfem::kokkos::array_type<type_real, dimension::dim> &weight,
      const specfem::compute::element_partial_derivatives &partial_derivatives,
      const specfem::compute::element_properties<
          medium_type::value, property_type::value> &properties,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &velocity,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &accelation) const;

  __inline__ static std::string to_string() { return "Stacey"; }

private:
  quadrature_points_type quadrature_points; ///< Quadrature points object.
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types>
      type; ///< type of the edge on an element on the boundary.
};
} // namespace boundary_conditions
} // namespace enums
} // namespace specfem

#endif // _ENUMS_BOUNDARY_CONDITIONS_STACEY2D_ELASTIC_HPP_