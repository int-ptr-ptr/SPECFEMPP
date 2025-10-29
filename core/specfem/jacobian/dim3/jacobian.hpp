#pragma once

#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

/**
 * Jacobian namespace contains overloaded functions for serial (without Kokkos)
 * and Kokkos implementations (using team policy)
 *
 */
namespace specfem::jacobian {

/**
 * @brief Compute global locations (x,z) from shape function matrix calcualted
 * at  \f$ (\xi, \gamma) \f$
 *
 * @param coorg Global control node locations (x_a, z_a)
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param gamma \f$ \gamma \f$ value of point
 * @param shape3D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
 * @return specfem::point::global_coordinates<specfem::dimension::type::dim3>
 * (x,y,z) value for the point
 */
specfem::point::global_coordinates<specfem::dimension::type::dim3>
compute_locations(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

/**
 * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param gamma \f$ \gamma \f$ value of point
 * @param dershape3D derivative of shape function matrix calculated at (xi,
 * eta, gamma)
    * @return std::tuple<type_real, type_real, type_real, type_real, type_real,
                            type_real, type_real, type_real, type_real> partial
 derivatives \f$ (\partial \xi/ \partial x, \partial \eta/ \partial
 * x,
 * \partial \xi/ \partial z, \partial \eta/ \partial z, \partial \gamma/
 \partial
 * z) \f$
 */
specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
compute_jacobian(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

template <typename CoordinateView, typename ShapeDerivativesView>
KOKKOS_FUNCTION
    specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
    compute_jacobian(const CoordinateView coordinates,
                     const ShapeDerivativesView shape_derivatives) {
  const int ngnod = coordinates.extent(0);
  type_real xxi = 0.0, yxi = 0.0, zxi = 0.0;
  type_real xeta = 0.0, yeta = 0.0, zeta = 0.0;
  type_real xgamma = 0.0, ygamma = 0.0, zgamma = 0.0;
  for (int in = 0; in < ngnod; in++) {
    xxi += shape_derivatives(0, in) * coordinates(in, 0);
    yxi += shape_derivatives(0, in) * coordinates(in, 1);
    zxi += shape_derivatives(0, in) * coordinates(in, 2);
    xeta += shape_derivatives(1, in) * coordinates(in, 0);
    yeta += shape_derivatives(1, in) * coordinates(in, 1);
    zeta += shape_derivatives(1, in) * coordinates(in, 2);
    xgamma += shape_derivatives(2, in) * coordinates(in, 0);
    ygamma += shape_derivatives(2, in) * coordinates(in, 1);
    zgamma += shape_derivatives(2, in) * coordinates(in, 2);
  }
  auto jacobian = xxi * (yeta * zgamma - ygamma * zeta) -
                  xeta * (yxi * zgamma - ygamma * zxi) +
                  xgamma * (yxi * zeta - yeta * zxi);
  const type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
  const type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
  const type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
  const type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
  const type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
  const type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
  const type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
  const type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
  const type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;
  return { xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, jacobian };
}

} // namespace specfem::jacobian
