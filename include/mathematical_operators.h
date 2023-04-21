#ifndef MATHEMATICAL_OPERATORS_H
#define MATHEMATICAL_OPERATORS_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

// using specfem::kokkos::simd_type =
// Kokkos::Experimental::native_simd<type_real>; using mask_type =
// Kokkos::Experimental::native_simd_mask<double>; using tag_type =
// Kokkos::Experimental::element_aligned_tag;

namespace specfem {
namespace mathematical_operators {

template <int NGLLZ, int NGLLX>
KOKKOS_FUNCTION void compute_gradients_2D(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const int ispec,
    const specfem::kokkos::DeviceView3d<specfem::kokkos::simd_type<> > xix,
    const specfem::kokkos::DeviceView3d<specfem::kokkos::simd_type<> > xiz,
    const specfem::kokkos::DeviceView3d<specfem::kokkos::simd_type<> > gammax,
    const specfem::kokkos::DeviceView3d<specfem::kokkos::simd_type<> > gammaz,
    const specfem::kokkos::StaticDeviceScratchView2d<
        specfem::kokkos::simd_type<>, NGLLZ, NGLLX>
        s_hprime_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<
        specfem::kokkos::simd_type<>, NGLLZ, NGLLX>
        s_hprime_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<
        specfem::kokkos::simd_type<>, NGLLZ, NGLLX>
        field_x,
    const specfem::kokkos::StaticDeviceScratchView2d<
        specfem::kokkos::simd_type<>, NGLLZ, NGLLX>
        field_z,
    specfem::kokkos::StaticDeviceScratchView2d<specfem::kokkos::simd_type<>,
                                               NGLLZ, NGLLX>
        s_duxdx,
    specfem::kokkos::StaticDeviceScratchView2d<specfem::kokkos::simd_type<>,
                                               NGLLZ, NGLLX>
        s_duxdz,
    specfem::kokkos::StaticDeviceScratchView2d<specfem::kokkos::simd_type<>,
                                               NGLLZ, NGLLX>
        s_duzdx,
    specfem::kokkos::StaticDeviceScratchView2d<specfem::kokkos::simd_type<>,
                                               NGLLZ, NGLLX>
        s_duzdz) {

  const int NGLL2 = NGLLZ * NGLLX;
  assert(xix.extent(1) == NGLLZ);
  assert(xix.extent(2) == NGLLX);
  assert(xiz.extent(1) == NGLLZ);
  assert(xiz.extent(2) == NGLLX);
  assert(gammax.extent(1) == NGLLZ);
  assert(gammax.extent(2) == NGLLX);
  assert(gammaz.extent(1) == NGLLZ);
  assert(gammaz.extent(2) == NGLLX);

  constexpr type_real NGLLX_INV = 1.0 / NGLLX;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
        const int iz = xz * NGLLX_INV;
        const int simd_ix = xz - iz * NGLLX;

        const specfem::kokkos::simd_type<> xixl = xix(ispec, iz, simd_ix);
        const specfem::kokkos::simd_type<> xizl = xiz(ispec, iz, simd_ix);
        const specfem::kokkos::simd_type<> gammaxl = gammax(ispec, iz, simd_ix);
        const specfem::kokkos::simd_type<> gammazl = gammaz(ispec, iz, simd_ix);

        specfem::kokkos::simd_type<> sum_hprime_x1(0.0);
        specfem::kokkos::simd_type<> sum_hprime_x3(0.0);
        specfem::kokkos::simd_type<> sum_hprime_z1(0.0);
        specfem::kokkos::simd_type<> sum_hprime_z3(0.0);

        for (int l = 0; l < NGLLZ; l++) {
          sum_hprime_x1 += s_hprime_xx(l, simd_ix) * field_x(iz, l);
          sum_hprime_x3 += s_hprime_xx(l, simd_ix) * field_z(iz, l);
          sum_hprime_z1 += s_hprime_zz(iz, l) * field_x(l, simd_ix);
          sum_hprime_z3 += s_hprime_zz(iz, l) * field_z(l, simd_ix);
        }

        const type_real debug_sum_hprime_x1 = sum_hprime_x1[0];
        const type_real debug_sum_hprime_x3 = sum_hprime_x3[0];
        const type_real debug_sum_hprime_z1 = sum_hprime_z1[0];
        const type_real debug_sum_hprime_z3 = sum_hprime_z3[0];
        // duxdx
        s_duxdx(iz, simd_ix) = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

        // duxdz
        s_duxdz(iz, simd_ix) = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

        // duzdx
        s_duzdx(iz, simd_ix) = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

        // duzdz
        s_duzdz(iz, simd_ix) = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;
      });

  return;
};

} // namespace mathematical_operators
} // namespace specfem

#endif
