#ifndef _MATHEMATICAL_OPERATORS_TPP
#define _MATHEMATICAL_OPERATORS_TPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using tag_type = Kokkos::Experimental::element_aligned_tag;
using mask_type = Kokkos::Experimental::simd_mask<type_real, Kokkos::Experimental::simd_abi::native<type_real>>;

namespace specfem {
namespace mathematical_operators {

template <int NGLL>
KOKKOS_FUNCTION void compute_gradients_2D(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const int ispec, const specfem::kokkos::DeviceView3d<type_real> xix,
    const specfem::kokkos::DeviceView3d<type_real> xiz,
    const specfem::kokkos::DeviceView3d<type_real> gammax,
    const specfem::kokkos::DeviceView3d<type_real> gammaz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_x,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_z,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        tfield_x,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        tfield_z,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duxdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duxdz,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duzdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duzdz) {


  constexpr int width = specfem::kokkos::simd_type<>::size();
  constexpr int NGLLZ = NGLL;
  constexpr int NGLLX = NGLL/width + (NGLL%width > 0);

  constexpr int NGLL2 = NGLLZ * NGLLX;
  assert(xix.extent(1) == NGLL);
  assert(xix.extent(2) == NGLL);
  assert(xiz.extent(1) == NGLL);
  assert(xiz.extent(2) == NGLL);
  assert(gammax.extent(1) == NGLL);
  assert(gammax.extent(2) == NGLL);
  assert(gammaz.extent(1) == NGLL);
  assert(gammaz.extent(2) == NGLL);

  constexpr type_real NGLL_INV = 1.0 / NGLLX;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
        const int iz = xz * NGLL_INV;
        const int ix = xz - iz * NGLLX;

        // const type_real xixl = xix(ispec, iz, ix);
        // const type_real xizl = xiz(ispec, iz, ix);
        // const type_real gammaxl = gammax(ispec, iz, ix);
        // const type_real gammazl = gammaz(ispec, iz, ix);

        specfem::kokkos::simd_type<> sum_hprime_x1(0.0);
        specfem::kokkos::simd_type<> sum_hprime_x3(0.0);
        specfem::kokkos::simd_type<> sum_hprime_z1(0.0);
        specfem::kokkos::simd_type<> sum_hprime_z3(0.0);

        specfem::kokkos::simd_type<> simd_xix(&xix(ispec, iz, ix), tag_type());
        specfem::kokkos::simd_type<> simd_xiz(&xiz(ispec, iz, ix), tag_type());
        specfem::kokkos::simd_type<> simd_gammax(&gammax(ispec, iz, ix), tag_type());
        specfem::kokkos::simd_type<> simd_gammaz(&gammaz(ispec, iz, ix), tag_type());


        for (int l = 0; l < NGLL; l += width) {

          mask_type mask([=](std::size_t lane) { return l + int(lane) < NGLL; });
          specfem::kokkos::simd_type<> simd_hprime_xx;
          specfem::kokkos::simd_type<> simd_hprime_zz;
          specfem::kokkos::simd_type<> simd_tfield_x;
          specfem::kokkos::simd_type<> simd_tfield_z;
          specfem::kokkos::simd_type<> simd_field_x;
          specfem::kokkos::simd_type<> simd_field_z;

          Kokkos::Experimental::where(mask, simd_hprime_xx).copy_from(&s_hprime_xx(l, ix), tag_type());
          Kokkos::Experimental::where(mask, simd_hprime_zz).copy_from(&s_hprime_zz(l, iz), tag_type());
          Kokkos::Experimental::where(mask, simd_tfield_x).copy_from(&tfield_x(l, iz), tag_type());
          Kokkos::Experimental::where(mask, simd_tfield_z).copy_from(&tfield_z(l, iz), tag_type());
          Kokkos::Experimental::where(mask, simd_field_x).copy_from(&field_x(l, ix), tag_type());
          Kokkos::Experimental::where(mask, simd_field_z).copy_from(&field_z(l, ix), tag_type());

        //   sum_hprime_x1 += simd_hprime_xx * simd_tfield_x;
        //   sum_hprime_x3 += simd_hprime_xx * simd_tfield_z;
        //   sum_hprime_z1 += simd_hprime_zz * simd_field_x;
        //   sum_hprime_z3 += simd_hprime_zz * simd_field_x;
        }

        specfem::kokkos::simd_type<> duxdx = simd_xix * sum_hprime_x1 + simd_gammax * sum_hprime_x3;
        specfem::kokkos::simd_type<> duxdz = simd_xiz * sum_hprime_x1 + simd_gammaz * sum_hprime_x3;
        specfem::kokkos::simd_type<> duzdx = simd_xix * sum_hprime_z1 + simd_gammax * sum_hprime_z3;
        specfem::kokkos::simd_type<> duzdz = simd_xiz * sum_hprime_z1 + simd_gammaz * sum_hprime_z3;

        duxdx.copy_to(&s_duxdx(iz, ix), tag_type());
        duxdz.copy_to(&s_duxdz(iz, ix), tag_type());
        duzdx.copy_to(&s_duzdx(iz, ix), tag_type());
        duzdz.copy_to(&s_duzdz(iz, ix), tag_type());

        // // duxdx
        // s_duxdx(iz, ix) = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

        // // duxdz
        // s_duxdz(iz, ix) = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

        // // duzdx
        // s_duzdx(iz, ix) = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

        // // duzdz
        // s_duzdz(iz, ix) = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;
      });

  return;
};

template <int NGLL>
KOKKOS_FUNCTION void add_contributions(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<int, NGLL, NGLL> s_iglob,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_1,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_2,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_3,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_4,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        field_dot_dot) {

  assert(wxgll.extent(0) == NGLL);
  assert(wzgll.extent(0) == NGLL);

  constexpr int NGLL2 = NGLL * NGLL;
  constexpr type_real NGLL_INV = 1.0 / NGLL;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
        const int iz = xz * NGLL_INV;
        const int ix = xz - iz * NGLL;

        type_real tempx1 = 0.0;
        type_real tempz1 = 0.0;
        type_real tempx3 = 0.0;
        type_real tempz3 = 0.0;

#pragma unroll
        for (int l = 0; l < NGLL; l++) {
          tempx1 += s_hprimewgll_xx(ix, l) * stress_integrand_1(iz, l);
          tempz1 += s_hprimewgll_xx(ix, l) * stress_integrand_2(iz, l);
          tempx3 += s_hprimewgll_zz(iz, l) * stress_integrand_3(l, ix);
          tempz3 += s_hprimewgll_zz(iz, l) * stress_integrand_4(l, ix);
        }

        const int iglob = s_iglob(iz, ix);
        const type_real sum_terms1 =
            -1.0 * (wzgll(iz) * tempx1) - (wxgll(ix) * tempx3);
        const type_real sum_terms3 =
            -1.0 * (wzgll(iz) * tempz1) - (wxgll(ix) * tempz3);
        Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
        Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
      });
}
} // namespace mathematical_operators
} // namespace specfem

#endif
