#ifndef _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
#define _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/interface.hpp"
#include "domain/impl/elements/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace {

// Do not pull velocity from global memory
template <int components, specfem::enums::element::boundary_tag tag>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, components>
get_velocity(
    const int &iglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot) {

  specfem::kokkos::array_type<type_real, components> velocity;
  return velocity;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 1>
get_velocity<1, specfem::enums::element::boundary_tag::stacey>(
    const int &iglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot) {

  specfem::kokkos::array_type<type_real, 1> velocity;

  velocity[0] = field_dot(iglob, 0);

  return velocity;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
get_velocity<2, specfem::enums::element::boundary_tag::stacey>(
    const int &iglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot) {

  specfem::kokkos::array_type<type_real, 2> velocity;

  velocity[0] = field_dot(iglob, 0);
  velocity[1] = field_dot(iglob, 1);

  return velocity;
};

template <int NGLL, int components>
KOKKOS_FORCEINLINE_FUNCTION void compute_gradient(
    const int ix, const int iz,
    const Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_hprime,
    const Kokkos::View<type_real[components][NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &sv_field,
    specfem::kokkos::array_type<type_real, components> &du_dxi,
    specfem::kokkos::array_type<type_real, components> &du_dgamma) {

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      du_dxi[icomponent] += s_hprime(iz, l) * sv_field(icomponent, l, ix);
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      du_dgamma[icomponent] += s_hprime(l, ix) * sv_field(icomponent, iz, l);
    }
  }

  return;
}

template <int NGLL, int components>
KOKKOS_FORCEINLINE_FUNCTION void compute_acceleration(
    const int ix, const int iz,
    const Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_hprimewgll,
    const Kokkos::View<type_real[components][NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        &s_stress_integrand_xi,
    const Kokkos::View<type_real[components][NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        &s_stress_integrand_gamma,
    specfem::kokkos::array_type<type_real, components> &acceleration) {

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      acceleration[icomponent] +=
          s_hprimewgll(iz, l) * s_stress_integrand_xi(icomponent, l, ix);
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int l = 0; l < NGLL; ++l) {
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      acceleration[icomponent] +=
          s_hprimewgll(ix, l) * s_stress_integrand_gamma(icomponent, iz, l);
    }
  }

  return;
}

} // namespace

template <class medium, class qp_type, class property, class BC>
specfem::domain::impl::kernels::element_kernel<medium, qp_type, property, BC>::
    element_kernel(
        const specfem::kokkos::DeviceView3d<int> ibool,
        const specfem::kokkos::DeviceView1d<int> ispec,
        const specfem::compute::partial_derivatives &partial_derivatives,
        const specfem::compute::properties &properties,
        const specfem::compute::boundaries &boundary_conditions,
        specfem::quadrature::quadrature *quadx,
        specfem::quadrature::quadrature *quadz, qp_type quadrature_points,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            field_dot_dot,
        specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
            mass_matrix)
    : ibool(ibool), ispec(ispec), quadx(quadx), quadz(quadz),
      quadrature_points(quadrature_points), field(field), field_dot(field_dot),
      field_dot_dot(field_dot_dot), mass_matrix(mass_matrix) {

#ifndef NDEBUG
  assert(field.extent(1) == medium::components);
  assert(field_dot_dot.extent(1) == medium::components);
  assert(mass_matrix.extent(1) == medium::components);
#endif

  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

  for (int icomponent = 0; icomponent < medium::components; ++icomponent) {
    execution_space[icomponent] = Kokkos::DefaultExecutionSpace();
  }

  this->xix = specfem::kokkos::DeviceView1d<type_real>(
      partial_derivatives.xix.data(), ispec.extent(0) * ngllx * ngllz);

  this->gammax = specfem::kokkos::DeviceView1d<type_real>(
      partial_derivatives.gammax.data(), ispec.extent(0) * ngllx * ngllz);

  this->xiz = specfem::kokkos::DeviceView1d<type_real>(
      partial_derivatives.xiz.data(), ispec.extent(0) * ngllx * ngllz);

  this->gammaz = specfem::kokkos::DeviceView1d<type_real>(
      partial_derivatives.gammaz.data(), ispec.extent(0) * ngllx * ngllz);

  __du_dxi = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::__du_dxi",
      medium::components, ispec.extent(0), ngllz, ngllx);

  __du_dgamma = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::__du_dgamma",
      medium::components, ispec.extent(0), ngllz, ngllx);

  __stress_integrand_xi = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::stress_integrand_xi",
      medium::components, ispec.extent(0), ngllz, ngllx);

  __stress_integrand_gamma = specfem::kokkos::DeviceView4d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::stress_integrand_gamma",
      medium::components, ispec.extent(0), ngllz, ngllx);

  element = specfem::domain::impl::elements::element<
      dimension, medium_type, quadrature_point_type, property, BC>(
      partial_derivatives, properties, boundary_conditions, quadrature_points);
  return;
}

template <class medium, class qp_type, class property, class BC>
void specfem::domain::impl::kernels::element_kernel<
    medium, qp_type, property, BC>::compute_mass_matrix() const {

  constexpr int components = medium::components;
  const int nelements = ispec.extent(0);

  if (nelements == 0)
    return;

  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::compute_mass_matrix",
      specfem::kokkos::DeviceTeam(ispec.extent(0), Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l = ispec(team_member.league_rank());

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec_l, iz, ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  mass_matrix_element;

              element.compute_mass_matrix_component(ispec_l, xz,
                                                    mass_matrix_element);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&mass_matrix(iglob, icomponent),
                                     wxgll(ix) * wzgll(iz) *
                                         mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();

  return;
}

template <class medium, class qp_type, class property, class BC>
template <specfem::enums::time_scheme::type time_scheme>
void specfem::domain::impl::kernels::element_kernel<
    medium, qp_type, property, BC>::mass_time_contribution(const type_real dt)
    const {

  constexpr int components = medium::components;
  const int nelements = ispec.extent(0);

  if (nelements == 0)
    return;

  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::add_mass_matrix_contribution",
      specfem::kokkos::DeviceTeam(ispec.extent(0), Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ielement = team_member.league_rank();
        const auto ispec_l = ispec(ielement);

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec_l, iz, ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  mass_matrix_element;

              specfem::kokkos::array_type<type_real, dimension::dim> weight;

              weight[0] = wxgll(ix);
              weight[1] = wzgll(iz);

              element.template mass_time_contribution<time_scheme>(
                  ispec_l, ielement, xz, dt, weight, mass_matrix_element);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                const type_real __mass_matrix = mass_matrix(iglob, icomponent);
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&mass_matrix(iglob, icomponent),
                                     mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();
  return;
}

template <class medium, class qp_type, class property, class BC>
void specfem::domain::impl::kernels::element_kernel<
    medium, qp_type, property, BC>::compute_stiffness_interaction() const {

  constexpr int components = medium::components;
  const int nelements = ispec.extent(0);

  constexpr int NGLL = 5;
  constexpr int NGLL2 = NGLL * NGLL;

  if (nelements == 0)
    return;

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  const specfem::kokkos::DeviceView1d<int> __ibool(
      ibool.data(), ibool.extent(0) * NGLL * NGLL);

  // s_hprime_xx
  // int scratch_size_0 =
  //     Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
  //                  specfem::kokkos::DevScratchSpace,
  //                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  // s_field
  int scratch_size_0 =
      Kokkos::View<type_real[NTHREADS * components + 1][NGLL][NGLL],
                   Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  const int nteams = nelements / NTHREADS + (nelements % NTHREADS != 0);

  specfem::kokkos::DeviceView2d<type_real> __s_du_dxi(
      __du_dxi.data(), __du_dxi.extent(0),
      __du_dxi.extent(1) * __du_dxi.extent(2) * __du_dxi.extent(3));

  specfem::kokkos::DeviceView2d<type_real> __s_du_dgamma(
      __du_dgamma.data(), __du_dgamma.extent(0),
      __du_dgamma.extent(1) * __du_dgamma.extent(2) * __du_dgamma.extent(3));

  // Compute gradients

  Kokkos::parallel_for(
      "specfem::domain::kernels::elements::compute_gradients_component",
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(nteams, NTHREADS,
                                                        NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size_0)),
      KOKKOS_CLASS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &team_member) {
        const int i2 =
            (team_member.league_rank() * NTHREADS + team_member.team_rank()) *
            NGLL * NGLL;

        // Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
        //              specfem::kokkos::DevScratchSpace,
        //              Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        //     s_hprime(team_member.team_scratch(0));

        Kokkos::View<type_real[NTHREADS * components + 1][NGLL][NGLL],
                     Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >
            shared_mem(team_member.team_scratch(0));

        // initialize the views
        // ---------------------------------------------------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, 1), [&](const int &) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);
                    shared_mem(0, iz, ix) = hprime_xx(iz, ix);
                  });
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [&](const int &ithread) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);
                    const int iglob = __ibool(i2 + xz);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      shared_mem(ithread * components + icomponent + 1, iz,
                                 ix) = field(iglob, icomponent);
                    }
                  });
            });
        // ---------------------------------------------------------------------------

        team_member.team_barrier();

        // Compute gradients
        // ---------------------------------------------------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [&](const int &ithread) {
              const auto s_hprime =
                  Kokkos::subview(shared_mem, 0, Kokkos::ALL(), Kokkos::ALL());
              const auto sv_field = Kokkos::subview(
                  shared_mem,
                  Kokkos::pair<int, int>(ithread * components + 1,
                                         ithread * components + components + 1),
                  Kokkos::ALL(), Kokkos::ALL());
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);

                    // compute gradient
                    specfem::kokkos::array_type<type_real, components> du_dxi;
                    specfem::kokkos::array_type<type_real, components>
                        du_dgamma;

                    compute_gradient<NGLL, components>(
                        ix, iz, s_hprime, sv_field, du_dxi, du_dgamma);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    // store gradient
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {

                      __s_du_dxi(icomponent, i2 + xz) = du_dxi[icomponent];
                      __s_du_dgamma(icomponent, i2 + xz) =
                          du_dgamma[icomponent];
                    }
                  });
            });
      });

  const int total_qps = ispec.extent(0) * NGLL2;

  const int nteams_1 = total_qps / 128 + (total_qps % 128 != 0);

  Kokkos::parallel_for(
      "specfem::domain::kernels::elements::compute_stress_integrand",
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(nteams_1, 4, 32),
      KOKKOS_CLASS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &team_member) {
        const int team_rank = team_member.league_rank();

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, 128), [&](const int &ivec) {
              const int iqp = team_rank * 128 + ivec;
              if (iqp >= total_qps)
                return;

              const int ielement = iqp / NGLL2;
              const int ispec_l = ielement;
              const int xz = iqp % NGLL2;

              int ix, iz;
              sub2ind(xz, NGLL, iz, ix);

              specfem::kokkos::array_type<type_real, components> du_dxi;
              specfem::kokkos::array_type<type_real, components> du_dgamma;

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                du_dxi[icomponent] = __du_dxi(icomponent, ispec_l, iz, ix);
                du_dgamma[icomponent] =
                    __du_dgamma(icomponent, ispec_l, iz, ix);
              }

              specfem::kokkos::array_type<type_real, components>
                  stress_integrand_xi;
              specfem::kokkos::array_type<type_real, components>
                  stress_integrand_gamma;

              element.compute_stress(ispec_l, ielement, iz, ix, du_dxi,
                                     du_dgamma, stress_integrand_xi,
                                     stress_integrand_gamma);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                __stress_integrand_xi(icomponent, ispec_l, iz, ix) =
                    stress_integrand_xi[icomponent];
                __stress_integrand_gamma(icomponent, ispec_l, iz, ix) =
                    stress_integrand_gamma[icomponent];
              }
            });
      });

  // Compute acceleration

  int scratch_size_t =
      2 * Kokkos::View<type_real[NTHREADS * components][NGLL][NGLL],
                       Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  scratch_size_t +=
      Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  specfem::kokkos::DeviceView2d<type_real> __s_stress_integrand_xi(
      __stress_integrand_xi.data(), __stress_integrand_xi.extent(0),
      __stress_integrand_xi.extent(1) * __stress_integrand_xi.extent(2) *
          __stress_integrand_xi.extent(3));

  specfem::kokkos::DeviceView2d<type_real> __s_stress_integrand_gamma(
      __stress_integrand_gamma.data(), __stress_integrand_gamma.extent(0),
      __stress_integrand_gamma.extent(1) * __stress_integrand_gamma.extent(2) *
          __stress_integrand_gamma.extent(3));

  constexpr int gamma_component = components * NTHREADS;

  Kokkos::parallel_for(
      "specfem::domain::kernels::elements::compute_acceleration",
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace,
                         Kokkos::LaunchBounds<128, 16> >(nteams, NTHREADS,
                                                         NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size_t)),
      KOKKOS_CLASS_LAMBDA(
          const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
              &team_member) {
        const int i2 =
            (team_member.league_rank() * NTHREADS + team_member.team_rank()) *
            NGLL * NGLL;

        Kokkos::View<type_real[2 * NTHREADS * components + 1][NGLL][NGLL],
                     Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >
            shared_mem(team_member.team_scratch(0));

        // initialize the views
        //
        // ---------------------------------------------------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, 1), [&](const int &) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);
                    shared_mem(0, iz, ix) =
                        hprime_xx(iz, ix) * wxgll(iz) * wzgll(ix);
                  });
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [&](const int &ithread) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      shared_mem(ithread * components + icomponent + 1, iz,
                                 ix) =
                          __s_stress_integrand_xi(icomponent, i2 + xz);
                      shared_mem(gamma_component + ithread * components +
                                     icomponent + 1,
                                 iz, ix) =
                          __s_stress_integrand_gamma(icomponent, i2 + xz);
                    }
                  });
            });
        //
        // ---------------------------------------------------------------------------

        team_member.team_barrier();

        // Compute gradients
        //
        // ---------------------------------------------------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [&](const int &ithread) {
              const auto s_hprimewgll =
                  Kokkos::subview(shared_mem, 0, Kokkos::ALL(), Kokkos::ALL());
              const auto s_stress_integrand_xi = Kokkos::subview(
                  shared_mem,
                  Kokkos::pair<int, int>(ithread * components + 1,
                                         ithread * components + components + 1),
                  Kokkos::ALL(), Kokkos::ALL());
              const auto s_stress_integrand_gamma = Kokkos::subview(
                  shared_mem,
                  Kokkos::pair<int, int>(
                      gamma_component + ithread * components + 1,
                      gamma_component + ithread * components + components + 1),
                  Kokkos::ALL(), Kokkos::ALL());
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [&](const int &xz) {
                    int ix, iz;
                    sub2ind(xz, NGLL, iz, ix);
                    const int iglob = __ibool(i2 + xz);

                    // compute gradient
                    specfem::kokkos::array_type<type_real, components>
                        acceleration;

                    compute_acceleration<NGLL, components>(
                        ix, iz, s_hprimewgll, s_stress_integrand_xi,
                        s_stress_integrand_gamma, acceleration);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    // store acceleration
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      Kokkos::atomic_add(&field_dot_dot(iglob, icomponent),
                                         acceleration[icomponent]);
                    }
                  });
            });
      });

  Kokkos::fence();

  return;
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
