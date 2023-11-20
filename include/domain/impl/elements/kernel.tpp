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

  if (nelements == 0)
    return;

  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();

  // s_hprime_xx, s_hprimewgll_xx
  int scratch_size =
      2 * quadrature_points.template shmem_size<
              type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

  // s_hprime_zz, s_hprimewgll_zz
  scratch_size +=
      2 * quadrature_points.template shmem_size<
              type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>();

  // s_field, s_stress_integrand_xi, s_stress_integrand_gamma
  scratch_size +=
      3 * Kokkos::View<type_real[NTHREADS][NGLL][NGLL][components],
                       Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  // s_iglob
  scratch_size +=
      Kokkos::View<int[NTHREADS][NGLL][NGLL][components], Kokkos::LayoutRight,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_interaction",
      specfem::kokkos::DeviceTeam(nelements / NTHREADS, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto team_rank = team_member.league_rank();
        // const auto ispec_l = ispec(ielement);

        // Instantiate shared views
        // ---------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));
        auto s_hprimewgll_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));
        auto s_hprimewgll_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_field =
            Kokkos::View<type_real[NTHREADS][NGLL][NGLL][components],
                         Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
                team_member.team_scratch(0));
        auto s_stress_integrand_xi =
            Kokkos::View<type_real[NTHREADS][NGLL][NGLL][components],
                         Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
                team_member.team_scratch(0));
        auto s_stress_integrand_gamma =
            Kokkos::View<type_real[NTHREADS][NGLL][NGLL][components],
                         Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
                team_member.team_scratch(0));
        auto s_iglob =
            Kokkos::View<int[NTHREADS][NGLL][NGLL][components],
                         Kokkos::LayoutRight, specfem::kokkos::DevScratchSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >(
                team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, 1), [&](const int &) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, ngllx * ngllx),
                  [=](const int xz) {
                    int iz, ix;
                    sub2ind(xz, ngllx, iz, ix);
                    s_hprime_xx(iz, ix, 0) = hprime_xx(iz, ix);
                    s_hprimewgll_xx(ix, iz, 0) = wxgll(iz) * hprime_xx(iz, ix);
                  });
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, 1), [&](const int &) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, ngllz * ngllz),
                  [=](const int xz) {
                    int iz, ix;
                    sub2ind(xz, ngllx, iz, ix);
                    s_hprime_zz(iz, ix, 0) = hprime_zz(iz, ix);
                    s_hprimewgll_zz(ix, iz, 0) = wzgll(iz) * hprime_zz(iz, ix);
                  });
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [=](const int thread_rank) {
              const int ielement = team_rank * NTHREADS + thread_rank;

              const int ispec_l = ispec(ielement);
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [=](const int xz) {
                    int iz, ix;
                    sub2ind(xz, ngllx, iz, ix);
                    const int iglob = ibool(ispec_l, iz, ix);
                    s_iglob(thread_rank, iz, ix, 0) = iglob;
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      s_field(thread_rank, iz, ix, icomponent) =
                          field(iglob, icomponent);
                      s_stress_integrand_xi(thread_rank, iz, ix, icomponent) =
                          0.0;
                      s_stress_integrand_gamma(thread_rank, iz, ix,
                                               icomponent) = 0.0;
                    }
                  });
            });

        //   Kokkos::parallel_for(
        //       quadrature_points.template
        //       TeamThreadRange<specfem::enums::axes::z,
        //                                                  specfem::enums::axes::x>(
        //           team_member),
        //       [&](const int xz) {
        //         int iz, ix;
        //         sub2ind(xz, ngllx, iz, ix);
        //         const int iglob = ibool(ispec_l, iz, ix);
        //         s_iglob(iz, ix, 0) = iglob;
        // #ifdef KOKKOS_ENABLE_CUDA
        // #pragma unroll
        // #endif
        //         for (int icomponent = 0; icomponent < components;
        //         ++icomponent) {
        //           s_field(iz, ix, icomponent) = field(iglob, icomponent);
        //           s_stress_integrand_xi(iz, ix, icomponent) = 0.0;
        //           s_stress_integrand_gamma(iz, ix, icomponent) = 0.0;
        //         }
        //       });

        // ------------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [=](const int thread_rank) {
              const int ielement = team_rank * NTHREADS + thread_rank;

              const int ispec_l = ispec(ielement);
              const auto sv_field = Kokkos::subview(
                  s_field, thread_rank, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [=](const int xz) {
                    int ix, iz;
                    sub2ind(xz, ngllx, iz, ix);

                    specfem::kokkos::array_type<type_real,
                                                medium_type::components>
                        dudxl;
                    specfem::kokkos::array_type<type_real,
                                                medium_type::components>
                        dudzl;

                    element.compute_gradient(ispec_l, ielement, xz, s_hprime_xx,
                                             s_hprime_zz, sv_field, dudxl,
                                             dudzl);

                    specfem::kokkos::array_type<type_real,
                                                medium_type::components>
                        stress_integrand_xi;
                    specfem::kokkos::array_type<type_real,
                                                medium_type::components>
                        stress_integrand_gamma;

                    element.compute_stress(ispec_l, ielement, xz, dudxl, dudzl,
                                           stress_integrand_xi,
                                           stress_integrand_gamma);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      s_stress_integrand_xi(thread_rank, iz, ix, icomponent) =
                          stress_integrand_xi[icomponent];
                      s_stress_integrand_gamma(thread_rank, iz, ix,
                                               icomponent) =
                          stress_integrand_gamma[icomponent];
                    }
                  });
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, NTHREADS),
            [=](const int thread_rank) {
              const int ielement = team_rank * NTHREADS + thread_rank;
              const int ispec_l = ispec(ielement);
              const auto sv_stress_integrand_xi =
                  Kokkos::subview(s_stress_integrand_xi, thread_rank,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
              const auto sv_stress_integrand_gamma =
                  Kokkos::subview(s_stress_integrand_gamma, thread_rank,
                                  Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                  [=](const int xz) {
                    int iz, ix;
                    sub2ind(xz, ngllx, iz, ix);

                    const int iglob = s_iglob(thread_rank, iz, ix, 0);
                    specfem::kokkos::array_type<type_real, dimension::dim>
                        weight;

                    weight[0] = wxgll(ix);
                    weight[1] = wzgll(iz);

                    specfem::kokkos::array_type<type_real,
                                                medium_type::components>
                        acceleration;

                    // only get velocity from global memory for stacey
                    // boundary
                    auto velocity =
                        get_velocity<components, BC::value>(iglob, field_dot);

                    element.compute_acceleration(
                        ispec_l, ielement, xz, weight, sv_stress_integrand_xi,
                        sv_stress_integrand_gamma, s_hprimewgll_xx,
                        s_hprimewgll_zz, velocity, acceleration);

#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
                    for (int icomponent = 0; icomponent < components;
                         ++icomponent) {
                      // Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                      Kokkos::atomic_add(&field_dot_dot(iglob, icomponent),
                                         acceleration[icomponent]);
                      // });
                    }
                  });
            });
      });

  Kokkos::fence();

  return;
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
