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

  __du_dx = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::du_dx",
      ispec.extent(0) * ngllx * ngllz * medium::components);

  __du_dz = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::du_dz",
      ispec.extent(0) * ngllx * ngllz * medium::components);

  __stress_integrand_xi = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::stress_integrand_xi",
      ispec.extent(0) * ngllx * ngllz * medium::components);

  __stress_integrand_gamma = specfem::kokkos::DeviceView1d<type_real>(
      "specfem::domain::impl::kernels::element_kernel::stress_integrand_gamma",
      ispec.extent(0) * ngllx * ngllz * medium::components);

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
  int scratch_size_0 =
      Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  // s_field
  scratch_size_0 +=
      Kokkos::View<type_real[NTHREADS][NGLL][NGLL], Kokkos::LayoutRight,
                   specfem::kokkos::DevScratchSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >::shmem_size();

  const int nteams = nelements / NTHREADS + (nelements % NTHREADS != 0);

  // Compute gradients
  for (int icomponent = 0; icomponent < components; ++icomponent) {

    Kokkos::parallel_for(
        "specfem::domain::kernels::elements::compute_gradients_component",
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(
            execution_space[icomponent], nteams, NTHREADS, NLANES)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size_0)),

        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
                &team_member) {
          const int ispec =
              team_member.league_rank() * NTHREADS + team_member.team_rank();
          const int i2 = ispec * NGLL * NGLL;
          const int i3 = i2 * components + icomponent * NGLL2;

          Kokkos::View<type_real[NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
              s_hprime(team_member.team_scratch(0));

          Kokkos::View<type_real[NTHREADS][NGLL][NGLL], Kokkos::LayoutRight,
                       specfem::kokkos::DevScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
              field_local(team_member.team_scratch(0));

          // initialize the views
          // ---------------------------------------------------------------------------
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, 1), [&](const int &) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                    [&](const int &xz) {
                      int ix, iz;
                      sub2ind(xz, NGLL, iz, ix);
                      s_hprime(iz, ix) = hprime_xx(iz, ix);
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

                      field_local(ithread, iz, ix) = field(iglob, icomponent);
                    });
              });
          // ---------------------------------------------------------------------------

          team_member.team_barrier();

          // Compute gradients
          // ---------------------------------------------------------------------------
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, NTHREADS),
              [&](const int &ithread) {
                const auto sv_field = Kokkos::subview(
                    field_local, ithread, Kokkos::ALL(), Kokkos::ALL());
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, NGLL * NGLL),
                    [&](const int &xz) {
                      int ix, iz;
                      sub2ind(xz, NGLL, iz, ix);

                      const auto partial_derivatives =
                          specfem::compute::element_partial_derivatives(
                              this->xix(i2 + xz), this->gammax(i2 + xz),
                              this->xiz(i2 + xz), this->gammaz(i2 + xz));

                      // compute gradient
                      const specfem::kokkos::array_type<type_real,
                                                        dimension::dim>
                          gradient_x = partial_derivatives
                                           .template compute_gradient<NGLL>(
                                               ix, iz, s_hprime, sv_field);

                      // store gradient
                      __du_dx(i3 + xz) = gradient_x[0];
                      __du_dz(i3 + xz) = gradient_x[1];
                    });
              });
        });
  }

  Kokkos::fence();

  return;
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
