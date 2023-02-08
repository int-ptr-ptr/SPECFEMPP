#include "../include/domain.h"
#include "../include/compute.h"
#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

specfem::Domain::Elastic::Elastic(
    const int ndim, const int nglob, specfem::compute::compute *compute,
    specfem::compute::properties *material_properties,
    specfem::compute::partial_derivatives *partial_derivatives,
    specfem::compute::sources *sources, quadrature::quadrature *quadx,
    quadrature::quadrature *quadz)
    : field(specfem::DeviceView2d<type_real>("specfem::Domain::Elastic::field",
                                             nglob, ndim)),
      field_dot(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot", nglob, ndim)),
      field_dot_dot(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::field_dot_dot", nglob, ndim)),
      rmass_inverse(specfem::DeviceView2d<type_real>(
          "specfem::Domain::Elastic::rmass_inverse", nglob, ndim)),
      compute(compute), material_properties(material_properties),
      partial_derivatives(partial_derivatives), sources(sources), quadx(quadx),
      quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  dux_dx = specfem::DeviceView3d<type_real>("specfem::Domin::Elastic::dux_dx",
                                            nspec, ngllz, ngllx);
  dux_dz = specfem::DeviceView3d<type_real>("specfem::Domin::Elastic::dux_dz",
                                            nspec, ngllz, ngllx);
  duz_dx = specfem::DeviceView3d<type_real>("specfem::Domin::Elastic::dux_dz",
                                            nspec, ngllz, ngllx);
  duz_dz = specfem::DeviceView3d<type_real>("specfem::Domin::Elastic::duz_dz",
                                            nspec, ngllz, ngllx);

  this->assign_views();

  this->nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->h_ispec_type(ispec) == elastic) {
      this->nelem_domain++;
    }
  }

  this->ispec_domain = specfem::DeviceView1d<int>(
      "specfem::Domain::Elastic::ispec_domain", this->nelem_domain);
  this->h_ispec_domain = Kokkos::create_mirror_view(ispec_domain);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (material_properties->h_ispec_type(ispec) == elastic) {
      this->h_ispec_domain(index) = ispec;
      index++;
    }
  }

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  return;
};

void specfem::Domain::Elastic::assign_views() {

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  const int nglob = field.extent(0);
  const int ndim = field.extent(1);
  // Initialize views
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::initiaze_views",
      specfem::DeviceMDrange<2>({ 0, 0 }, { nglob, ndim }),
      KOKKOS_CLASS_LAMBDA(const int iglob, const int idim) {
        this->field(iglob, idim) = 0;
        this->field_dot(iglob, idim) = 0;
        this->field_dot_dot(iglob, idim) = 0;
        this->rmass_inverse(iglob, idim) = 0;
      });

  // Compute the mass matrix
  specfem::DeviceScatterView2d<type_real> results(rmass_inverse);
  auto wxgll = quadx->get_w();
  auto wzgll = quadz->get_w();
  auto rho = this->material_properties->rho;
  auto ispec_type = this->material_properties->ispec_type;
  auto jacobian = this->partial_derivatives->jacobian;
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_mass_matrix",
      specfem::DeviceMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      KOKKOS_CLASS_LAMBDA(const int ispec, const int iz, const int ix) {
        int iglob = ibool(ispec, iz, ix);
        type_real rhol = rho(ispec, iz, ix);
        auto access = results.access();
        if (ispec_type(ispec) == elastic) {
          access(iglob, 0) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
          access(iglob, 1) +=
              wxgll(ix) * wzgll(iz) * rhol * jacobian(ispec, iz, ix);
        }
      });

  Kokkos::Experimental::contribute(rmass_inverse, results);

  // invert the mass matrix
  Kokkos::parallel_for(
      "specfem::Domain::Elastic::Invert_mass_matrix",
      specfem::DeviceRange(0, nglob), KOKKOS_CLASS_LAMBDA(const int iglob) {
        if (rmass_inverse(iglob, 0) > 0.0) {
          rmass_inverse(iglob, 0) = 1.0 / rmass_inverse(iglob, 0);
          rmass_inverse(iglob, 1) = 1.0 / rmass_inverse(iglob, 1);
        } else {
          rmass_inverse(iglob, 0) = 1.0;
          rmass_inverse(iglob, 1) = 1.0;
        }
      });

  return;
}

void specfem::Domain::Elastic::sync_field(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_field_dot(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_field_dot_dot(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::sync_rmass_inverse(specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

void specfem::Domain::Elastic::compute_stiffness_interaction() {

  const int ngllx = this->quadx->get_N();
  const int ngllz = this->quadz->get_N();
  const int ngllxz = ngllx * ngllz;
  const auto ibool = this->compute->ibool;
  const auto ispec_domain = this->ispec_domain;
  const auto xix = this->partial_derivatives->xix;
  const auto xiz = this->partial_derivatives->xiz;
  const auto gammax = this->partial_derivatives->gammax;
  const auto gammaz = this->partial_derivatives->gammaz;
  const auto jacobian = this->partial_derivatives->jacobian;
  const auto mu = this->material_properties->mu;
  const auto lambdaplus2mu = this->material_properties->lambdaplus2mu;
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();

  const int spectral_elems_per_block = 2;
  const int total_number_of_blocks =
      this->nelem_domain / spectral_elems_per_block;
  const int n_threads_per_warp = 32;
  const int n_threads_per_block = n_threads_per_warp * spectral_elems_per_block;

  int scratch_size =
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllx);
  scratch_size +=
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllz);
  scratch_size += 2 * specfem::DeviceScratchView3d<type_real>::shmem_size(
                          spectral_elems_per_block, ngllx, ngllz);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_gradient",
      specfem::DeviceTeam(total_number_of_blocks, n_threads_per_block, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(const specfem::DeviceTeam::member_type &team_member) {
        const int league_rank = team_member.league_rank();

        specfem::DeviceScratchView2d<type_real> s_hprime_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::DeviceScratchView2d<type_real> s_hprime_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::DeviceScratchView3d<type_real> s_tempx(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);
        specfem::DeviceScratchView3d<type_real> s_tempz(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);

        // -------------Load into scratch memory----------------------------
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, n_threads_per_block),
            [=](const int thread_id) {
              const int xz = thread_id % n_threads_per_warp;
              const int l_ispec = thread_id / n_threads_per_warp;
              const int g_ispec = ispec_domain(
                  spectral_elems_per_block * league_rank + l_ispec);
              if (xz < ngllxz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;
                int iglob = ibool(g_ispec, iz, ix);
                s_tempx(l_ispec, iz, ix) = this->field(iglob, 0);
                s_tempz(l_ispec, iz, ix) = this->field(iglob, 1);
              }
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
            [=](const int ij) {
              const int i = ij % ngllx;
              const int j = ij / ngllx;
              s_hprime_xx(j, i) = hprime_xx(j, i);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
            [=](const int ij) {
              const int i = ij % ngllz;
              const int j = ij / ngllz;
              s_hprime_zz(j, i) = hprime_zz(j, i);
            });
        //----------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, n_threads_per_block),
            [=](const int thread_id) {
              const int xz = thread_id % n_threads_per_warp;
              const int l_ispec = thread_id / n_threads_per_warp;
              const int g_ispec = ispec_domain(
                  spectral_elems_per_block * league_rank + l_ispec);
              if (xz < ngllxz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;

                type_real sum_hprime_x1 = 0;
                type_real sum_hprime_x3 = 0;
                type_real sum_hprime_z1 = 0;
                type_real sum_hprime_z3 = 0;

                for (int l = 0; l < ngllx; l++) {
                  sum_hprime_x1 += s_hprime_xx(ix, l) * s_tempx(l_ispec, iz, l);
                  sum_hprime_x3 += s_hprime_xx(ix, l) * s_tempz(l_ispec, iz, l);
                }

                for (int l = 0; l < ngllz; l++) {
                  sum_hprime_z1 += s_hprime_zz(iz, l) * s_tempx(l_ispec, l, ix);
                  sum_hprime_z3 += s_hprime_zz(iz, l) * s_tempz(l_ispec, l, ix);
                }

                const type_real xixl = xix(g_ispec, iz, ix);
                const type_real xizl = xiz(g_ispec, iz, ix);
                const type_real gammaxl = gammax(g_ispec, iz, ix);
                const type_real gammazl = gammaz(g_ispec, iz, ix);

                this->dux_dx(g_ispec, iz, ix) =
                    xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;
                this->dux_dz(g_ispec, iz, ix) =
                    xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

                this->duz_dx(g_ispec, iz, ix) =
                    xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;
                this->duz_dz(g_ispec, iz, ix) =
                    xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;
              }
            });
      });

  scratch_size =
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllx, ngllx);
  scratch_size +=
      specfem::DeviceScratchView2d<type_real>::shmem_size(ngllz, ngllz);
  scratch_size += 4 * specfem::DeviceScratchView3d<type_real>::shmem_size(
                          spectral_elems_per_block, ngllx, ngllz);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_stiffness_interaction",
      specfem::DeviceTeam(total_number_of_blocks, n_threads_per_block, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(const specfem::DeviceTeam::member_type &team_member) {
        const int league_rank = team_member.league_rank();

        specfem::DeviceScratchView2d<type_real> s_hprime_wgll_xx(
            team_member.team_scratch(0), ngllx, ngllx);
        specfem::DeviceScratchView2d<type_real> s_hprime_wgll_zz(
            team_member.team_scratch(0), ngllz, ngllz);
        specfem::DeviceScratchView3d<type_real> s_tempx1(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);
        specfem::DeviceScratchView3d<type_real> s_tempz1(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);
        specfem::DeviceScratchView3d<type_real> s_tempx3(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);
        specfem::DeviceScratchView3d<type_real> s_tempz3(
            team_member.team_scratch(0), spectral_elems_per_block, ngllz,
            ngllx);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllx * ngllx),
            [=](const int ij) {
              const int i = ij % ngllx;
              const int j = ij / ngllx;
              s_hprime_wgll_xx(i, j) = wxgll(j) * hprime_xx(j, i);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, ngllz * ngllz),
            [=](const int ij) {
              const int i = ij % ngllz;
              const int j = ij / ngllz;
              s_hprime_wgll_zz(i, j) = wzgll(j) * hprime_zz(j, i);
            });

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, n_threads_per_block),
            [=](const int thread_id) {
              const int xz = thread_id % n_threads_per_warp;
              const int l_ispec = thread_id / n_threads_per_warp;
              const int g_ispec = ispec_domain(
                  spectral_elems_per_block * league_rank + l_ispec);
              if (xz < ngllxz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;

                const type_real lambdal =
                    lambdaplus2mu(g_ispec, iz, ix) - 2 * mu(g_ispec, iz, ix);

                const type_real sigma_xx =
                    lambdaplus2mu(g_ispec, iz, ix) *
                        this->dux_dx(g_ispec, iz, ix) +
                    lambdal * this->duz_dz(g_ispec, iz, ix);
                const type_real sigma_zz =
                    lambdaplus2mu(g_ispec, iz, ix) *
                        this->duz_dz(g_ispec, iz, ix) +
                    lambdal * this->dux_dx(g_ispec, iz, ix);
                const type_real sigma_xz =
                    mu(g_ispec, iz, ix) * (this->duz_dx(g_ispec, iz, ix) +
                                           this->dux_dz(g_ispec, iz, ix));

                s_tempx1(l_ispec, iz, ix) = jacobian(g_ispec, iz, ix) *
                                            (sigma_xx * xix(g_ispec, iz, ix) +
                                             sigma_xz * xiz(g_ispec, iz, ix));
                s_tempz1(l_ispec, iz, ix) = jacobian(g_ispec, iz, ix) *
                                            (sigma_xz * xix(g_ispec, iz, ix) +
                                             sigma_zz * xiz(g_ispec, iz, ix));
                s_tempx3(l_ispec, ix, iz) =
                    jacobian(g_ispec, ix, iz) *
                    (sigma_xx * gammax(g_ispec, iz, ix) +
                     sigma_xz * gammaz(g_ispec, iz, ix));
                s_tempz3(l_ispec, ix, iz) =
                    jacobian(g_ispec, iz, ix) *
                    (sigma_xz * gammax(g_ispec, iz, ix) +
                     sigma_zz * gammaz(g_ispec, iz, ix));
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, n_threads_per_block),
            [=](const int thread_id) {
              const int xz = thread_id % n_threads_per_warp;
              const int l_ispec = thread_id / n_threads_per_warp;
              const int g_ispec = ispec_domain(
                  spectral_elems_per_block * league_rank + l_ispec);
              if (xz < ngllxz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;
                type_real tempx1 = 0;
                type_real tempz1 = 0;

                for (int l = 0; l < ngllx; l++) {
                  tempx1 += s_hprime_wgll_xx(ix, l) * s_tempx1(l_ispec, iz, l);
                  tempz1 += s_hprime_wgll_xx(ix, l) * s_tempz1(l_ispec, iz, l);
                }

                type_real tempx3 = 0;
                type_real tempz3 = 0;

                for (int l = 0; l < ngllx; l++) {
                  tempx3 += s_hprime_wgll_zz(iz, l) * s_tempx3(l_ispec, ix, l);
                  tempz3 += s_hprime_wgll_zz(iz, l) * s_tempz3(l_ispec, ix, l);
                }

                const int iglob = ibool(g_ispec, iz, ix);
                const type_real sum_terms1 =
                    -1.0 * (wzgll(iz) * tempx1) - (wxgll(ix) * tempx3);
                const type_real sum_terms3 =
                    -1.0 * (wzgll(iz) * tempz1) - (wxgll(ix) * tempz3);
                Kokkos::single(Kokkos::PerThread(team_member), [=] {
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 0),
                                     sum_terms1);
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 1),
                                     sum_terms3);
                });
              }
            });
      });

  return;
}

KOKKOS_IMPL_HOST_FUNCTION
void specfem::Domain::Elastic::divide_mass_matrix() {
  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::divide_mass_matrix",
      specfem::DeviceRange(0, nglob), KOKKOS_CLASS_LAMBDA(const int iglob) {
        this->field_dot_dot(iglob, 0) =
            this->field_dot_dot(iglob, 0) * this->rmass_inverse(iglob, 0);
        this->field_dot_dot(iglob, 1) =
            this->field_dot_dot(iglob, 1) * this->rmass_inverse(iglob, 1);
      });

  Kokkos::fence();

  return;
}

void specfem::Domain::Elastic::compute_source_interaction(
    const type_real timeval) {
  const int nsources = this->sources->source_array.extent(0);
  const int ngllz = this->sources->source_array.extent(1);
  const int ngllx = this->sources->source_array.extent(2);
  const int ngllxz = ngllx * ngllz;
  const auto ispec_array = this->sources->ispec_array;
  const auto ispec_type = this->material_properties->ispec_type;
  const auto stf_array = this->sources->stf_array;
  const auto source_array = this->sources->source_array;
  const auto ibool = this->compute->ibool;

  Kokkos::parallel_for(
      "specfem::Domain::Elastic::compute_source_interaction",
      specfem::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(const specfem::DeviceTeam::member_type &team_member) {
        int isource = team_member.league_rank();
        int ispec = ispec_array(isource);
        auto sv_ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

        if (ispec_type(ispec) == elastic) {

          type_real stf;

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team_member, 1),
              [=](const int &, type_real &lsum) {
                lsum = stf_array(isource).T->compute(timeval);
              },
              stf);

          team_member.team_barrier();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, ngllxz), [=](const int xz) {
                const int ix = xz % ngllz;
                const int iz = xz / ngllz;
                int iglob = sv_ibool(iz, ix);

                if (wave == p_sv) {
                  const type_real accelx =
                      source_array(isource, iz, ix, 0) * stf;
                  const type_real accelz =
                      source_array(isource, iz, ix, 1) * stf;
                  Kokkos::single(Kokkos::PerThread(team_member), [=] {
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                    Kokkos::atomic_add(&this->field_dot_dot(iglob, 1), accelz);
                  });
                } else {
                  const type_real accelx =
                      source_array(isource, iz, ix, 0) * stf;
                  Kokkos::atomic_add(&this->field_dot_dot(iglob, 0), accelx);
                }
              });
        }
      });

  Kokkos::fence();
  return;
}
