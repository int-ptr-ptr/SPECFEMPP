#ifndef _DOMAIN_TPP
#define _DOMAIN_TPP

#include "compute/interface.hpp"
#include "domain.hpp"
#include "impl/elements/interface.hpp"
#include "impl/receivers/interface.hpp"
#include "impl/sources/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

template <typename element_type>
using element_container =
    typename specfem::domain::impl::elements::container<element_type>;

template <class dimension, class medium, class qp_type, class... traits>
using element_type =
    typename specfem::domain::impl::elements::element<dimension, medium,
                                                      qp_type, traits...>;

template <typename source_type>
using source_container =
    typename specfem::domain::impl::sources::container<source_type>;

template <class dimension, class medium, class qp_type, class... traits>
using source_type =
    typename specfem::domain::impl::sources::source<dimension, medium, qp_type,
                                                    traits...>;

template <typename receiver_type>
using receiver_container =
    typename specfem::domain::impl::receivers::container<receiver_type>;

template <class dimension, class medium, class qp_type, class... traits>
using receiver_type =
    typename specfem::domain::impl::receivers::receiver<dimension, medium,
                                                        qp_type, traits...>;

template <class medium>
void initialize_views(
    const int &nglob,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {

  constexpr int components = medium::components;

  Kokkos::parallel_for(
      "specfem::domain::domain::initiaze_views",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_LAMBDA(const int iglob, const int idim) {
        field(iglob, idim) = 0;
        field_dot(iglob, idim) = 0;
        field_dot_dot(iglob, idim) = 0;
        rmass_inverse(iglob, idim) = 0;
      });
}

template <class medium, class qp_type>
void initialize_rmass_inverse(
    const specfem::kokkos::DeviceView3d<int> ibool,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::DeviceView1d<element_container<element_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &elements,
    const qp_type &quadrature_points,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        rmass_inverse) {
  // Compute the mass matrix

  constexpr int components = medium::components;
  constexpr auto value = medium::value;

  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);
  const int ngllxz = ngllx * ngllz;

  const int nglob = rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::domain::domain::rmass_matrix",
      specfem::kokkos::DeviceTeam(elements.extent(0), Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto element = elements(team_member.league_rank());
        const auto ispec = element.get_ispec();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);

              type_real mass_matrix_element[components];

              element.compute_mass_matrix_component(xz, mass_matrix_element);

              for (int icomponent = 0; icomponent < components; icomponent++) {
                Kokkos::single(Kokkos::PerThread(team_member), [&]() {
                  Kokkos::atomic_add(&rmass_inverse(iglob, icomponent),
                                     wxgll(ix) * wzgll(iz) *
                                         mass_matrix_element[icomponent]);
                });
              }
            });
      });

  Kokkos::fence();

  // Invert the mass matrix
  Kokkos::parallel_for(
      "specfem::domain::domain::invert_rmass_matrix",
      specfem::kokkos::DeviceMDrange<2, Kokkos::Iterate::Left>(
          { 0, 0 }, { nglob, components }),
      KOKKOS_LAMBDA(const int iglob, const int idim) {
        rmass_inverse(iglob, idim) = 1.0 / rmass_inverse(iglob, idim);
      });

  Kokkos::fence();

  return;
}

template <class medium, class qp_type>
void assign_elemental_properties(
    specfem::compute::partial_derivatives &partial_derivatives,
    specfem::compute::properties &properties,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot,
    specfem::kokkos::DeviceView1d<element_container<element_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &elements,
    const int &nspec, const int &ngllz, const int &ngllx, int &nelem_domain) {

  const auto value = medium::value;
  const auto components = medium::components;

  // count number of elements in this domain
  nelem_domain = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == value) {
      nelem_domain++;
    }
  }

  specfem::kokkos::DeviceView1d<int> ispec_domain(
      "specfem::domain::domain::h_ispec_domain", nelem_domain);
  specfem::kokkos::HostMirror1d<int> h_ispec_domain =
      Kokkos::create_mirror_view(ispec_domain);

  elements = specfem::kokkos::DeviceView1d<element_container<element_type<
      specfem::enums::element::dimension::dim2, medium, qp_type> > >(
      "specfem::domain::domain::elements", nelem_domain);

  // Get ispec for each element in this domain
  int index = 0;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (properties.h_ispec_type(ispec) == value) {
      h_ispec_domain(index) = ispec;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index == nelem_domain);
#endif

  Kokkos::deep_copy(ispec_domain, h_ispec_domain);

  // Assign elemental properties
  // ----------------------------------------------------------------------
  auto h_elements = Kokkos::create_mirror_view(elements);

  Kokkos::parallel_for(
      "specfem::domain::allocate_memory",
      specfem::kokkos::HostRange(0, h_elements.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        h_elements(i)
            .element = (element_type<specfem::enums::element::dimension::dim2,
                                     medium, qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                element_type<specfem::enums::element::dimension::dim2, medium,
                             qp_type,
                             specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(elements, h_elements);

  Kokkos::parallel_for(
      "specfem::domain::instantialize_element",
      specfem::kokkos::DeviceRange(0, ispec_domain.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        const int ispec = ispec_domain(i);
        auto &element = elements(ispec).element;
        new (element)
            element_type<specfem::enums::element::dimension::dim2, medium,
                         qp_type, specfem::enums::element::property::isotropic>(
                ispec, partial_derivatives, properties);
      });

  Kokkos::fence();
};

template <class medium, class qp_type>
void initialize_sources(
    specfem::compute::properties &properties,
    specfem::compute::partial_derivatives &partial_derivatives,
    specfem::compute::sources &compute_sources,
    specfem::kokkos::DeviceView1d<source_container<source_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &sources) {

  const auto value = medium::value;
  const auto h_ispec_type = properties.h_ispec_type;
  const auto ispec_array = compute_sources.ispec_array;
  const auto h_ispec_array = compute_sources.h_ispec_array;

  int nsources_domain = 0;

  // Count the number of sources in the domain
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (h_ispec_type(h_ispec_array(isource)) == value) {
      nsources_domain++;
    }
  }

  sources = specfem::kokkos::DeviceView1d<source_container<source_type<
      specfem::enums::element::dimension::dim2, medium, qp_type> > >(
      "specfem::domain::domain::sources", nsources_domain);

  specfem::kokkos::DeviceView1d<int> my_sources(
      "specfem::domain::domain::my_sources", nsources_domain);

  auto h_my_sources = Kokkos::create_mirror_view(my_sources);
  auto h_sources = Kokkos::create_mirror_view(sources);

  // Check if the source is in the domain
  int index = 0;
  for (int isource = 0; isource < ispec_array.extent(0); isource++) {
    if (h_ispec_type(h_ispec_array(isource)) == value) {
      h_my_sources(index) = isource;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index == nsources_domain);
#endif

  Kokkos::deep_copy(my_sources, h_my_sources);

  // Allocate memory for the sources on the device
  Kokkos::parallel_for(
      "specfem::domain::domain::allocate_memory",
      specfem::kokkos::HostRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int isource) {
        h_sources(isource).source =
            (source_type<specfem::enums::element::dimension::dim2, medium,
                         qp_type> *)
                Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                    source_type<specfem::enums::element::dimension::dim2,
                                medium, qp_type,
                                specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(sources, h_sources);

  // Initialize the sources
  Kokkos::parallel_for(
      "specfem::domain::domain::initialize_source",
      specfem::kokkos::DeviceRange(0, nsources_domain),
      KOKKOS_LAMBDA(const int isource) {
        const int ispec = ispec_array(my_sources(isource));

        specfem::forcing_function::stf *source_time_function =
            compute_sources.stf_array(my_sources(isource)).T;

        const auto sv_source_array =
            Kokkos::subview(compute_sources.source_array, my_sources(isource),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        auto &source = sources(isource).source;
        new (source)
            source_type<specfem::enums::element::dimension::dim2, medium,
                        qp_type, specfem::enums::element::property::isotropic>(
                ispec, properties, sv_source_array, source_time_function);
      });

  Kokkos::fence();
  return;
}

template <class medium, class qp_type>
void initialize_receivers(
    const specfem::compute::receivers &compute_receivers,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::properties &properties,
    specfem::kokkos::DeviceView1d<receiver_container<receiver_type<
        specfem::enums::element::dimension::dim2, medium, qp_type> > >
        &receivers) {

  const auto value = medium::value;
  const auto h_ispec_type = properties.h_ispec_type;
  const auto ispec_array = compute_receivers.ispec_array;
  const auto h_ispec_array = compute_receivers.h_ispec_array;
  const auto seis_types = compute_receivers.seismogram_types;

  int nreceivers_domain = 0;

  // Count the number of receivers in the domain
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) == value) {
      nreceivers_domain++;
    }
  }

  nreceivers_domain = nreceivers_domain * seis_types.extent(0);

  receivers = specfem::kokkos::DeviceView1d<receiver_container<receiver_type<
      specfem::enums::element::dimension::dim2, medium, qp_type> > >(
      "specfem::domain::domain::receivers", nreceivers_domain);

  specfem::kokkos::DeviceView1d<int> my_receivers(
      "specfem::domain::domain::my_receivers", nreceivers_domain);

  auto h_my_receivers = Kokkos::create_mirror_view(my_receivers);
  auto h_receivers = Kokkos::create_mirror_view(receivers);

  // Check if the receiver is in the domain
  int index = 0;
  for (int irec = 0; irec < ispec_array.extent(0); irec++) {
    if (h_ispec_type(h_ispec_array(irec)) == value) {
      h_my_receivers(index) = irec;
      index++;
    }
  }

#ifndef NDEBUG
  assert(index * seis_types.extent(0) == nreceivers_domain);
#endif

  Kokkos::deep_copy(my_receivers, h_my_receivers);

  // Allocate memory for the receivers on the device
  Kokkos::parallel_for(
      "specfem::domain::domain::allocate_memory",
      specfem::kokkos::HostRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int irec) {
        h_receivers(irec)
            .receiver = (receiver_type<specfem::enums::element::dimension::dim2,
                                       medium, qp_type> *)
            Kokkos::kokkos_malloc<specfem::kokkos::DevMemSpace>(sizeof(
                receiver_type<specfem::enums::element::dimension::dim2, medium,
                              qp_type,
                              specfem::enums::element::property::isotropic>));
      });

  Kokkos::deep_copy(receivers, h_receivers);

  Kokkos::parallel_for(
      "specfem::domain::domain::initialize_receiver",
      specfem::kokkos::DeviceRange(0, nreceivers_domain),
      KOKKOS_LAMBDA(const int inum) {
        const int irec = my_receivers(inum / seis_types.extent(0));
        const int iseis = inum % seis_types.extent(0);
        const int ispec = ispec_array(irec);
        const auto seis_type = seis_types(iseis);
        const type_real cos_rec = compute_receivers.cos_recs(irec);
        const type_real sin_rec = compute_receivers.sin_recs(irec);

        auto sv_receiver_array =
            Kokkos::subview(compute_receivers.receiver_array, irec, Kokkos::ALL,
                            Kokkos::ALL, Kokkos::ALL);

        // Subview the seismogram array at current receiver and current
        // seismogram value
        auto sv_receiver_seismogram =
            Kokkos::subview(compute_receivers.seismogram, Kokkos::ALL, iseis,
                            irec, Kokkos::ALL);
        auto sv_receiver_field =
            Kokkos::subview(compute_receivers.receiver_field, Kokkos::ALL, irec,
                            iseis, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        auto &receiver = receivers(inum).receiver;
        new (receiver)
            receiver_type<specfem::enums::element::dimension::dim2, medium,
                          qp_type,
                          specfem::enums::element::property::isotropic>(
                ispec, sin_rec, cos_rec, seis_type, sv_receiver_array,
                sv_receiver_seismogram, partial_derivatives, properties,
                sv_receiver_field);
      });

  return;
}

template <class medium, class qp_type>
specfem::domain::domain<medium, qp_type>::domain(
    const int nglob, const qp_type &quadrature_points,
    specfem::compute::compute *compute,
    specfem::compute::properties material_properties,
    specfem::compute::partial_derivatives partial_derivatives,
    specfem::compute::sources compute_sources,
    specfem::compute::receivers compute_receivers,
    specfem::quadrature::quadrature *quadx,
    specfem::quadrature::quadrature *quadz)
    : field(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::domain::domain::field", nglob, medium::components)),
      field_dot(specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
          "specfem::domain::domain::field_dot", nglob, medium::components)),
      field_dot_dot(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::domain::domain::field_dot_dot", nglob,
              medium::components)),
      rmass_inverse(
          specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>(
              "specfem::domain::domain::rmass_inverse", nglob,
              medium::components)),
      quadrature_points(quadrature_points), compute(compute), quadx(quadx),
      quadz(quadz) {

  this->h_field = Kokkos::create_mirror_view(this->field);
  this->h_field_dot = Kokkos::create_mirror_view(this->field_dot);
  this->h_field_dot_dot = Kokkos::create_mirror_view(this->field_dot_dot);
  this->h_rmass_inverse = Kokkos::create_mirror_view(this->rmass_inverse);

  const auto ibool = compute->ibool;
  const int nspec = ibool.extent(0);
  const int ngllz = ibool.extent(1);
  const int ngllx = ibool.extent(2);

  //----------------------------------------------------------------------------
  // Initialize views

  // In CUDA you can call class lambdas inside the constructors
  // Hence I need to use this function to initialize views
  initialize_views<medium>(nglob, this->field, this->field_dot,
                           this->field_dot_dot, this->rmass_inverse);

  // ----------------------------------------------------------------------------
  // Inverse of mass matrix

  assign_elemental_properties(partial_derivatives, material_properties,
                              this->field_dot_dot, this->elements, nspec, ngllz,
                              ngllx, this->nelem_domain);

  //----------------------------------------------------------------------------
  // Inverse of mass matrix

  initialize_rmass_inverse(compute->ibool, quadx->get_w(), quadz->get_w(),
                           this->elements, this->quadrature_points,
                           this->rmass_inverse);

  // ----------------------------------------------------------------------------
  // Initialize the sources

  initialize_sources(material_properties, partial_derivatives, compute_sources,
                     this->sources);

  // ----------------------------------------------------------------------------
  // Initialize the receivers
  initialize_receivers(compute_receivers, partial_derivatives,
                       material_properties, this->receivers);

  return;
};

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_field(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_field_dot(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot, field_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot, h_field_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_field_dot_dot(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::sync_rmass_inverse(
    specfem::sync::kind kind) {

  if (kind == specfem::sync::DeviceToHost) {
    Kokkos::deep_copy(h_rmass_inverse, rmass_inverse);
  } else if (kind == specfem::sync::HostToDevice) {
    Kokkos::deep_copy(rmass_inverse, h_rmass_inverse);
  } else {
    throw std::runtime_error("Could not recognize the kind argument");
  }

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::divide_mass_matrix() {

  constexpr int components = medium::components;
  const int nglob = this->rmass_inverse.extent(0);

  Kokkos::parallel_for(
      "specfem::domain::domain::divide_mass_matrix",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        this->field_dot_dot(iglob, idim) =
            this->field_dot_dot(iglob, idim) * this->rmass_inverse(iglob, idim);
      });

  Kokkos::fence();

  return;
}

template <class medium, class qp_type>
void specfem::domain::domain<medium, qp_type>::compute_stiffness_interaction() {

  constexpr int components = medium::components;
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  const auto wxgll = this->quadx->get_w();
  const auto wzgll = this->quadz->get_w();
  const auto ibool = this->compute->ibool;

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
      3 *
      quadrature_points
          .template shmem_size<type_real, components, specfem::enums::axes::x,
                               specfem::enums::axes::z>();

  // s_iglob
  scratch_size +=
      quadrature_points.template shmem_size<int, 1, specfem::enums::axes::x,
                                            specfem::enums::axes::z>();

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_gradients",
      specfem::kokkos::DeviceTeam(this->nelem_domain, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto element = this->elements(team_member.league_rank());
        const auto ispec = element.get_ispec();

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
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_stress_integrand_xi =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_stress_integrand_gamma =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));
        auto s_iglob = quadrature_points.template ScratchView<
            int, 1, specfem::enums::axes::z, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix, 0) = hprime_xx(iz, ix);
              s_hprimewgll_xx(ix, iz, 0) = wxgll(iz) * hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix, 0) = hprime_zz(iz, ix);
              s_hprimewgll_zz(ix, iz, 0) = wzgll(iz) * hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              const int iglob = ibool(ispec, iz, ix);
              s_iglob(iz, ix, 0) = iglob;
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_field(iz, ix, icomponent) = field(iglob, icomponent);
                s_stress_integrand_xi(iz, ix, icomponent) = 0.0;
                s_stress_integrand_gamma(iz, ix, icomponent) = 0.0;
              }
            });

        // ------------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              type_real dudxl[components];
              type_real dudzl[components];

              type_real stress_integrand_xi[components];
              type_real stress_integrand_gamma[components];

              element.compute_gradient(xz, s_hprime_xx, s_hprime_zz, s_field,
                                       dudxl, dudzl);

              element.compute_stress(xz, dudxl, dudzl, stress_integrand_xi,
                                     stress_integrand_gamma);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_stress_integrand_xi(iz, ix, icomponent) =
                    stress_integrand_xi[icomponent];
                s_stress_integrand_gamma(iz, ix, icomponent) =
                    stress_integrand_gamma[icomponent];
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [&](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);

              const int iglob = s_iglob(iz, ix, 0);
              const type_real wxglll = wxgll(ix);
              const type_real wzglll = wzgll(iz);

              auto sv_field_dot_dot =
                  Kokkos::subview(field_dot_dot, iglob, Kokkos::ALL());

              element.update_acceleration(
                  xz, wxglll, wzglll, s_stress_integrand_xi,
                  s_stress_integrand_gamma, s_hprimewgll_xx, s_hprimewgll_zz,
                  sv_field_dot_dot);
            });
      });

  Kokkos::fence();

  return;
}

template <typename medium, typename qp_type>
void specfem::domain::domain<medium, qp_type>::compute_source_interaction(
    const type_real timeval) {

  constexpr int components = medium::components;
  const int nsources = this->sources.extent(0);
  const auto ibool = this->compute->ibool;

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_source_interaction",
      specfem::kokkos::DeviceTeam(nsources, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        int isource = team_member.league_rank();
        const auto source = this->sources(isource);
        const int ispec = source.get_ispec();
        auto sv_ibool = Kokkos::subview(ibool, ispec, Kokkos::ALL, Kokkos::ALL);

        type_real stf;

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, 1),
            [=](const int &, type_real &lsum) {
              lsum = source.eval_stf(timeval);
            },
            stf);

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);

              type_real accel[components];
              auto sv_field_dot_dot =
                  Kokkos::subview(field_dot_dot, iglob, Kokkos::ALL());

              source.compute_interaction(xz, stf, accel);
              source.update_acceleration(accel, sv_field_dot_dot);
            });
      });

  Kokkos::fence();
  return;
}

template <typename medium, typename qp_type>
void specfem::domain::domain<medium, qp_type>::compute_seismogram(
    const int isig_step) {

  // Allocate scratch views for field, field_dot & field_dot_dot. Incase of
  // acostic domains when calculating displacement, velocity and acceleration
  // seismograms we need to compute the derivatives of the field variables. This
  // requires summing over all lagrange derivatives at all quadrature points
  // within the element. Scratch views speed up this computation by limiting
  // global memory accesses.

  constexpr int components = medium::components;
  const auto ibool = this->compute->ibool;
  const auto hprime_xx = this->quadx->get_hprime();
  const auto hprime_zz = this->quadz->get_hprime();
  // hprime_xx
  int scratch_size = quadrature_points.template shmem_size<
      type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>();

  // hprime_zz
  scratch_size += quadrature_points.template shmem_size<
      type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>();

  // field, field_dot, field_dot_dot
  scratch_size +=
      3 *
      quadrature_points
          .template shmem_size<type_real, components, specfem::enums::axes::z,
                               specfem::enums::axes::x>();

  Kokkos::parallel_for(
      "specfem::domain::domain::compute_seismogram",
      specfem::kokkos::DeviceTeam(this->receivers.extent(0), Kokkos::AUTO, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const int irec = team_member.league_rank();
        const auto receiver = this->receivers(irec);
        const int ispec = receiver.get_ispec();
        const auto seismogram_type = receiver.get_seismogram_type();

        // Instantiate shared views
        // ----------------------------------------------------------------
        auto s_hprime_xx = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::x, specfem::enums::axes::x>(
            team_member.team_scratch(0));

        auto s_hprime_zz = quadrature_points.template ScratchView<
            type_real, 1, specfem::enums::axes::z, specfem::enums::axes::z>(
            team_member.team_scratch(0));

        auto s_field =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        auto s_field_dot =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        auto s_field_dot_dot =
            quadrature_points.template ScratchView<type_real, components,
                                                   specfem::enums::axes::z,
                                                   specfem::enums::axes::x>(
                team_member.team_scratch(0));

        // Allocate shared views
        // ----------------------------------------------------------------
        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              s_hprime_xx(iz, ix, 0) = hprime_xx(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::z>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllz, iz, ix);
              s_hprime_zz(iz, ix, 0) = hprime_zz(iz, ix);
            });

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              int iglob = ibool(ispec, iz, ix);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; icomponent++) {
                s_field(iz, ix, icomponent) = field(iglob, icomponent);
                s_field_dot(iz, ix, icomponent) = field_dot(iglob, icomponent);
                s_field_dot_dot(iz, ix, icomponent) =
                    field_dot_dot(iglob, icomponent);
              }
            });

        // Get seismogram field
        // ----------------------------------------------------------------

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::z,
                                                       specfem::enums::axes::x>(
                team_member),
            [=](const int xz) {
              receiver.get_field(xz, isig_step, s_field, s_field_dot,
                                 s_field_dot_dot, s_hprime_xx, s_hprime_zz);
            });

        // compute seismograms components
        //-------------------------------------------------------------------
        switch (seismogram_type) {
        case specfem::enums::seismogram::type::displacement:
        case specfem::enums::seismogram::type::velocity:
        case specfem::enums::seismogram::type::acceleration:
          dimension::array_type<type_real> seismogram_components;
          Kokkos::parallel_reduce(
              quadrature_points.template TeamThreadRange<
                  specfem::enums::axes::z, specfem::enums::axes::x>(
                  team_member),
              [=](const int xz,
                  dimension::array_type<type_real> &l_seismogram_components) {
                receiver.compute_seismogram_components(xz, isig_step,
                                                       l_seismogram_components);
              },
              specfem::kokkos::Sum<dimension::array_type<type_real> >(
                  seismogram_components));
          Kokkos::single(Kokkos::PerTeam(team_member), [=] {
            receiver.compute_seismogram(isig_step, seismogram_components);
          });
          break;
        }
      });
}

#endif /* DOMAIN_HPP_ */