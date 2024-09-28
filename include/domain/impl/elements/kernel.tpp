#ifndef _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
#define _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP

#include "compute/interface.hpp"
#include "domain/impl/elements/acoustic/interface.hpp"
#include "domain/impl/elements/elastic/interface.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kernel.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include "_util/dump_discont_simfield.hpp"

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    element_kernel_base(
        const specfem::compute::assembly &assembly,
        const specfem::kokkos::HostView1d<int> h_element_kernel_index_mapping,
        const quadrature_points_type &quadrature_points)
    : nelements(h_element_kernel_index_mapping.extent(0)),
      element_kernel_index_mapping("specfem::domain::impl::kernels::element_"
                                   "kernel_base::element_kernel_index_mapping",
                                   nelements),
      h_element_kernel_index_mapping(h_element_kernel_index_mapping),
      points(assembly.mesh.points), quadrature(assembly.mesh.quadratures),
      partial_derivatives(assembly.partial_derivatives),
      properties(assembly.properties),
      boundary_conditions(assembly.boundaries.boundary_types),
      quadrature_points(quadrature_points),
      boundary_values(assembly.boundary_values.get_container<BoundaryTag>()),
      element(assembly, quadrature_points) {

  // Check if the elements being allocated to this kernel are of the correct
  // type
  for (int ispec = 0; ispec < nelements; ispec++) {
    const int ielement = h_element_kernel_index_mapping(ispec);
    if ((assembly.properties.h_element_types(ielement) != MediumTag) &&
        (assembly.properties.h_element_property(ielement) != PropertyTag)) {
      throw std::runtime_error("Invalid element detected in kernel");
    }
  }

  Kokkos::deep_copy(element_kernel_index_mapping,
                    h_element_kernel_index_mapping);
  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    compute_mass_matrix(
        const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) const {
  constexpr int components = medium_type::components;
  using PointMassType = specfem::point::field<DimensionType, MediumTag, false,
                                              false, false, true>;

  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::compute_mass_matrix",
      specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              const specfem::point::index index(ispec_l, iz, ix);

              const auto point_property =
                  [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
                specfem::point::properties<MediumTag, PropertyTag>
                    point_property;

                specfem::compute::load_on_device(index, properties,
                                                 point_property);
                return point_property;
              }();

              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2<true> {
                specfem::point::partial_derivatives2<true>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);
                return point_partial_derivatives;
              }();

              PointMassType point_mass;

              element.compute_mass_matrix_component(point_property,
                                                    point_partial_derivatives,
                                                    point_mass.mass_matrix);

              for (int icomponent = 0; icomponent < components; icomponent++) {
                point_mass.mass_matrix[icomponent] *= wgll(ix) * wgll(iz);
              }

              specfem::compute::atomic_add_on_device(index, point_mass, field);
            });
      });

  Kokkos::fence();

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
template <specfem::enums::time_scheme::type time_scheme>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    mass_time_contribution(
        const type_real dt,
        const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) const {

  constexpr int components = medium_type::components;
  using PointMassType = specfem::point::field<DimensionType, MediumTag, false,
                                              false, false, true>;

  if (nelements == 0)
    return;

  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  Kokkos::parallel_for(
      "specfem::domain::kernes::elements::add_mass_matrix_contribution",
      specfem::kokkos::DeviceTeam(nelements, Kokkos::AUTO, 1),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        const auto point_boundary_type = boundary_conditions(ispec_l);

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              const specfem::point::index index(ispec_l, iz, ix);

              const auto point_property =
                  [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
                specfem::point::properties<MediumTag, PropertyTag>
                    point_property;

                specfem::compute::load_on_device(index, properties,
                                                 point_property);
                return point_property;
              }();

              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2<true> {
                specfem::point::partial_derivatives2<true>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);
                return point_partial_derivatives;
              }();

              PointMassType point_mass;

              specfem::kokkos::array_type<type_real, dimension::dim> weight(
                  wgll(ix), wgll(iz));

              element.template mass_time_contribution<time_scheme>(
                  xz, dt, weight, point_partial_derivatives, point_property,
                  point_boundary_type, point_mass.mass_matrix);

              specfem::compute::atomic_add_on_device(index, point_mass, field);
            });
      });

  Kokkos::fence();
  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel_base<
    WavefieldType, DimensionType, MediumTag, PropertyTag, BoundaryTag,
    quadrature_points_type>::
    compute_stiffness_interaction(
        const int istep,
        const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) const {

  constexpr int components = medium_type::components;
  // Number of quadrature points
  constexpr int NGLL = quadrature_points_type::NGLL;
  // Element field type - represents which fields to fetch from global field
  // struct
  using ElementFieldType = specfem::element::field<
      NGLL, DimensionType, MediumTag, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false, false, false>;
  // Data structure used to store the element field - A scract view type
  using ElementFieldViewType = typename ElementFieldType::ViewType;
  // Quadrature type - represents data structure used to store element
  // quadrature
  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, specfem::dimension::type::dim2, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, true>;
  // Data structure used to field at GLL point - represents which field to
  // atomically update
  using PointAccelerationType =
      specfem::point::field<DimensionType, MediumTag, false, false, true,
                            false>;
  using PointVelocityType = specfem::point::field<DimensionType, MediumTag,
                                                  false, true, false, false>;

  if (nelements == 0)
    return;

  const auto hprime = quadrature.gll.hprime;
  const auto wgll = quadrature.gll.weights;
  const auto index_mapping = points.index_mapping;

  int scratch_size = ElementFieldType::shmem_size() +
                     2 * ElementFieldViewType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_"
      "interaction",
      specfem::kokkos::DeviceTeam(nelements, NTHREADS, NLANES)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            element_kernel_index_mapping(team_member.league_rank());

        const auto point_boundary_type = boundary_conditions(ispec_l);

        // Instantiate shared views
        // ---------------------------------------------------------------
        ElementFieldType element_field(team_member);
        ElementQuadratureType element_quadrature(team_member);
        ElementFieldViewType s_stress_integrand_xi(team_member.team_scratch(0));
        ElementFieldViewType s_stress_integrand_gamma(
            team_member.team_scratch(0));

        // ---------- Allocate shared views -------------------------------
        specfem::compute::load_on_device(team_member, quadrature,
                                         element_quadrature);
        specfem::compute::load_on_device(team_member, ispec_l, field,
                                         element_field);
        // ---------------------------------------------------------------

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              specfem::kokkos::array_type<type_real, medium_type::components>
                  dudxl;
              specfem::kokkos::array_type<type_real, medium_type::components>
                  dudzl;

              const specfem::point::index index(ispec_l, iz, ix);

              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2<true> {
                specfem::point::partial_derivatives2<true>
                    point_partial_derivatives;
                specfem::compute::load_on_device(index, partial_derivatives,
                                                 point_partial_derivatives);
                return point_partial_derivatives;
              }();

              element.compute_gradient(
                  xz, element_quadrature.hprime_gll, element_field.displacement,
                  point_partial_derivatives, point_boundary_type, dudxl, dudzl);
              
              //TODO copy values by better accessing, and handle non-acoustic case
              if constexpr(MediumTag == specfem::element::medium_tag::acoustic){
                //storing dchi/dn * dS, the normal derivative of chi, times the linear jacobian

                type_real nx, nz, det; //to store outward facing normal dotted with field
                //note that column i of the transformation is ortho to row 1-i (0-indexing)
                //of the inverse -- whether or not we have the right sign,
                // you just have to take my word for it, or write out the transformations

                type_real dvdx, dvdz, dvdxi, dvdga;

                //v derivs: v = L_{ix}(x) L_{iz}(z)
                dvdxi = hprime(ix,ix);
                dvdga = hprime(iz,iz);
                //(the problem was here)
                // dchidx
                dvdx = dvdxi * point_partial_derivatives.xix +
                            dvdga * point_partial_derivatives.gammax;

                // dchidz
                dvdz = dvdxi * point_partial_derivatives.xiz +
                            dvdga * point_partial_derivatives.gammaz;

                if(ix == ngllx-1){ //edge 0: +x
                  //n ~ (dz/dgamma, -dx/dgamma) ~~ ortho to d/dgamma vector
                  nx = point_partial_derivatives.xix;
                  nz = point_partial_derivatives.xiz;

                  //this is inv transform, we want Jacobian of base transform
                  det =1/fabs(point_partial_derivatives.xix*point_partial_derivatives.gammaz - 
                              point_partial_derivatives.xiz*point_partial_derivatives.gammax);
                  field.edge_values_x(ispec_l,0,iz,0) = (dudxl[0]*nx + dudzl[0]*nz)*det;    // dX/dn dS
                  field.edge_values_x(ispec_l,0,iz,1) = element_field.displacement(iz,ix,0);// X
                  field.edge_values_x(ispec_l,0,iz,2) = sqrt(nx*nx + nz*nz)*det;            // dS
                  field.edge_values_x(ispec_l,0,iz,11)= (dvdx*nx + dvdz*nz)*det;            // dv/dn dS (test fcn)
                  field.edge_values_x(ispec_l,0,iz,12)= dvdx;
                  field.edge_values_x(ispec_l,0,iz,13)= dvdz;
                  field.edge_values_x(ispec_l,0,iz,14)= nx;
                  field.edge_values_x(ispec_l,0,iz,15)= nz;
                  field.edge_values_x(ispec_l,0,iz,16)= det;
                }else if(ix == 0){//edge 2: -x
                  //n ~ (-dz/dgamma, dx/dgamma) ~~ ortho to d/dgamma vector
                  nx = -point_partial_derivatives.xix;
                  nz = -point_partial_derivatives.xiz;
                  det =1/fabs(point_partial_derivatives.xix*point_partial_derivatives.gammaz - 
                              point_partial_derivatives.xiz*point_partial_derivatives.gammax);
                  field.edge_values_x(ispec_l,1,iz,0) = (dudxl[0]*nx + dudzl[0]*nz)*det;    // dX/dn dS
                  field.edge_values_x(ispec_l,1,iz,1) = element_field.displacement(iz,ix,0);// X
                  field.edge_values_x(ispec_l,1,iz,2) = sqrt(nx*nx + nz*nz)*det;            // dS
                  field.edge_values_x(ispec_l,1,iz,11)= (dvdx*nx + dvdz*nz)*det;            // dv/dn dS (test fcn)
                  field.edge_values_x(ispec_l,1,iz,12)= dvdx;
                  field.edge_values_x(ispec_l,1,iz,13)= dvdz;
                  field.edge_values_x(ispec_l,1,iz,14)= nx;
                  field.edge_values_x(ispec_l,1,iz,15)= nz;
                  field.edge_values_x(ispec_l,1,iz,16)= det;
                }
                if(iz == ngllz-1){ //edge 1: +z
                  //n ~ (dz/dxi, -dx/dxi) ~~ ortho to d/dxi vector
                  nx = point_partial_derivatives.gammax;
                  nz = point_partial_derivatives.gammaz;
                  det =1/fabs(point_partial_derivatives.xix*point_partial_derivatives.gammaz - 
                              point_partial_derivatives.xiz*point_partial_derivatives.gammax);
                  field.edge_values_z(ispec_l,0,ix,0) = (dudxl[0]*nx + dudzl[0]*nz)*det;    // dX/dn dS
                  field.edge_values_z(ispec_l,0,ix,1) = element_field.displacement(iz,ix,0);// X
                  field.edge_values_z(ispec_l,0,ix,2) = sqrt(nx*nx + nz*nz)*det;            // dS
                  field.edge_values_z(ispec_l,0,ix,11)= (dvdx*nx + dvdz*nz)*det;            // dv/dn dS (test fcn)
                  field.edge_values_z(ispec_l,0,ix,12)= dvdx;
                  field.edge_values_z(ispec_l,0,ix,13)= dvdz;
                  field.edge_values_z(ispec_l,0,ix,14)= nx;
                  field.edge_values_z(ispec_l,0,ix,15)= nz;
                  field.edge_values_z(ispec_l,0,ix,16)= det;
                }else if(iz == 0){//edge 3: -z
                  //n ~ (dz/dxi, -dx/dxi) ~~ ortho to d/dxi vector
                  nx = -point_partial_derivatives.gammax;
                  nz = -point_partial_derivatives.gammaz;
                  det =1/fabs(point_partial_derivatives.xix*point_partial_derivatives.gammaz - 
                              point_partial_derivatives.xiz*point_partial_derivatives.gammax);
                  field.edge_values_z(ispec_l,1,ix,0) = (dudxl[0]*nx + dudzl[0]*nz)*det;    // dX/dn dS
                  field.edge_values_z(ispec_l,1,ix,1) = element_field.displacement(iz,ix,0);// X
                  field.edge_values_z(ispec_l,1,ix,2) = sqrt(nx*nx + nz*nz)*det;            // dS
                  field.edge_values_z(ispec_l,1,ix,11)= (dvdx*nx + dvdz*nz)*det;            // dv/dn dS (test fcn)
                  field.edge_values_z(ispec_l,1,ix,12)= dvdx;
                  field.edge_values_z(ispec_l,1,ix,13)= dvdz;
                  field.edge_values_z(ispec_l,1,ix,14)= nx;
                  field.edge_values_z(ispec_l,1,ix,15)= nz;
                  field.edge_values_z(ispec_l,1,ix,16)= det;
                }
              }

              specfem::kokkos::array_type<type_real, medium_type::components>
                  stress_integrand_xi;
              specfem::kokkos::array_type<type_real, medium_type::components>
                  stress_integrand_gamma;

              const auto point_property =
                  [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
                specfem::point::properties<MediumTag, PropertyTag>
                    point_property;

                specfem::compute::load_on_device(index, properties,
                                                 point_property);
                return point_property;
              }();

              element.compute_stress(xz, dudxl, dudzl,
                                     point_partial_derivatives, point_property,
                                     point_boundary_type, stress_integrand_xi,
                                     stress_integrand_gamma);
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
              for (int icomponent = 0; icomponent < components; ++icomponent) {
                s_stress_integrand_xi(iz, ix, icomponent) =
                    stress_integrand_xi[icomponent];
                s_stress_integrand_gamma(iz, ix, icomponent) =
                    stress_integrand_gamma[icomponent];
              }
            });

        team_member.team_barrier();

        Kokkos::parallel_for(
            quadrature_points.template TeamThreadRange<specfem::enums::axes::x,
                                                       specfem::enums::axes::z>(
                team_member),
            [&, istep](const int xz) {
              int iz, ix;
              sub2ind(xz, ngllx, iz, ix);
              constexpr auto tag = boundary_conditions_type::value;

              const specfem::kokkos::array_type<type_real, dimension::dim>
                  weight(wgll(ix), wgll(iz));

              PointAccelerationType acceleration;

              // Get velocity, partial derivatives, and properties
              // only if needed by the boundary condition
              // ---------------------------------------------------------------
              constexpr bool load_boundary_variables =
                  ((tag == specfem::element::boundary_tag::stacey) ||
                   (tag == specfem::element::boundary_tag::
                               composite_stacey_dirichlet));

              constexpr bool store_boundary_values =
                  ((BoundaryTag == specfem::element::boundary_tag::stacey) &&
                   (WavefieldType == specfem::wavefield::type::forward));

              const specfem::point::index index(ispec_l, iz, ix);

              const auto velocity = [&]() -> PointVelocityType {
                if constexpr (load_boundary_variables) {
                  PointVelocityType velocity_l;
                  specfem::compute::load_on_device(index, field, velocity_l);
                  return velocity_l;
                } else {
                  return {};
                }
              }();

              const auto point_partial_derivatives =
                  [&]() -> specfem::point::partial_derivatives2<true> {
                if constexpr (load_boundary_variables) {
                  specfem::point::partial_derivatives2<true>
                      point_partial_derivatives;
                  specfem::compute::load_on_device(index, partial_derivatives,
                                                   point_partial_derivatives);
                  return point_partial_derivatives;

                } else {
                  return {};
                }
              }();

              const auto point_property =
                  [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
                if constexpr (load_boundary_variables) {
                  specfem::point::properties<MediumTag, PropertyTag>
                      point_property;
                  specfem::compute::load_on_device(index, properties,
                                                   point_property);
                  return point_property;
                } else {
                  return specfem::point::properties<MediumTag, PropertyTag>();
                }
              }();
              // ---------------------------------------------------------------

              element.compute_acceleration(
                  xz, weight, s_stress_integrand_xi, s_stress_integrand_gamma,
                  element_quadrature.hprimew_gll, point_partial_derivatives,
                  point_property, point_boundary_type, velocity.velocity,
                  acceleration.acceleration);

              if constexpr (store_boundary_values) {
                specfem::compute::store_on_device(istep, index, acceleration,
                                                  boundary_values);
              }

              specfem::compute::atomic_add_on_device(index, acceleration,
                                                     field);

              // #ifdef KOKKOS_ENABLE_CUDA
              // #pragma unroll
              // #endif
              //               for (int icomponent = 0; icomponent < components;
              //               ++icomponent) {
              //                 Kokkos::single(Kokkos::PerThread(team_member),
              //                 [&]() {
              //                   Kokkos::atomic_add(&field.field_dot_dot(iglob,
              //                   icomponent),
              //                                      acceleration[icomponent]);
              //                 });
              //               }
            });
      });

  Kokkos::fence();

  //TODO handle non-acoustic case
  if constexpr(MediumTag == specfem::element::medium_tag::acoustic){


  int ngllx, ngllz;
  quadrature_points.get_ngll(&ngllx, &ngllz);

    
  //surface_integral (rho_inv * tilde chi * dchi/dn) dS ~ rho_inv * gllweight * linear_jac * dchi/dn
  //central flux: average; note that outward facing normal means adj needs flipping
  const auto fluxcalc = [&](const specfem::point::index index,int e_ind,
  specfem::kokkos::array_type<type_real, _DISCONT_SIMFIELD_EDGE_COMPONENTS> thisval,
  specfem::kokkos::array_type<type_real, _DISCONT_SIMFIELD_EDGE_COMPONENTS> adjval,
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> edgefield, int ispec, int pmind//TODO remove
  ) -> type_real {
    const auto point_property =
        [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
        specfem::point::properties<MediumTag, PropertyTag>
            point_property;
        specfem::compute::load_on_device(index, properties,
                                          point_property);
        return point_property;
    }();
    //TODO: change FAC to be a per-nearby-element quantity (FAC = a = alpha * cmax/hmax @ Grote et al. 2006)
    //                                            cmax was c^2, but for our problem should be 1/rho
    // what we have here should be an actual maximum, but this will work for constant field
    constexpr type_real FAC = 20 * (point_property.rho_inverse) / 70.7;

    //TODO remove debug:
    edgefield(ispec,pmind,e_ind,3) = (thisval[0] - adjval[0])/2; //normalderiv avg
    edgefield(ispec,pmind,e_ind,4) = (thisval[1] - adjval[1]);   //field jump
    edgefield(ispec,pmind,e_ind,5) = wgll(e_ind) * point_property.rho_inverse
          * (thisval[0] - adjval[0])/2;
    edgefield(ispec,pmind,e_ind,6) = FAC * (thisval[1] - adjval[1]) * thisval[2] * wgll(e_ind);
    edgefield(ispec,pmind,e_ind,7) = point_property.rho_inverse;
    edgefield(ispec,pmind,e_ind,8) = (hprime(e_ind,e_ind));
    edgefield(ispec,pmind,e_ind,9) = wgll(e_ind);
    edgefield(ispec,pmind,e_ind,10) =wgll(e_ind) * point_property.rho_inverse * (thisval[1] - adjval[1]) * (thisval[11]/2);
    //end debug

    return 
      wgll(e_ind) * point_property.rho_inverse * (thisval[1] - adjval[1]) * (thisval[11]/2)
      + wgll(e_ind) * point_property.rho_inverse * (thisval[0] - adjval[0])/2
      - FAC * (thisval[1] - adjval[1]) * thisval[2] * wgll(e_ind);
  };

  //returns a subview of the point corresponding to the edge adjacent to edge (ispec,iedge)
  const auto adjview = [&](int ispec,int iedge,int e_ind)
      -> specfem::kokkos::array_type<type_real, _DISCONT_SIMFIELD_EDGE_COMPONENTS> {
    const int adj_spec = field.mesh_adjacency(ispec,iedge,0);
    int adj_edge = field.mesh_adjacency(ispec,iedge,1);
    const bool adj_flip = (adj_edge & 4) != 0;
    adj_edge = 3 & adj_edge;
    if(adj_edge % 2 == 0){// left/right
      return Kokkos::subview(field.edge_values_x,adj_spec,(adj_edge & 2) >> 1,
      adj_flip ? (ngllz-1-e_ind):e_ind,Kokkos::ALL);
    }else{// top/bottom
      return Kokkos::subview(field.edge_values_z,adj_spec,(adj_edge & 2) >> 1,
      adj_flip ? (ngllx-1-e_ind):e_ind,Kokkos::ALL);
    }
  };

  Kokkos::parallel_for(nelements,[&](const int ind) { //TODO teams/deviceteam policy
    const auto ispec_l = element_kernel_index_mapping(ind);
    const auto point_boundary_type = boundary_conditions(ispec_l);
    
    Kokkos::parallel_for(ngllz,[&](const int e_ind) { //left/right bdries
      PointAccelerationType acceleration;
      if (point_boundary_type.right==specfem::element::boundary_tag::none &&
            field.mesh_adjacency(ispec_l,0,0) != -1) {//right; +x
        const specfem::point::index index(ispec_l, e_ind, ngllx-1);
        acceleration.acceleration[0] = fluxcalc(index,e_ind,
          Kokkos::subview(field.edge_values_x,ispec_l,0,e_ind,Kokkos::ALL),
          adjview(ispec_l,0,e_ind) ,field.edge_values_x,ispec_l,0
        );
        specfem::compute::atomic_add_on_device(index, acceleration,
                                                field);
      }
      if (point_boundary_type.left==specfem::element::boundary_tag::none &&
            field.mesh_adjacency(ispec_l,2,0) != -1) {//left; -x
        const specfem::point::index index(ispec_l, e_ind, 0);
        acceleration.acceleration[0] = fluxcalc(index,e_ind,
          Kokkos::subview(field.edge_values_x,ispec_l,1,e_ind,Kokkos::ALL),
          adjview(ispec_l,2,e_ind) ,field.edge_values_x,ispec_l,1
        );
        specfem::compute::atomic_add_on_device(index, acceleration,
                                                field);
      }
    });
    Kokkos::parallel_for(ngllx,[&](const int e_ind) { //top/bottom bdries
      PointAccelerationType acceleration;
      if (point_boundary_type.top==specfem::element::boundary_tag::none &&
            field.mesh_adjacency(ispec_l,1,0) != -1) {//top; +z
        const specfem::point::index index(ispec_l, ngllz-1, e_ind);
        acceleration.acceleration[0] = fluxcalc(index,e_ind,
          Kokkos::subview(field.edge_values_z,ispec_l,0,e_ind,Kokkos::ALL),
          adjview(ispec_l,1,e_ind) ,field.edge_values_z,ispec_l,0
        );
        specfem::compute::atomic_add_on_device(index, acceleration,
                                                field);
      }
      if (point_boundary_type.bottom==specfem::element::boundary_tag::none &&
            field.mesh_adjacency(ispec_l,3,0) != -1) {//bottom; -z
        const specfem::point::index index(ispec_l, 0, e_ind);
        acceleration.acceleration[0] = fluxcalc(index,e_ind,
          Kokkos::subview(field.edge_values_z,ispec_l,1,e_ind,Kokkos::ALL),
          adjview(ispec_l,3,e_ind) ,field.edge_values_z,ispec_l,1
        );
        specfem::compute::atomic_add_on_device(index, acceleration,
                                                field);
      }
    });
    

  });
  Kokkos::fence();

  //TODO remove this ifblock below

  
  Kokkos::deep_copy(field.h_edge_values_x, field.edge_values_x);
  Kokkos::deep_copy(field.h_edge_values_z, field.edge_values_z);
  _util::dump_discont_simfield_per_step(istep,"tmp/debug_dump",
    field,points);
  // if(istep == 50){
  //   std::cout << "\nCopying edge values ("<<field.h_edge_values_x.extent(3)<<"), dumping, then exiting\n";
  //   Kokkos::deep_copy(field.h_edge_values_x, field.edge_values_x);
  //   Kokkos::deep_copy(field.h_edge_values_z, field.edge_values_z);
  //   _util::dump_discont_simfield("debug_dump.dat",
  //     field,points);
  //   std::exit(0);
  // }


  }


  return;
}

// template <specfem::wavefield::type WavefieldType,
//           specfem::dimension::type DimensionType,
//           specfem::element::medium_tag MediumTag,
//           specfem::element::property_tag PropertyTag,
//           specfem::element::boundary_tag BoundaryTag,
//           typename quadrature_points_type>
// template <specfem::enums::boundaries::type edge>
// void _edge_iter_helper(const int &ngllx, const int &ngllz,
//           const specfem::point::boundary &boundary_type){

//   if (
//     (edge == specfem::enums::boundaries::type::LEFT &&
//       boundary_type.left == specfem::enums::element::boundary_tag::none)
//   ||(edge == specfem::enums::boundaries::type::TOP &&
//       boundary_type.top == specfem::enums::element::boundary_tag::none)
//   ||(edge == specfem::enums::boundaries::type::RIGHT &&
//       boundary_type.right == specfem::enums::element::boundary_tag::none)
//   ||(edge == specfem::enums::boundaries::type::BOTTOM &&
//       boundary_type.bottom == specfem::enums::element::boundary_tag::none)
//   ) {
//     return;
//   }
//   const int edgelen = (edge==specfem::enums::boundaries::type::LEFT
//                     || edge==specfem::enums::boundaries::type::RIGHT)?
//               ngllz: ngllx;

//   Kokkos::parallel_for(edgelen,
//     [&, istep](const int xz) {
//       const int ix = (edge==specfem::enums::boundaries::type::LEFT
//                    || edge==specfem::enums::boundaries::type::RIGHT)?
//           ((edge==specfem::enums::boundaries::type::LEFT)? 0:(ngllx-1)):xz;
//       const int iz = (edge==specfem::enums::boundaries::type::LEFT
//                    || edge==specfem::enums::boundaries::type::RIGHT)?
//           xz:((edge==specfem::enums::boundaries::type::TOP)? (ngllz-1):0);
//       constexpr auto tag = boundary_conditions_type::value;

//       const specfem::kokkos::array_type<type_real, dimension::dim>
//           weight(wgll(ix), wgll(iz));

//       PointAccelerationType acceleration;

//       constexpr bool load_boundary_variables = true;

//       const specfem::point::index index(ispec_l, iz, ix);

//       const auto velocity = [&]() -> PointVelocityType {
//         if constexpr (load_boundary_variables) {
//           PointVelocityType velocity_l;
//           specfem::compute::load_on_device(index, field, velocity_l);
//           return velocity_l;
//         } else {
//           return {};
//         }
//       }();

//       const auto point_partial_derivatives =
//           [&]() -> specfem::point::partial_derivatives2<true> {
//         if constexpr (load_boundary_variables) {
//           specfem::point::partial_derivatives2<true>
//               point_partial_derivatives;
//           specfem::compute::load_on_device(index, partial_derivatives,
//                                           point_partial_derivatives);
//           return point_partial_derivatives;

//         } else {
//           return {};
//         }
//       }();

//       const auto point_property =
//           [&]() -> specfem::point::properties<MediumTag, PropertyTag> {
//         if constexpr (load_boundary_variables) {
//           specfem::point::properties<MediumTag, PropertyTag>
//               point_property;
//           specfem::compute::load_on_device(index, properties,
//                                           point_property);
//           return point_property;
//         } else {
//           return specfem::point::properties<MediumTag, PropertyTag>();
//         }
//       }();
//       // ---------------------------------------------------------------
//       // element.compute_acceleration(
//       //     xz, weight, s_stress_integrand_xi, s_stress_integrand_gamma,
//       //     element_quadrature.hprimew_gll, point_partial_derivatives,
//       //     point_property, point_boundary_type, velocity.velocity,
//       //     acceleration.acceleration);
//       // compute bdry accel

//       if constexpr (store_boundary_values) {
//         specfem::compute::store_on_device(istep, index, acceleration,
//                                           boundary_values);
//       }

//       specfem::compute::atomic_add_on_device(index, acceleration,
//                                             field);

//     }
//   );
  
// }

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumType,
          specfem::element::property_tag PropertyTag,
          typename quadrature_points_type>
void specfem::domain::impl::kernels::element_kernel<
    specfem::wavefield::type::backward, DimensionType, MediumType, PropertyTag,
    specfem::element::boundary_tag::stacey,
    quadrature_points_type>::compute_stiffness_interaction(const int istep)
    const {

  constexpr int components = medium_type::components;
  // Number of quadrature points
  using PointAccelerationType =
      specfem::point::field<DimensionType, MediumType, false, false, true,
                            false>;

  Kokkos::parallel_for(
      "specfem::domain::impl::kernels::elements::compute_stiffness_"
      "interaction",
      specfem::kokkos::DeviceTeam(this->nelements, NTHREADS, NLANES),
      KOKKOS_CLASS_LAMBDA(
          const specfem::kokkos::DeviceTeam::member_type &team_member) {
        int ngllx, ngllz;
        this->quadrature_points.get_ngll(&ngllx, &ngllz);
        const auto ispec_l =
            this->element_kernel_index_mapping(team_member.league_rank());

        Kokkos::parallel_for(
            this->quadrature_points.template TeamThreadRange<
                specfem::enums::axes::z, specfem::enums::axes::x>(team_member),
            [&](const int xz) {
              int ix, iz;
              sub2ind(xz, ngllx, iz, ix);

              const specfem::point::index index(ispec_l, iz, ix);

              PointAccelerationType acceleration;
              specfem::compute::load_on_device(
                  istep, index, this->boundary_values, acceleration);

              specfem::compute::atomic_add_on_device(index, acceleration,
                                                     field);
            });
      });
}

#endif // _DOMAIN_IMPL_ELEMENTS_KERNEL_TPP
