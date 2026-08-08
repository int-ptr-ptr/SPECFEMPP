#pragma once

#include "specfem/algorithms/gradient.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::element_coupling::TMP_extra_kernel {
// TEMPORARY: figure out a better way of finalizing this design
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::element_connections::type ConnectionTag,
          specfem::element_coupling::flux_scheme_tag FluxSchemeTag,
          typename = void>
struct compute_coupling_extra_kernel {
  template <int NGLL, specfem::simulation::field_type wavefield>
  static void
  execute(const specfem::assembly::assembly<DimensionTag> &assembly) {}
};

// template <specfem::element::dimension_tag DimensionTag,
//           specfem::element_coupling::interface_tag InterfaceTag,
//           specfem::element::boundary_tag BoundaryTag,
//           specfem::element_connections::type ConnectionTag,
//           specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
// struct compute_coupling_extra_kernel<
//     DimensionTag, InterfaceTag, BoundaryTag, ConnectionTag, FluxSchemeTag,

//     decltype(specfem::assembly::nonconforming_interfaces_impl::flux_scheme_data<
//              DimensionTag, InterfaceTag, BoundaryTag, ConnectionTag,
//              FluxSchemeTag>::
//                  compute_coupling_extra_kernel(
//                      specfem::assembly::assembly<DimensionTag>()))> {
//   compute_coupling_extra_kernel(
//       const specfem::assembly::assembly<DimensionTag> &assembly) {
//     assembly.nonconforming_interfaces
//         .template get_interface_container<InterfaceTag, BoundaryTag,
//                                           ConnectionTag, FluxSchemeTag>()
//         .compute_coupling_extra_kernel(assembly);
//   }
// };
template <specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element_connections::type ConnectionTag>
struct compute_coupling_extra_kernel<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    ConnectionTag, specfem::element_coupling::flux_scheme_tag::natural,
    std::enable_if_t<
        InterfaceTag ==
            specfem::element_coupling::interface_tag::acoustic_elastic ||
        InterfaceTag ==
            specfem::element_coupling::interface_tag::elastic_acoustic>> {

  template <int NGLL, specfem::simulation::field_type wavefield>
  static void execute(
      const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
          &assembly) {
    constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
    constexpr auto FluxSchemeTag =
        specfem::element_coupling::flux_scheme_tag::natural;
    const auto &interface_container =
        assembly.nonconforming_interfaces.template get_interface_container<
            InterfaceTag, BoundaryTag, ConnectionTag, FluxSchemeTag>();
    const auto &flux_scheme_data = interface_container.flux_scheme_data;

    const int &num_faces = flux_scheme_data.self_faces.num_faces;
    if (num_faces == 0) {
      return;
    }
    const auto &ngll = flux_scheme_data.self_faces.face_view.n_points;

    constexpr auto self_medium =
        specfem::element_coupling::attributes<dimension_tag,
                                              InterfaceTag>::self_medium();
    constexpr auto ncomp_self =
        specfem::element::attributes<dimension_tag, self_medium>::components;
    using ChunkFieldView =
        specfem::chunk_element::displacement<1, NGLL, dimension_tag,
                                             self_medium, false>;
    using PointTags = specfem::tags::Tags<
        dimension_tag, self_medium, specfem::element::property_tag::isotropic,
        specfem::element::attenuation_tag::constant_isotropic,
        false /*using_simd*/>;
    using PointFieldView = specfem::point::displacement<PointTags>;
    using PointAccelView = specfem::point::acceleration<PointTags>;
    using QuadratureType = specfem::quadrature::lagrange_derivative<
        NGLL, dimension_tag,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const auto field =
        assembly.fields.template get_simulation_field<wavefield>();
    // compute normals
    using team_handle = Kokkos::TeamPolicy<>::member_type;
    Kokkos::parallel_for(
        "compute_coupling_extra_kernel",
        Kokkos::TeamPolicy<>(num_faces, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(ChunkFieldView::shmem_size() +
                                                 QuadratureType::shmem_size())),
        KOKKOS_LAMBDA(const team_handle &team) {
          const int &iface = team.league_rank();
          const auto face = flux_scheme_data.self_faces.face_view(iface);

          ChunkFieldView element_field(team.team_scratch(0));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, NGLL * NGLL * NGLL),
              [&](const int &ijk) {
                const int k = ijk % NGLL;
                const int ij = ijk / NGLL;
                const int j = ij % NGLL;
                const int i = ij / NGLL;

                specfem::point::index<dimension_tag> index(face.element_index,
                                                           k, j, i);
                PointFieldView point_field;
                specfem::assembly::load_on_device(index, field, point_field);
                for (int icomp = 0; icomp < ncomp_self; icomp++) {
                  element_field(0, index.iz, index.iy, index.ix, icomp) =
                      point_field(icomp);
                }
              });

          QuadratureType lagrange_derivative(team);
          specfem::assembly::load_on_device(team, assembly.mesh,
                                            lagrange_derivative);

          team.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &ij) {
                const int i = ij / NGLL;
                const int j = ij % NGLL;
                const auto index = face(i, j);
                using datatype = typename ChunkFieldView::simd::datatype;

                datatype df_dxi[ncomp_self] = { 0.0 };
                datatype df_deta[ncomp_self] = { 0.0 };
                datatype df_dgamma[ncomp_self] = { 0.0 };
                specfem::point::jacobian_matrix<dimension_tag, false,
                                                false /*using_simd*/>
                    point_jacobian_matrix;
                specfem::assembly::load_on_device(
                    index, assembly.jacobian_matrix, point_jacobian_matrix);

                specfem::point::index<dimension_tag> regular_index(
                    0, index.iz, index.iy,
                    index.ix); // ispec should be zero right now, since
                               // element_gradient uses it as a chunk ielem
                               // index.
                const auto grad = specfem::algorithms::impl::element_gradient(
                    element_field, regular_index, point_jacobian_matrix,
                    lagrange_derivative, df_dxi, df_deta, df_dgamma);
                regular_index.ispec = index.ispec;

                specfem::point::properties<PointTags> point_property;
                specfem::assembly::load_on_device(
                    regular_index, assembly.properties, point_property);
                const auto stress =
                    specfem::medium_physics::compute_stress<PointTags>(
                        point_property, grad);

                PointAccelView accel;
                for (int icomp = 0; icomp < ncomp_self; icomp++) {
                  accel(icomp) = 0;
                  for (int idim = 0;
                       idim < specfem::element::dimension<dimension_tag>::dim;
                       idim++) {
                    // take T directly, since we do not need the local->global
                    // conversion that divergence takes.
                    accel(icomp) += flux_scheme_data.normal_times_weight(
                                        iface, i, j, idim) *
                                    stress.T(icomp, idim);
                  }
                }

                specfem::point::boundary<BoundaryTag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(index, assembly.boundaries,
                                                  point_boundary);
                if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                                 acoustic_free_surface) {
                  specfem::boundary_conditions::apply_boundary_conditions(
                      point_boundary, accel);
                }

                specfem::assembly::atomic_add_on_device(index, field, accel);
              });
        });
  }
};
} // namespace specfem::element_coupling::TMP_extra_kernel
