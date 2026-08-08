#include "extra_kernel.hpp"
#include "specfem/boundary_conditions.hpp"
#include "specfem/medium_physics.hpp"

template <int NGLL, typename Tags>
void specfem::element_coupling::TMP_extra_kernel::compute_coupling_extra_kernel<
    NGLL, Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
        (Tags::interface_tag ==
             specfem::element_coupling::interface_tag::acoustic_elastic ||
         Tags::interface_tag ==
             specfem::element_coupling::interface_tag::elastic_acoustic)>>::
    execute(const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr auto flux_scheme_tag =
      specfem::element_coupling::flux_scheme_tag::natural;
  constexpr static auto connection_tag =
      specfem::element_connections::type::nonconforming;
  constexpr static auto interface_tag = Tags::interface_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto wavefield_tag = Tags::wavefield_tag;
  const auto &interface_container =
      assembly.nonconforming_interfaces.template get_interface_container<
          interface_tag, boundary_tag, connection_tag, flux_scheme_tag>();

  // ==================================================================
  //                     alpha - self-merging
  // ==================================================================
  {
    const auto &flux_scheme_data = interface_container.flux_scheme_data;

    const int &num_faces = flux_scheme_data.self_faces.num_faces;
    if (num_faces == 0) {
      return;
    }
    const auto &ngll = flux_scheme_data.self_faces.face_view.n_points;

    constexpr auto self_medium =
        specfem::element_coupling::attributes<dimension_tag,
                                              interface_tag>::self_medium();
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
        assembly.fields.template get_simulation_field<wavefield_tag>();
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

                specfem::point::boundary<boundary_tag, dimension_tag, false>
                    point_boundary;
                specfem::assembly::load_on_device(index, assembly.boundaries,
                                                  point_boundary);
                if constexpr (boundary_tag == specfem::element::boundary_tag::
                                                  acoustic_free_surface) {
                  specfem::boundary_conditions::apply_boundary_conditions(
                      point_boundary, accel);
                }

                specfem::assembly::atomic_add_on_device(index, field, accel);
              });
        });
  }
}
