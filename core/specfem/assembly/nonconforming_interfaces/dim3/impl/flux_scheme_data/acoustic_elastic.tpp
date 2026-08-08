#include "../unique_faces_container.hpp"
#include "flux_scheme_data.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/element/tags.hpp"

template <specfem::element::boundary_tag BoundaryTag,
          specfem::element_coupling::interface_tag InterfaceTag,
          specfem::element_connections::type ConnectionTag>
struct specfem::assembly::nonconforming_interfaces_impl::flux_scheme_data<
    specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag,
    ConnectionTag, specfem::element_coupling::flux_scheme_tag::natural,
    std::enable_if_t<
        InterfaceTag ==
            specfem::element_coupling::interface_tag::acoustic_elastic ||
        InterfaceTag ==
            specfem::element_coupling::interface_tag::elastic_acoustic>> {
  type_real self_neumann_merge_parameter;
  specfem::assembly::nonconforming_interfaces_impl::unique_faces_container
      self_faces;

  using FaceNormalView = typename specfem::data_access::Container<
      specfem::data_access::ContainerType::face,
      specfem::data_access::DataClassType::nonconforming_interface,
      specfem::element::dimension_tag::dim3>::
      tensor_type<type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for face normal vectors */
  FaceNormalView normal_times_weight;
  FaceNormalView::host_mirror_type h_normal_times_weight;

  flux_scheme_data() = default;

  flux_scheme_data(
      const specfem::assembly::element_intersections<
          specfem::element::dimension_tag::dim3> &element_intersections,
      const specfem::assembly::jacobian_matrix<
          specfem::element::dimension_tag::dim3> &jacobian_matrix,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::element_coupling::flux_scheme_configuration
          &flux_scheme_config) {

    self_neumann_merge_parameter =
        flux_scheme_config.has_scheme_parameter("self_neumann_merge_parameter")
            ? flux_scheme_config.get_scheme_parameter(
                  "self_neumann_merge_parameter")
            : 0;
    if (std::abs(self_neumann_merge_parameter - 0) < 1e-4) {
      self_neumann_merge_parameter = 0;
      return;
    }

    constexpr auto FluxSchemeTag =
        specfem::element_coupling::flux_scheme_tag::natural;
    const auto intersections_self =
        std::get<0>(element_intersections.get_intersections_on_host(
            specfem::element_connections::type::nonconforming, InterfaceTag,
            BoundaryTag, FluxSchemeTag));

    // we need to store these values
    this->self_faces = specfem::assembly::nonconforming_interfaces_impl::
        unique_faces_container(intersections_self);
    const auto &num_faces = self_faces.num_faces;
    const auto &ngll = self_faces.face_view.n_points;

    normal_times_weight = FaceNormalView(
        "specfem::assembly::nonconforming_interfaces::normal_times_weight",
        num_faces, ngll, ngll,
        specfem::element::dimension<
            specfem::element::dimension_tag::dim3>::dim);
    h_normal_times_weight = Kokkos::create_mirror_view(normal_times_weight);

    for (int iface = 0; iface < num_faces; ++iface) {
      const auto &self_face = self_faces.h_face_view(iface);
      const int &ispec = self_face.element_index;
      const auto &iface_type = self_face.face_type;

      for (int ipoint_i = 0; ipoint_i < ngll; ipoint_i++) {
        for (int ipoint_j = 0; ipoint_j < ngll; ipoint_j++) {
          const auto self_face_pt = self_face(ipoint_i, ipoint_j);
          const int iz = self_face_pt.iz;
          const int iy = self_face_pt.iy;
          const int ix = self_face_pt.ix;

          specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                          true, false>
              point_jacobian_matrix;
          specfem::point::index<specfem::element::dimension_tag::dim3, false>
              point_index{ ispec, iz, iy, ix };
          specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                          point_jacobian_matrix);

          const auto dn = point_jacobian_matrix.compute_normal(iface_type);
          const type_real face_factor = [&]() {
            switch (iface_type) {
            case specfem::mesh_entity::dim3::type::left:
            case specfem::mesh_entity::dim3::type::right:
              // Face in (iy, iz) plane; integrate over iy and iz
              return mesh.h_weights(iy) * mesh.h_weights(iz);
            case specfem::mesh_entity::dim3::type::bottom:
            case specfem::mesh_entity::dim3::type::top:
              // Face in (ix, iy) plane; integrate over ix and iy
              return mesh.h_weights(ix) * mesh.h_weights(iy);
            case specfem::mesh_entity::dim3::type::front:
            case specfem::mesh_entity::dim3::type::back:
              // Face in (ix, iz) plane; integrate over ix and iz
              return mesh.h_weights(ix) * mesh.h_weights(iz);
            default:
              KOKKOS_ABORT_WITH_LOCATION("Invalid face type");
              return static_cast<type_real>(0.0);
            }
          }() * self_neumann_merge_parameter;
          this->h_normal_times_weight(iface, ipoint_i, ipoint_j, 0) =
              dn(0) * face_factor;
          this->h_normal_times_weight(iface, ipoint_i, ipoint_j, 1) =
              dn(1) * face_factor;
          this->h_normal_times_weight(iface, ipoint_i, ipoint_j, 2) =
              dn(2) * face_factor;
        }
      }
    }
    Kokkos::deep_copy(normal_times_weight, h_normal_times_weight);
  }
};
