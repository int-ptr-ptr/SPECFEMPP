#include "unique_faces_container.hpp"

specfem::assembly::nonconforming_interfaces_impl::unique_faces_container::
    unique_faces_container(const FaceViewType::host_mirror_type &faces_source)
    : is_populated(true) {
  const int &ngll = faces_source.n_points;

  std::vector<specfem::mesh_entity::face<specfem::element::dimension_tag::dim3>>
      collected_faces;
  std::map<std::pair<int, specfem::mesh_entity::dim3::type>, int> face_indices;
  int num_self_faces = 0;

  // collect unique faces
  for (int iface = 0; iface < faces_source.N; ++iface) {
    const auto key = std::make_pair<int, specfem::mesh_entity::dim3::type>(
        faces_source(iface).element_index, faces_source(iface).face_type);

    if (face_indices.find(key) == face_indices.end()) {
      // face_indices does not yet have this (ispec, self_face_type)
      face_indices[key] = num_self_faces;
      collected_faces.push_back(
          { key.first, key.second, num_self_faces, false });
      num_self_faces++;
    }
  }

  // construct views from collected faces in both host and device space

  face_view = FaceViewType("specfem::assembly::nonconforming_interfaces_impl::"
                           "unique_faces_container::face_view",
                           num_self_faces, ngll);
  h_face_view = specfem::assembly::face_view_from_collected_faces(
      "specfem::assembly::nonconforming_interfaces_impl::"
      "unique_faces_container::h_face_view",
      collected_faces,
      specfem::mesh_entity::element<specfem::element::dimension_tag::dim3>(
          ngll, ngll, ngll));
  specfem::assembly::deep_copy(face_view, h_face_view);
  num_faces = num_self_faces;
}
