#pragma once

#include "specfem/assembly/element_intersections.hpp"

namespace specfem::assembly::nonconforming_interfaces_impl {

/**
 * @brief Stores faces from a FaceView (say, one side of an intersection), after
 * removing duplicates.
 */
struct unique_faces_container {
  using FaceViewType =
      specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;

  bool is_populated;
  FaceViewType face_view;
  FaceViewType::host_mirror_type h_face_view;
  int num_faces;

  unique_faces_container() : is_populated(false), num_faces(0) {}

  /**
   * @brief Generates the unique_faces_container from a given source, removing
   * duplicates.
   */
  unique_faces_container(const FaceViewType::host_mirror_type &faces_source);
};
} // namespace specfem::assembly::nonconforming_interfaces_impl
