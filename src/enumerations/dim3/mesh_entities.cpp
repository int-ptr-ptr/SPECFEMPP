#include "enumerations/mesh_entities.hpp"
#include <list>
#include <stdexcept>

const std::list<specfem::mesh_entity::dim3::type>
specfem::mesh_entity::dim3::faces_of_edge(
    const specfem::mesh_entity::dim3::type &edge) {
  switch (edge) {
  case specfem::mesh_entity::dim3::type::bottom_left:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::bottom_right:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::top_right:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::top_left:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::front_bottom:
    return { specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::bottom };
  case specfem::mesh_entity::dim3::type::front_top:
    return { specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::top };
  case specfem::mesh_entity::dim3::type::front_left:
    return { specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::front_right:
    return { specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::back_bottom:
    return { specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::bottom };
  case specfem::mesh_entity::dim3::type::back_top:
    return { specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::top };
  case specfem::mesh_entity::dim3::type::back_left:
    return { specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::back_right:
    return { specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::right };
  default:
    throw std::runtime_error("Invalid edge type");
  }
}

const std::list<specfem::mesh_entity::dim3::type>
specfem::mesh_entity::dim3::edges_of_corner(
    const specfem::mesh_entity::dim3::type &corner) {
  switch (corner) {
  case specfem::mesh_entity::dim3::type::bottom_front_left:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::front_left,
             specfem::mesh_entity::dim3::type::front_bottom };
  case specfem::mesh_entity::dim3::type::bottom_front_right:
    return { specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::front_bottom };
  case specfem::mesh_entity::dim3::type::bottom_back_left:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::back_left,
             specfem::mesh_entity::dim3::type::back_bottom };
  case specfem::mesh_entity::dim3::type::bottom_back_right:
    return { specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::back_right,
             specfem::mesh_entity::dim3::type::back_bottom };
  case specfem::mesh_entity::dim3::type::top_front_left:
    return { specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::front_left,
             specfem::mesh_entity::dim3::type::front_top };
  case specfem::mesh_entity::dim3::type::top_front_right:
    return { specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::front_top };
  case specfem::mesh_entity::dim3::type::top_back_left:
    return { specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::back_left,
             specfem::mesh_entity::dim3::type::back_top };
  case specfem::mesh_entity::dim3::type::top_back_right:
    return { specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::back_right,
             specfem::mesh_entity::dim3::type::back_top };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

const std::list<specfem::mesh_entity::dim3::type>
specfem::mesh_entity::dim3::corners_of_face(
    const specfem::mesh_entity::dim3::type &face) {
  switch (face) {
  case specfem::mesh_entity::dim3::type::bottom:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::bottom_back_right,
             specfem::mesh_entity::dim3::type::bottom_back_left };
  case specfem::mesh_entity::dim3::type::right:
    return { specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::bottom_back_right };
  case specfem::mesh_entity::dim3::type::top:
    return { specfem::mesh_entity::dim3::type::top_front_left,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::top_back_left };
  case specfem::mesh_entity::dim3::type::left:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::top_front_left,
             specfem::mesh_entity::dim3::type::top_back_left,
             specfem::mesh_entity::dim3::type::bottom_back_left };
  case specfem::mesh_entity::dim3::type::front:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_front_left };
  case specfem::mesh_entity::dim3::type::back:
    return { specfem::mesh_entity::dim3::type::bottom_back_left,
             specfem::mesh_entity::dim3::type::bottom_back_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::top_back_left };
  default:
    throw std::runtime_error("Invalid face type");
  }
}

const std::string specfem::mesh_entity::dim3::to_string(
    const specfem::mesh_entity::dim3::type &entity) {
  switch (entity) {
  case specfem::mesh_entity::dim3::type::bottom:
    return "bottom";
  case specfem::mesh_entity::dim3::type::right:
    return "right";
  case specfem::mesh_entity::dim3::type::top:
    return "top";
  case specfem::mesh_entity::dim3::type::left:
    return "left";
  case specfem::mesh_entity::dim3::type::front:
    return "front";
  case specfem::mesh_entity::dim3::type::back:
    return "back";
  case specfem::mesh_entity::dim3::type::bottom_left:
    return "bottom_left";
  case specfem::mesh_entity::dim3::type::bottom_right:
    return "bottom_right";
  case specfem::mesh_entity::dim3::type::top_right:
    return "top_right";
  case specfem::mesh_entity::dim3::type::top_left:
    return "top_left";
  case specfem::mesh_entity::dim3::type::front_bottom:
    return "front_bottom";
  case specfem::mesh_entity::dim3::type::front_top:
    return "front_top";
  case specfem::mesh_entity::dim3::type::front_left:
    return "front_left";
  case specfem::mesh_entity::dim3::type::front_right:
    return "front_right";
  case specfem::mesh_entity::dim3::type::back_bottom:
    return "back_bottom";
  case specfem::mesh_entity::dim3::type::back_top:
    return "back_top";
  case specfem::mesh_entity::dim3::type::back_left:
    return "back_left";
  case specfem::mesh_entity::dim3::type::back_right:
    return "back_right";
  case specfem::mesh_entity::dim3::type::bottom_front_left:
    return "bottom_front_left";
  case specfem::mesh_entity::dim3::type::bottom_front_right:
    return "bottom_front_right";
  case specfem::mesh_entity::dim3::type::bottom_back_left:
    return "bottom_back_left";
  case specfem::mesh_entity::dim3::type::bottom_back_right:
    return "bottom_back_right";
  case specfem::mesh_entity::dim3::type::top_front_left:
    return "top_front_left";
  case specfem::mesh_entity::dim3::type::top_front_right:
    return "top_front_right";
  case specfem::mesh_entity::dim3::type::top_back_left:
    return "top_back_left";
  case specfem::mesh_entity::dim3::type::top_back_right:
    return "top_back_right";
  default:
    throw std::runtime_error(
        std::string("specfem::mesh_entity::dim3::to_string does not handle ") +
        std::to_string(static_cast<int>(entity)));
    return "!ERR";
  }
}

specfem::mesh_entity::element<specfem::dimension::type::dim3>::element(
    const int ngllz, const int nglly, const int ngllx)
    : ngllz(ngllz), nglly(nglly), ngllx(ngllx), orderz(ngllz - 1),
      ordery(nglly - 1), orderx(ngllx - 1), size(ngllz * nglly * ngllx),
      ngll2d(ngllx * nglly), ngll(ngllx) {

  if (ngllz < 2 || nglly < 2 || ngllx < 2) {
    throw std::runtime_error(
        "ngllz, nglly, and ngllx must be at least 2 to define a 3D element");
  }

  if (ngllz != nglly || ngllz != ngllx) {
    throw std::runtime_error(
        "ngllz, nglly, and ngllx must be equal for a cubic 3D element");
  }

  // Initialize edge coordinates
  // xmin = left, xmax = right
  // zmin = bottom, zmax = top
  // ymin = front, ymax = back

  face_coordinates[specfem::mesh_entity::dim3::type::bottom] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = 0;
        const int iy = ipoint;
        const int ix = jpoint;
        return std::make_tuple(iz, iy, ix);
      };

  face_coordinates[specfem::mesh_entity::dim3::type::top] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = this->ngllz - 1;
        const int iy = ipoint;
        const int ix = jpoint;
        return std::make_tuple(iz, iy, ix);
      };

  face_coordinates[specfem::mesh_entity::dim3::type::front] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = ipoint;
        const int iy = 0;
        const int ix = jpoint;
        return std::make_tuple(iz, iy, ix);
      };

  face_coordinates[specfem::mesh_entity::dim3::type::back] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = ipoint;
        const int iy = this->nglly - 1;
        const int ix = jpoint;
        return std::make_tuple(iz, iy, ix);
      };

  face_coordinates[specfem::mesh_entity::dim3::type::left] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = ipoint;
        const int iy = jpoint;
        const int ix = 0;
        return std::make_tuple(iz, iy, ix);
      };

  face_coordinates[specfem::mesh_entity::dim3::type::right] =
      [*this](const int ipoint, const int jpoint) {
        const int iz = ipoint;
        const int iy = jpoint;
        const int ix = this->ngllx - 1;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::front_bottom] =
      [*this](const int point) {
        const int iz = 0;
        const int iy = 0;
        const int ix = point;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::back_bottom] =
      [*this](const int point) {
        const int iz = 0;
        const int iy = this->nglly - 1;
        const int ix = point;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::front_top] =
      [*this](const int point) {
        const int iz = this->ngllz - 1;
        const int iy = 0;
        const int ix = point;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::back_top] =
      [*this](const int point) {
        const int iz = this->ngllz - 1;
        const int iy = this->nglly - 1;
        const int ix = point;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::bottom_left] =
      [*this](const int point) {
        const int iz = 0;
        const int iy = point;
        const int ix = 0;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::bottom_right] =
      [*this](const int point) {
        const int iz = 0;
        const int iy = point;
        const int ix = this->ngllx - 1;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::top_left] =
      [*this](const int point) {
        const int iz = this->ngllz - 1;
        const int iy = point;
        const int ix = 0;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::top_right] =
      [*this](const int point) {
        const int iz = this->ngllz - 1;
        const int iy = point;
        const int ix = this->ngllx - 1;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::front_left] =
      [*this](const int point) {
        const int iz = point;
        const int iy = 0;
        const int ix = 0;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::front_right] =
      [*this](const int point) {
        const int iz = point;
        const int iy = 0;
        const int ix = this->ngllx - 1;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::back_left] =
      [*this](const int point) {
        const int iz = point;
        const int iy = this->nglly - 1;
        const int ix = 0;
        return std::make_tuple(iz, iy, ix);
      };

  edge_coordinates[specfem::mesh_entity::dim3::type::back_right] =
      [*this](const int point) {
        const int iz = point;
        const int iy = this->nglly - 1;
        const int ix = this->ngllx - 1;
        return std::make_tuple(iz, iy, ix);
      };

  // Initialize corner coordinates
  corner_coordinates[specfem::mesh_entity::dim3::type::bottom_front_left] =
      std::make_tuple(0, 0, 0);
  corner_coordinates[specfem::mesh_entity::dim3::type::bottom_front_right] =
      std::make_tuple(0, 0, this->ngllx - 1);
  corner_coordinates[specfem::mesh_entity::dim3::type::bottom_back_left] =
      std::make_tuple(0, this->nglly - 1, 0);
  corner_coordinates[specfem::mesh_entity::dim3::type::bottom_back_right] =
      std::make_tuple(0, this->nglly - 1, this->ngllx - 1);
  corner_coordinates[specfem::mesh_entity::dim3::type::top_front_left] =
      std::make_tuple(this->ngllz - 1, 0, 0);
  corner_coordinates[specfem::mesh_entity::dim3::type::top_front_right] =
      std::make_tuple(this->ngllz - 1, 0, this->ngllx - 1);
  corner_coordinates[specfem::mesh_entity::dim3::type::top_back_left] =
      std::make_tuple(this->ngllz - 1, this->nglly - 1, 0);
  corner_coordinates[specfem::mesh_entity::dim3::type::top_back_right] =
      std::make_tuple(this->ngllz - 1, this->nglly - 1, this->ngllx - 1);
}

int specfem::mesh_entity::element<specfem::dimension::type::dim3>::
    number_of_points_on_orientation(
        const specfem::mesh_entity::dim3::type &entity) const {

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces, entity))
    return ngll2d;

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges, entity))
    return ngll;

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                     entity))
    return 1;

  throw std::runtime_error("Invalid entity type");
}

std::tuple<int, int, int>
specfem::mesh_entity::element<specfem::dimension::type::dim3>::map_coordinates(
    const specfem::mesh_entity::dim3::type &entity, const int point) const {

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                     entity)) {
    throw std::runtime_error("Corner mapping requires no point index");
  }

  if (point < 0 || point >= this->number_of_points_on_orientation(entity)) {
    throw std::runtime_error("Point index out of bounds");
  }

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges, entity))
    return edge_coordinates.at(entity)(point);

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                     entity)) {
    const int ipoint = point % ngll;
    const int jpoint = point / ngll;
    return face_coordinates.at(entity)(ipoint, jpoint);
  }

  throw std::runtime_error("Unknown entity type");
}

std::tuple<int, int, int>
specfem::mesh_entity::element<specfem::dimension::type::dim3>::map_coordinates(
    const specfem::mesh_entity::dim3::type &corner) const {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                      corner)) {
    throw std::runtime_error("The argument is not a corner");
  }

  return corner_coordinates.at(corner);
}

/**
 *
 *    7----6
 *   /|   /|
 *  4----5 |
 *  | 3--|-2
 *  |/   |/
 *  0----1
 */
std::vector<int> specfem::mesh_entity::nodes_on_orientation(
    const specfem::mesh_entity::dim3::type &entity) {

  switch (entity) {
  // Faces
  case specfem::mesh_entity::dim3::type::left:
    return { 0, 4, 7, 3 };
  case specfem::mesh_entity::dim3::type::right:
    return { 1, 2, 6, 5 };
  case specfem::mesh_entity::dim3::type::front:
    return { 0, 1, 5, 4 };
  case specfem::mesh_entity::dim3::type::back:
    return { 3, 7, 6, 2 };
  case specfem::mesh_entity::dim3::type::bottom:
    return { 0, 3, 2, 1 };
  case specfem::mesh_entity::dim3::type::top:
    return { 4, 5, 6, 7 };
  // Edges
  case specfem::mesh_entity::dim3::type::bottom_left:
    return { 0, 3 };
  case specfem::mesh_entity::dim3::type::bottom_right:
    return { 1, 2 };
  case specfem::mesh_entity::dim3::type::top_left:
    return { 4, 7 };
  case specfem::mesh_entity::dim3::type::top_right:
    return { 5, 6 };
  case specfem::mesh_entity::dim3::type::front_bottom:
    return { 0, 1 };
  case specfem::mesh_entity::dim3::type::front_top:
    return { 4, 5 };
  case specfem::mesh_entity::dim3::type::front_left:
    return { 0, 4 };
  case specfem::mesh_entity::dim3::type::front_right:
    return { 1, 5 };
  case specfem::mesh_entity::dim3::type::back_bottom:
    return { 3, 2 };
  case specfem::mesh_entity::dim3::type::back_top:
    return { 7, 6 };
  case specfem::mesh_entity::dim3::type::back_left:
    return { 3, 7 };
  case specfem::mesh_entity::dim3::type::back_right:
    return { 2, 6 };
  // Corners
  case specfem::mesh_entity::dim3::type::bottom_front_left:
    return { 0 };
  case specfem::mesh_entity::dim3::type::bottom_front_right:
    return { 1 };
  case specfem::mesh_entity::dim3::type::bottom_back_left:
    return { 3 };
  case specfem::mesh_entity::dim3::type::bottom_back_right:
    return { 2 };
  case specfem::mesh_entity::dim3::type::top_front_left:
    return { 4 };
  case specfem::mesh_entity::dim3::type::top_front_right:
    return { 5 };
  case specfem::mesh_entity::dim3::type::top_back_left:
    return { 7 };
  case specfem::mesh_entity::dim3::type::top_back_right:
    return { 6 };
  default:
    throw std::runtime_error("The provided entity is not a valid mesh entity");
  }
}

std::array<specfem::mesh_entity::dim3::type, 4>
specfem::mesh_entity::edges_of_face(
    const specfem::mesh_entity::dim3::type &face) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      face)) {
    throw std::runtime_error("The argument is not a face");
  }

  switch (face) {
  case specfem::mesh_entity::dim3::type::bottom:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::back_bottom,
             specfem::mesh_entity::dim3::type::front_bottom };
  case specfem::mesh_entity::dim3::type::right:
    return { specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::back_right };
  case specfem::mesh_entity::dim3::type::top:
    return { specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::front_top,
             specfem::mesh_entity::dim3::type::back_top };
  case specfem::mesh_entity::dim3::type::left:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::back_left,
             specfem::mesh_entity::dim3::type::front_left };
  case specfem::mesh_entity::dim3::type::front:
    return { specfem::mesh_entity::dim3::type::front_bottom,
             specfem::mesh_entity::dim3::type::front_top,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::front_left };
  case specfem::mesh_entity::dim3::type::back:
    return { specfem::mesh_entity::dim3::type::back_bottom,
             specfem::mesh_entity::dim3::type::back_top,
             specfem::mesh_entity::dim3::type::back_right,
             specfem::mesh_entity::dim3::type::back_left };
  default:
    throw std::runtime_error("Invalid face type");
  }
}

std::array<specfem::mesh_entity::dim3::type, 4>
specfem::mesh_entity::corners_of_face(
    const specfem::mesh_entity::dim3::type &face) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      face)) {
    throw std::runtime_error("The argument is not a face");
  }

  switch (face) {
  case specfem::mesh_entity::dim3::type::bottom:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::bottom_back_right,
             specfem::mesh_entity::dim3::type::bottom_back_left };
  case specfem::mesh_entity::dim3::type::right:
    return { specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::bottom_back_right };
  case specfem::mesh_entity::dim3::type::top:
    return { specfem::mesh_entity::dim3::type::top_front_left,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::top_back_left };
  case specfem::mesh_entity::dim3::type::left:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::top_front_left,
             specfem::mesh_entity::dim3::type::top_back_left,
             specfem::mesh_entity::dim3::type::bottom_back_left };
  case specfem::mesh_entity::dim3::type::front:
    return { specfem::mesh_entity::dim3::type::bottom_front_left,
             specfem::mesh_entity::dim3::type::bottom_front_right,
             specfem::mesh_entity::dim3::type::top_front_right,
             specfem::mesh_entity::dim3::type::top_front_left };
  case specfem::mesh_entity::dim3::type::back:
    return { specfem::mesh_entity::dim3::type::bottom_back_left,
             specfem::mesh_entity::dim3::type::bottom_back_right,
             specfem::mesh_entity::dim3::type::top_back_right,
             specfem::mesh_entity::dim3::type::top_back_left };
  default:
    throw std::runtime_error("Invalid face type");
  }
}

std::list<specfem::mesh_entity::dim3::type>
specfem::mesh_entity::faces_of_corner(
    const specfem::mesh_entity::dim3::type &corner) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                      corner)) {
    throw std::runtime_error("The argument is not a corner");
  }

  switch (corner) {
  case specfem::mesh_entity::dim3::type::bottom_front_left:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::bottom_front_right:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::bottom_back_left:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::bottom_back_right:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::top_front_left:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::top_front_right:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::front,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::top_back_left:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::top_back_right:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::back,
             specfem::mesh_entity::dim3::type::right };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

std::list<specfem::mesh_entity::dim3::type>
specfem::mesh_entity::edges_of_corner(
    const specfem::mesh_entity::dim3::type &corner) {
  switch (corner) {
  case specfem::mesh_entity::dim3::type::bottom_front_left:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::front_left,
             specfem::mesh_entity::dim3::type::front_bottom };
  case specfem::mesh_entity::dim3::type::bottom_front_right:
    return { specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::front_bottom };
  case specfem::mesh_entity::dim3::type::bottom_back_left:
    return { specfem::mesh_entity::dim3::type::bottom_left,
             specfem::mesh_entity::dim3::type::back_left,
             specfem::mesh_entity::dim3::type::back_bottom };
  case specfem::mesh_entity::dim3::type::bottom_back_right:
    return { specfem::mesh_entity::dim3::type::bottom_right,
             specfem::mesh_entity::dim3::type::back_right,
             specfem::mesh_entity::dim3::type::back_bottom };
  case specfem::mesh_entity::dim3::type::top_front_left:
    return { specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::front_left,
             specfem::mesh_entity::dim3::type::front_top };
  case specfem::mesh_entity::dim3::type::top_front_right:
    return { specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::front_right,
             specfem::mesh_entity::dim3::type::front_top };
  case specfem::mesh_entity::dim3::type::top_back_left:
    return { specfem::mesh_entity::dim3::type::top_left,
             specfem::mesh_entity::dim3::type::back_left,
             specfem::mesh_entity::dim3::type::back_top };
  case specfem::mesh_entity::dim3::type::top_back_right:
    return { specfem::mesh_entity::dim3::type::top_right,
             specfem::mesh_entity::dim3::type::back_right,
             specfem::mesh_entity::dim3::type::back_top };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

std::list<specfem::mesh_entity::dim3::type> specfem::mesh_entity::faces_of_edge(
    const specfem::mesh_entity::dim3::type &edge) {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges,
                                      edge)) {
    throw std::runtime_error("The argument is not an edge");
  }

  switch (edge) {
  case specfem::mesh_entity::dim3::type::bottom_left:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::bottom_right:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::top_left:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::left };
  case specfem::mesh_entity::dim3::type::top_right:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::right };
  case specfem::mesh_entity::dim3::type::front_bottom:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::front };
  case specfem::mesh_entity::dim3::type::front_top:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::front };
  case specfem::mesh_entity::dim3::type::front_left:
    return { specfem::mesh_entity::dim3::type::left,
             specfem::mesh_entity::dim3::type::front };
  case specfem::mesh_entity::dim3::type::front_right:
    return { specfem::mesh_entity::dim3::type::right,
             specfem::mesh_entity::dim3::type::front };
  case specfem::mesh_entity::dim3::type::back_bottom:
    return { specfem::mesh_entity::dim3::type::bottom,
             specfem::mesh_entity::dim3::type::back };
  case specfem::mesh_entity::dim3::type::back_top:
    return { specfem::mesh_entity::dim3::type::top,
             specfem::mesh_entity::dim3::type::back };
  case specfem::mesh_entity::dim3::type::back_left:
    return { specfem::mesh_entity::dim3::type::left,
             specfem::mesh_entity::dim3::type::back };
  case specfem::mesh_entity::dim3::type::back_right:
    return { specfem::mesh_entity::dim3::type::right,
             specfem::mesh_entity::dim3::type::back };
  default:
    throw std::runtime_error("Invalid edge type");
  }
}
