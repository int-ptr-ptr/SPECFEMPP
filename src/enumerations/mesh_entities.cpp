#include "enumerations/mesh_entities.hpp"
#include <list>
#include <stdexcept>

std::list<specfem::mesh_entity::dim2::type>
specfem::mesh_entity::dim2::edges_of_corner(
    const specfem::mesh_entity::dim2::type &corner) {
  switch (corner) {
  case specfem::mesh_entity::dim2::type::top_left:
    return { specfem::mesh_entity::dim2::type::top,
             specfem::mesh_entity::dim2::type::left };
  case specfem::mesh_entity::dim2::type::top_right:
    return { specfem::mesh_entity::dim2::type::top,
             specfem::mesh_entity::dim2::type::right };
  case specfem::mesh_entity::dim2::type::bottom_right:
    return { specfem::mesh_entity::dim2::type::bottom,
             specfem::mesh_entity::dim2::type::right };
  case specfem::mesh_entity::dim2::type::bottom_left:
    return { specfem::mesh_entity::dim2::type::bottom,
             specfem::mesh_entity::dim2::type::left };
  default:
    throw std::runtime_error("Invalid corner type");
  }
}

const std::string specfem::mesh_entity::dim2::to_string(
    const specfem::mesh_entity::dim2::type &entity) {
  switch (entity) {
  case specfem::mesh_entity::dim2::type::bottom:
    return "bottom";
  case specfem::mesh_entity::dim2::type::right:
    return "right";
  case specfem::mesh_entity::dim2::type::top:
    return "top";
  case specfem::mesh_entity::dim2::type::left:
    return "left";
  case specfem::mesh_entity::dim2::type::bottom_left:
    return "bottom_left";
  case specfem::mesh_entity::dim2::type::bottom_right:
    return "bottom_right";
  case specfem::mesh_entity::dim2::type::top_right:
    return "top_right";
  case specfem::mesh_entity::dim2::type::top_left:
    return "top_left";
  default:
    throw std::runtime_error(
        std::string("specfem::mesh_entity::dim2::to_string does not handle ") +
        std::to_string(static_cast<int>(entity)));
    return "!ERR";
  }
}

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
