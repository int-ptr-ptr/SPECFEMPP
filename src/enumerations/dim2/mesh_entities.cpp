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
