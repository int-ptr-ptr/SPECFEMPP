#pragma once
#include "parameter_parser/mesh_modifiers.hpp"
#include "enumerations/interface_resolution.hpp"

// #include <fstream>
// #include <iostream>
#include <ostream>

template <specfem::dimension::type DimensionType>
std::shared_ptr<specfem::mesh::modifiers<DimensionType> >
specfem::runtime_configuration::mesh_modifiers::instantiate_mesh_modifiers() {
  std::shared_ptr<specfem::mesh::modifiers<DimensionType> > modifiers =
      std::make_shared<specfem::mesh::modifiers<DimensionType> >();

  load_subdivisions(*modifiers);
  load_interface_rules(*modifiers);

  return modifiers;
}

template <specfem::dimension::type DimensionType>
void specfem::runtime_configuration::mesh_modifiers::load_subdivisions(
    specfem::mesh::modifiers<DimensionType> &modifiers) {
  if (const YAML::Node &subdivisions_node =
          mesh_modifiers_node["subdivisions"]) {
    if (!subdivisions_node.IsSequence()) {
      std::ostringstream message;

      message << "Error reading specfem mesh modifiers.\n";
      message << "\"subdivisions\" must be a sequence.\n";

      throw std::runtime_error(message.str());
    }
    for (YAML::Node subdiv_entry : subdivisions_node) {
      int subx = 1;
      int subz = 1;
      int materialID;
      if (const YAML::Node &materialentry = subdiv_entry["material"]) {
        try {
          materialID = materialentry.as<int>();
        } catch (YAML::ParserException &e) {
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: subdivisions.\n";
          message << "\"material\" must be an integer.\n";
          message << e.what();
          throw std::runtime_error(message.str());
        }
      } else {
        std::ostringstream message;
        message << "Error reading specfem mesh modifiers: subdivisions.\n";
        message
            << "\"material\" must be specified for each subdivision entry.\n";
        throw std::runtime_error(message.str());
      }
      if (const YAML::Node &subdiv_z = subdiv_entry["z"]) {
        try {
          subz = subdiv_z.as<int>();
          if (subz < 1) {
            std::ostringstream message;
            message << "Error reading specfem mesh modifiers: subdivisions.\n";
            message << "\"z\" must be a positive integer.\n";
            throw std::runtime_error(message.str());
          }
        } catch (YAML::ParserException &e) {
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: subdivisions.\n";
          message << "\"z\" must be a positive integer.\n";
          message << e.what();
          throw std::runtime_error(message.str());
        }
      }
      if (const YAML::Node &subdiv_x = subdiv_entry["x"]) {
        try {
          subx = subdiv_x.as<int>();
          if (subx < 1) {
            std::ostringstream message;
            message << "Error reading specfem mesh modifiers: subdivisions.\n";
            message << "\"x\" must be a positive integer.\n";
            throw std::runtime_error(message.str());
          }
        } catch (YAML::ParserException &e) {
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: subdivisions.\n";
          message << "\"x\" must be a positive integer.\n";
          message << e.what();
          throw std::runtime_error(message.str());
        }
      }
      // minus 1 for zero-indexing.
      modifiers.set_subdivision(materialID - 1, std::make_tuple(subx, subz));
    }
  }
}

template <specfem::dimension::type DimensionType>
void specfem::runtime_configuration::mesh_modifiers::load_interface_rules(
    specfem::mesh::modifiers<DimensionType> &modifiers) {
  if (const YAML::Node &interface_node =
          mesh_modifiers_node["interface-rules"]) {

    if (!interface_node.IsSequence()) {
      std::ostringstream message;

      message << "Error reading specfem mesh modifiers.\n";
      message << "\"interface-rules\" must be a sequence.\n";

      throw std::runtime_error(message.str());
    }

    for (YAML::Node interface_entry : interface_node) {
      int material1;
      int material2;
      specfem::enums::interface_resolution::type rule = enums::interface_resolution::type::UNKNOWN;

      if (const YAML::Node &materialentry = interface_entry["material1"]) {
        try {
          material1 = materialentry.as<int>();
        } catch (YAML::ParserException &e) {
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: interface-rules.\n";
          message << "\"material1\" must be an integer.\n";
          message << e.what();
          throw std::runtime_error(message.str());
        }
      } else {
        std::ostringstream message;
        message << "Error reading specfem mesh modifiers: interface-rules.\n";
        message
            << "\"material1\" must be specified for each entry.\n";
        throw std::runtime_error(message.str());
      }

      if (const YAML::Node &materialentry = interface_entry["material2"]) {
        try {
          material2 = materialentry.as<int>();
        } catch (YAML::ParserException &e) {
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: interface-rules.\n";
          message << "\"material2\" must be an integer.\n";
          message << e.what();
          throw std::runtime_error(message.str());
        }
      } else {
        std::ostringstream message;
        message << "Error reading specfem mesh modifiers: interface-rules.\n";
        message
            << "\"material2\" must be specified for each entry.\n";
        throw std::runtime_error(message.str());
      }

      if (const YAML::Node &resolution_type = interface_entry["rule"]) {
        rule = specfem::enums::interface_resolution::from_string(resolution_type.as<std::string>());
        if(rule == enums::interface_resolution::type::UNKNOWN){
          std::ostringstream message;
          message << "Error reading specfem mesh modifiers: interface-rules.\n";
          message
              << "cannot parse rule: \"" << resolution_type.as<std::string>() << "\".\n";
          throw std::runtime_error(message.str());
        }
      } else {
        std::ostringstream message;
        message << "Error reading specfem mesh modifiers: interface-rules.\n";
        message
            << "\"rule\" must be specified for each entry.\n";
        throw std::runtime_error(message.str());
      }
      modifiers.set_interface_resolution_rule(material1, material2, rule);
    }
  }
}
