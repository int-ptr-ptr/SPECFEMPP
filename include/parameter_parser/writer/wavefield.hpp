#ifndef _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP
#define _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP

#include "compute/assembly/assembly.hpp"
#include "reader/reader.hpp"
#include "writer/writer.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {
class wavefield {

public:
  wavefield(const std::string output_format, const std::string output_folder,
            const specfem::simulation::type type)
      : output_format(output_format), output_folder(output_folder),
        simulation_type(type) {}

  wavefield(const YAML::Node &Node, const specfem::simulation::type type);

  std::shared_ptr<specfem::writer::writer> instantiate_wavefield_writer(
      const specfem::compute::assembly &assembly) const;

  std::shared_ptr<specfem::reader::reader> instantiate_wavefield_reader(
      const specfem::compute::assembly &assembly) const;

  inline specfem::simulation::type get_simulation_type() const {
    return this->simulation_type;
  }

private:
  std::string output_format;                 ///< format of output file
  std::string output_folder;                 ///< Path to output folder
  specfem::simulation::type simulation_type; ///< Type of simulation
};
} // namespace runtime_configuration
} // namespace specfem

#endif /* _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP */
