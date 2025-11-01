#pragma once

#include "../../MPI_environment.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly.hpp"
#include "specfem/source.hpp"
#include "utilities/strings.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::testing::kernel_compare {

/**
 * @brief Stores information for each kernel compare test. Meshes are stored in
 * here, since tests and meshes should be in 1-1 correspondence.
 *
 */
struct Test {
public:
  Test(const YAML::Node &Node, specfem::MPI::MPI *mpi);

  const std::string name;
  const std::string description;
  const std::string nonconforming_database_file;
  const specfem::mesh::mesh<specfem::dimension::type::dim2> nonconforming_mesh;
  const specfem::mesh::mesh<specfem::dimension::type::dim2> conforming_mesh;

private:
  // values in assembly need to be references. Store them here
  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::dimension::type::dim2> > >
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim2> > >
      receivers;
  const specfem::quadrature::quadratures quadrature;

public:
  specfem::assembly::assembly<specfem::dimension::type::dim2>
      nonconforming_assembly;
  specfem::assembly::assembly<specfem::dimension::type::dim2>
      conforming_assembly;
};

class mesh_list : public ::testing::Test,
                  public std::vector<specfem::testing::kernel_compare::Test> {

protected:
  mesh_list();
};

} // namespace specfem::testing::kernel_compare
