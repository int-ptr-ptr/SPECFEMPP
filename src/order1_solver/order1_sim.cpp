// from specfem2d.cpp
#include "compute/interface.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/interface.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// end from specfem2d.cpp


#include "mesh/mesh.hpp"
#include "specfem_mpi/specfem_mpi.hpp"
#include "enumerations/simulation.hpp"

#include "source/source.hpp"
#include "receiver/receiver.hpp"

#include "compute/assembly/assembly.hpp"

#include "quadrature/quadrature.hpp"
#include "specfem_setup.hpp"
#include <iostream>

struct simulation_params {
  const type_real t0 = 0.0;
  const type_real dt = 1e-3;
  const int nsteps = 1000;
  const type_real angle = 0.0;
  static constexpr auto simulation_type = specfem::simulation::type::forward;
  std::vector<specfem::enums::seismogram::type> seismo_types{specfem::enums::seismogram::type::displacement};
  const int nseismo_steps = 1000;
};

const auto quadrature = []() {
    /// Gauss-Lobatto-Legendre quadrature with 5 GLL points
    const specfem::quadrature::gll::gll gll(0, 0, 5);
    return specfem::quadrature::quadratures(gll);
  };

specfem::compute::assembly
    generate_assembly(const specfem::mesh::mesh &mesh,
                      const simulation_params &params) {

    const int nsteps = params.nsteps;
    const int nseismo_steps = params.nseismo_steps;
    const type_real t0 = params.t0;
    const type_real dt =  params.dt;
    const auto seismo_type = params.seismo_types;
    const auto simulation = params.simulation_type;


    std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers;
    std::vector<std::shared_ptr<specfem::sources::source> > sources;

    specfem::compute::assembly assembly(mesh, quadrature(), sources, receivers,
            seismo_type, t0, dt, nsteps, nseismo_steps, simulation);

    return assembly;
}



int main(int argc, char **argv) {
  // TODO sources / receivers
// https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html


  

  return 0;
}