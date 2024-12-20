#include "mesh/mesh.hpp"
#include "IO/fortranio/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "mesh/IO/fortran/read_mesh_database.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

specfem::mesh::mesh::mesh(const std::string filename,
                          const specfem::MPI::MPI *mpi) {

  std::ifstream stream;
  stream.open(filename);

  if (!stream.is_open()) {
    throw std::runtime_error("Could not open database file");
  }

  try {
    auto [nspec, npgeo, nproc] =
        specfem::mesh::IO::fortran::read_mesh_database_header(stream, mpi);
    this->nspec = nspec;
    this->npgeo = npgeo;
    this->nproc = nproc;
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->control_nodes.coord = specfem::mesh::IO::fortran::read_coorg_elements(
        stream, this->npgeo, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->parameters = specfem::mesh::properties(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  this->control_nodes.ngnod = this->parameters.ngnod;
  this->control_nodes.nspec = this->nspec;
  this->control_nodes.knods = specfem::kokkos::HostView2d<int>(
      "specfem::mesh::knods", this->parameters.ngnod, this->nspec);

  int nspec_all = mpi->reduce(this->parameters.nspec, specfem::MPI::sum);
  int nelem_acforcing_all =
      mpi->reduce(this->parameters.nelem_acforcing, specfem::MPI::sum);
  int nelem_acoustic_surface_all =
      mpi->reduce(this->parameters.nelem_acoustic_surface, specfem::MPI::sum);

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::mesh::IO::fortran::read_mesh_database_attenuation(stream, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->materials =
        specfem::mesh::materials(stream, this->parameters.numat, this->nspec,
                                 this->control_nodes.knods, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // try {
  //   materials = specfem::mesh::IO::fortran::read_material_properties(
  //       stream, this->parameters.numat, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->material_ind = specfem::mesh::material_ind(
  //       stream, this->parameters.ngnod, this->nspec, this->parameters.numat,
  //       this->control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->interface = specfem::mesh::interfaces::interface(stream, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  int ninterfaces;
  int max_interface_size;

  specfem::IO::fortran_read_line(stream, &ninterfaces, &max_interface_size);

  try {
    this->boundaries = specfem::mesh::boundaries(
        stream, this->parameters.nspec, this->parameters.nelemabs,
        this->parameters.nelem_acforcing,
        this->parameters.nelem_acoustic_surface, this->control_nodes.knods,
        mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // try {
  //   this->boundaries.absorbing_boundary = specfem::mesh::absorbing_boundary(
  //       stream, this->parameters.nelemabs, this->parameters.nspec, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->boundaries.forcing_boundary = specfem::mesh::forcing_boundary(
  //       stream, this->parameters.nelem_acforcing, this->parameters.nspec,
  //       mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   this->boundaries.acoustic_free_surface =
  //       specfem::mesh::acoustic_free_surface(
  //           stream, this->parameters.nelem_acoustic_surface,
  //           this->control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  try {
    this->coupled_interfaces = specfem::mesh::coupled_interfaces(
        stream, this->parameters.num_fluid_solid_edges,
        this->parameters.num_fluid_poro_edges,
        this->parameters.num_solid_poro_edges, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->tangential_nodes = specfem::mesh::elements::tangential_elements(
        stream, this->parameters.nnodes_tangential_curve);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    this->axial_nodes = specfem::mesh::elements::axial_elements(
        stream, this->parameters.nelem_on_the_axis, this->nspec, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  // Check if database file was read completely
  if (stream.get() && !stream.eof()) {
    throw std::runtime_error("The Database file wasn't fully read. Is there "
                             "anything written after axial elements?");
  }

  stream.close();

  // Print material properties

  mpi->cout("Material systems:\n"
            "------------------------------");

  mpi->cout("Number of material systems = " +
            std::to_string(this->materials.n_materials) + "\n\n");

  const auto l_elastic_isotropic =
      this->materials.elastic_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      this->materials.acoustic_isotropic.material_properties;

  for (const auto material : l_elastic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() ==
         this->materials.n_materials);

  this->tags = specfem::mesh::tags(this->materials, this->boundaries);

  return;
}

std::string specfem::mesh::mesh::print() const {

  int n_elastic;
  int n_acoustic;

  Kokkos::parallel_reduce(
      "specfem::mesh::mesh::print", specfem::kokkos::HostRange(0, this->nspec),
      KOKKOS_CLASS_LAMBDA(const int ispec, int &n_elastic, int &n_acoustic) {
        if (this->materials.material_index_mapping(ispec).type ==
            specfem::element::medium_tag::elastic) {
          n_elastic++;
        } else if (this->materials.material_index_mapping(ispec).type ==
                   specfem::element::medium_tag::acoustic) {
          n_acoustic++;
        }
      },
      n_elastic, n_acoustic);

  std::ostringstream message;

  message
      << "Spectral element information:\n"
      << "------------------------------\n"
      << "Total number of spectral elements : " << this->nspec << "\n"
      << "Total number of spectral elements assigned to elastic material : "
      << n_elastic << "\n"
      << "Total number of spectral elements assigned to acoustic material : "
      << n_acoustic << "\n"
      << "Total number of geometric points : " << this->npgeo << "\n";

  return message.str();
}
