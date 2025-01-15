#pragma once

#include "IO/fortranio/interface.hpp"
#include "IO/interface.hpp"
#include "IO/mesh/impl/fortran/read_boundaries.hpp"
#include "IO/mesh/impl/fortran/read_elements.hpp"
#include "IO/mesh/impl/fortran/read_interfaces.hpp"
#include "IO/mesh/impl/fortran/read_material_properties.hpp"
#include "IO/mesh/impl/fortran/read_mesh_database.hpp"
#include "IO/mesh/impl/fortran/read_parameters.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "material/material.hpp"
#include "mesh/mesh.hpp"
#include "mesh/tags/tags.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"

// External/Standard Libraries
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "double_read_material_properties.cpp"

static specfem::kokkos::HostView2d<type_real>
read_coorg_elements(std::ifstream &stream1, std::ifstream &stream2,
                    const int npgeo1, const int npgeo2,
                    const specfem::MPI::MPI *mpi) {

  int ipoin = 0;

  type_real coorgi, coorgj;
  specfem::kokkos::HostView2d<type_real> coorg("specfem::mesh::coorg", ndim,
                                               npgeo1 + npgeo2);

  for (int i = 0; i < npgeo1; i++) {
    specfem::IO::fortran_read_line(stream1, &ipoin, &coorgi, &coorgj);
    if (ipoin < 1 || ipoin > npgeo1) {
      throw std::runtime_error("Error reading coordinates");
    }
    // coorg stores the x,z for every control point
    // coorg([0, 2), i) = [x, z]
    coorg(0, ipoin - 1) = coorgi;
    coorg(1, ipoin - 1) = coorgj;
  }
  for (int i = 0; i < npgeo2; i++) {
    specfem::IO::fortran_read_line(stream2, &ipoin, &coorgi, &coorgj);
    if (ipoin < 1 || ipoin > npgeo2) {
      throw std::runtime_error("Error reading coordinates");
    }
    // coorg stores the x,z for every control point
    // coorg([0, 2), i) = [x, z]
    coorg(0, ipoin - 1 + npgeo1) = coorgi;
    coorg(1, ipoin - 1 + npgeo1) = coorgj;
  }

  return coorg;
}

static specfem::mesh::parameters<specfem::dimension::type::dim2>
read_mesh_parameters(std::ifstream &stream1, std::ifstream &stream2,
                     int &numat1, const specfem::MPI::MPI *mpi) {
  // ---------------------------------------------------------------------
  // reading mesh properties

  // int numat1;           ///< Total number of different materials
  int ngnod1;           ///< Number of control nodes
  int nspec1;           ///< Number of spectral elements
  int pointsdisp1;      // Total number of points to display (Only used for
                        // visualization)
  int nelemabs1;        ///< Number of elements on absorbing boundary
  int nelem_acforcing1; ///< Number of elements on acoustic forcing boundary
  int nelem_acoustic_surface1;  ///< Number of elements on acoustic surface
  int num_fluid_solid_edges1;   ///< Number of solid-fluid edges
  int num_fluid_poro_edges1;    ///< Number of fluid-poroelastic edges
  int num_solid_poro_edges1;    ///< Number of solid-poroelastic edges
  int nnodes_tangential_curve1; ///< Number of elements on tangential curve
  int nelem_on_the_axis1;       ///< Number of axial elements

  bool plot_lowerleft_corner_only1;

  int numat2;           ///< Total number of different materials
  int ngnod2;           ///< Number of control nodes
  int nspec2;           ///< Number of spectral elements
  int pointsdisp2;      // Total number of points to display (Only used for
                        // visualization)
  int nelemabs2;        ///< Number of elements on absorbing boundary
  int nelem_acforcing2; ///< Number of elements on acoustic forcing boundary
  int nelem_acoustic_surface2;  ///< Number of elements on acoustic surface
  int num_fluid_solid_edges2;   ///< Number of solid-fluid edges
  int num_fluid_poro_edges2;    ///< Number of fluid-poroelastic edges
  int num_solid_poro_edges2;    ///< Number of solid-poroelastic edges
  int nnodes_tangential_curve2; ///< Number of elements on tangential curve
  int nelem_on_the_axis2;       ///< Number of axial elements

  bool plot_lowerleft_corner_only2;

  specfem::IO::fortran_read_line(stream1, &numat1, &ngnod1, &nspec1,
                                 &pointsdisp1, &plot_lowerleft_corner_only1);
  specfem::IO::fortran_read_line(stream2, &numat2, &ngnod2, &nspec2,
                                 &pointsdisp2, &plot_lowerleft_corner_only2);
  if (plot_lowerleft_corner_only1 != plot_lowerleft_corner_only2) {
    throw std::runtime_error(
        "plot_lowerleft_corner_only is not the same between input files!");
  }

  // ---------------------------------------------------------------------
  if (ngnod1 != 9 || ngnod2 != 9) {
    std::ostringstream error_message;
    error_message << "Number of control nodes per element must be 9, but is "
                  << ngnod1 << " for stream1 and " << ngnod2 << " for stream2\n"
                  << "Currently, there is a bug when NGNOD == 4 \n";
    throw std::runtime_error(error_message.str());
  }

  specfem::IO::fortran_read_line(
      stream1, &nelemabs1, &nelem_acforcing1, &nelem_acoustic_surface1,
      &num_fluid_solid_edges1, &num_fluid_poro_edges1, &num_solid_poro_edges1,
      &nnodes_tangential_curve1, &nelem_on_the_axis1);
  specfem::IO::fortran_read_line(
      stream2, &nelemabs2, &nelem_acforcing2, &nelem_acoustic_surface2,
      &num_fluid_solid_edges2, &num_fluid_poro_edges2, &num_solid_poro_edges2,
      &nnodes_tangential_curve2, &nelem_on_the_axis2);
  // ----------------------------------------------------------------------

  mpi->sync_all();

  return { numat1 + numat2,
           ngnod1,
           nspec1 + nspec2,
           pointsdisp1 + pointsdisp2,
           nelemabs1 + nelemabs2,
           nelem_acforcing1 + nelem_acforcing2,
           nelem_acoustic_surface1 + nelem_acoustic_surface2,
           num_fluid_solid_edges1 + num_fluid_solid_edges2,
           num_fluid_poro_edges1 + num_fluid_poro_edges2,
           num_solid_poro_edges1 + num_solid_poro_edges2,
           nnodes_tangential_curve1 + nnodes_tangential_curve2,
           nelem_on_the_axis1 + nelem_on_the_axis2,
           plot_lowerleft_corner_only1 };
}

namespace _util {

specfem::mesh::mesh<specfem::dimension::type::dim2>
read_mesh(const std::string filename1, const std::string filename2,
          const specfem::MPI::MPI *mpi) {

  // Declaring empty mesh objects
  specfem::mesh::mesh<specfem::dimension::type::dim2> mesh;
  constexpr int ndim = 2;

  // Open the database file
  std::ifstream stream1;
  stream1.open(filename1);
  std::ifstream stream2;
  stream2.open(filename2);

  if (!stream1.is_open()) {
    throw std::runtime_error("Could not open database file 1");
  }
  if (!stream2.is_open()) {
    throw std::runtime_error("Could not open database file 2");
  }
  int nspec1, npgeo1, nproc1;
  int nspec2, npgeo2, nproc2;

  try {
    std::tie(nspec1, npgeo1, nproc1) =
        specfem::IO::mesh::impl::fortran::read_mesh_database_header(stream1,
                                                                    mpi);
    std::tie(nspec2, npgeo2, nproc2) =
        specfem::IO::mesh::impl::fortran::read_mesh_database_header(stream2,
                                                                    mpi);
    mesh.nspec = nspec1 + nspec2;
    mesh.npgeo = npgeo1 + npgeo2;
    mesh.nproc = nproc1;
  } catch (std::runtime_error &e) {
    throw;
  }

  // Mesh class to be populated from the database file.
  try {
    mesh.control_nodes.coord =
        read_coorg_elements(stream1, stream2, npgeo1, npgeo2, mpi);

  } catch (std::runtime_error &e) {
    throw;
  }

  int numat1;
  try {
    mesh.parameters = read_mesh_parameters(stream1, stream2, numat1, mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  mesh.control_nodes.ngnod = mesh.parameters.ngnod;
  mesh.control_nodes.nspec = mesh.nspec;
  mesh.control_nodes.knods = specfem::kokkos::HostView2d<int>(
      "specfem::mesh::knods", mesh.parameters.ngnod, mesh.nspec);

  int nspec_all = mpi->reduce(mesh.parameters.nspec, specfem::MPI::sum);
  int nelem_acforcing_all =
      mpi->reduce(mesh.parameters.nelem_acforcing, specfem::MPI::sum);
  int nelem_acoustic_surface_all =
      mpi->reduce(mesh.parameters.nelem_acoustic_surface, specfem::MPI::sum);

  try {
    auto [n_sls, attenuation_f0_reference, read_velocities_at_f0] =
        specfem::IO::mesh::impl::fortran::read_mesh_database_attenuation(
            stream1, mpi);
    specfem::IO::mesh::impl::fortran::read_mesh_database_attenuation(stream2,
                                                                     mpi);
  } catch (std::runtime_error &e) {
    throw;
  }

  try {
    mesh.materials = _util::read_material_properties(
        stream1, stream2, mesh.parameters.numat, mesh.nspec,
        mesh.control_nodes.knods, numat1, nspec1, npgeo1, mpi);
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

  specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2> bdry_abs(0);
  specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2> bdry_afs(
      0);
  specfem::mesh::forcing_boundary<specfem::dimension::type::dim2> bdry_forcing(
      0);
  mesh.boundaries = specfem::mesh::boundaries<specfem::dimension::type::dim2>(
      bdry_abs, bdry_afs, bdry_forcing);
  specfem::IO::fortran_read_line(stream1, &ninterfaces, &max_interface_size);
  specfem::IO::fortran_read_line(stream2, &ninterfaces, &max_interface_size);

  // try {
  //   mesh.boundaries = specfem::IO::mesh::impl::fortran::read_boundaries(
  //       stream, mesh.parameters.nspec, mesh.parameters.nelemabs,
  //       mesh.parameters.nelem_acoustic_surface,
  //       mesh.parameters.nelem_acforcing, mesh.control_nodes.knods, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

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

  // try {
  //   mesh.coupled_interfaces =
  //       specfem::IO::mesh::impl::fortran::read_coupled_interfaces(
  //           stream, mesh.parameters.num_fluid_solid_edges,
  //           mesh.parameters.num_fluid_poro_edges,
  //           mesh.parameters.num_solid_poro_edges, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.tangential_nodes =
  //       specfem::IO::mesh::impl::fortran::read_tangential_elements(
  //           stream, mesh.parameters.nnodes_tangential_curve);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // try {
  //   mesh.axial_nodes = specfem::IO::mesh::impl::fortran::read_axial_elements(
  //       stream, mesh.parameters.nelem_on_the_axis, mesh.nspec, mpi);
  // } catch (std::runtime_error &e) {
  //   throw;
  // }

  // Check if database file was read completely
  // if (stream1.get() && !stream1.eof()) {
  //   throw std::runtime_error("The Database 1 file wasn't fully read. Is there
  //   "
  //                            "anything written after axial elements?");
  // }

  stream1.close();
  // if (stream2.get() && !stream2.eof()) {
  //   throw std::runtime_error("The Database 2 file wasn't fully read. Is there
  //   "
  //                            "anything written after axial elements?");
  // }

  stream2.close();

  // Print material properties

  mpi->cout("Material systems:\n"
            "------------------------------");

  mpi->cout("Number of material systems = " +
            std::to_string(mesh.materials.n_materials) + "\n\n");

  const auto l_elastic_isotropic =
      mesh.materials.elastic_isotropic.material_properties;
  const auto l_acoustic_isotropic =
      mesh.materials.acoustic_isotropic.material_properties;

  for (const auto material : l_elastic_isotropic) {
    mpi->cout(material.print());
  }

  for (const auto material : l_acoustic_isotropic) {
    mpi->cout(material.print());
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() ==
         mesh.materials.n_materials);

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);

  return mesh;
}

} // namespace _util
