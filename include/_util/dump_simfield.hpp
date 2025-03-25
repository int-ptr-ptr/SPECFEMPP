#ifndef __UTIL_DUMP_DISCONT_SIMFIELD_
#define __UTIL_DUMP_DISCONT_SIMFIELD_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"

#include "_util/edge_storages.hpp"
#include "adjacency_graph/adjacency_graph.hpp"

#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>

template <typename T, int dim, typename ViewType>
static void _stream_view(std::ofstream &stream, const ViewType &view) {
  T value;
  const char *val = (char *)&value;
  int extents[dim];
  for (int i = 0; i < dim; i++)
    extents[i] = view.extent(i);
  stream << "<" << typeid(T).name() << "(size=" << sizeof(T) << "B)>["
         << extents[0];
  for (int i = 1; i < dim; i++)
    stream << "," << extents[i];
  stream << "]";
  if constexpr (dim == 1) {
    for (int i = 0; i < extents[0]; i++) {
      value = view(i);
      stream.write(val, sizeof(T));
    }
  } else if constexpr (dim == 2) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        value = view(i, j);
        stream.write(val, sizeof(T));
      }
    }
  } else if constexpr (dim == 3) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        for (int k = 0; k < extents[2]; k++) {
          value = view(i, j, k);
          stream.write(val, sizeof(T));
        }
      }
    }
  } else if constexpr (dim == 4) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        for (int k = 0; k < extents[2]; k++) {
          for (int l = 0; l < extents[3]; l++) {
            value = view(i, j, k, l);
            stream.write(val, sizeof(T));
          }
        }
      }
    }
  } else {
    static_assert(dim > 0 && dim <= 4, "dim not supported!");
  }
}

namespace _util {

static std::string tostr(specfem::element::boundary_tag tag) {
  switch (tag) {
  case specfem::element::boundary_tag::none:
    return "none";
  case specfem::element::boundary_tag::acoustic_free_surface:
    return "acoustic_free_surface";
  case specfem::element::boundary_tag::stacey:
    return "stacey";
  case specfem::element::boundary_tag::composite_stacey_dirichlet:
    return "composite_stacey_dirichlet";
  default:
    return "UNKNOWN";
  }
};

template <typename edgequad, int datacapacity>
void dump_edge_container(
    std::ofstream &dump,
    _util::edge_manager::edge_storage<edgequad, datacapacity> &edge_storage) {

  // FF dynamics
  dump << "acoustic_acoustic_nderiv";
  _stream_view<type_real, 3>(
      dump, edge_storage.acoustic_acoustic_interface.h_medium1_field_nderiv);

  // FS dynamics
  dump << "acoustic_elastic_sn";
  _stream_view<type_real, 2>(
      dump, edge_storage.acoustic_elastic_interface.h_disp_dot_normal);
}
template <typename edgequad, int datacapacity>
void dump_edge_container(
    const std::string &filename,
    _util::edge_manager::edge_storage<edgequad, datacapacity> &edge_storage) {

  std::ofstream dump;
  dump.open(filename);
  dump_edge_container(dump, edge_storage);
  dump.close();
}

template <typename edgequad, int datacapacity>
void dump_edge_container_statics(
    std::ofstream &dump,
    _util::edge_manager::edge_storage<edgequad, datacapacity> &edge_storage) {
  dump << "acoustic_acoustic_ispecs";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_acoustic_interface.h_medium1_index_mapping);
  dump << "elastic_elastic_ispecs";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.elastic_elastic_interface.h_medium1_index_mapping);
  dump << "acoustic_elastic_ispecs1";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_medium1_index_mapping);
  dump << "acoustic_elastic_ispecs2";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_medium2_index_mapping);

  dump << "edge_type_refs";
  Kokkos::View<specfem::enums::edge::type[4], specfem::kokkos::HostMemSpace>
      edge_type_refs("dump_edge_container_statics:edge_type_refs");
  edge_type_refs(0) = specfem::enums::edge::type::RIGHT;
  edge_type_refs(1) = specfem::enums::edge::type::TOP;
  edge_type_refs(2) = specfem::enums::edge::type::LEFT;
  edge_type_refs(3) = specfem::enums::edge::type::BOTTOM;
  _stream_view<unsigned int, 1>(dump, edge_type_refs);

  dump << "acoustic_acoustic_edgetypes";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_acoustic_interface.h_medium1_edge_type);
  dump << "elastic_elastic_edgetypes";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.elastic_elastic_interface.h_medium1_edge_type);
  dump << "acoustic_elastic_edgetypes1";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_medium1_edge_type);
  dump << "acoustic_elastic_edgetypes2";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_medium2_edge_type);

  dump << "acoustic_acoustic_interface_inds1";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_acoustic_interface.h_interface_medium1_index);
  dump << "elastic_elastic_interface_inds1";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.elastic_elastic_interface.h_interface_medium1_index);
  dump << "acoustic_elastic_interface_inds1";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_interface_medium1_index);
  dump << "acoustic_acoustic_interface_inds2";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_acoustic_interface.h_interface_medium2_index);
  dump << "elastic_elastic_interface_inds2";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.elastic_elastic_interface.h_interface_medium2_index);
  dump << "acoustic_elastic_interface_inds2";
  _stream_view<unsigned int, 1>(
      dump, edge_storage.acoustic_elastic_interface.h_interface_medium2_index);

  dump << "acoustic_acoustic_interface_paramstart1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_acoustic_interface.h_interface_medium1_param_start);
  dump << "elastic_elastic_interface_paramstart1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.elastic_elastic_interface.h_interface_medium1_param_start);
  dump << "acoustic_elastic_interface_paramstart1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_elastic_interface.h_interface_medium1_param_start);
  dump << "acoustic_acoustic_interface_paramstart2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_acoustic_interface.h_interface_medium2_param_start);
  dump << "elastic_elastic_interface_paramstart2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.elastic_elastic_interface.h_interface_medium2_param_start);
  dump << "acoustic_elastic_interface_paramstart2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_elastic_interface.h_interface_medium2_param_start);

  dump << "acoustic_acoustic_interface_paramend1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_acoustic_interface.h_interface_medium1_param_end);
  dump << "elastic_elastic_interface_paramend1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.elastic_elastic_interface.h_interface_medium1_param_end);
  dump << "acoustic_elastic_interface_paramend1";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_elastic_interface.h_interface_medium1_param_end);
  dump << "acoustic_acoustic_interface_paramend2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_acoustic_interface.h_interface_medium2_param_end);
  dump << "elastic_elastic_interface_paramend2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.elastic_elastic_interface.h_interface_medium2_param_end);
  dump << "acoustic_elastic_interface_paramend2";
  _stream_view<type_real, 1>(
      dump,
      edge_storage.acoustic_elastic_interface.h_interface_medium2_param_end);

  dump << "acoustic_acoustic_interface_mortartrans1";
  _stream_view<type_real, 3>(dump, edge_storage.acoustic_acoustic_interface
                                       .h_interface_medium1_mortar_transfer);
  dump << "elastic_elastic_interface_mortartrans1";
  _stream_view<type_real, 3>(dump, edge_storage.elastic_elastic_interface
                                       .h_interface_medium1_mortar_transfer);
  dump << "acoustic_elastic_interface_mortartrans1";
  _stream_view<type_real, 3>(dump, edge_storage.acoustic_elastic_interface
                                       .h_interface_medium1_mortar_transfer);
  dump << "acoustic_acoustic_interface_mortartrans2";
  _stream_view<type_real, 3>(dump, edge_storage.acoustic_acoustic_interface
                                       .h_interface_medium2_mortar_transfer);
  dump << "elastic_elastic_interface_mortartrans2";
  _stream_view<type_real, 3>(dump, edge_storage.elastic_elastic_interface
                                       .h_interface_medium2_mortar_transfer);
  dump << "acoustic_elastic_interface_mortartrans2";
  _stream_view<type_real, 3>(dump, edge_storage.acoustic_elastic_interface
                                       .h_interface_medium2_mortar_transfer);

  dump << "acoustic_acoustic_interface_jw";
  _stream_view<type_real, 2>(dump,
                             edge_storage.acoustic_acoustic_interface
                                 .h_interface_surface_jacobian_times_weight);
  dump << "elastic_elastic_interface_jw";
  _stream_view<type_real, 2>(dump,
                             edge_storage.elastic_elastic_interface
                                 .h_interface_surface_jacobian_times_weight);
  dump << "acoustic_elastic_interface_jw";
  _stream_view<type_real, 2>(dump,
                             edge_storage.acoustic_elastic_interface
                                 .h_interface_surface_jacobian_times_weight);

  // FF specific statics
  dump << "acoustic_acoustic_normal";
  _stream_view<type_real, 3>(
      dump, edge_storage.acoustic_acoustic_interface.h_medium1_edge_normal);
  dump << "acoustic_acoustic_contranormal";
  _stream_view<type_real, 3>(dump, edge_storage.acoustic_acoustic_interface
                                       .h_medium1_edge_contravariant_normal);
  dump << "acoustic_acoustic_relaxparam";
  _stream_view<type_real, 1>(dump, edge_storage.acoustic_acoustic_interface
                                       .h_interface_relaxation_parameter);
  dump << "acoustic_acoustic_dLn1";
  _stream_view<type_real, 3>(
      dump, edge_storage.acoustic_acoustic_interface
                .h_interface_medium1_mortar_transfer_deriv_times_n);
  dump << "acoustic_acoustic_dLn2";
  _stream_view<type_real, 3>(
      dump, edge_storage.acoustic_acoustic_interface
                .h_interface_medium2_mortar_transfer_deriv_times_n);

  // FS specific statics
  dump << "acoustic_elastic_normal";
  _stream_view<type_real, 3>(
      dump, edge_storage.acoustic_elastic_interface.h_medium2_edge_normal);
}
template <typename edgequad, int datacapacity>
void dump_edge_container_statics(
    const std::string &filename,
    _util::edge_manager::edge_storage<edgequad, datacapacity> &edge_storage) {
  std::ofstream dump;
  dump.open(filename);
  dump_edge_container_statics(dump, edge_storage);
  dump.close();
}

template <int num_sides>
void dump_adjacency_graph(
    const std::string &filename,
    const specfem::adjacency_graph::adjacency_graph<num_sides> &graph) {
  int nspec = graph.get_size();
  Kokkos::View<int *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      elems("_util::dump_adjacency_graph::elems", nspec);
  Kokkos::View<int8_t *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      sides("_util::dump_adjacency_graph::sides", nspec);
  Kokkos::View<bool *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      flips("_util::dump_adjacency_graph::flips", nspec);
  for (int i = 0; i < nspec; i++) {
    for (int side = 0; side < num_sides; side++) {
      specfem::adjacency_graph::adjacency_pointer adj =
          graph.get_adjacency(i, side);
      elems(i, side) = adj.elem;
      sides(i, side) = adj.side;
      flips(i, side) = adj.flip;
    }
  }
  std::ofstream dump;
  dump.open(filename);
  dump << "elem";
  _stream_view<int, 2>(dump, elems);
  dump << "side";
  _stream_view<int8_t, 2>(dump, sides);
  dump << "flips";
  _stream_view<bool, 2>(dump, flips);
  dump.close();
}

template <specfem::wavefield::simulation_field WavefieldType>
void dump_simfield(
    std::ofstream &dump,
    const specfem::compute::simulation_field<WavefieldType> &simfield,
    const specfem::compute::points &points, bool skip_statics = false) {

  if (!skip_statics) {
    Kokkos::deep_copy(points.h_coord, points.coord);
    dump << "pts";
    _stream_view<type_real, 4>(dump, points.h_coord);
    Kokkos::deep_copy(simfield.h_index_mapping, simfield.index_mapping);
    dump << "index_mapping";
    _stream_view<unsigned int, 3>(dump, simfield.h_index_mapping);
    Kokkos::deep_copy(simfield.h_assembly_index_mapping,
                      simfield.assembly_index_mapping);
    dump << "assembly_index_mapping";
    _stream_view<int, 2>(dump, simfield.h_assembly_index_mapping);
    Kokkos::deep_copy(simfield.acoustic.h_mass_inverse,
                      simfield.acoustic.mass_inverse);
    dump << "acoustic_mass_inverse";
    _stream_view<type_real, 2>(dump, simfield.acoustic.h_mass_inverse);
    Kokkos::deep_copy(simfield.elastic.h_mass_inverse,
                      simfield.elastic.mass_inverse);
    dump << "elastic_mass_inverse";
    _stream_view<type_real, 2>(dump, simfield.elastic.h_mass_inverse);
  }
  // //dump mesh adjacency
  // dump << "mesh_adj";
  // _stream_view<int,3>(dump,simfield.h_mesh_adjacency);
  // //dump ispec map
  // dump << "ispec_map";
  // _stream_view<int,2>(dump,simfield.h_assembly_ispec_mapping);
  // dump acoustic, elastic
  Kokkos::deep_copy(simfield.acoustic.h_field, simfield.acoustic.field);
  dump << "acoustic_field";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_field);
  Kokkos::deep_copy(simfield.elastic.h_field, simfield.elastic.field);
  dump << "elastic_field";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_field);
  Kokkos::deep_copy(simfield.acoustic.h_field_dot, simfield.acoustic.field_dot);
  dump << "acoustic_field_dot";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_field_dot);
  Kokkos::deep_copy(simfield.elastic.h_field_dot, simfield.elastic.field_dot);
  dump << "elastic_field_dot";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_field_dot);
  Kokkos::deep_copy(simfield.acoustic.h_field_dot_dot,
                    simfield.acoustic.field_dot_dot);
  dump << "acoustic_field_ddot";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_field_dot_dot);
  Kokkos::deep_copy(simfield.elastic.h_field_dot_dot,
                    simfield.elastic.field_dot_dot);
  dump << "elastic_field_ddot";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_field_dot_dot);
  // //dump edge values
  // dump << "edge_values_x";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_x);
  // dump << "edge_values_z";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_z);
}
template <specfem::wavefield::simulation_field WavefieldType>
void dump_simfield(
    const std::string &filename,
    const specfem::compute::simulation_field<WavefieldType> &simfield,
    const specfem::compute::points &points, bool skip_statics = false) {
  std::ofstream dump;
  dump.open(filename);
  dump_simfield(dump, simfield, points, skip_statics);
  dump.close();
}

void dump_simfield_statics(std::ofstream &dump,
                           specfem::compute::assembly &assembly) {
  const auto &simfield = assembly.fields.forward;
  Kokkos::deep_copy(assembly.mesh.points.h_coord, assembly.mesh.points.coord);
  // dump points
  dump << "pts";
  _stream_view<type_real, 4>(dump, assembly.mesh.points.h_coord);
  Kokkos::deep_copy(simfield.h_index_mapping, simfield.index_mapping);
  dump << "index_mapping";
  _stream_view<unsigned int, 3>(dump, simfield.h_index_mapping);
  // //dump mesh adjacency
  // dump << "mesh_adj";
  // _stream_view<int,3>(dump,simfield.h_mesh_adjacency);
  // //dump ispec map
  // dump << "ispec_map";
  // _stream_view<int,2>(dump,simfield.h_assembly_ispec_mapping);
  // dump acoustic, elastic
  Kokkos::deep_copy(simfield.h_assembly_index_mapping,
                    simfield.assembly_index_mapping);
  dump << "assembly_index_mapping";
  _stream_view<int, 2>(dump, simfield.h_assembly_index_mapping);
  Kokkos::deep_copy(simfield.acoustic.h_mass_inverse,
                    simfield.acoustic.mass_inverse);
  dump << "acoustic_mass_inverse";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_mass_inverse);
  Kokkos::deep_copy(simfield.elastic.h_mass_inverse,
                    simfield.elastic.mass_inverse);
  dump << "elastic_mass_inverse";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_mass_inverse);
  // //dump edge values
  // dump << "edge_values_x";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_x);
  // dump << "edge_values_z";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_z);

  dump << "medium_type_refs";
  Kokkos::View<int[2], specfem::kokkos::HostMemSpace> medium_type_refs(
      "dump_simfield_statics:medium_type_refs");
  medium_type_refs(0) =
      static_cast<int>(specfem::element::medium_tag::acoustic);
  medium_type_refs(1) = static_cast<int>(specfem::element::medium_tag::elastic);
  _stream_view<int, 1>(dump, medium_type_refs);
}
void dump_simfield_statics(const std::string &filename,
                           specfem::compute::assembly &assembly) {
  std::ofstream dump;
  dump.open(filename);
  dump_simfield_statics(dump, assembly);
  dump.close();
}

void dump_simfield_per_step(const int istep, const std::string &filename,
                            specfem::compute::assembly &assembly) {
  dump_simfield(filename + std::to_string(istep) + ".dat",
                assembly.fields.forward, assembly.mesh.points, true);
  if (istep == 0) {
    dump_simfield_statics(filename + "statics.dat", assembly);
  }
}
template <typename edgequad, int datacapacity>
void dump_simfield_per_step(
    const int istep, const std::string &filename,
    specfem::compute::assembly &assembly,
    _util::edge_manager::edge_storage<edgequad, datacapacity> &edge_storage) {
  std::ofstream dump;
  dump.open(filename + std::to_string(istep) + ".dat");
  dump_simfield(dump, assembly.fields.forward, assembly.mesh.points, true);
  dump_edge_container(dump, edge_storage);
  dump.close();
  if (istep == 0) {
    dump.open(filename + "statics.dat");
    dump_simfield_statics(dump, assembly);
    dump_edge_container_statics(dump, edge_storage);
    dump.close();
  }
}

void init_dirs(const boost::filesystem::path &dirname, bool clear = true) {
  if (clear) {
    boost::filesystem::remove_all(dirname);
  }
  boost::filesystem::create_directories(dirname);
}
} // namespace _util
#endif
