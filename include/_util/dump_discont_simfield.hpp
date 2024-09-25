#ifndef __UTIL_DUMP_DISCONT_SIMFIELD_
#define __UTIL_DUMP_DISCONT_SIMFIELD_

#include "compute/fields/discontinuous/discont_field_impl.hpp"
#include "compute/fields/discontinuous/discont_simulation_field.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"

#include <Kokkos_Core.hpp>
#include <string>
#include <iostream>
#include <fstream>


template <typename T,int dim, typename ViewType>
static void _stream_view(std::ofstream &stream, const ViewType &view){
  T value;
  const char* val = (char*)&value;
  int extents[dim];
  for (int i = 0; i < dim; i++)extents[i] = view.extent(i);
  stream << "View<"<<typeid(T).name()<<"(size="<<sizeof(T)<<"B)>["<<extents[0];
  for (int i =1; i < dim; i++)stream << "," << extents[i];
  stream <<"]";
  if constexpr(dim == 1){
    for(int i = 0; i < extents[0]; i++){
      value = view(i);
      stream.write(val,sizeof(T));
    }
  }else if constexpr(dim == 2){
    for(int i = 0; i < extents[0]; i++){
    for(int j = 0; j < extents[1]; j++){
      value = view(i,j);
      stream.write(val,sizeof(T));
    }}
  }else if constexpr(dim == 3){
    for(int i = 0; i < extents[0]; i++){
    for(int j = 0; j < extents[1]; j++){
    for(int k = 0; k < extents[2]; k++){
      value = view(i,j,k);
      stream.write(val,sizeof(T));
    }}}
  }else if constexpr(dim == 4){
    for(int i = 0; i < extents[0]; i++){
    for(int j = 0; j < extents[1]; j++){
    for(int k = 0; k < extents[2]; k++){
    for(int l = 0; l < extents[3]; l++){
      value = view(i,j,k,l);
      stream.write(val,sizeof(T));
    }}}}
  }else{
    static_assert(false, "dim not supported!");
  }
}

namespace _util{


  template <specfem::wavefield::type WavefieldType>
  void dump_discont_simfield(const std::string &filename,
      const specfem::compute::discontinuous_simulation_field<WavefieldType> &simfield,
      const specfem::compute::points &points){
    std::ofstream dump;
    dump.open(filename);
    //dump points
    dump << "pts";
    _stream_view<type_real,4>(dump,points.h_coord);
    //dump mesh adjacency
    dump << "mesh_adj";
    _stream_view<int,3>(dump,simfield.h_mesh_adjacency);
    //dump ispec map
    dump << "ispec_map";
    _stream_view<int,2>(dump,simfield.h_assembly_ispec_mapping);
    //dump acoustic, elastic
    dump << "acoustic_field";
    _stream_view<type_real,4>(dump,simfield.acoustic.h_field);
    dump << "elastic_field";
    _stream_view<type_real,4>(dump,simfield.elastic.h_field);
    //dump edge values
    dump << "edge_values_x";
    _stream_view<type_real,4>(dump,simfield.h_edge_values_x);
    dump << "edge_values_z";
    _stream_view<type_real,4>(dump,simfield.h_edge_values_z);

    dump.close();
  }

  static int dump_interval = 5;
  static int laststep = -dump_interval;
  template <specfem::wavefield::type WavefieldType>
  void dump_discont_simfield_per_step(const int istep, const std::string &filename,
      const specfem::compute::discontinuous_simulation_field<WavefieldType> &simfield,
      const specfem::compute::points &points){
    if(istep >= laststep + dump_interval){
      dump_discont_simfield(filename + std::to_string(istep)+".dat",simfield,points);
      laststep = istep;
    }
  }
}
#endif