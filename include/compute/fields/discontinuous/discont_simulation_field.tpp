#ifndef _COMPUTE_FIELDS_DISCONTINUOUS_DISCONT_SIMULATION_FIELD_TPP_
#define _COMPUTE_FIELDS_DISCONTINUOUS_DISCONT_SIMULATION_FIELD_TPP_

#include "compute/fields/discontinuous/discont_field_impl.hpp"
#include "compute/fields/discontinuous/discont_field_impl.tpp"
#include "compute/fields/discontinuous/discont_simulation_field.hpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>


template <specfem::wavefield::type WavefieldType>
specfem::compute::discontinuous_simulation_field<WavefieldType>::discontinuous_simulation_field(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties)
    //:specfem::compute::simulation_field<WavefieldType>::simulation_field(mesh,properties)
{
  //return;


  this->nspec = mesh.points.nspec;
  this->ngllz = mesh.points.ngllz;
  this->ngllx = mesh.points.ngllx;
  this->nglob = this->nspec * this->ngllz * this->ngllx;
  this->index_mapping = mesh.points.index_mapping;
  this->h_index_mapping = mesh.points.h_index_mapping;

  assembly_ispec_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::DevMemSpace>(
          "specfem::compute::simulation_field::ispec_mapping", this->nspec);

  h_assembly_ispec_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::HostMemSpace>(
          Kokkos::create_mirror_view(assembly_ispec_mapping));

  for (int ispec = 0; ispec < this->nspec; ispec++) {
    for (int itype = 0; itype < specfem::element::ntypes; itype++) {
      h_assembly_ispec_mapping(ispec, itype) = -1;
    }
  }

  auto acoustic_index =
      Kokkos::subview(h_assembly_ispec_mapping, Kokkos::ALL,
                      static_cast<int>(acoustic_type::medium_tag));

  auto elastic_index =
      Kokkos::subview(h_assembly_ispec_mapping, Kokkos::ALL,
                      static_cast<int>(elastic_type::medium_tag));

  elastic =
      specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                         specfem::element::medium_tag::elastic>(
          mesh, properties, elastic_index);

  acoustic = specfem::compute::impl::discontinuous_field_impl<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(
      mesh, properties, acoustic_index);

  Kokkos::deep_copy(assembly_ispec_mapping, h_assembly_ispec_mapping);


  //TODO generate adjacencies in mesh, then just set pointers to them;
  //this is just placeholder code using the global indices

  mesh_adjacency = //(ispec,edge,data_index) data_index(0: ispec, 1: edge+flip*4)
       specfem::kokkos::DeviceView3d<int, Kokkos::LayoutLeft>(
      "specfem::compute::simulation_field::mesh_adj", this->nspec,4,2);
  h_mesh_adjacency = specfem::kokkos::HostMirror3d<int, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(mesh_adjacency));

  std::vector<int> spec_from_glob(this->nglob);
  std::vector<int8_t> edge_from_glob(this->nglob);
  for (int iglob = 0; iglob < this->nglob; iglob++) {
    spec_from_glob[iglob] = -1; //init as null
  }
  for (int ispec = 0; ispec < this->nspec; ispec++) {
    for (int edge = 0; edge < 4; edge++){
      h_mesh_adjacency(ispec,edge,0) = -1; //init as null
    }
    
  }

  /*
    using the above vectors, we initialize index_mapping, by setting adjacencies
    according to shared global indices.

      if we have not encountered this global index:
        mark it encountered and point to this (ispec,edge)
      
      else:
        match (ispec,edge) to the stored values in *_from_glob[iglob]
   */
  const auto link_edges = [&](int ispec1, int edge1, int ispec2, int edge2){
    const int flip_bit = 4&(edge1 ^ edge2);
    h_mesh_adjacency(ispec1,edge1 & 3, 0) = ispec2;
    h_mesh_adjacency(ispec2,edge2 & 3, 0) = ispec1;
    h_mesh_adjacency(ispec1,edge1 & 3, 1) = (edge2 & 3) | flip_bit;
    h_mesh_adjacency(ispec2,edge2 & 3, 1) = (edge1 & 3) | flip_bit;
      // (flip:1b)(edge:2b) - edge = opposite edge, get from edge_from_glob
      // flip = 0 if local coordinates increase in same direction
      // we assume edge&4 flag is set on one side of the edge based on
      // local coordinate
  };
  const auto verify_adjacencies = [&](int ispec, int ix, int iz, int edge) {
    int iglob = h_index_mapping(ispec,iz,ix);
    if(spec_from_glob[iglob] == -1){
      //not yet defined, so define it
      spec_from_glob[iglob] = ispec;
      edge_from_glob[iglob] = edge;
    }else{
      //this matches another point, so set adjacency
      link_edges(ispec,edge,spec_from_glob[iglob],edge_from_glob[iglob]);
    }
  };

  for (int ispec = 0; ispec < this->nspec; ispec++) {
    //edge 0 - +x
    verify_adjacencies(ispec,this->ngllx-1,1,            0);
    verify_adjacencies(ispec,this->ngllx-1,this->ngllz-2,4);
    //edge 1 - +z
    verify_adjacencies(ispec,1,            this->ngllz-1,1);
    verify_adjacencies(ispec,this->ngllx-2,this->ngllz-1,5);
    //edge 2 - -x
    verify_adjacencies(ispec,0,1,            2);
    verify_adjacencies(ispec,0,this->ngllz-2,6);
    //edge 3 - -z
    verify_adjacencies(ispec,1,            0,3);
    verify_adjacencies(ispec,this->ngllx-2,0,7);
  }


    
  Kokkos::deep_copy(mesh_adjacency, h_mesh_adjacency);
  edge_values_x = //(ispec,edge,pt,data_index)
       specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::simulation_field::edge_values_x", this->nspec,2,this->ngllz,_DISCONT_SIMFIELD_EDGE_COMPONENTS);//TODO change last index to be dependent on #components?
  h_edge_values_x = specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(edge_values_x));

  edge_values_z = //(ispec,edge,pt,data_index)
       specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::simulation_field::edge_values_z", this->nspec,2,this->ngllx,_DISCONT_SIMFIELD_EDGE_COMPONENTS);//TODO change last index to be dependent on #components?
  h_edge_values_z = specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(edge_values_z));

  return;
}

#endif /* _COMPUTE_FIELDS_DISCONT_SIMULATION_FIELD_TPP_ */
