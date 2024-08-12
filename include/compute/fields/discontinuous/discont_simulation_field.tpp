#ifndef _COMPUTE_FIELDS_DISCONTINUOUS_DISCONT_SIMULATION_FIELD_TPP_
#define _COMPUTE_FIELDS_DISCONTINUOUS_DISCONT_SIMULATION_FIELD_TPP_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/impl/field_impl.tpp"
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
  this->index_mapping = specfem::kokkos::DeviceView3d<int>(
      "specfem::compute::discontinuous_simulation_field::index_mapping",this->nspec,this->ngllz,this->ngllx);
  this->h_index_mapping = specfem::kokkos::HostMirror3d<int> (Kokkos::create_mirror_view(this->index_mapping));
  this->assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::DevMemSpace>(
          "specfem::compute::discontinuous_simulation_field::assembly_index_mapping", this->nglob);

  this->h_assembly_index_mapping =
      Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
                   specfem::kokkos::HostMemSpace>(
          Kokkos::create_mirror_view(this->assembly_index_mapping));

  int iglob = 0;
  int elastic_count = 0;
  int acoustic_count = 0;
  //if (element_type(ispec) == MediumTag) {
  for (int ispec = 0; ispec < this->nspec; ++ispec) {
    // increase the count only if current element is of the medium type
    bool do_elastic = (properties.h_element_types(ispec) == specfem::element::medium_tag::elastic);
    bool do_acoustic = (properties.h_element_types(ispec) == specfem::element::medium_tag::acoustic);
    for (int iz = 0; iz < this->ngllz; ++iz) {
      for (int ix = 0; ix < this->ngllx; ++ix) {
        
        this->h_index_mapping(ispec,iz,ix) = iglob;
        if(do_elastic){
          this->h_assembly_index_mapping(iglob, static_cast<int>(elastic_type::medium_tag)) = elastic_count;
          elastic_count++;
        }
        if(do_acoustic){
          this->h_assembly_index_mapping(iglob, static_cast<int>(acoustic_type::medium_tag)) = acoustic_count;
          acoustic_count++;
        }
        iglob++;
      }
    }
  }
  Kokkos::deep_copy(this->index_mapping, this->h_index_mapping);
  Kokkos::deep_copy(this->assembly_index_mapping, this->h_assembly_index_mapping);

  this->elastic =
      specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
                                         specfem::element::medium_tag::elastic>(elastic_count);

  this->acoustic = specfem::compute::impl::field_impl<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic>(acoustic_count);

  return;
}

#endif /* _COMPUTE_FIELDS_DISCONT_SIMULATION_FIELD_TPP_ */
