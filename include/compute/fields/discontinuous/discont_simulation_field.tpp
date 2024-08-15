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

  return;
}

#endif /* _COMPUTE_FIELDS_DISCONT_SIMULATION_FIELD_TPP_ */
