#ifndef _COMPUTE_FIELDS_DISCONTINUOUS_SIMULATION_FIELD_HPP_
#define _COMPUTE_FIELDS_DISCONTINUOUS_SIMULATION_FIELD_HPP_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/simulation_field.hpp"
#include "element/field.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "point/field.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {

template <specfem::wavefield::type WavefieldType>
struct discontinuous_simulation_field: public simulation_field<WavefieldType> {

  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;

  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;

  discontinuous_simulation_field() = default;

  discontinuous_simulation_field(const specfem::compute::mesh &mesh,
                   const specfem::compute::properties &properties);

  // template <specfem::element::medium_tag medium>
  // KOKKOS_INLINE_FUNCTION
  //     specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //     medium> get_field() const {
  //   if constexpr (medium == specfem::element::medium_tag::elastic) {
  //     return elastic;
  //   } else if constexpr (medium == specfem::element::medium_tag::acoustic) {
  //     return acoustic;
  //   } else {
  //     static_assert("medium type not supported");
  //   }
  // }

  template <specfem::wavefield::type DestinationWavefieldType>
  void operator=(const simulation_field<DestinationWavefieldType> &rhs) {
    const discontinuous_simulation_field<DestinationWavefieldType>* testptr = 
        dynamic_cast<const discontinuous_simulation_field<DestinationWavefieldType>*>(&rhs);
    //TODO
    if(testptr != nullptr){
      this->nglob = rhs.nglob;
      this->assembly_index_mapping = rhs.assembly_index_mapping;
      this->h_assembly_index_mapping = rhs.h_assembly_index_mapping;
      this->elastic = rhs.elastic;
      this->acoustic = rhs.acoustic;
      return;
    }
    this->nglob = rhs.nglob;
    this->assembly_index_mapping = rhs.assembly_index_mapping;
    this->h_assembly_index_mapping = rhs.h_assembly_index_mapping;
    this->elastic = rhs.elastic;
    this->acoustic = rhs.acoustic;
  }

//============================================================================
//declarations from simulation_field
//============================================================================
  // int nglob = 0;
  // int nspec;
  // int ngllz;
  // int ngllx;
  // specfem::kokkos::DeviceView3d<int> index_mapping;
  // specfem::kokkos::HostMirror3d<int> h_index_mapping;
  // Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
  //              specfem::kokkos::DevMemSpace>
  //     assembly_index_mapping;
  // Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
  //              specfem::kokkos::HostMemSpace>
  //     h_assembly_index_mapping;
  // specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //                                    specfem::element::medium_tag::elastic>
  //     elastic;
  // specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //                                    specfem::element::medium_tag::acoustic>
  //     acoustic;
//============================================================================
};

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_DISCONT_discontinuous_simulation_field_HPP_ */
