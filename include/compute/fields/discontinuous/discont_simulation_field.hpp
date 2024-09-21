#ifndef _COMPUTE_FIELDS_DISCONTINUOUS_SIMULATION_FIELD_HPP_
#define _COMPUTE_FIELDS_DISCONTINUOUS_SIMULATION_FIELD_HPP_

#include "compute/fields/discontinuous/discont_field_impl.hpp"
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
struct discontinuous_simulation_field{

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
  void operator=(const discontinuous_simulation_field<DestinationWavefieldType> &rhs) {
    this->nglob = rhs.nglob;
    this->nspec = rhs.nspec;
    this->ngllz = rhs.ngllz;
    this->ngllx = rhs.ngllx;
    this->index_mapping = rhs.index_mapping;
    this->h_index_mapping = rhs.h_index_mapping;
    this->assembly_ispec_mapping = rhs.assembly_ispec_mapping;
    this->h_assembly_ispec_mapping = rhs.h_assembly_ispec_mapping;
    this->elastic = rhs.elastic;
    this->acoustic = rhs.acoustic;
    this->mesh_adjacency = rhs.mesh_adjacency;
    this->h_mesh_adjacency = rhs.h_mesh_adjacency;
    this->edge_values_x = rhs.edge_values_x;
    this->h_edge_values_x = rhs.h_edge_values_x;
    this->edge_values_z = rhs.edge_values_z;
    this->h_edge_values_z = rhs.h_edge_values_z;
    return;
  }



  template <specfem::element::medium_tag MediumType>
  KOKKOS_INLINE_FUNCTION int get_nglob() const {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return elastic.nglob;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return acoustic.nglob;
    } else {
      static_assert("medium type not supported");
    }
  }
//============================================================================
//declarations from simulation_field
//============================================================================
  int nglob = 0;
  int nspec;
  int ngllz;
  int ngllx;
  specfem::kokkos::DeviceView3d<int> index_mapping;
  specfem::kokkos::HostMirror3d<int> h_index_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::DevMemSpace>
      assembly_ispec_mapping;
  Kokkos::View<int * [specfem::element::ntypes], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      h_assembly_ispec_mapping;
  // specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //                                    specfem::element::medium_tag::elastic>
  //     elastic;
  // specfem::compute::impl::field_impl<specfem::dimension::type::dim2,
  //                                    specfem::element::medium_tag::acoustic>
  //     acoustic;
//============================================================================
  specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic>
      elastic;
  specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::acoustic>
      acoustic;

  //TODO change to be more efficient?
  specfem::kokkos::DeviceView3d<int, Kokkos::LayoutLeft> mesh_adjacency;
  specfem::kokkos::HostMirror3d<int, Kokkos::LayoutLeft> h_mesh_adjacency;


//TODO figure this size out; current debugging: (du/dn ds, dudx,dudz, nx, nz, det, fluxresult, wgll, rho_inv, du/dn ds from fluxcalc, du/dn ds adjacent)
#define _DISCONT_SIMFIELD_EDGE_COMPONENTS 17
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> edge_values_x;
  specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft> h_edge_values_x;
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> edge_values_z;
  specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft> h_edge_values_z;

  void copy_to_host() { sync_fields<specfem::sync::kind::DeviceToHost>(); }

  void copy_to_device() { sync_fields<specfem::sync::kind::HostToDevice>(); }

private:
  template <specfem::sync::kind sync> void sync_fields() {
    elastic.sync_fields<sync>();
    acoustic.sync_fields<sync>();
    //TODO add sync for edge values
  }
};

template <specfem::wavefield::type WavefieldType1,
          specfem::wavefield::type WavefieldType2>
void deep_copy(discontinuous_simulation_field<WavefieldType1> &dst,
               const discontinuous_simulation_field<WavefieldType2> &src) {
  dst.nglob = src.nglob;
  specfem::compute::deep_copy(dst.elastic, src.elastic);
  specfem::compute::deep_copy(dst.acoustic, src.acoustic);
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void load_on_device(
    const int iglob,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration,
                          StoreMassMatrix> &point_field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  load_on_device(specfem::point::index(ispec, iz, ix),field, point_field);
  return;
  throw std::runtime_error("Attempting to load a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void load_on_host(
    const int iglob,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration,
                          StoreMassMatrix> &point_field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  load_on_host(specfem::point::index(ispec, iz, ix),field, point_field);
  throw std::runtime_error("Attempting to load a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void load_on_device(
    const specfem::point::index &index,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration,
                          StoreMassMatrix> &point_field) {
  const auto curr_field = [&]()
      -> specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    point_field.displacement =
        Kokkos::subview(curr_field.field, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }
  if constexpr (StoreVelocity) {
    point_field.velocity =
        Kokkos::subview(curr_field.field_dot, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }
  if constexpr (StoreAcceleration) {
    point_field.acceleration =
        Kokkos::subview(curr_field.field_dot_dot, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  if constexpr (StoreMassMatrix) {
    point_field.mass_matrix =
        Kokkos::subview(curr_field.mass_inverse, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void load_on_host(
    const specfem::point::index &index,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::point::field<specfem::dimension::type::dim2, MediumType,
                          StoreDisplacement, StoreVelocity, StoreAcceleration,
                          StoreMassMatrix> &point_field) {

  const auto curr_field = [&]()
      -> specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  if constexpr (StoreDisplacement) {
    point_field.displacement =
        Kokkos::subview(curr_field.h_field, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  if constexpr (StoreVelocity) {
    point_field.velocity =
        Kokkos::subview(curr_field.h_field_dot, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  if constexpr (StoreAcceleration) {
    point_field.acceleration =
        Kokkos::subview(curr_field.h_field_dot_dot, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  if constexpr (StoreMassMatrix) {
    point_field.mass_matrix =
        Kokkos::subview(curr_field.h_mass_inverse, index.ispec, index.iz, index.ix, Kokkos::ALL);
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void store_on_device(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  store_on_device(specfem::point::index(ispec, iz, ix),point_field,field);
  return;

  throw std::runtime_error("Attempting to store a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void store_on_host(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  store_on_host(specfem::point::index(ispec, iz, ix),point_field,field);
  return;
  throw std::runtime_error("Attempting to store a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void store_on_device(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.field(index.ispec,index.iz,index.ix, icomp) = point_field.displacement[icomp];
    }
    if constexpr (StoreVelocity) {
      curr_field.field_dot(index.ispec,index.iz,index.ix, icomp) = point_field.velocity[icomp];
    }
    if constexpr (StoreAcceleration) {
      curr_field.field_dot_dot(index.ispec,index.iz,index.ix, icomp) = point_field.acceleration[icomp];
    }
    if constexpr (StoreMassMatrix) {
      curr_field.mass_inverse(index.ispec,index.iz,index.ix, icomp) = point_field.mass_matrix[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void store_on_host(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.h_field(index.ispec,index.iz,index.ix, icomp) = point_field.displacement[icomp];
    }
    if constexpr (StoreVelocity) {
      curr_field.h_field_dot(index.ispec,index.iz,index.ix, icomp) = point_field.velocity[icomp];
    }
    if constexpr (StoreAcceleration) {
      curr_field.h_field_dot_dot(index.ispec,index.iz,index.ix, icomp) =
          point_field.acceleration[icomp];
    }
    if constexpr (StoreMassMatrix) {
      curr_field.h_mass_inverse(index.ispec,index.iz,index.ix, icomp) = point_field.mass_matrix[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void add_on_device(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  add_on_device(specfem::point::index(ispec, iz, ix),point_field,field);
  return;
  throw std::runtime_error("Attempting to add a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void add_on_host(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  add_on_host(specfem::point::index(ispec, iz, ix),point_field,field);
  return;
  throw std::runtime_error("Attempting to add a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void add_on_device(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.field(index.ispec,index.iz,index.ix, icomp) += point_field.displacement[icomp];
    }
    if constexpr (StoreVelocity) {
      curr_field.field_dot(index.ispec,index.iz,index.ix, icomp) += point_field.velocity[icomp];
    }
    if constexpr (StoreAcceleration) {
      curr_field.field_dot_dot(index.ispec,index.iz,index.ix, icomp) += point_field.acceleration[icomp];
    }
    if constexpr (StoreMassMatrix) {
      curr_field.mass_inverse(index.ispec,index.iz,index.ix, icomp) += point_field.mass_matrix[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void add_on_host(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {


  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      curr_field.h_field(index.ispec,index.iz,index.ix, icomp) += point_field.displacement[icomp];
    }
    if constexpr (StoreVelocity) {
      curr_field.h_field_dot(index.ispec,index.iz,index.ix, icomp) += point_field.velocity[icomp];
    }
    if constexpr (StoreAcceleration) {
      curr_field.h_field_dot_dot(index.ispec,index.iz,index.ix, icomp) +=
          point_field.acceleration[icomp];
    }
    if constexpr (StoreMassMatrix) {
      curr_field.h_mass_inverse(index.ispec,index.iz,index.ix, icomp) += point_field.mass_matrix[icomp];
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void atomic_add_on_device(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  atomic_add_on_device(specfem::point::index(ispec, iz, ix),point_field,field);
  return;
  throw std::runtime_error("Attempting to atomic-add a global index on a discontinuous field. Use 3D indexing!");

}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void atomic_add_on_host(
    const int iglob,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {
  const int ispec = iglob / (field.ngllx * field.ngllz);
  int ix = iglob % (field.ngllx * field.ngllz);
  const int iz = ix / field.ngllx;
  ix = ix % field.ngllx;
  atomic_add_on_host(specfem::point::index(ispec, iz, ix),point_field,field);
  return;
  throw std::runtime_error("Attempting to atomic-add a global index on a discontinuous field. Use 3D indexing!");
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
KOKKOS_FUNCTION void atomic_add_on_device(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      Kokkos::atomic_add(&curr_field.field(index.ispec,index.iz,index.ix, icomp),
                         point_field.displacement[icomp]);
    }
    if constexpr (StoreVelocity) {
      Kokkos::atomic_add(&curr_field.field_dot(index.ispec,index.iz,index.ix, icomp),
                         point_field.velocity[icomp]);
    }
    if constexpr (StoreAcceleration) {
      Kokkos::atomic_add(&curr_field.field_dot_dot(index.ispec,index.iz,index.ix, icomp),
                         point_field.acceleration[icomp]);
    }
    if constexpr (StoreMassMatrix) {
      Kokkos::atomic_add(&curr_field.mass_inverse(index.ispec,index.iz,index.ix, icomp),
                         point_field.mass_matrix[icomp]);
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType,
          specfem::element::medium_tag MediumType, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix>
void atomic_add_on_host(
    const specfem::point::index &index,
    const specfem::point::field<
        specfem::dimension::type::dim2, MediumType, StoreDisplacement,
        StoreVelocity, StoreAcceleration, StoreMassMatrix> &point_field,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  auto curr_field = [&]() -> specfem::compute::impl::discontinuous_field_impl<
                              specfem::dimension::type::dim2, MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  for (int icomp = 0; icomp < components; ++icomp) {
    if constexpr (StoreDisplacement) {
      Kokkos::atomic_add(&curr_field.h_field(index.ispec,index.iz,index.ix, icomp),
                         point_field.displacement[icomp]);
    }
    if constexpr (StoreVelocity) {
      Kokkos::atomic_add(&curr_field.h_field_dot(index.ispec,index.iz,index.ix, icomp),
                         point_field.velocity[icomp]);
    }
    if constexpr (StoreAcceleration) {
      Kokkos::atomic_add(&curr_field.h_field_dot_dot(index.ispec,index.iz,index.ix, icomp),
                         point_field.acceleration[icomp]);
    }
    if constexpr (StoreMassMatrix) {
      Kokkos::atomic_add(&curr_field.h_mass_inverse(index.ispec,index.iz,index.ix, icomp),
                         point_field.mass_matrix[icomp]);
    }
  }

  return;
}

template <specfem::wavefield::type WavefieldType, int NGLL,
          specfem::element::medium_tag MediumType, typename MemberType,
          typename MemorySpace, typename MemoryTraits, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
          std::enable_if_t<std::is_same_v<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>,
                           bool> = true>
KOKKOS_FUNCTION void load_on_device(
    const MemberType &team, const int ispec,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::element::field<NGLL, specfem::dimension::type::dim2, MediumType,
                            MemorySpace, MemoryTraits, StoreDisplacement,
                            StoreVelocity, StoreAcceleration, StoreMassMatrix>
        &element_field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               specfem::kokkos::DevExecSpace>,
                "This function should only be called with device execution "
                "space");

  const auto curr_field = [&]()
      -> specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int iz, ix;
        sub2ind(xz, NGLL, iz, ix);

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            element_field.displacement(iz, ix, icomp) =
                curr_field.field(ispec,iz,ix, icomp);
          }
          if constexpr (StoreVelocity) {
            element_field.velocity(iz, ix, icomp) =
                curr_field.field_dot(ispec,iz,ix, icomp);
          }
          if constexpr (StoreAcceleration) {
            element_field.acceleration(iz, ix, icomp) =
                curr_field.field_dot_dot(ispec,iz,ix, icomp);
          }
          if constexpr (StoreMassMatrix) {
            element_field.mass_matrix(iz, ix, icomp) =
                curr_field.mass_inverse(ispec,iz,ix, icomp);
          }
        }
      });

  return;
}

template <specfem::wavefield::type WavefieldType, int NGLL,
          specfem::element::medium_tag MediumType, typename MemberType,
          typename MemorySpace, typename MemoryTraits, bool StoreDisplacement,
          bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
          std::enable_if_t<std::is_same_v<typename MemberType::execution_space::
                                              scratch_memory_space,
                                          MemorySpace>,
                           bool> = true>
void load_on_host(
    const MemberType &team, const int ispec,
    const specfem::compute::discontinuous_simulation_field<WavefieldType> &field,
    specfem::element::field<NGLL, specfem::dimension::type::dim2, MediumType,
                            MemorySpace, MemoryTraits, StoreDisplacement,
                            StoreVelocity, StoreAcceleration, StoreMassMatrix>
        &element_field) {

  constexpr int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumType>::components;

  static_assert(
      std::is_same_v<typename MemberType::execution_space, Kokkos::HostSpace>,
      "This function should only be called with host execution space");

  const auto curr_field = [&]()
      -> specfem::compute::impl::discontinuous_field_impl<specfem::dimension::type::dim2,
                                            MediumType> {
    if constexpr (MediumType == specfem::element::medium_tag::elastic) {
      return field.elastic;
    } else if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
      return field.acoustic;
    } else {
      static_assert("medium type not supported");
    }
  }();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int iz, ix;
        sub2ind(xz, NGLL, iz, ix);

        for (int icomp = 0; icomp < components; ++icomp) {
          if constexpr (StoreDisplacement) {
            element_field.displacement(iz, ix, icomp) =
                curr_field.h_field(ispec,iz,ix, icomp);
          }
          if constexpr (StoreVelocity) {
            element_field.velocity(iz, ix, icomp) =
                curr_field.h_field_dot(ispec,iz,ix, icomp);
          }
          if constexpr (StoreAcceleration) {
            element_field.acceleration(iz, ix, icomp) =
                curr_field.h_field_dot_dot(ispec,iz,ix, icomp);
          }
          if constexpr (StoreMassMatrix) {
            element_field.mass_matrix(iz, ix, icomp) =
                curr_field.h_mass_inverse(ispec,iz,ix, icomp);
          }
        }
      });

  return;
}
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_DISCONT_discontinuous_simulation_field_HPP_ */
