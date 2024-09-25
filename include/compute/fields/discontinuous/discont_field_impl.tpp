#ifndef _COMPUTE_FIELDS_DISCONTINUOUS_IMPL_FIELD_IMPL_TPP_
#define _COMPUTE_FIELDS_DISCONTINUOUS_IMPL_FIELD_IMPL_TPP_

#include "compute/fields/impl/field_impl.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::discontinuous_field_impl<DimensionType, MediumTag>::discontinuous_field_impl(
    const int nspec, const int ngllz, const int ngllx)
    : nglob(nspec * ngllx * ngllz),
      field("specfem::compute::fields::field", nspec,ngllz, ngllx, medium_type::components),
      h_field(Kokkos::create_mirror_view(field)),
      field_dot("specfem::compute::fields::field_dot", nspec,ngllz, ngllx,
                medium_type::components),
      h_field_dot(Kokkos::create_mirror_view(field_dot)),
      field_dot_dot("specfem::compute::fields::field_dot_dot", nspec,ngllz, ngllx,
                    medium_type::components),
      h_field_dot_dot(Kokkos::create_mirror_view(field_dot_dot)),
      mass_inverse("specfem::compute::fields::mass_inverse", nspec,ngllz, ngllx,
                    medium_type::components),
      h_mass_inverse(Kokkos::create_mirror_view(mass_inverse)) {}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
specfem::compute::impl::discontinuous_field_impl<DimensionType, MediumTag>::discontinuous_field_impl(
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
        assembly_index_mapping) {
  const auto index_mapping = mesh.points.h_index_mapping;
  const auto element_type = properties.h_element_types;
  const int nspec = mesh.points.nspec;
  const int ngllz = mesh.points.ngllz;
  const int ngllx = mesh.points.ngllx;

  // Count the total number of distinct global indices for the medium
  int count = 0;

  for (int ispec = 0; ispec < nspec; ++ispec) {
    // increase the count only if current element is of the medium type
    if (element_type(ispec) == MediumTag) {
      assembly_index_mapping(ispec) = count;
      count++;
    }
  }

  nglob = count * ngllz * ngllx;

  field = specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field", count,ngllz,ngllx, medium_type::components);
  h_field = specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field));
  field_dot = specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot", count,ngllz,ngllx, medium_type::components);
  h_field_dot = specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(field_dot));
  field_dot_dot = specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::field_dot_dot", count,ngllz,ngllx,
      medium_type::components);
  h_field_dot_dot =
      specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
          Kokkos::create_mirror_view(field_dot_dot));
  mass_inverse = specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft>(
      "specfem::compute::fields::mass_inverse", count,ngllz,ngllx, medium_type::components);
  h_mass_inverse = specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft>(
      Kokkos::create_mirror_view(mass_inverse));

  Kokkos::parallel_for(
      "specfem::compute::fields::field_impl::initialize_field",
      specfem::kokkos::HostRange(0, count), [=](const int &ispec) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            type_real x = mesh.points.h_coord(0,ispec,iz,ix);
            type_real z = mesh.points.h_coord(1,ispec,iz,ix);

            for (int icomp = 0; icomp < medium_type::components; ++icomp) {
              //TODO initial condition: field is bump
#define _DSF_BUMP_CENTER_X_ 200
#define _DSF_BUMP_CENTER_Z_ 150
#define _DSF_BUMP_TWOSIG2_ 1250

#define _DSF_DISPL_X_ (x-_DSF_BUMP_CENTER_X_)
#define _DSF_DISPL_Z_ (z-_DSF_BUMP_CENTER_Z_)
              h_field(ispec,iz,ix, icomp) = exp(-(
                    _DSF_DISPL_X_ * _DSF_DISPL_X_
                  + _DSF_DISPL_Z_ * _DSF_DISPL_Z_)
                  /_DSF_BUMP_TWOSIG2_);
              h_field_dot(ispec,iz,ix, icomp) = 0.0;
              h_field_dot_dot(ispec,iz,ix, icomp) = 0.0;
              h_mass_inverse(ispec,iz,ix, icomp) = 0.0;
            }
          }
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(field, h_field);
  Kokkos::deep_copy(field_dot, h_field_dot);
  Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  Kokkos::deep_copy(mass_inverse, h_mass_inverse);

  return;
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
template <specfem::sync::kind sync>
void specfem::compute::impl::discontinuous_field_impl<DimensionType, MediumTag>::sync_fields()
    const {
  if constexpr (sync == specfem::sync::kind::DeviceToHost) {
    Kokkos::deep_copy(h_field, field);
    Kokkos::deep_copy(h_field_dot, field_dot);
    Kokkos::deep_copy(h_field_dot_dot, field_dot_dot);
  } else if constexpr (sync == specfem::sync::kind::HostToDevice) {
    Kokkos::deep_copy(field, h_field);
    Kokkos::deep_copy(field_dot, h_field_dot);
    Kokkos::deep_copy(field_dot_dot, h_field_dot_dot);
  }
}

#endif /* _COMPUTE_FIELDS_IMPL_FIELD_IMPL_TPP_ */

// template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &specfem::compute::(const int &iglob,
//   const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &field_dot(const int &iglob,
//                                               const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field_dot(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &field_dot_dot(const int &iglob,
//                                                   const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.field_dot_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.field_dot_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_field_dot_dot(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_field_dot_dot(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_field_dot_dot(index, icomp);
//     }
//   }

//   template <typename medium>
//   KOKKOS_INLINE_FUNCTION type_real &mass_inverse(const int &iglob,
//                                                  const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.mass_inverse(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.mass_inverse(index, icomp);
//     }
//   }

//   template <typename medium>
//   inline type_real &h_mass_inverse(const int &iglob, const int &icomp) {
//     if constexpr (std::is_same_v<medium, elastic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return elastic.h_mass_inverse(index, icomp);
//     } else if constexpr (std::is_same_v<medium, acoustic_type>) {
//       int index =
//           h_assembly_index_mapping(iglob, static_cast<int>(medium::value));
//       return acoustic.h_mass_inverse(index, icomp);
//     }
//   }
