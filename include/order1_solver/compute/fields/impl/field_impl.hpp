#ifndef _ORDER1SOLVER_COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_
#define _ORDER1SOLVER_COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/properties/interface.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
namespace impl {
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
class o1_field_impl {
public:
  using medium_type = specfem::medium::medium<DimensionType, MediumTag>;

  constexpr static int components = medium_type::components;

  o1_field_impl() = default;

  o1_field_impl(
      const specfem::compute::mesh &mesh,
      const specfem::compute::properties &properties,
      Kokkos::View<int *, Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
          assembly_index_mapping);

  o1_field_impl(
    const int nspec, const int ngllz, const int ngllx);

  template <specfem::sync::kind sync> void sync_fields() const;

  int nglob;
  int ncomponents;
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> field;
  specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft> h_field;
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> field_dot;
  specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft> h_field_dot;
  specfem::kokkos::DeviceView4d<type_real, Kokkos::LayoutLeft> mass_inverse;
  specfem::kokkos::HostMirror4d<type_real, Kokkos::LayoutLeft> h_mass_inverse;
};
} // namespace impl

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
void deep_copy(impl::o1_field_impl<DimensionType, MediumTag> &dst,
               const impl::o1_field_impl<DimensionType, MediumTag> &src) {
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::deep_copy(dst.h_field, src.h_field);
  Kokkos::deep_copy(dst.field_dot, src.field_dot);
  Kokkos::deep_copy(dst.h_field_dot, src.h_field_dot);
}

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_IMPL_FIELD_IMPL_HPP_ */
