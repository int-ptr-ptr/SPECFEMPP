#pragma once

#include "compute/assembly/assembly.hpp"
#include "compute/fields/simulation_field.hpp"
#include <any>
#include <utility>
#include <vector>

template <specfem::wavefield::simulation_field WavefieldType>
static void
mass_matrix_combine(specfem::compute::simulation_field<WavefieldType> &field,
                    const int iglobsrc, const int iglobdest) {
  if (field.nglob <= iglobdest || field.nglob <= iglobsrc) {
    return;
  }
  int f_iglobsrc = field.h_assembly_index_mapping(
      iglobsrc, static_cast<int>(specfem::element::medium_tag::acoustic));
  int f_iglobdst = field.h_assembly_index_mapping(
      iglobdest, static_cast<int>(specfem::element::medium_tag::acoustic));
  if (f_iglobsrc >= 0 && f_iglobdst >= 0) {
    field.template get_field<specfem::element::medium_tag::acoustic>()
        .h_mass_inverse(f_iglobdst, 0) +=
        field.template get_field<specfem::element::medium_tag::acoustic>()
            .h_mass_inverse(f_iglobsrc, 0);
  }
  f_iglobsrc = field.h_assembly_index_mapping(
      iglobsrc, static_cast<int>(specfem::element::medium_tag::elastic));
  f_iglobdst = field.h_assembly_index_mapping(
      iglobdest, static_cast<int>(specfem::element::medium_tag::elastic));
  if (f_iglobsrc >= 0 && f_iglobdst >= 0) {
    field.template get_field<specfem::element::medium_tag::elastic>()
        .h_mass_inverse(f_iglobdst, 0) +=
        field.template get_field<specfem::element::medium_tag::elastic>()
            .h_mass_inverse(f_iglobsrc, 0);
    field.template get_field<specfem::element::medium_tag::elastic>()
        .h_mass_inverse(f_iglobdst, 1) +=
        field.template get_field<specfem::element::medium_tag::elastic>()
            .h_mass_inverse(f_iglobsrc, 1);
  }
}
static void mass_matrix_combine(specfem::compute::fields &fields,
                                const int iglobsrc, const int iglobdest) {
  mass_matrix_combine(fields.buffer, iglobsrc, iglobdest);
  mass_matrix_combine(fields.forward, iglobsrc, iglobdest);
  mass_matrix_combine(fields.adjoint, iglobsrc, iglobdest);
  mass_matrix_combine(fields.backward, iglobsrc, iglobdest);
}

namespace _util {
void periodify_LR(specfem::compute::assembly &assembly) {
  std::vector<std::pair<type_real, int> > unmatched_left;
  std::vector<std::pair<
      type_real, specfem::point::index<specfem::dimension::type::dim2> > >
      unmatched_right;
  const type_real xmin = assembly.mesh.points.xmin;
  const type_real xmax = assembly.mesh.points.xmax;
  const type_real epsx = (xmax - xmin) * 1e-5;
  const type_real epsy =
      (assembly.mesh.points.zmax - assembly.mesh.points.zmin) * 1e-5;

  for (int ispec = 0; ispec < assembly.mesh.nspec; ispec++) {
    for (int iz = 0; iz < assembly.mesh.ngllz; iz++) {
      for (int ix = 0; ix < assembly.mesh.ngllx; ix++) {
        int iglob = assembly.mesh.points.h_index_mapping(ispec, iz, ix);
        type_real x = assembly.mesh.points.h_coord(0, ispec, iz, ix);
        type_real y = assembly.mesh.points.h_coord(1, ispec, iz, ix);
        if (x - xmin < epsx) {
          // did we already see this y?
          bool found = false;
          for (int i = 0; i < unmatched_right.size(); i++) {
            if (std::fabs(y - unmatched_right[i].first) < epsy) {
              // yes we did. Pop this element and copy global values
              found = true;
              const auto ind_right = unmatched_right[i].second;
              int &iglobR = assembly.mesh.points.h_index_mapping(
                  ind_right.ispec, ind_right.iz, ind_right.ix);
              mass_matrix_combine(assembly.fields, iglobR, iglob);
              iglobR = iglob;
              unmatched_right[i] = unmatched_right.back();
              unmatched_right.pop_back();
              break;
            }
          }
          if (found) {
            continue;
          }
          // not matched; so add it to waitlist
          unmatched_left.push_back(std::make_pair(y, iglob));
        } else if (xmax - x < epsx) {
          // did we already see this y?
          bool found = false;
          for (int i = 0; i < unmatched_left.size(); i++) {
            if (std::fabs(y - unmatched_left[i].first) < epsy) {
              // yes we did. Pop this element and copy global values
              found = true;
              mass_matrix_combine(
                  assembly.fields, iglob,
                  assembly.mesh.points.h_index_mapping(ispec, iz, ix));
              assembly.mesh.points.h_index_mapping(ispec, iz, ix) =
                  unmatched_left[i].second;
              unmatched_left[i] = unmatched_left.back();
              unmatched_left.pop_back();
              break;
            }
          }
          if (found) {
            continue;
          }
          // not matched; so add it to waitlist
          unmatched_right.push_back(std::make_pair(
              y, specfem::point::index<specfem::dimension::type::dim2>(
                     ispec, iz, ix)));
        }
      }
    }
  }
  Kokkos::deep_copy(assembly.mesh.points.index_mapping,
                    assembly.mesh.points.h_index_mapping);
  assembly.fields.copy_to_device();
}
} // namespace _util
