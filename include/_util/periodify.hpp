#pragma once

#include "compute/assembly/assembly.hpp"
#include "compute/fields/simulation_field.hpp"
#include "mesh/mesh.hpp"
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

template <typename bdtype, typename callback>
static void bdkill(bdtype &bd, const callback &on_LR, const int nelems_init) {
  std::vector<int> spec_keep;
  std::vector<specfem::enums::boundaries::type> edge_keep;
  for (int ibd = 0; ibd < nelems_init; ibd++) {
    const int ispec = bd.index_mapping(ibd);
    const specfem::enums::boundaries::type edge = bd.type(ibd);
    int inod1, inod2;
    switch (edge) {
    case specfem::enums::boundaries::TOP_LEFT:
      inod1 = 3;
      inod2 = 3;
      break;
    case specfem::enums::boundaries::TOP_RIGHT:
      inod1 = 2;
      inod2 = 2;
      break;
    case specfem::enums::boundaries::BOTTOM_LEFT:
      inod1 = 0;
      inod2 = 0;
      break;
    case specfem::enums::boundaries::BOTTOM_RIGHT:
      inod1 = 1;
      inod2 = 1;
      break;
    case specfem::enums::boundaries::TOP:
      inod1 = 2;
      inod2 = 3;
      break;
    case specfem::enums::boundaries::LEFT:
      inod1 = 0;
      inod2 = 3;
      break;
    case specfem::enums::boundaries::RIGHT:
      inod1 = 1;
      inod2 = 2;
      break;
    case specfem::enums::boundaries::BOTTOM:
      inod1 = 0;
      inod2 = 1;
      break;
    }
    // keep elems only when they are not on LR.
    if (!(on_LR(ispec, inod1) && on_LR(ispec, inod2))) {
      spec_keep.push_back(ispec);
      edge_keep.push_back(edge);
    }
  }
  const int nelem = spec_keep.size();
  bd = bdtype(nelem);
  for (int i = 0; i < nelem; i++) {
    bd.index_mapping(i) = spec_keep[i];
    bd.type(i) = edge_keep[i];
  }
  std::cout << "bdkill: " << nelems_init << " -> " << nelem << std::endl;
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

  int num_combined = 0;

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
              num_combined++;
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
              num_combined++;
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
  std::cout << "periodify_LR: " << num_combined << " combined iglobs"
            << std::endl;
}

void kill_LR_BCs(specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) {
  type_real xmin = mesh.control_nodes.coord(0, 0);
  type_real xmax = xmin;
  for (int i = 1; i < mesh.control_nodes.coord.extent(1); i++) {
    const type_real x = mesh.control_nodes.coord(0, i);
    xmin = std::min(xmin, x);
    xmax = std::max(xmax, x);
  }
  type_real epsx = (xmax - xmin) * 1e-5;
  const auto on_LR = [&](int ispec, int inod) {
    const int node = mesh.control_nodes.knods(inod, ispec);
    const type_real x = mesh.control_nodes.coord(0, node);
    return (xmax - x < epsx) || (x - xmin < epsx);
  };
  bdkill(mesh.boundaries.absorbing_boundary, on_LR,
         mesh.boundaries.absorbing_boundary.nelements);
  bdkill(mesh.boundaries.acoustic_free_surface, on_LR,
         mesh.boundaries.acoustic_free_surface.nelem_acoustic_surface);
}

} // namespace _util
