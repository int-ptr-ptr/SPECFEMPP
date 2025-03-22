#include "mesh/modifiers/modifiers.hpp"
#include "enumerations/interface_resolution.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

// apply() handled in apply*.cpp in this directory

//===== display / debug / info =====
template <specfem::dimension::type DimensionType>
std::string
specfem::mesh::modifiers<DimensionType>::subdivisions_to_string() const {
  std::string repr =
      "subdivisions (set: " + std::to_string(subdivisions.size()) + "):";
#define BUFSIZE 50
  char buf[BUFSIZE];
  for (const auto &[matID, subs] : subdivisions) {
    repr += "\n  - material %d: " + dimtuple<int, dim>::subdiv_str(subs) +
            " cell subdivision";
  }
#undef BUFSIZE
  return repr;
}

template <specfem::dimension::type DimensionType>
std::string specfem::mesh::modifiers<DimensionType>::to_string() const {
  std::string repr = "mesh modifiers: \n";
  repr += subdivisions_to_string();

  return repr;
}

//===== setting modifiers =====
template <specfem::dimension::type DimensionType>
void specfem::mesh::modifiers<DimensionType>::set_subdivision(
    const int material, subdiv_tuple subdivs) {
  subdivisions.insert(std::make_pair(material, subdivs));
  does_any_modifications = true;
}

template <specfem::dimension::type DimensionType>
void specfem::mesh::modifiers<DimensionType>::set_interface_resolution_rule(
    const int material1, const int material2,
    const specfem::enums::interface_resolution::type rule) {
  int mat1, mat2;
  if (material1 == material2) {
    std::ostringstream message;
    message << "Error setting interface resolution rule.\n";
    message << "cannot resolve rule between the same material: " << material1
            << ".\n";
    throw std::runtime_error(message.str());
  } else if (material1 > material2) {
    mat1 = material2;
    mat2 = material1;
  } else {
    mat1 = material1;
    mat2 = material2;
  }
  // now, mat1 < mat2. set rule
  interface_resolutions.insert(
      std::make_pair(enumerate_pairs(mat1, mat2), rule));
  if (rule != enums::interface_resolution::type::UNKNOWN) {
    does_any_modifications = true;
  }
}
//===== getting modifiers =====
template <specfem::dimension::type DimensionType>
typename specfem::mesh::modifiers<DimensionType>::subdiv_tuple
specfem::mesh::modifiers<DimensionType>::get_subdivision(
    const int material) const {
  auto got = subdivisions.find(material);
  if (got == subdivisions.end()) {
    // default: no subdividing (1 subdiv in z, 1 in x)
    return specfem::mesh::dimtuple<int, dim>::ones();
  } else {
    return got->second;
  }
}
template <specfem::dimension::type DimensionType>
typename specfem::mesh::modifiers<DimensionType>::subdiv_tuple
specfem::mesh::modifiers<DimensionType>::get_subdivision(
    const specfem::mesh::materials::material_specification &matspec) const {
  return get_subdivision(matspec.database_index);
}

template <specfem::dimension::type DimensionType>
specfem::enums::interface_resolution::type
specfem::mesh::modifiers<DimensionType>::get_interface_resolution_rule(
    const int material1, const int material2) const {
  int mat1 = material1, mat2 = material2;
  if (mat1 > mat2) {
    std::swap(mat1, mat2);
  }
  auto got = interface_resolutions.find(enumerate_pairs(mat1, mat2));
  if (got == interface_resolutions.end()) {
    return specfem::enums::interface_resolution::type::UNKNOWN;
  } else {
    return got->second;
  }
}

template <specfem::dimension::type DimensionType>
std::vector<int> specfem::mesh::modifiers<DimensionType>::partition_materials(
    const int num_materials) const {
  std::vector<int> partitions(num_materials);
  int part = 0; // current partition ID
  for (int imat = 0; imat < num_materials; imat++) {
    // if continuous to prior, set to that
    bool found_part = false;
    for (int jmat = 0; jmat < imat; jmat++) {
      if (specfem::enums::interface_resolution::requires_assembly(
              get_interface_resolution_rule(jmat, imat))) {
        partitions[imat] = partitions[jmat];
        found_part = true;
      }
    }
    // otherwise, new partition
    if (found_part) {
      continue;
    }
    partitions[imat] = part;
    part++;
  }
  return partitions;
}

template class specfem::mesh::modifiers<specfem::dimension::type::dim2>;
