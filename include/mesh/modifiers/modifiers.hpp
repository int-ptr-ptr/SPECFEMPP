#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/interface_resolution.hpp"
#include "mesh/mesh.hpp"

#include <string>
#include <tuple>
#include <unordered_map>

namespace specfem {
namespace mesh {

// dimtuple<T, dim> gives tuple<T,...> (size = dim)
// dimtuple<T, toset, set> gives tuple<T,...> (size = toset + sizeof(set))
template <typename T, int toset, typename... set> struct dimtuple {
  // using type = typename std::conditional<toset <= 0, std::tuple<set...>,
  // typename dimtuple<T,toset-1,T,set...>::type>::type;
  using unravelstruct = dimtuple<T, toset - 1, T, set...>;
  using type = typename unravelstruct::type;
  static inline std::string subdiv_str(type tup) {
    return unravelstruct::subdiv_str(tup) + ((toset > 1) ? "x" : "") +
           std::to_string(std::get<toset - 1>(tup));
  }
  static inline auto ones() {
    return std::tuple_cat(unravelstruct::ones(), std::tuple<int>{ 1 });
  }
};
template <typename T, typename... set> struct dimtuple<T, 0, set...> {
  using type = std::tuple<set...>;
  static inline std::string subdiv_str(type tup) { return ""; }
  static inline auto ones() { return std::tuple<>{}; }
};

template <specfem::dimension::type DimensionType> class modifiers {
public:
  static constexpr int dim = specfem::dimension::dimension<DimensionType>::dim;
  using subdiv_tuple = typename dimtuple<int, dim>::type;
  modifiers() : does_any_modifications(false), interface_resolutions() {}

  //===== application =====
  void apply(specfem::mesh::mesh<DimensionType> &mesh) const;

  //===== display / debug / info =====
  std::string to_string() const;
  std::string subdivisions_to_string() const;

  //===== setting modifiers =====
  void set_subdivision(const int material, subdiv_tuple subdivisions);
  void set_interface_resolution_rule(
      const int material1, const int material2,
      const specfem::enums::interface_resolution::type rule);
  //===== getting modifiers =====
  specfem::mesh::modifiers<DimensionType>::subdiv_tuple
  get_subdivision(const int material) const;
  specfem::enums::interface_resolution::type
  get_interface_resolution_rule(const int material1, const int material2) const;
  std::vector<int> partition_materials(const int num_materials) const;

private:
  bool does_any_modifications;
  std::unordered_map<int, subdiv_tuple> subdivisions; ///< map
                                                      ///< materialID ->
                                                      ///< (subdivide_z,
                                                      ///< subdivide_x)
  std::unordered_map<int, specfem::enums::interface_resolution::type>
      interface_resolutions; ///< map
  ///< enumerate_pairs(materialID1,MaterialID2) ->
  ///< resolution_rule
  static int enumerate_pairs(int a, int b) {
    // a <= b (just in case we need to resolve internally in the future)
    return a + ((b * (b + 1)) / 2);
  }
};
} // namespace mesh
} // namespace specfem
