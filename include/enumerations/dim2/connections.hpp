#pragma once

#include "../mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

template <> class connection_mapping<specfem::dimension::type::dim2> {
private:
  using ElementIndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

  connection_mapping(const int ngllz, const int ngllx,
                     ElementIndexView element1, ElementIndexView element2)
      : ngllz(ngllz), ngllx(ngllx), element1(element1), element2(element2) {
    if (element1.extent(0) != element2.extent(0)) {
      throw std::runtime_error(
          "Elements must have the same number of control nodes.");
    }
  }

  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &from,
                  const specfem::mesh_entity::dim2::type &to, const int iz,
                  const int ix) const;

  std::tuple<int, int>
  map_coordinates(const specfem::mesh_entity::dim2::type &from,
                  const specfem::mesh_entity::dim2::type &to) const;

private:
  int ngllz;
  int ngllx;

  ElementIndexView element1;
  ElementIndexView element2;
};

} // namespace specfem::connections
