#include "mesh/coupled_interfaces/coupled_interfaces.hpp"
#include "mesh/coupled_interfaces/interface_container.hpp"
#include "mesh/coupled_interfaces/interface_container.tpp"

specfem::mesh::coupled_interfaces coupled_interfaces(
    specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>
        elastic_acoustic,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>
        acoustic_poroelastic,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::elastic,
        specfem::element::medium_tag::poroelastic>
        elastic_poroelastic) {}

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
std::variant<
    specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::elastic,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get() const {
  if constexpr (medium1 == specfem::element::medium_tag::elastic &&
                medium2 == specfem::element::medium_tag::acoustic) {
    return elastic_acoustic;
  } else if constexpr (medium1 == specfem::element::medium_tag::acoustic &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return acoustic_poroelastic;
  } else if constexpr (medium1 == specfem::element::medium_tag::elastic &&
                       medium2 == specfem::element::medium_tag::poroelastic) {
    return elastic_poroelastic;
  }
}

// Explicitly instantiate template member function
template int
specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                   specfem::element::medium_tag::acoustic>::
    get_spectral_elem_index<specfem::element::medium_tag::elastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                   specfem::element::medium_tag::acoustic>::
    get_spectral_elem_index<specfem::element::medium_tag::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::element::medium_tag::acoustic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::acoustic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::element::medium_tag::acoustic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::poroelastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::elastic>(
        const int interface_index) const;

template int
specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                   specfem::element::medium_tag::poroelastic>::
    get_spectral_elem_index<specfem::element::medium_tag::poroelastic>(
        const int interface_index) const;

// Explicitly instantiate template member function
template std::variant<
    specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::elastic,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>() const;

template std::variant<
    specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::elastic,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>() const;

template std::variant<
    specfem::mesh::interface_container<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::poroelastic>,
    specfem::mesh::interface_container<
        specfem::element::medium_tag::elastic,
        specfem::element::medium_tag::poroelastic> >
specfem::mesh::coupled_interfaces::coupled_interfaces::get<
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::poroelastic>() const;
