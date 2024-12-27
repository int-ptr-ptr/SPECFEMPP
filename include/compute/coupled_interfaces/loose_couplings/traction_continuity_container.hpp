#pragma once

#include "coupled_interface/loose/fluxes/traction_continuity.hpp"
#include "interface_geometry.hpp"
#include "interface_quadrature.hpp"

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::traction_continuity::container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::elastic, QuadratureType,
          QuadratureType>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::elastic, QuadratureType, 2, true> {

private:
public:
  container() = default;

  // void operator=(const
  // specfem::coupled_interface::loose::flux::traction_continuity::container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::elastic, QuadratureType> &rhs) {
  //   specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::elastic, QuadratureType,
  //         QuadratureType>::operator=(rhs);
  // }
  // /// chi field normal derivative
  // EdgeScalarView medium1_chi_nderiv;
  // EdgeScalarView medium2_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_chi_nderiv;

protected:
  container(int num_medium1_edges, int num_medium2_edges, int num_interfaces)
      : specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<DimensionType,
                                      specfem::element::medium_tag::acoustic,
                                      specfem::element::medium_tag::elastic,
                                      QuadratureType, QuadratureType>(
                num_interfaces),
        specfem::compute::loose::interface_normal_container<
            DimensionType, specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::elastic, QuadratureType, 2, true>(
            num_medium2_edges) {}
  template <int medium, bool on_device>
  KOKKOS_INLINE_FUNCTION void compute_edge_intermediate(int index) {}
};
