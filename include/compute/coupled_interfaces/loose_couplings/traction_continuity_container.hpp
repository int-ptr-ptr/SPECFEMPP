#pragma once

#include "coupled_interface/loose/fluxes/traction_continuity.hpp"
#include "interface_quadrature.hpp"

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::traction_continuity::container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::elastic, QuadratureType,
          QuadratureType> {

private:
  using EdgeScalarView = Kokkos::View<type_real * [QuadratureType::NGLL],
                                      Kokkos::DefaultExecutionSpace>;

public:
  // /// chi field normal derivative
  // EdgeScalarView medium1_chi_nderiv;
  // EdgeScalarView medium2_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_chi_nderiv;
};
