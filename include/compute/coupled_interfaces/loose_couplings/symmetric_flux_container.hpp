#pragma once

#include "coupled_interface/loose/fluxes/symmetric_flux.hpp"
#include "interface_quadrature.hpp"

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::symmetric_flux::container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType,
          QuadratureType> {

private:
  using EdgeScalarView = Kokkos::View<type_real * [QuadratureType::NGLL],
                                      Kokkos::DefaultExecutionSpace>;

public:
  // /// Shape function normal derivative
  // EdgeScalarView medium1_shape_nderiv;
  // EdgeScalarView medium2_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_shape_nderiv;

  // /// chi field normal derivative
  // EdgeScalarView medium1_chi_nderiv;
  // EdgeScalarView medium2_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_chi_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_chi_nderiv;
};

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::symmetric_flux::container<
    DimensionType, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType,
          QuadratureType> {

private:
  using EdgeScalarView = Kokkos::View<type_real * [QuadratureType::NGLL],
                                      Kokkos::DefaultExecutionSpace>;
  using EdgeVectorView =
      Kokkos::View<type_real *
                       [QuadratureType::NGLL]
                           [specfem::dimension::dimension<DimensionType>::dim],
                   Kokkos::DefaultExecutionSpace>;

public:
  // /// Shape function normal derivative
  // EdgeScalarView medium1_shape_nderiv;
  // EdgeScalarView medium2_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_shape_nderiv;

  // /// disp field normal derivative
  // EdgeVectorView medium1_disp_nderiv;
  // EdgeVectorView medium2_disp_nderiv;
  // typename EdgeVectorView::HostMirror h_medium1_disp_nderiv;
  // typename EdgeVectorView::HostMirror h_medium2_disp_nderiv;
};
