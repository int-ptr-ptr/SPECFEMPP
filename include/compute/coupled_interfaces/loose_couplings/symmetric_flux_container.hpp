#pragma once

#include "coupled_interface/loose/fluxes/symmetric_flux.hpp"
#include "interface_geometry.hpp"
#include "interface_quadrature.hpp"

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::symmetric_flux::container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType,
          QuadratureType>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType, 1, false>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType, 2, false> {

private:
  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<QuadratureType>;

public:
  container() = default;
  void operator=(
      const specfem::coupled_interface::loose::flux::symmetric_flux::container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType> &rhs) {
    specfem::compute::loose::interface_normal_container<
        DimensionType, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::acoustic, QuadratureType, 1,
        false>::operator=(rhs);
    specfem::compute::loose::interface_normal_container<
        DimensionType, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::acoustic, QuadratureType, 2,
        false>::operator=(rhs);
    specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
        DimensionType, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::acoustic, QuadratureType,
        QuadratureType>::operator=(rhs);
  }
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

protected:
  container(int num_medium1_edges, int num_medium2_edges, int num_interfaces)
      : specfem::compute::loose::interface_normal_container<
            DimensionType, specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::acoustic, QuadratureType, 1, false>(
            num_medium1_edges),
        specfem::compute::loose::interface_normal_container<
            DimensionType, specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::acoustic, QuadratureType, 2, false>(
            num_medium2_edges),
        specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<DimensionType,
                                      specfem::element::medium_tag::acoustic,
                                      specfem::element::medium_tag::acoustic,
                                      QuadratureType, QuadratureType>(
                num_interfaces) {}
};

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct specfem::coupled_interface::loose::flux::symmetric_flux::container<
    DimensionType, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          DimensionType, specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::acoustic, QuadratureType,
          QuadratureType>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, specfem::element::medium_tag::elastic,
          specfem::element::medium_tag::elastic, QuadratureType, 1, false>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, specfem::element::medium_tag::elastic,
          specfem::element::medium_tag::elastic, QuadratureType, 2, false> {

private:
  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<QuadratureType>;
  using EdgeVectorView =
      specfem::compute::loose::EdgeVectorView<DimensionType, QuadratureType>;

public:
  container() = default;
  // void operator=(const
  // specfem::coupled_interface::loose::flux::symmetric_flux::container<
  //         DimensionType, specfem::element::medium_tag::elastic,
  //         specfem::element::medium_tag::elastic, QuadratureType> &rhs) {
  //   specfem::compute::loose::interface_normal_container<DimensionType,specfem::element::medium_tag::elastic,specfem::element::medium_tag::elastic,QuadratureType,1,false>::operator=(rhs);
  //   specfem::compute::loose::interface_normal_container<DimensionType,specfem::element::medium_tag::elastic,specfem::element::medium_tag::elastic,QuadratureType,2,false>::operator=(rhs);
  //   specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::acoustic, QuadratureType,
  //         QuadratureType>::operator=(rhs);
  // }

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

protected:
  container(int num_medium1_edges, int num_medium2_edges, int num_interfaces)
      : specfem::compute::loose::interface_normal_container<
            DimensionType, specfem::element::medium_tag::elastic,
            specfem::element::medium_tag::elastic, QuadratureType, 1, false>(
            num_medium1_edges),
        specfem::compute::loose::interface_normal_container<
            DimensionType, specfem::element::medium_tag::elastic,
            specfem::element::medium_tag::elastic, QuadratureType, 2, false>(
            num_medium2_edges),
        specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<DimensionType,
                                      specfem::element::medium_tag::acoustic,
                                      specfem::element::medium_tag::acoustic,
                                      QuadratureType, QuadratureType>(
                num_interfaces) {}
};
