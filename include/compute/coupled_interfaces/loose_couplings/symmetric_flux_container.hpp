#pragma once

#include "Kokkos_Core.hpp"
#include "coupled_interface/loose/fluxes/symmetric_flux.hpp"
#include "interface_geometry.hpp"
#include "interface_quadrature.hpp"

namespace specfem {
namespace compute {
namespace loosely_coupled_interface {

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct temp_aux_container {
private:
  using IntersectionQuadrature = QuadratureType;
  using EdgeQuadrature = QuadratureType;
  using RealView = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  using EdgeField1View = specfem::compute::loose::EdgeFieldView<
      DimensionType, specfem::element::medium_tag::acoustic, QuadratureType>;
  using EdgeField2View = specfem::compute::loose::EdgeFieldView<
      DimensionType, specfem::element::medium_tag::elastic, QuadratureType>;

public:
  temp_aux_container(int num_medium1_edges = 0)
      : medium1_field_vel("temp_aux_container::medium1_field_vel",
                          num_medium1_edges),
        h_medium1_field_vel(Kokkos::create_mirror_view(medium1_field_vel)),
        medium1_field_nderiv_noncontra(
            "temp_aux_container::medium1_field_deriv", num_medium1_edges),
        h_medium1_field_nderiv_noncontra(
            Kokkos::create_mirror_view(medium1_field_nderiv_noncontra)) {}

  EdgeField1View medium1_field_vel;
  typename EdgeField1View::HostMirror h_medium1_field_vel;
  EdgeField1View medium1_field_nderiv_noncontra;
  typename EdgeField1View::HostMirror h_medium1_field_nderiv_noncontra;
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType>
struct symmetric_flux_container;

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux_container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::acoustic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          QuadratureType, QuadratureType>,
      specfem::compute::loose::interface_contravariant_normal_container<
          DimensionType, QuadratureType, 1, true>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, QuadratureType, 1, true>,
      specfem::compute::loose::single_medium_interface_container,
      temp_aux_container<DimensionType, QuadratureType> {
private:
  using IntersectionQuadrature = QuadratureType;
  using EdgeQuadrature = QuadratureType;
  using RealView = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  using EdgeField1View = specfem::compute::loose::EdgeFieldView<
      DimensionType, specfem::element::medium_tag::acoustic, QuadratureType>;
  using EdgeField2View = specfem::compute::loose::EdgeFieldView<
      DimensionType, specfem::element::medium_tag::elastic, QuadratureType>;
  using TransferTensorView =
      Kokkos::View<type_real *
                       [IntersectionQuadrature::NGLL][EdgeQuadrature::NGLL],
                   Kokkos::DefaultExecutionSpace>;

public:
  symmetric_flux_container() = default;
  //   void operator=(
  //       const
  //       specfem::coupled_interface::loose::flux::symmetric_flux::container<
  //           DimensionType, specfem::element::medium_tag::acoustic,
  //           specfem::element::medium_tag::acoustic, QuadratureType> &rhs) {
  //     specfem::compute::loose::interface_contravariant_normal_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::acoustic, QuadratureType, 1,
  //         false>::operator=(rhs);
  //     specfem::compute::loose::interface_contravariant_normal_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::acoustic, QuadratureType, 2,
  //         false>::operator=(rhs);
  //     specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::acoustic, QuadratureType,
  //         QuadratureType>::operator=(rhs);
  //   }
  // /// Shape function normal derivative
  // EdgeScalarView medium1_shape_nderiv;
  // EdgeScalarView medium2_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium1_shape_nderiv;
  // typename EdgeScalarView::HostMirror h_medium2_shape_nderiv;

  /// chi field normal derivative
  EdgeField1View medium1_field_nderiv;
  typename EdgeField1View::HostMirror h_medium1_field_nderiv;

  RealView interface_relaxation_parameter;
  typename RealView::HostMirror h_interface_relaxation_parameter;

  TransferTensorView interface_medium1_mortar_transfer_deriv_times_n;
  TransferTensorView interface_medium2_mortar_transfer_deriv_times_n;
  typename TransferTensorView::HostMirror
      h_interface_medium1_mortar_transfer_deriv_times_n;
  typename TransferTensorView::HostMirror
      h_interface_medium2_mortar_transfer_deriv_times_n;

protected:
  symmetric_flux_container(int num_medium1_edges, int num_medium2_edges,
                           int num_interfaces)
      : specfem::compute::loose::interface_contravariant_normal_container<
            DimensionType, QuadratureType, 1, true>(num_medium1_edges),
        specfem::compute::loose::interface_normal_container<
            DimensionType, QuadratureType, 1, true>(num_medium1_edges),
        specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<QuadratureType, QuadratureType>(
                num_interfaces),
        medium1_field_nderiv("specfem::coupled_interface::loose::flux::"
                             "symmetric_flux::container.medium1_field_nderiv",
                             num_medium1_edges),
        h_medium1_field_nderiv(
            Kokkos::create_mirror_view(medium1_field_nderiv)),
        interface_relaxation_parameter(
            "specfem::coupled_interface::loose::flux::symmetric_flux::"
            "container.interface_relaxation_parameter",
            num_interfaces),
        h_interface_relaxation_parameter(
            Kokkos::create_mirror_view(interface_relaxation_parameter)),
        interface_medium1_mortar_transfer_deriv_times_n(
            "specfem::coupled_interface::loose::flux::symmetric_flux::"
            "container.interface_medium1_mortar_transfer_deriv_times_n",
            num_interfaces),
        h_interface_medium1_mortar_transfer_deriv_times_n(
            Kokkos::create_mirror_view(
                interface_medium1_mortar_transfer_deriv_times_n)),
        interface_medium2_mortar_transfer_deriv_times_n(
            "specfem::coupled_interface::loose::flux::symmetric_flux::"
            "container.interface_medium2_mortar_transfer_deriv_times_n",
            num_interfaces),
        h_interface_medium2_mortar_transfer_deriv_times_n(
            Kokkos::create_mirror_view(
                interface_medium2_mortar_transfer_deriv_times_n)),
        temp_aux_container<DimensionType, QuadratureType>(num_medium1_edges) {}
};

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct symmetric_flux_container<
    DimensionType, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          QuadratureType, QuadratureType>,
      specfem::compute::loose::interface_contravariant_normal_container<
          DimensionType, QuadratureType, 1, true>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, QuadratureType, 1, true>,
      specfem::compute::loose::single_medium_interface_container {

private:
  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<QuadratureType>;
  using EdgeVectorView =
      specfem::compute::loose::EdgeVectorView<DimensionType, QuadratureType>;

public:
  symmetric_flux_container() = default;
  // void operator=(const
  // specfem::coupled_interface::loose::flux::symmetric_flux::container<
  //         DimensionType, specfem::element::medium_tag::elastic,
  //         specfem::element::medium_tag::elastic, QuadratureType> &rhs) {
  //   specfem::compute::loose::interface_contravariant_normal_container<DimensionType,specfem::element::medium_tag::elastic,specfem::element::medium_tag::elastic,QuadratureType,1,false>::operator=(rhs);
  //   specfem::compute::loose::interface_contravariant_normal_container<DimensionType,specfem::element::medium_tag::elastic,specfem::element::medium_tag::elastic,QuadratureType,2,false>::operator=(rhs);
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
  symmetric_flux_container(int num_medium1_edges, int num_medium2_edges,
                           int num_interfaces)
      : specfem::compute::loose::interface_contravariant_normal_container<
            DimensionType, QuadratureType, 1, true>(num_medium1_edges),
        specfem::compute::loose::interface_normal_container<
            DimensionType, QuadratureType, 1, true>(num_medium2_edges),
        specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<QuadratureType, QuadratureType>(
                num_interfaces) {}
};

} // namespace loosely_coupled_interface
} // namespace compute
} // namespace specfem
