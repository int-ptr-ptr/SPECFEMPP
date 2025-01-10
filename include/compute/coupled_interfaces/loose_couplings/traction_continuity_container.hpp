#pragma once

#include "coupled_interface/loose/fluxes/traction_continuity.hpp"
#include "interface_geometry.hpp"
#include "interface_quadrature.hpp"

namespace specfem {
namespace compute {
namespace loosely_coupled_interface {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType>
struct traction_continuity_container;

template <specfem::dimension::type DimensionType, typename QuadratureType>
struct traction_continuity_container<
    DimensionType, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic, QuadratureType>
    : specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
          QuadratureType, QuadratureType>,
      specfem::compute::loose::interface_normal_container<
          DimensionType, QuadratureType, 2, true> {

private:
public:
  traction_continuity_container() = default;

  // void operator=(const
  // specfem::coupled_interface::loose::flux::traction_continuity::container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::elastic, QuadratureType> &rhs) {
  //   specfem::coupled_interface::loose::quadrature::mortar_transfer_container<
  //         DimensionType, specfem::element::medium_tag::acoustic,
  //         specfem::element::medium_tag::elastic, QuadratureType,
  //         QuadratureType>::operator=(rhs);
  // }

  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<QuadratureType>;
  EdgeScalarView disp_dot_normal;
  typename EdgeScalarView::HostMirror h_disp_dot_normal;

protected:
  traction_continuity_container(int num_medium1_edges, int num_medium2_edges,
                                int num_interfaces)
      : specfem::coupled_interface::loose::quadrature::
            mortar_transfer_container<QuadratureType, QuadratureType>(
                num_interfaces),
        specfem::compute::loose::interface_normal_container<
            DimensionType, QuadratureType, 2, true>(num_medium2_edges),
        disp_dot_normal("specfem::coupled_interface::loose::flux::traction_"
                        "continuity::container.disp_dot_normal",
                        num_medium2_edges),
        h_disp_dot_normal(Kokkos::create_mirror_view(disp_dot_normal)) {}
};

} // namespace loosely_coupled_interface
} // namespace compute
} // namespace specfem
