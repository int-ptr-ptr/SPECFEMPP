#pragma once

namespace specfem {
namespace coupled_interface {
namespace loose {
namespace quadrature {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename EdgeQuadrature,
          typename IntersectionQuadrature>
struct mortar_transfer_container {
protected:
  using TransferTensorView =
      Kokkos::View<type_real *
                       [IntersectionQuadrature::NGLL][EdgeQuadrature::NGLL],
                   Kokkos::DefaultExecutionSpace>;

public:
  static constexpr int NGLL_INTERSECTION = IntersectionQuadrature::NGLL;

  TransferTensorView interface_medium1_mortar_transfer;
  typename TransferTensorView::HostMirror h_interface_medium1_mortar_transfer;
  TransferTensorView interface_medium2_mortar_transfer;
  typename TransferTensorView::HostMirror h_interface_medium2_mortar_transfer;

  // TODO remove when edge_storages scheme is deprecated
  template <int ngllcapacity>
  void from_edge_data_mortar_trans(
      const int interface_ind,
      const type_real (&a_mortar_trans)[ngllcapacity][ngllcapacity],
      const type_real (&b_mortar_trans)[ngllcapacity][ngllcapacity]) {
    for (int igll1 = 0;
         igll1 < ngllcapacity && igll1 < IntersectionQuadrature::NGLL;
         igll1++) {
      for (int igll2 = 0; igll2 < ngllcapacity && igll2 < EdgeQuadrature::NGLL;
           igll2++) {
        h_interface_medium1_mortar_transfer(interface_ind, igll1, igll2) =
            a_mortar_trans[igll1][igll2];
        h_interface_medium2_mortar_transfer(interface_ind, igll1, igll2) =
            b_mortar_trans[igll1][igll2];
      }
    }
  }
  template <int ngllcapacity>
  void to_edge_data_mortar_trans(
      const int interface_ind,
      type_real (&a_mortar_trans)[ngllcapacity][ngllcapacity],
      type_real (&b_mortar_trans)[ngllcapacity][ngllcapacity]) {
    for (int igll1 = 0;
         igll1 < ngllcapacity && igll1 < IntersectionQuadrature::NGLL;
         igll1++) {
      for (int igll2 = 0; igll2 < ngllcapacity && igll2 < EdgeQuadrature::NGLL;
           igll2++) {
        a_mortar_trans[igll1][igll2] =
            h_interface_medium1_mortar_transfer(interface_ind, igll1, igll2);
        b_mortar_trans[igll1][igll2] =
            h_interface_medium2_mortar_transfer(interface_ind, igll1, igll2);
      }
    }
  }
  mortar_transfer_container() = default;
  // void operator=(const
  // mortar_transfer_container<DimensionType,MediumTag1,MediumTag2,EdgeQuadrature,IntersectionQuadrature>
  // &rhs) {
  //   this->interface_medium1_mortar_transfer =
  //   rhs.interface_medium1_mortar_transfer;
  //   this->h_interface_medium1_mortar_transfer =
  //   rhs.h_interface_medium1_mortar_transfer;
  //   this->interface_medium2_mortar_transfer =
  //   rhs.interface_medium2_mortar_transfer;
  //   this->h_interface_medium2_mortar_transfer =
  //   rhs.h_interface_medium2_mortar_transfer;
  // }

protected:
  mortar_transfer_container(int num_interfaces)
      : interface_medium1_mortar_transfer(
            "specfem::coupled_interface::loose::quadrature::mortar_transfer_"
            "container.interface_medium1_mortar_transfer",
            num_interfaces),
        h_interface_medium1_mortar_transfer(
            Kokkos::create_mirror_view(interface_medium1_mortar_transfer)),
        interface_medium2_mortar_transfer(
            "specfem::coupled_interface::loose::quadrature::mortar_transfer_"
            "container.interface_medium2_mortar_transfer",
            num_interfaces),
        h_interface_medium2_mortar_transfer(
            Kokkos::create_mirror_view(interface_medium2_mortar_transfer)) {}
};

} // namespace quadrature
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
