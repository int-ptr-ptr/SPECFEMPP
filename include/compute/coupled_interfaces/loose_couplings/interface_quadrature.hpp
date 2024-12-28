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
  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<IntersectionQuadrature>;

public:
  static constexpr int NGLL_INTERFACE = IntersectionQuadrature::NGLL;

  TransferTensorView interface_medium1_mortar_transfer;
  typename TransferTensorView::HostMirror h_interface_medium1_mortar_transfer;
  TransferTensorView interface_medium2_mortar_transfer;
  typename TransferTensorView::HostMirror h_interface_medium2_mortar_transfer;
  EdgeScalarView interface_surface_jacobian;
  typename EdgeScalarView::HostMirror h_interface_surface_jacobian;

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

#define mortar_trans(medium, on_device)                                        \
  [&]() {                                                                      \
    if constexpr (medium == 1) {                                               \
      if constexpr (on_device) {                                               \
        return interface_medium1_mortar_transfer;                              \
      } else {                                                                 \
        return h_interface_medium1_mortar_transfer;                            \
      }                                                                        \
    } else if constexpr (medium == 2) {                                        \
      if constexpr (on_device) {                                               \
        return interface_medium2_mortar_transfer;                              \
      } else {                                                                 \
        return h_interface_medium2_mortar_transfer;                            \
      }                                                                        \
    } else {                                                                   \
      static_assert(false, "Medium can only be 1 or 2!");                      \
    }                                                                          \
  }()
  template <int medium, bool on_device>
  type_real edge_to_mortar(const int edge_index, const int mortar_index,
                           const type_real *quantity) {
    type_real val = 0;
    for (int i = 0; i < EdgeQuadrature::NGLL; i++) {
      val += mortar_trans(medium, on_device)(edge_index, mortar_index, i) *
             quantity[i];
    }
    return val;
  }
#undef mortar_trans
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
            Kokkos::create_mirror_view(interface_medium2_mortar_transfer)),
        interface_surface_jacobian(
            "specfem::coupled_interface::loose::quadrature::mortar_transfer_"
            "container.interface_surface_jacobian",
            num_interfaces),
        h_interface_surface_jacobian(
            Kokkos::create_mirror_view(interface_surface_jacobian)) {}
};

} // namespace quadrature
} // namespace loose
} // namespace coupled_interface
} // namespace specfem
