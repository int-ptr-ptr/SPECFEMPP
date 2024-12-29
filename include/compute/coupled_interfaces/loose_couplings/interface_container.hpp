#pragma once

#include "edge/loose/edge_access.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace compute {
namespace loose {

template <int NGLL>
KOKKOS_INLINE_FUNCTION void
point_from_edge(int &iz, int &ix, const specfem::enums::edge::type edge,
                const int iedge) {
  switch (edge) {
  case specfem::enums::edge::type::RIGHT:
    ix = NGLL - 1;
    iz = iedge;
    return;
  case specfem::enums::edge::type::TOP:
    ix = iedge;
    iz = NGLL - 1;
    return;
  case specfem::enums::edge::type::LEFT:
    ix = 0;
    iz = iedge;
    return;
  case specfem::enums::edge::type::BOTTOM:
    ix = iedge;
    iz = 0;
    return;
  default:
    return;
  }
};

template <specfem::dimension::type DimensionType, typename QuadratureType>
using EdgeVectorView =
    Kokkos::View<type_real *
                     [QuadratureType::NGLL]
                         [specfem::dimension::dimension<DimensionType>::dim],
                 Kokkos::DefaultExecutionSpace>;
template <typename QuadratureType>
using EdgeScalarView = Kokkos::View<type_real * [QuadratureType::NGLL],
                                    Kokkos::DefaultExecutionSpace>;
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename QuadratureType>
using EdgeFieldView =
    Kokkos::View<type_real *
                     [QuadratureType::NGLL][specfem::element::attributes<
                         DimensionType, MediumTag>::components()],
                 Kokkos::DefaultExecutionSpace>;

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType,
          typename FluxScheme>
struct interface_container : FluxScheme::container<DimensionType, MediumTag1,
                                                   MediumTag2, QuadratureType> {
private:
  using super = typename FluxScheme::container<DimensionType, MediumTag1,
                                               MediumTag2, QuadratureType>;

  using IndexView =
      Kokkos::View<int *, Kokkos::DefaultExecutionSpace>; ///< Underlying view
                                                          ///< type to store
                                                          ///< indices
  using EdgeTypeView =
      Kokkos::View<specfem::enums::edge::type *,
                   Kokkos::DefaultExecutionSpace>; ///< Underlying view type to
                                                   ///< store edge types
  using RealView = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;

public:
  static constexpr specfem::dimension::type dim_type = DimensionType;
  static constexpr specfem::element::medium_tag medium1_type =
      MediumTag1; ///< Self medium of the interface
  static constexpr specfem::element::medium_tag medium2_type =
      MediumTag2; ///< Other medium of the interface

  template <int medium>
  static constexpr specfem::element::medium_tag medium_type = [] {
    static_assert(medium == 1 || medium == 2, "Medium can only be 1 or 2!");
    if constexpr (medium == 1) {
      return MediumTag1;
    } else {
      return MediumTag2;
    }
  }();
  static constexpr int NGLL_EDGE = QuadratureType::NGLL;
  using EdgeQuadrature = QuadratureType;

  // void operator=(const
  // interface_container<DimensionType,MediumTag1,MediumTag2,QuadratureType,FluxScheme>
  // &rhs) {
  //     this->num_medium1_edges = rhs.num_medium1_edges;
  //     this->num_medium2_edges = rhs.num_medium2_edges;
  //     this->num_interfaces = rhs.num_interfaces;
  //     this->medium1_index_mapping = rhs.medium1_index_mapping;
  //     this->h_medium1_index_mapping = rhs.h_medium1_index_mapping;
  //     this->medium2_index_mapping = rhs.medium2_index_mapping;
  //     this->h_medium2_index_mapping = rhs.h_medium2_index_mapping;
  //     this->medium1_edge_type = rhs.medium1_edge_type;
  //     this->h_medium1_edge_type = rhs.h_medium1_edge_type;
  //     this->medium2_edge_type = rhs.medium2_edge_type;
  //     this->h_medium2_edge_type = rhs.h_medium2_edge_type;

  //     this->interface_medium1_index = rhs.interface_medium1_index;
  //     this->h_interface_medium1_index = rhs.h_interface_medium1_index;
  //     this->interface_medium2_index = rhs.interface_medium2_index;
  //     this->h_interface_medium2_index = rhs.h_interface_medium2_index;
  //     this->interface_medium1_param_start =
  //     rhs.interface_medium1_param_start;
  //     this->h_interface_medium1_param_start =
  //     rhs.h_interface_medium1_param_start;
  //     this->interface_medium2_param_start =
  //     rhs.interface_medium2_param_start;
  //     this->h_interface_medium2_param_start =
  //     rhs.h_interface_medium2_param_start; this->interface_medium1_param_end
  //     = rhs.interface_medium1_param_end; this->h_interface_medium1_param_end
  //     = rhs.h_interface_medium1_param_end; this->interface_medium2_param_end
  //     = rhs.interface_medium2_param_end; this->h_interface_medium2_param_end
  //     = rhs.h_interface_medium2_param_end; super::operator=(rhs);

  //     //todo remove

  //     this->a_NORMAL = rhs.a_NORMAL;
  //     this->h_a_NORMAL = rhs.h_a_NORMAL;
  //     this->b_NORMAL = rhs.b_NORMAL;
  //     this->h_b_NORMAL = rhs.h_b_NORMAL;
  //     this->a_DET = rhs.a_DET;
  //     this->h_a_DET = rhs.h_a_DET;
  //     this->b_DET = rhs.b_DET;
  //     this->h_b_DET = rhs.h_b_DET;
  //     this->a_DS = rhs.a_DS;
  //     this->h_a_DS = rhs.h_a_DS;
  //     this->b_DS = rhs.b_DS;
  //     this->h_b_DS = rhs.h_b_DS;
  //     this->a_SPEEDPARAM = rhs.a_SPEEDPARAM;
  //     this->h_a_SPEEDPARAM = rhs.h_a_SPEEDPARAM;
  //     this->b_SPEEDPARAM = rhs.b_SPEEDPARAM;
  //     this->h_b_SPEEDPARAM = rhs.h_b_SPEEDPARAM;
  //     this->a_FIELDNDERIV = rhs.a_FIELDNDERIV;
  //     this->h_a_FIELDNDERIV = rhs.h_a_FIELDNDERIV;
  //     this->b_FIELDNDERIV = rhs.b_FIELDNDERIV;
  //     this->h_b_FIELDNDERIV = rhs.h_b_FIELDNDERIV;
  //     this->a_SHAPENDERIV = rhs.a_SHAPENDERIV;
  //     this->h_a_SHAPENDERIV = rhs.h_a_SHAPENDERIV;
  //     this->b_SHAPENDERIV = rhs.b_SHAPENDERIV;
  //     this->h_b_SHAPENDERIV = rhs.h_b_SHAPENDERIV;
  //     this->interface_relax_param = rhs.interface_relax_param;
  //     this->h_interface_relax_param = rhs.h_interface_relax_param;
  //   }
  interface_container() = default;
  interface_container(const int num_medium1_edges, const int num_medium2_edges,
                      const int num_interfaces)
      : num_medium1_edges(num_medium1_edges),
        num_medium2_edges(num_medium2_edges), num_interfaces(num_interfaces),
        medium1_index_mapping("specfem::compute::loose::interface_container."
                              "medium1_index_mapping",
                              num_medium1_edges),
        h_medium1_index_mapping(
            Kokkos::create_mirror_view(medium1_index_mapping)),
        medium2_index_mapping("specfem::compute::loose::interface_container."
                              "medium2_index_mapping",
                              num_medium2_edges),
        h_medium2_index_mapping(
            Kokkos::create_mirror_view(medium2_index_mapping)),
        super(num_medium1_edges, num_medium2_edges, num_interfaces) {}

  int num_medium1_edges;
  int num_medium2_edges;
  int num_interfaces;

  IndexView medium1_index_mapping; ///< Spectral element index for every edge on
                                   ///< self medium
  IndexView medium2_index_mapping; ///< Spectral element index for every edge on
                                   ///< other medium

  EdgeTypeView medium1_edge_type; ///< Edge orientation for every edge on self
                                  ///< medium
  EdgeTypeView medium2_edge_type; ///< Edge orientation for every edge on other
                                  ///< medium

  IndexView::HostMirror h_medium1_index_mapping; ///< Host mirror for @ref
                                                 ///< medium1_index_mapping
  IndexView::HostMirror h_medium2_index_mapping; ///< Host mirror for @ref
                                                 ///< medium2_index_mapping

  EdgeTypeView::HostMirror h_medium1_edge_type; ///< Host mirror for @ref
                                                ///< medium1_edge_type
  EdgeTypeView::HostMirror h_medium2_edge_type; ///< Host mirror for @ref
                                                ///< medium2_edge_type

  /// reference indices to edges on the medium1 and medium2 sides
  IndexView interface_medium1_index;
  IndexView interface_medium2_index;
  IndexView::HostMirror h_interface_medium1_index;
  IndexView::HostMirror h_interface_medium2_index;

  /// start and end parameters for edges
  RealView interface_medium1_param_start;
  RealView interface_medium1_param_end;
  RealView interface_medium2_param_start;
  RealView interface_medium2_param_end;
  RealView::HostMirror h_interface_medium1_param_start;
  RealView::HostMirror h_interface_medium1_param_end;
  RealView::HostMirror h_interface_medium2_param_start;
  RealView::HostMirror h_interface_medium2_param_end;

  // These are temporary until we figure out where to put them all
  using EdgeScalarView =
      specfem::compute::loose::EdgeScalarView<QuadratureType>;
  using EdgeVectorView =
      specfem::compute::loose::EdgeVectorView<DimensionType, QuadratureType>;
  using EdgeQuadView =
      Kokkos::View<type_real * [QuadratureType::NGLL][QuadratureType::NGLL],
                   Kokkos::DefaultExecutionSpace>;

  EdgeVectorView a_NORMAL;
  EdgeVectorView b_NORMAL;
  typename EdgeVectorView::HostMirror h_a_NORMAL;
  typename EdgeVectorView::HostMirror h_b_NORMAL;
  EdgeScalarView a_DET;
  EdgeScalarView b_DET;
  typename EdgeScalarView::HostMirror h_a_DET;
  typename EdgeScalarView::HostMirror h_b_DET;
  EdgeScalarView a_DS;
  EdgeScalarView b_DS;
  typename EdgeScalarView::HostMirror h_a_DS;
  typename EdgeScalarView::HostMirror h_b_DS;
  EdgeVectorView a_FIELDNDERIV;
  EdgeVectorView b_FIELDNDERIV;
  typename EdgeVectorView::HostMirror h_a_FIELDNDERIV;
  typename EdgeVectorView::HostMirror h_b_FIELDNDERIV;
  EdgeScalarView a_SPEEDPARAM;
  EdgeScalarView b_SPEEDPARAM;
  typename EdgeScalarView::HostMirror h_a_SPEEDPARAM;
  typename EdgeScalarView::HostMirror h_b_SPEEDPARAM;
  EdgeQuadView a_SHAPENDERIV;
  EdgeQuadView b_SHAPENDERIV;
  typename EdgeQuadView::HostMirror h_a_SHAPENDERIV;
  typename EdgeQuadView::HostMirror h_b_SHAPENDERIV;

  RealView interface_relax_param;
  RealView::HostMirror h_interface_relax_param;

  // TODO change out the design to fit closer to specfem?
  // TODO include UseSIMD=true case.

  template <int medium, bool on_device>
  KOKKOS_FUNCTION
      specfem::edge::loose::positions<DimensionType, QuadratureType, false>
      get_positions(const int edge_index,
                    specfem::compute::assembly &assembly) {
    specfem::edge::loose::positions<DimensionType, QuadratureType, false>
        edge_positions;
    int ispec;
    specfem::enums::edge::type edge;
    if constexpr (medium == 1) {
      if constexpr (on_device == true) {
        ispec = medium1_index_mapping(edge_index);
        edge = medium1_edge_type(edge_index);
      } else {
        ispec = h_medium1_index_mapping(edge_index);
        edge = h_medium1_edge_type(edge_index);
      }
    } else if constexpr (medium == 2) {
      if constexpr (on_device == true) {
        ispec = medium2_index_mapping(edge_index);
        edge = medium2_edge_type(edge_index);
      } else {
        ispec = h_medium2_index_mapping(edge_index);
        edge = h_medium2_edge_type(edge_index);
      }
    } else {
      static_assert(false, "Medium can only be 1 or 2!");
    }
    int ix, iz;
#pragma unroll
    for (int igll = 0; igll < NGLL_EDGE; igll++) {
      point_from_edge<NGLL_EDGE>(iz, ix, edge, igll);
      edge_positions.x[igll] = assembly.mesh.points.coord(0, ispec, iz, ix);
      edge_positions.z[igll] = assembly.mesh.points.coord(1, ispec, iz, ix);
    }
    return edge_positions;
  }

  //   template<int medium, bool on_device,
  //           specfem::element::medium_tag MediumType, bool StoreDisplacement,
  //           bool StoreVelocity, bool StoreAcceleration, bool StoreMassMatrix,
  //           bool UseSIMD>
  //   KOKKOS_FUNCTION
  //   void load_field(
  //         const int edge_index, specfem::compute::assembly& assembly,
  //          specfem::point::field<DimensionType, MediumType,
  //          StoreDisplacement, StoreVelocity, StoreAcceleration,
  //          StoreMassMatrix, UseSIMD>* point_arr
  //         ){
  //     int ispec;
  //     specfem::enums::edge::type edge;
  //     if constexpr(medium == 1){
  //       static_assert(MediumType == MediumTag1, "Attemping access for medium
  //       1 with an incompatible point::field"); if constexpr(on_device ==
  //       true){
  //         ispec = medium1_index_mapping(edge_index);
  //         edge = medium1_edge_type(edge_index);
  //       }else{
  //         ispec = h_medium1_index_mapping(edge_index);
  //         edge = h_medium1_edge_type(edge_index);
  //       }
  //     }else if constexpr(medium == 2){
  //       static_assert(MediumType == MediumTag2, "Attemping access for medium
  //       2 with an incompatible point::field"); if constexpr(on_device ==
  //       true){
  //         ispec = medium2_index_mapping(edge_index);
  //         edge = medium2_edge_type(edge_index);
  //       }else{
  //         ispec = h_medium2_index_mapping(edge_index);
  //         edge = h_medium2_edge_type(edge_index);
  //       }
  //     }else{
  //       static_assert(false,"Medium can only be 1 or 2!");
  //     }
  //     int ix, iz;
  // #pragma unroll
  //     for(int igll = 0; igll < NGLL_EDGE; igll++){
  //       point_from_edge<NGLL_EDGE>(iz,ix,edge,igll);
  //       specfem::point::index<DimensionType> index(ispec,iz,ix);
  //       if constexpr(on_device == true){
  //         specfem::compute::load_on_device(index,assembly.fields.forward,point_arr[igll]);
  //       }else{
  //         specfem::compute::load_on_host(index,assembly.fields.forward,point_arr[igll]);
  //       }
  //     }
  //   }

#define impl_field_access_arg_types                                            \
  const int edge_index, specfem::compute::assembly &assembly,                  \
      specfem::point::field<DimensionType, MediumType, StoreDisplacement,      \
                            StoreVelocity, StoreAcceleration, StoreMassMatrix, \
                            UseSIMD> *point_arr
#define impl_field_access_args edge_index, assembly, point_arr
#define impl_field_access_template                                             \
  int medium, bool on_device, specfem::element::medium_tag MediumType,         \
      bool StoreDisplacement, bool StoreVelocity, bool StoreAcceleration,      \
      bool StoreMassMatrix, bool UseSIMD

  template <impl_field_access_template, typename on_host_call,
            typename on_device_call>
  KOKKOS_INLINE_FUNCTION void impl_field_access(impl_field_access_arg_types,
                                                on_host_call on_host_func,
                                                on_device_call on_device_func) {
    int ispec;
    specfem::enums::edge::type edge;
    if constexpr (medium == 1) {
      static_assert(
          MediumType == MediumTag1,
          "Attemping access for medium 1 with an incompatible point::field");
      if constexpr (on_device == true) {
        ispec = medium1_index_mapping(edge_index);
        edge = medium1_edge_type(edge_index);
      } else {
        ispec = h_medium1_index_mapping(edge_index);
        edge = h_medium1_edge_type(edge_index);
      }
    } else if constexpr (medium == 2) {
      static_assert(
          MediumType == MediumTag2,
          "Attemping access for medium 2 with an incompatible point::field");
      if constexpr (on_device == true) {
        ispec = medium2_index_mapping(edge_index);
        edge = medium2_edge_type(edge_index);
      } else {
        ispec = h_medium2_index_mapping(edge_index);
        edge = h_medium2_edge_type(edge_index);
      }
    } else {
      static_assert(false, "Medium can only be 1 or 2!");
    }
    int ix, iz;
#pragma unroll
    for (int igll = 0; igll < NGLL_EDGE; igll++) {
      point_from_edge<NGLL_EDGE>(iz, ix, edge, igll);
      specfem::point::index<DimensionType> index(ispec, iz, ix);
      if constexpr (on_device == true) {
        on_device_func(index, point_arr[igll]);
      } else {
        on_host_func(index, point_arr[igll]);
      }
    }
  }

#define impl_field_access_lambdify_index_field_point(func)                     \
  [&](specfem::point::index<DimensionType> index,                              \
      specfem::point::field<DimensionType, MediumType, StoreDisplacement,      \
                            StoreVelocity, StoreAcceleration, StoreMassMatrix, \
                            UseSIMD> &point) {                                 \
    func(index, assembly.fields.forward, point);                               \
  }
#define impl_field_access_lambdify_index_point_field(func)                     \
  [&](specfem::point::index<DimensionType> index,                              \
      specfem::point::field<DimensionType, MediumType, StoreDisplacement,      \
                            StoreVelocity, StoreAcceleration, StoreMassMatrix, \
                            UseSIMD> &point) {                                 \
    func(index, point, assembly.fields.forward);                               \
  }

  template <impl_field_access_template>
  KOKKOS_FUNCTION void store_field(impl_field_access_arg_types) {
    impl_field_access<medium, on_device>(
        impl_field_access_args,
        impl_field_access_lambdify_index_point_field(
            specfem::compute::store_on_host),
        impl_field_access_lambdify_index_point_field(
            specfem::compute::store_on_device));
  }
  template <impl_field_access_template>
  KOKKOS_FUNCTION void atomic_add_to_field(impl_field_access_arg_types) {
    impl_field_access<medium, on_device>(
        impl_field_access_args,
        impl_field_access_lambdify_index_point_field(
            specfem::compute::atomic_add_on_host),
        impl_field_access_lambdify_index_point_field(
            specfem::compute::atomic_add_on_device));
  }
  template <impl_field_access_template>
  KOKKOS_FUNCTION void load_field(impl_field_access_arg_types) {
    impl_field_access<medium, on_device>(
        impl_field_access_args,
        impl_field_access_lambdify_index_field_point(
            specfem::compute::load_on_host),
        impl_field_access_lambdify_index_field_point(
            specfem::compute::load_on_device));
  }
#undef impl_field_access_arg_types
#undef impl_field_access_args
#undef impl_field_access_lambdify_index_field_point
#undef impl_field_access_lambdify_index_point_field
#undef impl_field_access_template

  // these called before the intersection loop to store intermediate values

  template <int medium, bool on_device>
  void compute_edge_intermediates(specfem::compute::assembly &assembly) {
    if constexpr (on_device == true) {
      static_assert(false, "on_device == true not yet supported.");
    }
    if constexpr (medium != 1 && medium != 2) {
      static_assert(false, "Medium can only be 1 or 2!");
    }
    for (int i = 0; i <
                    [&] {
                      if constexpr (medium == 1) {
                        return num_medium1_edges;
                      } else {
                        return num_medium2_edges;
                      }
                    }();
         i++) {
      FluxScheme::template kernel<DimensionType, MediumTag1, MediumTag2,
                                  QuadratureType>::
          template compute_edge_intermediate<medium, on_device>(i, assembly,
                                                                *this);
    }
  }

  template <bool on_device, typename internal_call>
  void foreach_interface(internal_call kernel) {
    if constexpr (on_device == true) {
      static_assert(false, "on_device == true not yet supported.");
    } else {
      for (int interface = 0; interface < num_interfaces; interface++) {
        kernel(interface);
      }
    }
  }
};

} // namespace loose
} // namespace compute
} // namespace specfem
