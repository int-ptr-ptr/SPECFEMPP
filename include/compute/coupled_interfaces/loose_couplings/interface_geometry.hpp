#pragma once

#include "interface_container.hpp"

namespace specfem {
namespace compute {
namespace loose {

/*
store edge normals (outward facing) on the given medium index (1 or 2).
if normalize, then the vectors are set to magnitude 1. otherwise, they have
length dS (the 1d (2d) Jacobian along the edge (surface))
*/
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType,
          int medium, bool normalize>
struct interface_normal_container;

/*
geometry variables take the same priors, so they should be computed at the same
time.
*/
template <int medium, bool on_device, bool UseSIMD, typename ContainerType>
KOKKOS_INLINE_FUNCTION void
compute_geometry(specfem::compute::assembly &assembly, ContainerType &container,
                 int edge_index) {
  int ispec;
  specfem::enums::edge::type edge;
  if constexpr (medium == 1) {
    if constexpr (on_device == true) {
      ispec = container.medium1_index_mapping(edge_index);
      edge = container.medium1_edge_type(edge_index);
    } else {
      ispec = container.h_medium1_index_mapping(edge_index);
      edge = container.h_medium1_edge_type(edge_index);
    }
  } else if constexpr (medium == 2) {
    if constexpr (on_device == true) {
      ispec = container.medium2_index_mapping(edge_index);
      edge = container.medium2_edge_type(edge_index);
    } else {
      ispec = container.h_medium2_index_mapping(edge_index);
      edge = container.h_medium2_edge_type(edge_index);
    }
  } else {
    static_assert(false, "Medium can only be 1 or 2!");
  }

  using PointPartialDerivativesType =
      specfem::point::partial_derivatives<specfem::dimension::type::dim2, true,
                                          UseSIMD>;
  int ix, iz;
  for (int i = 0; i < ContainerType::NGLL_EDGE; i++) {
    specfem::compute::loose::point_from_edge<ContainerType::NGLL_EDGE>(iz, ix,
                                                                       edge, i);
    specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz, ix);

    PointPartialDerivativesType ppd;
    specfem::compute::load_on_host(index, assembly.partial_derivatives, ppd);
    // type_real dvdxi = assembly->mesh.quadratures.h_hprime(ix,ix);
    // type_real dvdga = assembly->mesh.quadratures.h_hprime(iz,iz);
    // type_real dvdx = dvdxi * ppd.xix +
    //                  dvdga * ppd.gammax;
    // type_real dvdz = dvdxi * ppd.xiz +
    //                  dvdga * ppd.gammaz;
    type_real det = 1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
    type_real nx, nz;
    switch (edge) {
    case specfem::enums::edge::type::RIGHT:
      nx = ppd.xix;
      nz = ppd.xiz;
      break;
    case specfem::enums::edge::type::TOP:
      nx = ppd.gammax;
      nz = ppd.gammaz;
      break;
    case specfem::enums::edge::type::LEFT:
      nx = -ppd.xix;
      nz = -ppd.xiz;
      break;
    case specfem::enums::edge::type::BOTTOM:
      nx = -ppd.gammax;
      nz = -ppd.gammaz;
      break;
    }
    if constexpr (std::is_base_of<
                      interface_normal_container<
                          ContainerType::dim_type, ContainerType::medium1_type,
                          ContainerType::medium2_type,
                          typename ContainerType::EdgeQuadrature, medium,
                          false>,
                      ContainerType>()) {
      if constexpr (medium == 1) {
        if constexpr (on_device == true) {
          container.medium1_edge_normal(edge_index, i, 0) = nx * det;
          container.medium1_edge_normal(edge_index, i, 1) = nz * det;
        } else {
          container.h_medium1_edge_normal(edge_index, i, 0) = nx * det;
          container.h_medium1_edge_normal(edge_index, i, 1) = nz * det;
        }
      } else if constexpr (medium == 2) {
        if constexpr (on_device == true) {
          container.medium2_edge_normal(edge_index, i, 0) = nx * det;
          container.medium2_edge_normal(edge_index, i, 1) = nz * det;
        } else {
          container.h_medium2_edge_normal(edge_index, i, 0) = nx * det;
          container.h_medium2_edge_normal(edge_index, i, 1) = nz * det;
        }
      }
    }
    if constexpr (std::is_base_of<
                      interface_normal_container<
                          ContainerType::dim_type, ContainerType::medium1_type,
                          ContainerType::medium2_type,
                          typename ContainerType::EdgeQuadrature, medium, true>,
                      ContainerType>()) {
      type_real inv_mag = 1 / sqrt(nx * nx + nz * nz);
      if constexpr (medium == 1) {
        if constexpr (on_device == true) {
          container.medium1_edge_normal(edge_index, i, 0) = nx * inv_mag;
          container.medium1_edge_normal(edge_index, i, 1) = nz * inv_mag;
        } else {
          container.h_medium1_edge_normal(edge_index, i, 0) = nx * inv_mag;
          container.h_medium1_edge_normal(edge_index, i, 1) = nz * inv_mag;
        }
      } else if constexpr (medium == 2) {
        if constexpr (on_device == true) {
          container.medium2_edge_normal(edge_index, i, 0) = nx * inv_mag;
          container.medium2_edge_normal(edge_index, i, 1) = nz * inv_mag;
        } else {
          container.h_medium2_edge_normal(edge_index, i, 0) = nx * inv_mag;
          container.h_medium2_edge_normal(edge_index, i, 1) = nz * inv_mag;
        }
      }
    }
    // e.data[EDGEIND_NX][i] = nx;
    // e.data[EDGEIND_NZ][i] = nz;
    // e.data[EDGEIND_DET][i] = det;
    // e.data[EDGEIND_DS][i] = sqrt(nx * nx + nz * nz) * det; // dS
    // for (int ishape = 0; ishape < e.ngll; ishape++) {
    //   int ixshape, izshape;
    //   point_from_id(ixshape, izshape, e.parent.bdry, ishape);
    //   // v = L_{ixshape}(x) L_{izshape}(z);
    //   // hprime(i,j) = L_j'(t_i)
    //   type_real dvdxi =
    //       assembly->mesh.quadratures.gll.h_hprime(ix, ixshape) *
    //       (iz == izshape);
    //   type_real dvdga =
    //       assembly->mesh.quadratures.gll.h_hprime(iz, izshape) *
    //       (ix == ixshape);
    //   type_real dvdx = dvdxi * ppd.xix + dvdga * ppd.gammax;
    //   type_real dvdz = dvdxi * ppd.xiz + dvdga * ppd.gammaz;
    //   e.data[EDGEIND_SHAPENDERIV + ishape][i] =
    //       (dvdx * nx + dvdz * nz) * det;
    // }
  }
}

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType,
          bool normalize>
struct interface_normal_container<DimensionType, MediumTag1, MediumTag2,
                                  QuadratureType, 1, normalize> {
public:
  EdgeVectorView<DimensionType, QuadratureType> medium1_edge_normal;
  typename EdgeVectorView<DimensionType, QuadratureType>::HostMirror
      h_medium1_edge_normal;
  interface_normal_container() = default;

  // void operator=(const
  // interface_normal_container<DimensionType,MediumTag1,MediumTag2,QuadratureType,1,normalize>
  // &rhs) {
  //   this->medium1_edge_normal = rhs.medium1_edge_normal;
  //   this->h_medium1_edge_normal = rhs.h_medium1_edge_normal;
  // }

protected:
  interface_normal_container(int num_edges)
      : medium1_edge_normal("specfem::compute::loose::interface_normal_"
                            "container.medium1_edge_normal",
                            num_edges),
        h_medium1_edge_normal(Kokkos::create_mirror_view(medium1_edge_normal)) {
  }
};

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag1,
          specfem::element::medium_tag MediumTag2, typename QuadratureType,
          bool normalize>
struct interface_normal_container<DimensionType, MediumTag1, MediumTag2,
                                  QuadratureType, 2, normalize> {
public:
  EdgeVectorView<DimensionType, QuadratureType> medium2_edge_normal;
  typename EdgeVectorView<DimensionType, QuadratureType>::HostMirror
      h_medium2_edge_normal;
  interface_normal_container() = default;
  // void operator=(const
  // interface_normal_container<DimensionType,MediumTag1,MediumTag2,QuadratureType,2,normalize>
  // &rhs) {
  //   this->medium2_edge_normal = rhs.medium2_edge_normal;
  //   this->h_medium2_edge_normal = rhs.h_medium2_edge_normal;
  // }

protected:
  interface_normal_container(int num_edges)
      : medium2_edge_normal("specfem::compute::loose::interface_normal_"
                            "container.medium2_edge_normal",
                            num_edges),
        h_medium2_edge_normal(Kokkos::create_mirror_view(medium2_edge_normal)) {
  }
};

} // namespace loose
} // namespace compute
} // namespace specfem
