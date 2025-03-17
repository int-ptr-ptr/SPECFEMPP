#pragma once

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"
#include "compute/assembly/assembly.hpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/sources/sources.hpp"
#include "decl/Kokkos_Declare_SERIAL.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/field.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include <cmath>
#include <utility>
#include <vector>

using tb_point_tuple =
    std::tuple<specfem::point::index<specfem::dimension::type::dim2>, type_real,
               type_real, type_real, type_real, type_real, type_real,
               type_real>;

template <specfem::element::medium_tag MediumType>
std::vector<tb_point_tuple> static tb_points(
    specfem::compute::assembly &assembly, bool include_top,
    bool include_bottom);

template <specfem::element::medium_tag MediumType>
KOKKOS_INLINE_FUNCTION void
get_field_derivs(type_real *dfdxi, type_real *dfdga, int ispec, int iz, int ix,
                 specfem::compute::assembly &assembly) {
  const int ncomp = specfem::element::attributes<specfem::dimension::type::dim2,
                                                 MediumType>::components();

  specfem::point::field<specfem::dimension::type::dim2, MediumType, true, false,
                        false, false, false>
      disp;

#pragma unroll
  for (int icomp = 0; icomp < ncomp; icomp++) {
    dfdxi[icomp] = 0;
    dfdga[icomp] = 0;
  }

  for (int k = 0; k < assembly.mesh.ngllx; k++) {
    specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz, k);
    specfem::compute::load_on_host(index, assembly.fields.forward, disp);
#pragma unroll
    for (int icomp = 0; icomp < ncomp; icomp++) {
      dfdxi[icomp] += assembly.mesh.quadratures.gll.h_hprime(ix, k) *
                      disp.displacement(icomp);
    }
  }
  for (int k = 0; k < assembly.mesh.ngllz; k++) {
    specfem::point::index<specfem::dimension::type::dim2> index(ispec, k, ix);
    specfem::compute::load_on_host(index, assembly.fields.forward, disp);
#pragma unroll
    for (int icomp = 0; icomp < ncomp; icomp++) {
      dfdga[icomp] += assembly.mesh.quadratures.gll.h_hprime(iz, k) *
                      disp.displacement(icomp);
    }
  }
}

namespace _util {
namespace sourceboundary {

template <specfem::element::medium_tag MediumType> class kernel {
public:
  kernel(specfem::compute::assembly &assembly, bool dotop, bool dobottom)
      : assembly(assembly) {
    const auto pts = tb_points<MediumType>(assembly, dotop, dobottom);
    num_pts = pts.size();
    inds = decltype(inds)("_util::sourceboundary::kernel::inds", num_pts);
    coord = decltype(coord)("_util::sourceboundary::kernel::coord", num_pts);
    norm = decltype(norm)("_util::sourceboundary::kernel::norm", num_pts);
    norm_contra = decltype(norm_contra)(
        "_util::sourceboundary::kernel::norm_contra", num_pts);
    jac = decltype(jac)("_util::sourceboundary::kernel::jac", num_pts);
    velstore =
        decltype(velstore)("_util::sourceboundary::kernel::velstore", num_pts);

    h_inds = Kokkos::create_mirror_view(inds);
    h_coord = Kokkos::create_mirror_view(coord);
    h_jac = Kokkos::create_mirror_view(jac);
    h_velstore = Kokkos::create_mirror_view(velstore);
    h_norm = Kokkos::create_mirror_view(norm);
    h_norm_contra = Kokkos::create_mirror_view(norm_contra);

    for (int i = 0; i < num_pts; i++) {
      h_inds(i) = std::get<0>(pts[i]);
      h_coord(i, 0) = std::get<1>(pts[i]);
      h_coord(i, 1) = std::get<2>(pts[i]);
      h_norm(i, 0) = std::get<3>(pts[i]);
      h_norm(i, 1) = std::get<4>(pts[i]);
      h_norm_contra(i, 0) = std::get<5>(pts[i]);
      h_norm_contra(i, 1) = std::get<6>(pts[i]);
      h_jac(i) = std::get<7>(pts[i]);
    }

    Kokkos::deep_copy(inds, h_inds);
    Kokkos::deep_copy(coord, h_coord);
    Kokkos::deep_copy(jac, h_jac);
    Kokkos::deep_copy(norm, h_norm);
    Kokkos::deep_copy(norm_contra, h_norm_contra);

    std::cout << "Injector kernel created with " << num_pts << " nodes."
              << std::endl;
  }

  void absorb(
      // type_real kx, type_real kz, type_real phase,
      //                      type_real amplitude
  ) {
    Kokkos::parallel_for(
        num_pts, KOKKOS_LAMBDA(const int iworker) {
          specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                false, false, true, false, false>
              accel;
          const auto ind = inds(iworker);
          int ix, iz;
          const type_real nx = norm(iworker, 0);
          const type_real nz = norm(iworker, 1);
          const type_real nxi = norm_contra(iworker, 0);
          const type_real nga = norm_contra(iworker, 1);
          type_real dfdxi[ncomp];
          type_real dfdga[ncomp];
          get_field_derivs<MediumType>(dfdxi, dfdga, ind.ispec, ind.iz, ind.ix,
                                       assembly);

          // int (n . grad phi) dS
          // phi = amp cos(min(0, k . coord - phase))
          // const type_real waveparam =
          //     kx * coord(iworker, 0) + kz * coord(iworker, 1) - phase;

          if constexpr (MediumType == specfem::element::medium_tag::acoustic) {
            specfem::point::properties<
                specfem::dimension::type::dim2, MediumType,
                specfem::element::property_tag::isotropic, false>
                ppt;
            specfem::compute::load_on_device(ind, assembly.properties, ppt);
            const type_real c = std::sqrt(ppt.kappa());
            type_real dfdn = dfdxi[0] * nxi + dfdga[0] * nga;
            accel.acceleration(0) =
                -(velstore(iworker, 0) / c - dfdn) * 0.5 * jac(iworker);
            // accel.acceleration(0) += (kx * nx + kz * nz) * amplitude *
            //                          std::sin(waveparam < 0 ? waveparam : 0)
            //                          * jac(iworker);
          } else if constexpr (MediumType ==
                               specfem::element::medium_tag::elastic) {

            // for now, we just copy code from
            // include/boundary_conditions/stacey/stacey.tpp
            specfem::point::properties<
                specfem::dimension::type::dim2, MediumType,
                specfem::element::property_tag::isotropic, false>
                ppt;
            specfem::compute::load_on_device(ind, assembly.properties, ppt);

            // type_real dfdn = dfdxi[0] * nx + dfdga[0] * nz;
            type_real vn =
                velstore(iworker, 0) * nx + velstore(iworker, 1) * nz;

            type_real factor[2];

            for (int icomp = 0; icomp < 2; ++icomp) {
              factor[icomp] = ((vn * norm(iworker, icomp)) *
                               (ppt.rho_vp() - ppt.rho_vs())) +
                              velstore(iworker, icomp) * ppt.rho_vs();
            }

            accel.acceleration(0) =
                static_cast<type_real>(-1.0) * factor[0] * jac(iworker);
            accel.acceleration(1) =
                static_cast<type_real>(-1.0) * factor[1] * jac(iworker);
          }

          specfem::compute::atomic_add_on_device(ind, accel,
                                                 assembly.fields.forward);
        });
    Kokkos::fence();
  }

  void store_velocity(type_real dt_inc) {
    Kokkos::parallel_for(
        num_pts, KOKKOS_LAMBDA(const int iworker) {
          specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                false, true, true, false, false>
              vel;
          const auto ind = inds(iworker);
          specfem::compute::load_on_device(ind, assembly.fields.forward, vel);

#pragma unroll
          for (int icomp = 0; icomp < ncomp; icomp++) {
            velstore(iworker, icomp) =
                vel.velocity(icomp) + dt_inc * vel.acceleration(icomp);
          }
        });
    Kokkos::fence();
  }

private:
  static constexpr int ncomp =
      specfem::element::attributes<specfem::dimension::type::dim2,
                                   MediumType>::components();
  specfem::compute::assembly &assembly;
  int num_pts;
  using IndsType = specfem::kokkos::DeviceView1d<
      specfem::point::index<specfem::dimension::type::dim2> >;
  using CoordsType = Kokkos::View<type_real *[2], Kokkos::LayoutRight,
                                  specfem::kokkos::DevMemSpace>;
  using JacType = specfem::kokkos::DeviceView1d<type_real>;
  using VelStoreType = Kokkos::View<type_real *[ncomp], Kokkos::LayoutRight,
                                    specfem::kokkos::DevMemSpace>;

  IndsType inds;
  IndsType::HostMirror h_inds;
  CoordsType coord;
  CoordsType::HostMirror h_coord;
  CoordsType norm;
  CoordsType::HostMirror h_norm;
  CoordsType norm_contra;
  CoordsType::HostMirror h_norm_contra;
  JacType jac;
  JacType::HostMirror h_jac;
  VelStoreType velstore;
  typename VelStoreType::HostMirror h_velstore;
};

} // namespace sourceboundary
} // namespace _util

template <specfem::element::medium_tag MediumType>

std::vector<tb_point_tuple> static tb_points(
    specfem::compute::assembly &assembly, bool include_top,
    bool include_bottom) {
  // get min,max, and epsilon for coordinate.
  const auto &coord = assembly.mesh.control_nodes.h_coord;
  type_real ymin = coord(0, 0, 0);
  type_real ymax = ymin;
  for (int ispec = 1; ispec < coord.extent(1); ispec++) {
    for (int inod = 1; inod < coord.extent(2); inod++) {
      const type_real y = coord(1, ispec, inod);
      ymin = std::min(ymin, y);
      ymax = std::max(ymax, y);
    }
  }
  type_real epsy = (ymax - ymin) * 1e-5;

  // append to vec all pts on bdry: <index, x, z, J*w @ point>
  std::vector<tb_point_tuple> vec;
  specfem::point::index<specfem::dimension::type::dim2> index(0, 0, 0);
  specfem::point::partial_derivatives<specfem::dimension::type::dim2, false,
                                      false>
      ppd;
  for (int &ispec = index.ispec; ispec < coord.extent(1); ispec++) {
    bool onbdry[9];
    if (assembly.element_types.medium_tags(ispec) != MediumType) {
      continue;
    }
    for (int inod = 0; inod < coord.extent(2); inod++) {
      const type_real y = coord(1, ispec, inod);
      onbdry[inod] = (y - ymin < epsy) && include_bottom ||
                     (ymax - y < epsy) && include_top;
    }

#define mesh_x                                                                 \
  (assembly.mesh.points.h_coord(0, index.ispec, index.iz, index.ix))
#define mesh_y                                                                 \
  (assembly.mesh.points.h_coord(1, index.ispec, index.iz, index.ix))
    if (onbdry[0] && onbdry[1]) {
      // bottom
      index.iz = 0;
      index.ix = 0;
      for (int &ix = index.ix; ix < assembly.mesh.ngllx; ix++) {
        specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                       ppd);
        const type_real &w = assembly.mesh.quadratures.gll.h_weights(ix);
        const type_real det =
            1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
        const type_real J =
            det * std::sqrt(ppd.gammax * ppd.gammax + ppd.gammaz * ppd.gammaz);
        const type_real nx = -ppd.gammax;
        const type_real nz = -ppd.gammaz;
        const type_real contra_nxi = nx * ppd.xix + nz * ppd.xiz;
        const type_real contra_nga = nx * ppd.gammax + nz * ppd.gammaz;
        const type_real mag = std::sqrt(nx * nx + nz * nz);
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, nx / mag, nz / mag,
                                      contra_nxi / mag, contra_nga / mag,
                                      J * w));
      }
    }
    if (onbdry[2] && onbdry[1]) {
      // right
      index.iz = 0;
      index.ix = assembly.mesh.ngllx - 1;
      for (int &iz = index.iz; iz < assembly.mesh.ngllz; iz++) {
        specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                       ppd);
        const type_real &w = assembly.mesh.quadratures.gll.h_weights(iz);
        const type_real det =
            1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
        const type_real J =
            det * std::sqrt(ppd.xix * ppd.xix + ppd.xiz * ppd.xiz);
        const type_real nx = ppd.xix;
        const type_real nz = ppd.xiz;
        const type_real contra_nxi = nx * ppd.xix + nz * ppd.xiz;
        const type_real contra_nga = nx * ppd.gammax + nz * ppd.gammaz;
        const type_real mag = std::sqrt(nx * nx + nz * nz);
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, nx / mag, nz / mag,
                                      contra_nxi / mag, contra_nga / mag,
                                      J * w));
      }
    }
    if (onbdry[2] && onbdry[3]) {
      // top
      index.iz = assembly.mesh.ngllz - 1;
      index.ix = 0;
      for (int &ix = index.ix; ix < assembly.mesh.ngllx; ix++) {
        specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                       ppd);
        const type_real &w = assembly.mesh.quadratures.gll.h_weights(ix);
        const type_real det =
            1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
        const type_real J =
            det * std::sqrt(ppd.gammax * ppd.gammax + ppd.gammaz * ppd.gammaz);
        const type_real nx = ppd.gammax;
        const type_real nz = ppd.gammaz;
        const type_real contra_nxi = nx * ppd.xix + nz * ppd.xiz;
        const type_real contra_nga = nx * ppd.gammax + nz * ppd.gammaz;
        const type_real mag = std::sqrt(nx * nx + nz * nz);
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, nx / mag, nz / mag,
                                      contra_nxi / mag, contra_nga / mag,
                                      J * w));
      }
    }
    if (onbdry[0] && onbdry[3]) {
      // left
      index.iz = 0;
      index.ix = 0;
      for (int &iz = index.iz; iz < assembly.mesh.ngllz; iz++) {
        specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                       ppd);
        const type_real &w = assembly.mesh.quadratures.gll.h_weights(iz);
        const type_real det =
            1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
        const type_real J =
            det * std::sqrt(ppd.xix * ppd.xix + ppd.xiz * ppd.xiz);
        const type_real nx = -ppd.xix;
        const type_real nz = -ppd.xiz;
        const type_real contra_nxi = nx * ppd.xix + nz * ppd.xiz;
        const type_real contra_nga = nx * ppd.gammax + nz * ppd.gammaz;
        const type_real mag = std::sqrt(nx * nx + nz * nz);
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, nx / mag, nz / mag,
                                      contra_nxi / mag, contra_nga / mag,
                                      J * w));
      }
    }
  }
  return vec;
}
