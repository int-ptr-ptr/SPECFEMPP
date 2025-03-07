#pragma once

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"
#include "compute/assembly/assembly.hpp"
#include "compute/fields/simulation_field.hpp"
#include "compute/sources/sources.hpp"
#include "enumerations/dimension.hpp"
#include "kokkos_abstractions.h"
#include "point/field.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include <cmath>
#include <utility>
#include <vector>

template <specfem::element::medium_tag MediumType>
std::vector<std::tuple<
    specfem::point::index<specfem::dimension::type::dim2>, type_real, type_real,
    type_real> > static bottom_points(specfem::compute::assembly &assembly);

namespace _util {
namespace sourceboundary {

template <specfem::element::medium_tag MediumType> class kernel {
public:
  kernel(specfem::compute::assembly &assembly) : assembly(assembly) {
    const auto pts = bottom_points<MediumType>(assembly);
    num_pts = pts.size();
    inds = decltype(inds)("_util::sourceboundary::kernel::inds", num_pts);
    coord = decltype(coord)("_util::sourceboundary::kernel::coord", num_pts);
    jac = decltype(jac)("_util::sourceboundary::kernel::jac", num_pts);

    h_inds = Kokkos::create_mirror_view(inds);
    h_coord = Kokkos::create_mirror_view(coord);
    h_jac = Kokkos::create_mirror_view(jac);

    for (int i = 0; i < num_pts; i++) {
      h_inds(i) = std::get<0>(pts[i]);
      h_coord(i, 0) = std::get<1>(pts[i]);
      h_coord(i, 1) = std::get<2>(pts[i]);
      h_jac(i) = std::get<3>(pts[i]);
    }

    nx = 0;
    nz = -1;

    Kokkos::deep_copy(inds, h_inds);
    Kokkos::deep_copy(coord, h_coord);
    Kokkos::deep_copy(jac, h_jac);

    std::cout << "Injector kernel created with " << num_pts << " nodes."
              << std::endl;
  }

  void force_planar_wave(type_real kx, type_real kz, type_real phase,
                         type_real amplitude) {
    Kokkos::parallel_for(
        num_pts, KOKKOS_LAMBDA(const int iworker) {
          specfem::point::field<specfem::dimension::type::dim2, MediumType,
                                false, false, true, false, false>
              accel;
          // int (n . grad phi) dS
          // phi = amp cos(min(0, k . coord - phase))
          const type_real waveparam =
              kx * coord(iworker, 0) + kz * coord(iworker, 1) - phase;
          accel.acceleration(0) = (kx * nx + kz * nz) * amplitude *
                                  std::sin(waveparam < 0 ? waveparam : 0) *
                                  jac(iworker);
          const auto ind = inds(iworker);

          // specfem::point::field<specfem::dimension::type::dim2, MediumType,
          //                       true, true, true, false, false>
          //     read;
          // specfem::compute::load_on_device(ind, assembly.fields.forward,
          // read); const auto dis = read.displacement(0); const auto vel =
          // read.velocity(0); const auto acc = read.acceleration(0); const auto
          // locx = coord(iworker,0); const auto locz = coord(iworker,1); const
          // auto jac_ = jac(iworker); if(std::abs(dis) > 100){
          //   read.displacement(0) = read.displacement(0);
          // }
          specfem::compute::atomic_add_on_device(ind, accel,
                                                 assembly.fields.forward);
        });
  }

private:
  specfem::compute::assembly &assembly;
  type_real nx;
  type_real nz;
  int num_pts;
  using IndsType = specfem::kokkos::HostView1d<
      specfem::point::index<specfem::dimension::type::dim2> >;
  using CoordsType = specfem::kokkos::HostView1d<type_real[2]>;
  using JacType = specfem::kokkos::HostView1d<type_real>;

  IndsType inds;
  IndsType::HostMirror h_inds;
  CoordsType coord;
  CoordsType::HostMirror h_coord;
  JacType jac;
  JacType::HostMirror h_jac;
};

} // namespace sourceboundary
} // namespace _util

template <specfem::element::medium_tag MediumType>

std::vector<std::tuple<
    specfem::point::index<specfem::dimension::type::dim2>, type_real, type_real,
    type_real> > static bottom_points(specfem::compute::assembly &assembly) {
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
  std::vector<std::tuple<specfem::point::index<specfem::dimension::type::dim2>,
                         type_real, type_real, type_real> >
      vec;
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
      onbdry[inod] = (y - ymin < epsy);
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
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, J * w));
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
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, J * w));
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
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, J * w));
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
        vec.push_back(std::make_tuple(index, mesh_x, mesh_y, J * w));
      }
    }
  }
  return vec;
}
