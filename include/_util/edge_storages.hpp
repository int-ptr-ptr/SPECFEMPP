#ifndef __UTIL_EDGE_STORAGES_HPP_
#define __UTIL_EDGE_STORAGES_HPP_

namespace _util {
namespace edge_manager {
struct edge;
template <int ngllcapacity> struct edge_intersection;

template <typename edgequad, int datacapacity> struct edge_storage;

template <typename edgequad, int datacapacity>
edge_storage<edgequad, datacapacity> *edge_storage_instance;
} // namespace edge_manager
} // namespace _util
#include "compute/coupled_interfaces/loose_couplings/interface_container.hpp"
#include "compute/coupled_interfaces/loose_couplings/symmetric_flux_container.hpp"
#include "compute/coupled_interfaces/loose_couplings/traction_continuity_container.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include <functional>
#include <vector>

namespace _util {
namespace edge_manager {

// different quadrature rules? consider GLL -> GL for intersections, since
// endpoints aren't needed
struct quadrature_rule {
  int nquad;
  std::vector<type_real> t; // knots
  std::vector<type_real> w; // weights
  std::vector<type_real> L; // Lagrange polynomial coeffs

  quadrature_rule(int nquad)
      : nquad(nquad), t(std::vector<type_real>(nquad)),
        w(std::vector<type_real>(nquad)),
        L(std::vector<type_real>(nquad * nquad)) {}
  quadrature_rule(quadrature_rule &other)
      : nquad(other.nquad), t(std::vector<type_real>(other.t)),
        w(std::vector<type_real>(other.w)), L(std::vector<type_real>(other.L)) {
  }

  type_real integrate(const type_real *f);
  type_real deriv(const type_real *f, const type_real t);
  type_real interpolate(const type_real *f, const type_real t);

  template <int ngllcapacity>
  void sample_L(type_real buf[][ngllcapacity], const type_real *t_vals,
                const int t_size);
};

quadrature_rule gen_GLL(int ngll);

struct edge {
  int id;
  specfem::enums::edge::type bdry;
  specfem::element::medium_tag medium;
  edge() : id(-1), bdry(specfem::enums::edge::type::NONE) {}
  // edge(const edge& other) : id(other.id), bdry(other.bdry) {}
};

template <int ngllcapacity> struct edge_intersection {
  int id;
  int a_ref_ind;
  int b_ref_ind;
  type_real a_mortar_trans[ngllcapacity][ngllcapacity];
  type_real b_mortar_trans[ngllcapacity][ngllcapacity];
  type_real ds[ngllcapacity];
  type_real a_param_start, a_param_end;
  type_real b_param_start, b_param_end;
  int a_ngll, b_ngll, ngll;
  type_real relax_param;
  type_real quad_weight[ngllcapacity];
  edge_intersection()
      : ngll(0), a_ref_ind(-1), b_ref_ind(-1), a_ngll(0), b_ngll(0) {}

  type_real a_to_mortar(const int node_index, const type_real *quantity);
  type_real b_to_mortar(const int node_index, const type_real *quantity);
};

template <typename edgequad, int datacapacity> struct edge_storage {
public:
  static constexpr int ngll = edgequad::NGLL;
  edge_storage(std::vector<edge> edges, specfem::compute::assembly &assembly);

  void build_intersections_on_host();
  edge_intersection<ngll> get_intersection_on_host(int intersection);

  edge_intersection<ngll> load_intersection(const int intersectionID);
  void store_intersection(const int intersectionID,
                          const edge_intersection<ngll> &intersection);

  bool intersect(const int a, const int b,
                 edge_intersection<ngll> &intersection);

  void load_all_intersections();
  void store_all_intersections();

  int num_edges() const { return n_edges; }
  int num_intersections() const { return n_intersections; }
  specfem::kokkos::HostView2d<type_real> &get_intersection_data_on_host() {
    if (!intersection_data_built) {
      throw std::runtime_error("Attempting a get_intersection_data_on_host() "
                               "before the intersections array was built!");
    }
    return h_intersection_data;
  }
  void initialize_intersection_data(int capacity);

  bool interface_structs_initialized;
  specfem::compute::loose::interface_container<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::elastic, edgequad,
      specfem::coupled_interface::loose::flux::type::traction_continuity>
      acoustic_elastic_interface;
  specfem::compute::loose::interface_container<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::acoustic, edgequad,
      specfem::coupled_interface::loose::flux::type::symmetric_flux>
      acoustic_acoustic_interface;
  specfem::compute::loose::interface_container<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::medium_tag::elastic, edgequad,
      specfem::coupled_interface::loose::flux::type::symmetric_flux>
      elastic_elastic_interface;

private:
  int n_edges;
  std::vector<edge> edges;
  std::vector<edge> acoustic_edges;
  std::vector<edge> elastic_edges;
  std::vector<int> edge_sorted_inds;
  std::vector<specfem::element::medium_tag> edge_media;
  std::vector<int> intersection_edge_a;
  std::vector<int> intersection_edge_b;

  // specfem::kokkos::DeviceView1d<edge_data<ngll, datacapacity> >
  //     edge_data_container;
  // specfem::kokkos::HostView1d<edge_data<ngll, datacapacity> >
  //     h_edge_data_container;

  int n_intersections;
  // specfem::kokkos::DeviceView1d<edge_intersection<ngll> >
  //     intersection_container;
  // specfem::kokkos::HostView1d<edge_intersection<ngll> >
  //     h_intersection_container;
  std::vector<int> intersection_sorted_inds;

  bool intersection_data_built;
  specfem::kokkos::DeviceView2d<type_real> intersection_data;
  specfem::kokkos::HostView2d<type_real> h_intersection_data;

  specfem::compute::assembly &assembly;
};

} // namespace edge_manager
} // namespace _util

#include "_util/edge_storages.tpp"
#endif
