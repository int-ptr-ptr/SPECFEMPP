#ifndef __UTIL_EDGE_STORAGES_TPP_
#define __UTIL_EDGE_STORAGES_TPP_

#include "_util/edge_storages.hpp"
#include "edge/loose/edge_access.hpp"
#include <array>
#include <cmath>
#include <iostream>

namespace _util {
namespace edge_manager {


template<typename ViewType, typename HostMirrorType>
void init_views(ViewType& view, HostMirrorType& host_mirror, int size, std::string label){
  view = ViewType(label,size);
  host_mirror = Kokkos::create_mirror_view(view);
}

template <typename edgequad, int datacapacity>
edge_intersection<edge_storage<edgequad, datacapacity>::ngll> edge_storage<edgequad, datacapacity>::get_intersection_on_host(int intersection){
  return load_intersection(intersection);
}

template <typename edgequad, int datacapacity>
edge_storage<edgequad, datacapacity>::edge_storage(std::vector<edge> edges,
      specfem::compute::assembly& assembly)
    : n_edges(edges.size()), edges(edges),
      // edge_data_container(
      //     specfem::kokkos::DeviceView1d<edge_data<ngll, datacapacity> >(
      //         "_util::edge_manager::edge_storage::edge_data", n_edges)),
      // h_edge_data_container(Kokkos::create_mirror_view(edge_data_container)),
      edge_sorted_inds(n_edges),edge_media(n_edges),
      interface_structs_initialized(false),
      assembly(assembly) {
  //count media of edges
  for (int i = 0; i < n_edges; i++) {
    edges[i].medium = assembly.element_types.medium_tags(edges[i].id);
    edge_media[i] = edges[i].medium;
    if(edges[i].medium == specfem::element::medium_tag::acoustic){
      edge_sorted_inds[i] = acoustic_edges.size();
      acoustic_edges.push_back(edges[i]);
    }else{
      edge_sorted_inds[i] = elastic_edges.size();
      elastic_edges.push_back(edges[i]);
    }
  }
  build_intersections_on_host();

  edge_storage_instance<edgequad,datacapacity> = this;
}

template <typename edgequad, int datacapacity>
void edge_storage<edgequad, datacapacity>::initialize_intersection_data(int capacity){
  intersection_data = specfem::kokkos::DeviceView2d<type_real>(
              "_util::edge_manager::edge_storage::edge_data", n_intersections, capacity);
  h_intersection_data = Kokkos::create_mirror_view(intersection_data);

  intersection_data_built = true;
}


/**
 * @brief Checks whether or not these edges intersect. If they do, intersection
 * is modified to contain the intersection. That is, the parameter_start/end
 * fields in intersection& are set.
 *
 * @param a the first edge
 * @param b the second edge
 * @param intersection the struct to store the intersection into if there is a
 * nonzero intersection. This reference may update the parameter_start/end
 * variables even if there's no intersection.
 * @return true if a nonzero intersection occurs between these two edges
 * @return false if no nonzero intersection occurs between these two edges
 */
template <typename edgequad, int datacapacity>
template <int intersection_nquad>
bool edge_storage<edgequad, datacapacity>::intersect(const int a_edge_index,
               const int b_edge_index,
               edge_intersection<intersection_nquad> &intersection) {
#define intersect_eps 1e-3
#define intersect_eps2 (intersect_eps * intersect_eps)
  quadrature_rule gll = gen_GLL(ngll);
  quadrature_rule interquad = gen_GLL(intersection_nquad);

  //maybe do an AABB check for performance? the box can be precomputed per edge.


  type_real anodex[ngll];
  type_real anodez[ngll];
  type_real bnodex[ngll];
  type_real bnodez[ngll];
  int ix, iz;
  for(int igll = 0; igll < ngll; igll++){
    specfem::compute::loose::point_from_edge<ngll>(iz,ix,edges[a_edge_index].bdry,igll);
    anodex[igll] = assembly.mesh.points.coord(0, edges[a_edge_index].id, iz, ix);
    anodez[igll] = assembly.mesh.points.coord(1, edges[a_edge_index].id, iz, ix);
    specfem::compute::loose::point_from_edge<ngll>(iz,ix,edges[b_edge_index].bdry,igll);
    bnodex[igll] = assembly.mesh.points.coord(0, edges[b_edge_index].id, iz, ix);
    bnodez[igll] = assembly.mesh.points.coord(1, edges[b_edge_index].id, iz, ix);
    intersection.quad_weight[igll] = gll.w[igll];
  }

  //approximate edges as linear segments
  constexpr int subdivisions = 10;
  constexpr type_real h = 2.0/subdivisions;

  type_real ax[subdivisions+1];
  type_real bx[subdivisions+1];
  type_real az[subdivisions+1];
  type_real bz[subdivisions+1];

  for(int i = 0; i < subdivisions+1; i++){
    ax[i] = gll.interpolate(anodex, i*h-1.0);
    az[i] = gll.interpolate(anodez, i*h-1.0);
    bx[i] = gll.interpolate(bnodex, i*h-1.0);
    bz[i] = gll.interpolate(bnodez, i*h-1.0);
  }

  const auto line_intersection = [&gll](type_real ax0, type_real az0, type_real ax1, type_real az1,
                type_real bx0, type_real bz0, type_real bx1, type_real bz1,
                type_real& alow, type_real& ahigh, type_real& blow, type_real& bhigh) -> bool {
    type_real adx = ax1-ax0;
    type_real bdx = bx1-bx0;
    type_real adz = az1-az0;
    type_real bdz = bz1-bz0;
    type_real cross = adx * bdz - bdx * adz;
    type_real sin2 = cross * cross / ((adx * adx + adz * adz) * (bdx * bdx + bdz * bdz));
    if (sin2 > 1e-2){
      //not parallel, but there is an interval for which the lines are within eps distance
      //parameters where intersections occur: use cramer's rule
      type_real c1 = bx0-ax0;
      type_real c2 = bz0-az0;
      type_real ta = (bdz*c1 - bdx*c2)/cross;
      type_real tb = (adz*c1 - adx*c2)/cross;

      //this is the distance from the intersection where the lines are intersect_eps distance apart
      type_real permitted_dist = intersect_eps / sqrt(2 - 2*sqrt(1 - sin2));
      //and the distance in parameter space to achieve that:
      type_real a_shift = permitted_dist / sqrt(adx*adx + adz*adz);
      type_real b_shift = permitted_dist / sqrt(bdx*bdx + bdz*bdz);
      alow = std::max((type_real)0.0, ta - a_shift);
      blow = std::max((type_real)0.0, tb - b_shift);
      ahigh = std::min((type_real)1.0, ta + a_shift);
      bhigh = std::min((type_real)1.0, tb + b_shift);

    //confirm that the intersection occurs inside the segments
      if(ahigh - alow <= 0 || bhigh - blow <= 0){
        return false;
      }

      return true;
    }
    // sin^2 theta <= eps, so parallel; orth project points to find distance between lines


    //(a0-b0) - ad( (a0 - b0) . (ad)/|ad|^2 )

    // initial_point_deviation . a_direction
    type_real dot_over_mag2 =
        ((ax0 - bx0) * adx + (az0 - bz0) * adz) / (adx * adx + adz * adz);
    //proj initial_point_deviation perp to {a_direction}
    type_real orthx = (ax0 - bx0) - adx * dot_over_mag2;
    type_real orthz = (az0 - bz0) - adz * dot_over_mag2;

    if (orthx * orthx + orthz * orthz > intersect_eps2) {
      // distance between lines is greater than eps, so no intersection
      return false;
    }

    //the lines intersect. Find start and end parameters. We use [0,1] for parameter
    // (a0 + ad*ta = b0 + bd*tb)


    // find (a0 + ad*tb0 ~ b0) and (a0 + ad*tb0 ~ b1): do a projection
    type_real tb0 = ((bx0 - ax0) * adx + (bz0 - az0) * adz) / (adx * adx + adz * adz);
    type_real tb1 = ((bx1 - ax0) * adx + (bz1 - az0) * adz) / (adx * adx + adz * adz);

    alow = std::max((type_real)0.0, std::min(tb0, tb1));
    ahigh = std::min((type_real)1.0, std::max(tb0, tb1));

    //confirm that the intersection occurs inside the segment
    if(ahigh - alow <= 0){
      return false;
    }

    //we have an intersection. Set blow,bhigh and return true

    if (fabs(bdx) > fabs(bdz)) {
      blow = (ax0 + alow * adx - bx0) / bdx;
      bhigh = (ax0 + ahigh * adx - bx0) / bdx;
    } else {
      blow = (az0 + alow * adz - bz0) / bdz;
      bhigh = (az0 + ahigh * adz - bz0) / bdz;
    }
    return true;
  };

  float a_param_start = 1, a_param_end = -1;
  float b_param_start = 1, b_param_end = -1;



  //this is O(n^2) check. this really needs to be optimized
  float alow,ahigh,blow,bhigh;
  for(int ia = 0; ia < subdivisions; ia++){
    for(int ib = 0; ib < subdivisions; ib++){
      if (line_intersection(ax[ia],az[ia],ax[ia+1],az[ia+1],
                           bx[ib],bz[ib],bx[ib+1],bz[ib+1],
                           alow,ahigh,blow,bhigh)){
        //push out parameters. We assume intersection is convex
        a_param_start = std::min(a_param_start, (ia+alow)*h-1);
        a_param_end = std::max(a_param_end, (ia+ahigh)*h-1);
        b_param_start = std::min(b_param_start, (ib+blow)*h-1);
        b_param_end = std::max(b_param_end, (ib+bhigh)*h-1);
      }
    }
  }
  if(a_param_end - a_param_start < intersect_eps && b_param_end - b_param_start < intersect_eps){
    //intersection (within segments) is too small
    return false;
  }
  intersection.a_param_start = a_param_start;
  intersection.a_param_end = a_param_end;
  intersection.b_param_start = b_param_start;
  intersection.b_param_end = b_param_end;

  //TODO this is all wrong when param_start != -1 and param_end != -1.
  //fix it.

  // populate mortar transfer functions by computing reference parameters for even spacing
  type_real t_samples[intersection_nquad];

  //this uses the linear approximations, but we should consider a different scheme (locate_point?)

  type_real a_len = 0, b_len = 0;//length of intersection as measured on either side
  type_real a_sublens[subdivisions];
  type_real b_sublens[subdivisions];

  // subdivision on which intersection starts. since we are in positive range, casting floors
  //start at greatest int where (ih-1 <= param_start)
  //recall 2/h == subdivisions
  int a_ind_start = std::max(0,static_cast<int>((a_param_start + 1) * subdivisions * 0.5));
  type_real a_subind_start = 0.5*subdivisions*(a_param_start + 1) - a_ind_start;
  int b_ind_start = std::max(0,static_cast<int>((b_param_start + 1) * subdivisions * 0.5));
  type_real b_subind_start = 0.5*subdivisions*(b_param_start + 1) - b_ind_start;

  //end at smallest int where ((i+1)h-1 >= param_end).
  int a_ind_end = std::min(subdivisions-1,subdivisions-1-static_cast<int>(subdivisions-(a_param_end + 1) * subdivisions * 0.5));
  type_real a_subind_end = 0.5*subdivisions*(a_param_end + 1) - a_ind_end;
  int b_ind_end = std::min(subdivisions-1,subdivisions-1-static_cast<int>(subdivisions-(b_param_end + 1) * subdivisions * 0.5));
  type_real b_subind_end = 0.5*subdivisions*(b_param_end + 1) - b_ind_end;
  for(int i = a_ind_start; i <= a_ind_end; i++){
    type_real dx = ax[i+1] - ax[i];
    type_real dz = az[i+1] - az[i];
    a_sublens[i] = sqrt(dx*dx + dz*dz);
    a_len += a_sublens[i];
  }
  //for a_len, we overcounted on the endpoints
  a_len -= a_sublens[a_ind_start] * a_subind_start;
  a_len -= a_sublens[a_ind_end] * (1-a_subind_end);

  for(int i = b_ind_start; i <= b_ind_end; i++){
    type_real dx = bx[i+1] - bx[i];
    type_real dz = bz[i+1] - bz[i];
    b_sublens[i] = sqrt(dx*dx + dz*dz);
    b_len += b_sublens[i];
  }
  b_len -= b_sublens[b_ind_start] * b_subind_start;
  b_len -= b_sublens[b_ind_end] * (1-b_subind_end);

#ifdef ___DEBUG___
  ASSERT(0 <= a_subind_start && a_subind_start <= 1+1e-6, "start: "+std::to_string(a_param_start)+" ("+std::to_string(a_ind_start)+","+std::to_string(a_subind_start)+")");
  ASSERT(0 <= a_subind_end && a_subind_end <= 1+1e-6, "end: "+std::to_string(a_param_end)+" ("+std::to_string(a_ind_end)+","+std::to_string(a_subind_end)+")");
  ASSERT(0 <= b_subind_start && b_subind_start <= 1+1e-6, "start: "+std::to_string(b_param_start)+" ("+std::to_string(b_ind_start)+","+std::to_string(b_subind_start)+")");
  ASSERT(0 <= b_subind_end && b_subind_end <= 1+1e-6, "end: "+std::to_string(b_param_end)+" ("+std::to_string(b_ind_end)+","+std::to_string(b_subind_end)+")");
#endif

  for (int i = 0; i < intersection_nquad; i++) {
    // avg / 2, since parameter for quadrature ranges from -1 to 1 (len 2)
    //the goal is to space nodes so ds is constant.
    intersection.ds[i] = (a_len + b_len)/4;
  }

  //even spacing is len_desired = (igll/(nquad-1) - 1) * len(intersection)    igll = 0:nquad
  //find parameters int_{param_start}^{t_desired} |dr|  = len_desired


  //this is the length at the end of this segment. We know t lies on this segment if len_inc >= len_desired
  // [parameter space] (index*h - 1)
  //       param_start                           t_desired (want)
  // |----------+--------------|--------|-------------------------------|
  //
  // [segment index space] ((parameter + 1)/h)
  //  ---------- <- subind_start                                     segment
  // |----------+--------------|--------|---------------+---------------|
  // ind_start                                           --------------- <- delta / sublens[segment-1]
  //
  // [length space] (sum_start^index sublen )
  //            0                            len_desired (known)       len_inc
  // |----------+--------------|--------|---------------+---------------|
  //                                     ------------------------------- <- sublens[segment-1]
  //                                                     --------------- <- delta = len_inc - len_desired
  //                                        (ratio by linear approx)

  // indexwise   ---  (segment+1) = t_desired/h + delta
  // lengthwise  ---  len_inc                   = len_desired + delta*sublen[segment-1]
  type_real len_inc = a_sublens[a_ind_start] * (1-a_subind_start);

  int segment = a_ind_start+1;
#ifdef ___DEBUG___
  std::string compute_log = "  init(seg="+std::to_string(segment)+",inc="+std::to_string(len_inc)+")";
#endif
  for (int i = 0; i < intersection_nquad; i++) {
    type_real len_desired = 0.5*(1+gll.t[i])*a_len;
#ifdef ___DEBUG___
  compute_log += "\n  newtarget(len_desired="+std::to_string(len_desired)+")";
#endif
    while(len_inc < len_desired && segment <= a_ind_end){
      len_inc += a_sublens[segment];
      segment++;
#ifdef ___DEBUG___
  compute_log += "\n  inc(seg="+std::to_string(segment)+",inc="+std::to_string(len_inc)+")";
#endif
    }
    t_samples[i] = h*(segment + (len_desired - len_inc)/a_sublens[segment-1]) - 1;
#ifdef ___DEBUG___
  compute_log += "\n  select(t["+std::to_string(i)+"]="+std::to_string(t_samples[i])+")";
#endif
  }
#ifdef ___DEBUG___
  const auto stringify_arr = [](type_real* arr, int size){
    std::string st = "[" + std::to_string(arr[0]);
    for(int i = 1; i < size; i++){
      st += ", " + std::to_string(arr[i]);
    }
    st += "]";
    return st;
  };
  ASSERT(-1-1e-6 <= t_samples[0] && t_samples[intersection_nquad-1] <= 1+1e-6,
    "bad t-samples\ntsamples (side1): log\n"+compute_log+"\nlen = "+std::to_string(a_len)
    +"\nstart: "+std::to_string(a_param_start)+" ("+std::to_string(a_ind_start)+","+std::to_string(a_subind_start)+") "
    +"end: "+std::to_string(a_param_end)+" ("+std::to_string(a_ind_end)+","+std::to_string(a_subind_end)+")"
    +"\nsublens = "+stringify_arr(a_sublens,subdivisions));
#endif
  gll.sample_L(intersection.a_mortar_trans, t_samples, intersection_nquad);


  segment = b_ind_start+1;
  len_inc = b_sublens[b_ind_start] * (1-b_subind_start);
#ifdef ___DEBUG___
  compute_log = "  init(seg="+std::to_string(segment)+",inc="+std::to_string(len_inc)+")";
#endif
  for (int i = 0; i < intersection_nquad; i++) {
    type_real len_desired = 0.5*(1+gll.t[i])*b_len;
#ifdef ___DEBUG___
  compute_log += "\n  newtarget(len_desired="+std::to_string(len_desired)+")";
#endif
    while(len_inc < len_desired && segment <= b_ind_end){
      len_inc += b_sublens[segment];
      segment++;
#ifdef ___DEBUG___
  compute_log += "\n  inc(seg="+std::to_string(segment)+",inc="+std::to_string(len_inc)+")";
#endif
    }
    t_samples[i] = h*(segment + (len_desired - len_inc)/b_sublens[segment-1]) - 1;
#ifdef ___DEBUG___
  compute_log += "\n  select(t["+std::to_string(i)+"]="+std::to_string(t_samples[i])+")";
#endif
  }
#ifdef ___DEBUG___
  ASSERT(-1-1e-6 <= t_samples[0] && t_samples[intersection_nquad-1] <= 1+1e-6,
    "bad t-samples\ntsamples (side2): log\n"+compute_log+"\nlen = "+std::to_string(b_len)
    +"\nstart: "+std::to_string(b_param_start)+" ("+std::to_string(b_ind_start)+","+std::to_string(b_subind_start)+") "
    +"end: "+std::to_string(b_param_end)+" ("+std::to_string(b_ind_end)+","+std::to_string(b_subind_end)+")"
    +"\nsublens = "+stringify_arr(b_sublens,subdivisions));
#endif
  gll.sample_L(intersection.b_mortar_trans, t_samples, intersection_nquad);
  intersection.ngll = intersection_nquad;
  intersection.a_ngll = edgequad::NGLL;
  intersection.b_ngll = edgequad::NGLL;
  return true;
#undef intersect_eps2
#undef intersect_eps
}

template <typename edgequad, int datacapacity>
void edge_storage<edgequad, datacapacity>::build_intersections_on_host() {
  int intersections_acoustic_acoustic = 0;
  int intersections_acoustic_elastic = 0;
  int intersections_elastic_elastic = 0;
  std::vector<edge_intersection<ngll> > intersections;
  edge_intersection<ngll> intersection;
  intersection_sorted_inds = std::vector<int>();
  // foreach unordered pair (edge[i], edge[j]), j != i
  for (int i = 0; i < n_edges; i++) {
    for (int j = i + 1; j < n_edges; j++) {
      // if there is an intersection, store it.
      if (intersect(i, j, intersection)) {
        intersection.a_ref_ind = i;
        intersection.b_ref_ind = j;
        intersections.push_back(intersection);
        if(edge_media[i] == specfem::element::medium_tag::acoustic
        &&edge_media[j] == specfem::element::medium_tag::acoustic){
          intersection_sorted_inds.push_back(intersections_acoustic_acoustic);
          intersections_acoustic_acoustic++;
        }else if(edge_media[i] == specfem::element::medium_tag::elastic
        &&edge_media[j] == specfem::element::medium_tag::elastic){
          intersection_sorted_inds.push_back(intersections_elastic_elastic);
          intersections_elastic_elastic++;
        }else{
          intersection_sorted_inds.push_back(intersections_acoustic_elastic);
          intersections_acoustic_elastic++;
        }
      }
    }
  }
  n_intersections = intersections.size();
  // intersection_container =
  //     specfem::kokkos::DeviceView1d<edge_intersection<ngll> >(
  //         "_util::edge_manager::edge_storage::edge_data", n_intersections);
  // h_intersection_container = Kokkos::create_mirror_view(intersection_container);

  //temporary: initialize with no interfaces
  acoustic_elastic_interface = decltype(acoustic_elastic_interface)(acoustic_edges.size(), elastic_edges.size(), intersections_acoustic_elastic);
  acoustic_acoustic_interface = decltype(acoustic_acoustic_interface)(acoustic_edges.size(), acoustic_edges.size(), intersections_acoustic_acoustic);
  elastic_elastic_interface = decltype(elastic_elastic_interface)(elastic_edges.size(), elastic_edges.size(), intersections_acoustic_elastic);
  for (int i = 0; i < n_edges; i++) {
    const int sorted_ind = edge_sorted_inds[i];
    if(edge_media[i] == specfem::element::medium_tag::acoustic){
      acoustic_elastic_interface.h_medium1_index_mapping(sorted_ind) = edges[i].id;
      acoustic_elastic_interface.h_medium1_edge_type(sorted_ind) = edges[i].bdry;

      acoustic_acoustic_interface.h_medium1_index_mapping(sorted_ind) = edges[i].id;
      acoustic_acoustic_interface.h_medium1_edge_type(sorted_ind) = edges[i].bdry;
      acoustic_acoustic_interface.h_medium2_index_mapping(sorted_ind) = edges[i].id;
      acoustic_acoustic_interface.h_medium2_edge_type(sorted_ind) = edges[i].bdry;
    }else{
      acoustic_elastic_interface.h_medium2_index_mapping(sorted_ind) = edges[i].id;
      acoustic_elastic_interface.h_medium2_edge_type(sorted_ind) = edges[i].bdry;

      elastic_elastic_interface.h_medium1_index_mapping(sorted_ind) = edges[i].id;
      elastic_elastic_interface.h_medium1_edge_type(sorted_ind) = edges[i].bdry;
      elastic_elastic_interface.h_medium2_index_mapping(sorted_ind) = edges[i].id;
      elastic_elastic_interface.h_medium2_edge_type(sorted_ind) = edges[i].bdry;
    }
  }
  interface_structs_initialized = true;

  intersection_edge_a = std::vector<int>(n_intersections);
  intersection_edge_b = std::vector<int>(n_intersections);

  for (int i = 0; i < n_intersections; i++) {
    int a_ind = intersections[i].a_ref_ind;
    int b_ind = intersections[i].b_ref_ind;
    intersection_edge_a[i] = a_ind;
    intersection_edge_b[i] = b_ind;
    if(edge_media[a_ind] == specfem::element::medium_tag::elastic
    &&edge_media[b_ind] == specfem::element::medium_tag::acoustic){
        // swap a and b
        int tmpi;
        type_real tmpr;
        #define swpi(a, b)                                                             \
          {                                                                            \
            tmpi = a;                                                                  \
            a = b;                                                                     \
            b = tmpi;                                                                  \
          }
        #define swpr(a, b)                                                             \
          {                                                                            \
            tmpr = a;                                                                  \
            a = b;                                                                     \
            b = tmpr;                                                                  \
          }
                  swpi(intersections[i].a_ref_ind, intersections[i].b_ref_ind);
                  swpi(intersections[i].a_ngll, intersections[i].b_ngll);
                  swpr(intersections[i].a_param_start, intersections[i].b_param_start);
                  swpr(intersections[i].a_param_end, intersections[i].b_param_end);
                  for (int igll1 = 0; igll1 < ngll; igll1++) {
                    for (int igll2 = 0; igll2 < ngll; igll2++) {
                      swpr(intersections[i].a_mortar_trans[igll1][igll2],
                          intersections[i].b_mortar_trans[igll1][igll2])
                    }
                  }
        swpi(a_ind,b_ind);
        #undef swpi
        #undef swpr
    }
    store_intersection(i,intersections[i]);
  }
}

type_real quadrature_rule::integrate(const type_real *f) {
  type_real sum = 0;
  for (int i = 0; i < nquad; i++) {
    sum += f[i] * w[i];
  }
  return sum;
}
type_real quadrature_rule::deriv(const type_real *f, const type_real t) {
  // f(t) = sum_{j} f[j] L_j(t) = sum_{ij} f[j] L_{ji} i t^(i-1)
  type_real tim1 = 1; // t^(i-1)
  type_real sum = 0;
  for (int i = 1; i < nquad; i++) {
    for (int j = 0; j < nquad; j++) {
      sum += f[j] * L[j * nquad + i] * i * tim1;
    }
    tim1 *= t;
  }
  return sum;
}
type_real quadrature_rule::interpolate(const type_real *f, const type_real t) {
  // f(t) = sum_{j} f[j] L_j(t) = sum_{ij} f[j] L_{ji} t^i
  type_real ti = 1; // t^i
  type_real sum = 0;
  for (int i = 0; i < nquad; i++) {
    for (int j = 0; j < nquad; j++) {
      sum += f[j] * L[j * nquad + i] * ti;
    }
    ti *= t;
  }
  return sum;
}

quadrature_rule gen_GLL(int ngll) {
  // TODO should we set a builder for a general rule?
  if (ngll == 5) {
    quadrature_rule gll(ngll);
    gll.t[0] = -1.000000000000000;
    gll.t[1] = -0.654653670707977;
    gll.t[2] = 0.000000000000000;
    gll.t[3] = 0.654653670707977;
    gll.t[4] = 1.000000000000000;
    gll.w[0] = 0.100000000000000;
    gll.w[1] = 0.544444444444444;
    gll.w[2] = 0.711111111111111;
    gll.w[3] = 0.544444444444444;
    gll.w[4] = 0.100000000000000;
    gll.L[0] = -0.000000000000000;
    gll.L[1] = 0.375000000000000;
    gll.L[2] = -0.375000000000000;
    gll.L[3] = -0.875000000000000;
    gll.L[4] = 0.875000000000000;
    gll.L[5] = 0.000000000000000;
    gll.L[6] = -1.336584577695453;
    gll.L[7] = 2.041666666666667;
    gll.L[8] = 1.336584577695453;
    gll.L[9] = -2.041666666666667;
    gll.L[10] = 1.000000000000000;
    gll.L[11] = -0.000000000000000;
    gll.L[12] = -3.333333333333333;
    gll.L[13] = 0.000000000000000;
    gll.L[14] = 2.333333333333333;
    gll.L[15] = -0.000000000000000;
    gll.L[16] = 1.336584577695453;
    gll.L[17] = 2.041666666666667;
    gll.L[18] = -1.336584577695453;
    gll.L[19] = -2.041666666666667;
    gll.L[20] = 0.000000000000000;
    gll.L[21] = -0.375000000000000;
    gll.L[22] = -0.375000000000000;
    gll.L[23] = 0.875000000000000;
    gll.L[24] = 0.875000000000000;

    return gll;
  }

  throw std::runtime_error("gen_GLL only supports ngll=5 right now.");
}

template <int ngllcapacity>
void quadrature_rule::sample_L(type_real buf[][ngllcapacity], const type_real *t_vals,
                               const int t_size) {
  for (int it = 0; it < t_size; it++) { // foreach t
    type_real tpow = 1;
    // reset accum
    for (int iL = 0; iL < nquad; iL++) {
      buf[it][iL] = 0;
    }
    for (int ipow = 0; ipow < nquad; ipow++) {
      // buf += L_{:,ipow} * t^ipow
      for (int iL = 0; iL < nquad; iL++) {
        buf[it][iL] += tpow * L[iL * nquad + ipow];
      }
      tpow *= t_vals[it];
    }
  }
}

template <int ngllcapacity>
type_real edge_intersection<ngllcapacity>::a_to_mortar(const int mortar_index,
                                                       const type_real *quantity) {
  type_real val = 0;
  for (int i = 0; i < a_ngll; i++) {
    val += a_mortar_trans[mortar_index][i] * quantity[i];
  }
  return val;
}
template <int ngllcapacity>
type_real edge_intersection<ngllcapacity>::b_to_mortar(const int mortar_index,
                                                       const type_real *quantity) {
  type_real val = 0;
  for (int i = 0; i < b_ngll; i++) {
    val += b_mortar_trans[mortar_index][i] * quantity[i];
  }
  return val;
}

template<typename edgequad, int datacapacity>
edge_intersection<edge_storage<edgequad, datacapacity>::ngll> edge_storage<edgequad, datacapacity>::load_intersection(const int id) {
  edge_intersection<edge_storage<edgequad, datacapacity>::ngll> intersection;
  const int a_ind = intersection_edge_a[id];
  const int b_ind = intersection_edge_b[id];
  intersection.a_ref_ind = a_ind;
  intersection.b_ref_ind = b_ind;
  const int sorted_ind = intersection_sorted_inds[id];
#define transfer_interface_vals(interface) {     \
    intersection.a_param_start = interface.h_interface_medium1_param_start(sorted_ind);\
    intersection.b_param_start = interface.h_interface_medium2_param_start(sorted_ind);\
    intersection.a_param_end = interface.h_interface_medium1_param_end(sorted_ind);\
    intersection.b_param_end = interface.h_interface_medium2_param_end(sorted_ind);\
    intersection.a_ngll = interface.NGLL_EDGE;\
    intersection.b_ngll = interface.NGLL_EDGE;\
    intersection.ngll = interface.NGLL_INTERFACE;\
    interface.to_edge_data_mortar_trans(sorted_ind,intersection.a_mortar_trans, intersection.b_mortar_trans);\
    for(int igll = 0; igll < interface.NGLL_INTERFACE; igll++){\
      intersection.ds[igll] = interface.h_interface_surface_jacobian_times_weight(sorted_ind,igll);\
    }\
  }
  if(edge_media[a_ind] == specfem::element::medium_tag::acoustic
  &&edge_media[b_ind] == specfem::element::medium_tag::acoustic){
      intersection.relax_param = acoustic_acoustic_interface.h_interface_relaxation_parameter(sorted_ind);
      transfer_interface_vals(acoustic_acoustic_interface);
  }else if(edge_media[a_ind] == specfem::element::medium_tag::elastic
  &&edge_media[b_ind] == specfem::element::medium_tag::elastic){
      transfer_interface_vals(elastic_elastic_interface);
  }else if(edge_media[a_ind] == specfem::element::medium_tag::acoustic
  &&edge_media[b_ind] == specfem::element::medium_tag::elastic){
      transfer_interface_vals(acoustic_elastic_interface);
  }
  return intersection;
#undef transfer_interface_vals

}
template<typename edgequad, int datacapacity>
void edge_storage<edgequad, datacapacity>::store_intersection(const int id, const edge_intersection<edge_storage<edgequad, datacapacity>::ngll>& intersection) {
  const int a_ind = intersection.a_ref_ind;
  const int b_ind = intersection.b_ref_ind;
  const int sorted_ind = intersection_sorted_inds[id];
#define transfer_interface_vals(interface) {     \
    interface.h_interface_medium1_index(sorted_ind) = edge_sorted_inds[a_ind];\
    interface.h_interface_medium2_index(sorted_ind) = edge_sorted_inds[b_ind];\
    interface.h_interface_medium1_param_start(sorted_ind) = intersection.a_param_start;\
    interface.h_interface_medium2_param_start(sorted_ind) = intersection.b_param_start;\
    interface.h_interface_medium1_param_end(sorted_ind) = intersection.a_param_end;\
    interface.h_interface_medium2_param_end(sorted_ind) = intersection.b_param_end;\
    interface.from_edge_data_mortar_trans(sorted_ind,intersection.a_mortar_trans, intersection.b_mortar_trans);\
    for(int igll = 0; igll < interface.NGLL_INTERFACE; igll++){\
      interface.h_interface_surface_jacobian_times_weight(sorted_ind,igll) = intersection.ds[igll] * intersection.quad_weight[igll];\
    }\
  }
  if(edge_media[a_ind] == specfem::element::medium_tag::acoustic
  &&edge_media[b_ind] == specfem::element::medium_tag::acoustic){
      transfer_interface_vals(acoustic_acoustic_interface);
  }else if(edge_media[a_ind] == specfem::element::medium_tag::elastic
  &&edge_media[b_ind] == specfem::element::medium_tag::elastic){
      transfer_interface_vals(elastic_elastic_interface);
  }else if(edge_media[a_ind] == specfem::element::medium_tag::acoustic
  &&edge_media[b_ind] == specfem::element::medium_tag::elastic){
      transfer_interface_vals(acoustic_elastic_interface);
  }
  // h_intersection_container(id) = intersection;
#undef transfer_interface_vals
}

template<typename edgequad, int datacapacity>
void edge_storage<edgequad, datacapacity>::load_all_intersections() {
  if(!interface_structs_initialized){
    throw std::runtime_error("attempting to run edge_storage.load_all_intersections() prior to interface struct initialization.");
  }
  for (int i = 0; i < n_intersections; i++) {
    load_intersection(i);
  }
}
template<typename edgequad, int datacapacity>
void edge_storage<edgequad, datacapacity>::store_all_intersections() {
  if(!interface_structs_initialized){
    throw std::runtime_error("attempting to run edge_storage.store_all_intersections() prior to interface struct initialization.");
  }
  for (int i = 0; i < n_intersections; i++) {
    // store_intersection(i, h_intersection_container(i));
  }
}





} // namespace edge_manager
} // namespace _util

#endif
