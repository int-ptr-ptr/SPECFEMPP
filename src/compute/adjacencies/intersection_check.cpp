// this file should only be #included from adjacency_map.cpp, just to isolate
// this intersection check code.
#include "compute/adjacencies/adjacency_map.hpp"
#include "macros.hpp"

template <typename NumericType>
NumericType gll_interp(const NumericType *f, const NumericType t,
                       const int nquad) {
  ASSERT(nquad == 5, "gll_interp only ngll=5 supported right now.");
  constexpr NumericType L[5 * 5] = {
    -0.000000000000000, 0.375000000000000,  -0.375000000000000,
    -0.875000000000000, 0.875000000000000,  0.000000000000000,
    -1.336584577695453, 2.041666666666667,  1.336584577695453,
    -2.041666666666667, 1.000000000000000,  -0.000000000000000,
    -3.333333333333333, 0.000000000000000,  2.333333333333333,
    -0.000000000000000, 1.336584577695453,  2.041666666666667,
    -1.336584577695453, -2.041666666666667, 0.000000000000000,
    -0.375000000000000, -0.375000000000000, 0.875000000000000,
    0.875000000000000
  };
  // f(t) = sum_{j} f[j] L_j(t) = sum_{ij} f[j] L_{ji} t^i
  NumericType ti = 1; // t^i
  NumericType sum = 0;
  for (int i = 0; i < nquad; i++) {
    for (int j = 0; j < nquad; j++) {
      sum += f[j] * L[j * nquad + i] * ti;
    }
    ti *= t;
  }
  return sum;
}

// this code came from dg-phase2:include/_util/edge_storages.tpp
// we should consider two things:
// increase order of approximation (edge is subdivided into linear elements),
// maybe newton's method should be employed to NGLL order have proper quadrature
// structures optimize (it's probably enough to consider a rough check (BVH,
// contiguous element tracing, etc.) outside of this method)

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
bool intersect(const type_real *anodex, const type_real *anodez,
               const int nglla, const type_real *bnodex,
               const type_real *bnodez, const int ngllb,
               specfem::compute::adjacencies::nonconforming_edge &intersection,
               const bool a_left_side, const bool a_flip, const bool b_flip) {
#define intersect_eps 1e-3
#define throwout_threshold 1e-2
#define intersect_eps2 (intersect_eps * intersect_eps)

  // maybe do an AABB check for performance? the box can be precomputed per
  // edge.

  // approximate edges as linear segments
  constexpr int subdivisions = 10;
  constexpr type_real h = 2.0 / subdivisions;

  type_real ax[subdivisions + 1];
  type_real bx[subdivisions + 1];
  type_real az[subdivisions + 1];
  type_real bz[subdivisions + 1];

  for (int i = 0; i < subdivisions + 1; i++) {
    type_real param = i * h - 1.0;
    ax[i] = gll_interp(anodex, param, nglla);
    az[i] = gll_interp(anodez, param, nglla);
    bx[i] = gll_interp(bnodex, param, ngllb);
    bz[i] = gll_interp(bnodez, param, ngllb);
  }

  const auto line_intersection = [](type_real ax0, type_real az0, type_real ax1,
                                    type_real az1, type_real bx0, type_real bz0,
                                    type_real bx1, type_real bz1,
                                    type_real &alow, type_real &ahigh,
                                    type_real &blow, type_real &bhigh) -> bool {
    type_real adx = ax1 - ax0;
    type_real bdx = bx1 - bx0;
    type_real adz = az1 - az0;
    type_real bdz = bz1 - bz0;
    type_real cross = adx * bdz - bdx * adz;
    type_real sin2 =
        cross * cross / ((adx * adx + adz * adz) * (bdx * bdx + bdz * bdz));
    if (sin2 > 1e-2) {
      if (sin2 > 1) {
        sin2 = 1;
      }
      // not parallel, but there is an interval for which the lines are within
      // eps distance parameters where intersections occur: use cramer's rule
      type_real c1 = bx0 - ax0;
      type_real c2 = bz0 - az0;
      type_real ta = (bdz * c1 - bdx * c2) / cross;
      type_real tb = (adz * c1 - adx * c2) / cross;

      // this is the distance from the intersection where the lines are
      // intersect_eps distance apart
      type_real permitted_dist =
          (intersect_eps / 2) / sqrt(2 - 2 * sqrt(1 - sin2));
      // and the distance in parameter space to achieve that:
      type_real a_shift = permitted_dist / sqrt(adx * adx + adz * adz);
      type_real b_shift = permitted_dist / sqrt(bdx * bdx + bdz * bdz);
      alow = std::max((type_real)0.0, ta - a_shift);
      blow = std::max((type_real)0.0, tb - b_shift);
      ahigh = std::min((type_real)1.0, ta + a_shift);
      bhigh = std::min((type_real)1.0, tb + b_shift);

      // confirm that the intersection occurs inside the segments
      if (ahigh - alow <= 0 || bhigh - blow <= 0) {
        return false;
      }

      return true;
    }
    // sin^2 theta <= eps, so parallel; orth project points to find distance
    // between lines

    //(a0-b0) - ad( (a0 - b0) . (ad)/|ad|^2 )

    // initial_point_deviation . a_direction
    type_real dot_over_mag2 =
        ((ax0 - bx0) * adx + (az0 - bz0) * adz) / (adx * adx + adz * adz);
    // proj initial_point_deviation perp to {a_direction}
    type_real orthx = (ax0 - bx0) - adx * dot_over_mag2;
    type_real orthz = (az0 - bz0) - adz * dot_over_mag2;

    if (orthx * orthx + orthz * orthz > intersect_eps2) {
      // distance between lines is greater than eps, so no intersection
      return false;
    }

    // the lines intersect. Find start and end parameters. We use [0,1] for
    // parameter
    //  (a0 + ad*ta = b0 + bd*tb)

    // find (a0 + ad*tb0 ~ b0) and (a0 + ad*tb0 ~ b1): do a projection
    type_real tb0 =
        ((bx0 - ax0) * adx + (bz0 - az0) * adz) / (adx * adx + adz * adz);
    type_real tb1 =
        ((bx1 - ax0) * adx + (bz1 - az0) * adz) / (adx * adx + adz * adz);

    alow = std::max((type_real)0.0, std::min(tb0, tb1));
    ahigh = std::min((type_real)1.0, std::max(tb0, tb1));

    // confirm that the intersection occurs inside the segment
    if (ahigh - alow <= 0) {
      return false;
    }

    // we have an intersection. Set blow,bhigh and return true

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

  // this is O(n^2) check. this really needs to be optimized
  float alow, ahigh, blow, bhigh;
  for (int ia = 0; ia < subdivisions; ia++) {
    for (int ib = 0; ib < subdivisions; ib++) {
      if (line_intersection(ax[ia], az[ia], ax[ia + 1], az[ia + 1], bx[ib],
                            bz[ib], bx[ib + 1], bz[ib + 1], alow, ahigh, blow,
                            bhigh)) {
        // push out parameters. We assume intersection is convex
        a_param_start = std::min(a_param_start, (ia + alow) * h - 1);
        a_param_end = std::max(a_param_end, (ia + ahigh) * h - 1);
        b_param_start = std::min(b_param_start, (ib + blow) * h - 1);
        b_param_end = std::max(b_param_end, (ib + bhigh) * h - 1);
      }
    }
  }
  if (a_param_end - a_param_start < throwout_threshold &&
      b_param_end - b_param_start < throwout_threshold) {
    // intersection (within segments) is too small
    return false;
  }

  if (a_flip) {
    std::swap(a_param_start, a_param_end);
  }
  if (b_flip) {
    std::swap(b_param_start, b_param_end);
  }
  if (a_left_side) {
    intersection.param_startL = a_param_start;
    intersection.param_endL = a_param_end;
    intersection.param_startR = b_param_start;
    intersection.param_endR = b_param_end;
  } else {
    intersection.param_startR = a_param_start;
    intersection.param_endR = a_param_end;
    intersection.param_startL = b_param_start;
    intersection.param_endL = b_param_end;
  }
  return true;
#undef intersect_eps2
#undef intersect_eps
}
