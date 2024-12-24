#pragma once

#include "datatypes/simd.hpp"

namespace specfem {
namespace edge {
namespace loose {

template <specfem::dimension::type DimensionType, typename QuadratureType,
          bool UseSIMD>
struct positions;

template <typename QuadratureType, bool UseSIMD>
struct positions<specfem::dimension::type::dim2, QuadratureType, UseSIMD> {
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static int NGLL = QuadratureType::NGLL;
  using value_type =
      typename simd::datatype; ///< Value type to store properties

  value_type x[NGLL];
  value_type z[NGLL];

  KOKKOS_FUNCTION
  positions() = default;

  KOKKOS_FUNCTION
  positions(const value_type *x, const value_type *z) {
#pragma unroll
    for (int igll = 0; igll < NGLL; igll++) {
      this->x[igll] = x[igll];
      this->z[igll] = z[igll];
    }
  }
};

} // namespace loose
} // namespace edge
} // namespace specfem
