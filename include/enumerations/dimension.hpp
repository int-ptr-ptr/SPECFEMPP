#ifndef _ENUMERATIONS_DIMENSION_HPP_
#define _ENUMERATIONS_DIMENSION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace enums {
namespace element {
/**
 * @namespace dimensionality property of the element
 *
 */
namespace dimension {
/**
 * @brief 2D element
 *
 */
class dim2 {
public:
  constexpr static int dim = 2; ///< Dimensionality of the element
};
/**
 * @brief 3D element
 *
 */
class dim3 {
public:
  constexpr static int dim = 3; ///< Dimensionality of the element
};
} // namespace dimension
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_DIMENSION_HPP_ */
