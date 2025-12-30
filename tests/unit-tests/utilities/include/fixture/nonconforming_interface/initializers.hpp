#pragma once

// all non-specific declarations for NCIs
#include "specfem/data_access/accessor.hpp"
namespace specfem::test_fixture {

/**
 * @brief Manages views of field values along an edge.
 *
 * @tparam Initializer initializer type: must inherit
 * EdgeFunctionInitializer2D::Base
 */
template <typename Initializer> struct EdgeFunction2D;
/**
 * @brief Initializes views of field values along an edge.
 *
 */
namespace EdgeFunctionInitializer2D {
struct EdgeFunctionInitializer2D {};
template <typename AnalyticalFunctionInitializer,
          typename EdgePointsInitializer>
struct FromAnalyticalFunction;
} // namespace EdgeFunctionInitializer2D

/**
 * @brief Manages views of the transfer function.
 *
 * @tparam Initializer initializer type: must inherit
 * TransferFunctionInitializer2D::Base
 */
template <typename Initializer> struct TransferFunction2D;
namespace TransferFunctionInitializer2D {
struct TransferFunctionInitializer2D {};
template <typename EdgeQuadratureInitializer,
          typename IntersectionQuadratureInitializer>
struct FromQuadratureRules;
} // namespace TransferFunctionInitializer2D

/**
 * @brief Provides helpers for computing the quadrature weights on an
 * intersection. Provides a means of patching the corresponding intersection
 * data component.
 *
 * @tparam Initializer An initializer of type IntersectionFactorInitializer2D
 */
template <typename Initializer> struct IntersectionFactor2D;
/**
 * @brief Gives the quadrature weights for a particular rule, accounting for the
 * Jacobian determinant and integral not being over [-1,1].
 *
 */
namespace IntersectionFactorInitializer2D {
struct IntersectionFactorInitializer2D {};
} // namespace IntersectionFactorInitializer2D

/**
 * @brief Manages the data of an intersection (everything but field values),
 * providing a patch for AccessorPack.
 *
 * @tparam Initializer initializer type: must inherit from
 * IntersectionDataInitializer2D
 * @tparam AccessorTypes list of accessors to derive the AccessorPack.
 */
template <typename Initializer,
          specfem::data_access::AccessorType... AccessorTypes>
struct IntersectionDataPack2D;

/**
 * @brief Specifies a two-sided intersection (self and coupled). Each
 * IntersectionDataInitializer2D gives a transfer function for each side,
 * and the integration weights (with any potential Jacobian scaling) for
 * intersection integration.
 *
 */
namespace IntersectionDataInitializer2D {
struct IntersectionDataInitializer2D {};
} // namespace IntersectionDataInitializer2D

// =================================================================================================
// We may wish to move these to somewhere else in the future: these are not
// exclusive to NCIs

/**
 * @brief Provides helper functions for a quadrature rule (Lagrange polynomial
 * evaluation, weight computation, etc.)
 *
 */
template <typename QuadraturePoints> struct QuadratureRule;
/**
 * @brief Gives the quadrature points for a particular rule.
 *
 */
namespace QuadraturePoints {
struct QuadraturePoints {};
} // namespace QuadraturePoints

/**
 * @brief Manages 1-parameter functions. These can be used, say for
 * edge-coordinate analytical fields
 */
namespace AnalyticalFunctionType {
struct AnalyticalFunctionType {};
} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
