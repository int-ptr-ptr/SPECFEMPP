#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/dimension.hpp"
#include "initializers.hpp"
#include "intersection_factor.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem_setup.hpp"
#include "transfer_function.hpp"

#include <type_traits>
static constexpr specfem::dimension::type dimension_tag_ =
    specfem::dimension::type::dim2;
namespace specfem::test_fixture {

template <specfem::interface::interface_tag InterfaceTag, typename... Accessors>
struct IntersectionAccessorPack
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          dimension_tag_, false>,
      public Accessors... {
  static constexpr specfem::connections::type connection_tag =
      specfem::connections::type::nonconforming;
  static constexpr specfem::interface::interface_tag interface_tag =
      InterfaceTag;
  static constexpr specfem::dimension::type dimension_tag = dimension_tag_;

  constexpr static size_t n_accessors = sizeof...(Accessors);
  using packed_accessors = std::tuple<Accessors...>;

  KOKKOS_INLINE_FUNCTION
  IntersectionAccessorPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;

  static std::string description() {
    return std::string("IntersectionAccessorPack (...)");
  }
};

template <typename Initializer,
          specfem::data_access::AccessorType... AccessorTypes>
struct IntersectionDataPack2D {};

namespace IntersectionDataInitializer2D {

template <typename QuadraturePointsType, typename EdgeCoordinateField>
struct Conforming : IntersectionDataInitializer2D {
  using QuadraturePoints = QuadraturePointsType;
  static_assert(std::is_base_of_v<
                    specfem::test_fixture::QuadraturePoints::QuadraturePoints,
                    QuadraturePoints>,
                "Conforming IntersectionDataInitializer needs a "
                "QuadraturePoints parameter in its first argument!");

  using EdgeCoordinates = EdgeCoordinateField;
  static_assert(
      std::is_base_of_v<
          specfem::test_fixture::AnalyticalFunctionType::AnalyticalFunctionType,
          EdgeCoordinates>,
      "Conforming IntersectionDataInitializer needs an AnalyticalFunctionType "
      "parameter in its second argument!");

  using SelfTransferInitializer =
      TransferFunctionInitializer2D::Identity<QuadraturePoints::nquad>;
  using CoupledTransferInitializer =
      TransferFunctionInitializer2D::Identity<QuadraturePoints::nquad>;

  using IntersectionFactorInitializer =
      IntersectionFactorInitializer2D::FromIntersectionKnots<QuadraturePoints,
                                                             EdgeCoordinates>;
};
} // namespace IntersectionDataInitializer2D

} // namespace specfem::test_fixture
