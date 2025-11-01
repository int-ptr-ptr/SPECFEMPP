
#pragma once

#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"

#include "specfem/assembly/coupled_interfaces/dim2/data_access/impl/load_access_compatibility.hpp"

namespace specfem::assembly::coupled_interfaces_impl {

/**
 * @brief Container for 2D nonconforming interface data storage and access
 *
 * Manages interface data between different physical media (elastic-acoustic)
 * with specific boundary conditions. Stores edge factors and normal vectors
 * for interface computations in 2D spectral element simulations.
 *
 * TODO: consider same physical media
 *
 * @tparam InterfaceTag Type of interface (ELASTIC_ACOUSTIC or ACOUSTIC_ELASTIC)
 * @tparam BoundaryTag Boundary condition type (NONE, STACEY, etc.)
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct interface_container<specfem::dimension::type::dim2, InterfaceTag,
                           BoundaryTag,
                           specfem::connections::type::nonconforming>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2> {
public:
  /** @brief Dimension tag for 2D specialization */
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  constexpr static auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  constexpr static auto boundary_tag = BoundaryTag;
  /** @brief Medium type on the self side of the interface */
  constexpr static auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  /** @brief Medium type on the coupled side of the interface */
  constexpr static auto coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();

public:
  /** @brief Base container type alias */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::edge,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2>;
  /** @brief View type for edge scaling factors */
  using EdgeFactorView = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for edge normal vectors */
  using EdgeNormalView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  /** @brief View type for transfer function */
  using TransferFunctionView = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  /** @brief Device view for edge scaling factors */
  EdgeFactorView intersection_factor;
  /** @brief Device view for edge normal vectors */
  EdgeNormalView intersection_normal;
  /** @brief Device view for transfer function on self */
  TransferFunctionView transfer_function;
  /** @brief Device view for transfer function on coupled side */
  TransferFunctionView transfer_function_other;

  /** @brief Host mirror for edge scaling factors */
  EdgeFactorView::HostMirror h_intersection_factor;
  /** @brief Host mirror for edge normal vectors */
  EdgeNormalView::HostMirror h_intersection_normal;
  /** @brief Device view for transfer function on self */
  TransferFunctionView::HostMirror h_transfer_function;
  /** @brief Device view for transfer function on coupled side */
  TransferFunctionView::HostMirror h_transfer_function_other;

public:
  /**
   * @brief Constructs interface container with mesh and geometry data
   *
   * @param ngllz Number of GLL points in z-direction
   * @param ngllx Number of GLL points in x-direction
   * @param edge_types Edge type information from mesh
   * @param jacobian_matrix Jacobian transformation data
   * @param mesh Mesh connectivity and geometry
   */
  interface_container(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<specfem::dimension::type::dim2>
          &edge_types,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  /** @brief Default constructor */
  interface_container() = default;

  /**
   * @brief Loads interface data at specified index into point
   *
   * Template function that loads edge factor and normal vector data
   * from either device or host memory into the provided point object.
   *
   * @tparam on_device If true, loads from device memory; if false, from host
   * @tparam IndexType Type of index (must have iedge and ipoint members)
   * @tparam PointType Type of point (must have intersection_factor and
   * intersection_normal)
   * @param index Edge and point indices for data location
   * @param point Output point object to store loaded data
   */
  template <bool on_device, typename IndexType, typename PointType,
            typename std::enable_if_t<
                specfem::data_access::is_point<PointType>::value, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                             PointType &point) const {
    if constexpr (on_device) {
      point.edge_factor = intersection_factor(index.iedge, index.ipoint);
      point.intersection_normal(0) =
          intersection_normal(index.iedge, index.ipoint, 0);
      point.intersection_normal(1) =
          intersection_normal(index.iedge, index.ipoint, 1);
    } else {
      point.edge_factor = h_intersection_factor(index.iedge, index.ipoint);
      point.intersection_normal(0) =
          h_intersection_normal(index.iedge, index.ipoint, 0);
      point.intersection_normal(1) =
          h_intersection_normal(index.iedge, index.ipoint, 1);
    }
    return;
  }

  /**
   * @brief Loads interface data at specified index into edge
   *
   * Template function that loads edge factor and normal vector data
   * from either device or host memory into the provided edge object.
   *
   * @tparam on_device If true, loads from device memory; if false, from host
   * @tparam IndexType Type of index
   * @tparam EdgeType Type of edge (must have intersection_factor and
   * intersection_normal)
   * @param index Edge and point indices for data location
   * @param edge Output edge object to store loaded data
   */
  template <bool on_device, typename IndexType, typename EdgeType,
            typename std::enable_if_t<
                specfem::data_access::is_chunk_edge<EdgeType>::value, int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void impl_load(const IndexType &index,
                                             EdgeType &edge) const {
    // is there a better way of recovering global index?
    const auto team = index.get_policy_index();
    const int &offset = team.league_rank() * EdgeType::chunk_size;
    const int &num_edges = index.nedges();

    if constexpr (specfem::assembly::coupled_interfaces_impl::
                      stores_intersection_factor<EdgeType>::value ||
                  specfem::assembly::coupled_interfaces_impl::
                      stores_intersection_normal<EdgeType>::value) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team,
                                  num_edges * EdgeType::n_quad_intersection),
          [&](const int &ichunkmortar) {
            const int imortar = ichunkmortar % EdgeType::n_quad_intersection;
            const int iedge = ichunkmortar / EdgeType::n_quad_intersection;

            const int &local_slot = iedge;
            const int &container_slot = offset + iedge;

            if constexpr (specfem::assembly::coupled_interfaces_impl::
                              stores_intersection_factor<EdgeType>::value) {
              if constexpr (on_device) {
                edge.intersection_factor(local_slot, imortar) =
                    intersection_factor(container_slot, imortar);
              } else {
                edge.intersection_factor(local_slot, imortar) =
                    h_intersection_factor(container_slot, imortar);
              }
            }

            if constexpr (specfem::assembly::coupled_interfaces_impl::
                              stores_intersection_normal<EdgeType>::value) {
              if constexpr (on_device) {
                edge.intersection_normal(local_slot, imortar, 0) =
                    intersection_normal(container_slot, imortar, 0);
                edge.intersection_normal(local_slot, imortar, 1) =
                    intersection_normal(container_slot, imortar, 1);
              } else {
                edge.intersection_normal(local_slot, imortar, 0) =
                    h_intersection_normal(container_slot, imortar, 0);
                edge.intersection_normal(local_slot, imortar, 1) =
                    h_intersection_normal(container_slot, imortar, 1);
              }
            }
          });
    }

    if constexpr (specfem::assembly::coupled_interfaces_impl::
                      stores_transfer_function_self<EdgeType>::value ||
                  specfem::assembly::coupled_interfaces_impl::
                      stores_transfer_function_coupled<EdgeType>::value) {
      specfem::execution::for_each_level(
          index.get_iterator(),
          [&](const typename IndexType::iterator_type::index_type
                  &iterator_index) {
            const auto point_index = iterator_index.get_index();

            const int &local_slot = point_index.iedge;
            const int &container_slot = offset + point_index.iedge;
            const int &ipoint = point_index.ipoint;

            if constexpr (on_device) {
              for (int i = 0; i < EdgeType::n_quad_intersection; i++) {
                if constexpr (specfem::assembly::coupled_interfaces_impl::
                                  stores_transfer_function_self<
                                      EdgeType>::value) {
                  edge.transfer_function_self(local_slot, ipoint, i) =
                      transfer_function(container_slot, i, ipoint);
                }
                if constexpr (specfem::assembly::coupled_interfaces_impl::
                                  stores_transfer_function_coupled<
                                      EdgeType>::value) {
                  edge.transfer_function_coupled(local_slot, ipoint, i) =
                      transfer_function_other(container_slot, i, ipoint);
                }
              }
            } else {
              for (int i = 0; i < EdgeType::n_quad_intersection; i++) {
                if constexpr (specfem::assembly::coupled_interfaces_impl::
                                  stores_transfer_function_self<
                                      EdgeType>::value) {
                  edge.transfer_function_self(local_slot, ipoint, i) =
                      h_transfer_function(container_slot, i, ipoint);
                }
                if constexpr (specfem::assembly::coupled_interfaces_impl::
                                  stores_transfer_function_coupled<
                                      EdgeType>::value) {
                  edge.transfer_function_coupled(local_slot, ipoint, i) =
                      h_transfer_function_other(container_slot, i, ipoint);
                }
              }
            }
          });
    }
  }
};
} // namespace specfem::assembly::coupled_interfaces_impl
