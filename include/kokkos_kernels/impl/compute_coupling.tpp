#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "compute_coupling.hpp"
#include "enumerations/connections.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_intersection_iterator.hpp"
#include "execution/for_all.hpp"
#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/point.hpp"
#include "specfem/point/interface_index.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

#include "algorithms/integrate/integrate1d.hpp"

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::weakly_conforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto connection_tag =
      specfem::connections::type::weakly_conforming;
  constexpr static auto interface_tag = InterfaceTag;
  constexpr static auto boundary_tag = BoundaryTag;
  constexpr static auto wavefield = WavefieldType;

  constexpr static auto self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();

  const auto &coupled_interfaces = assembly.coupled_interfaces;
  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  if (self_edges.extent(0) == 0 && coupled_edges.extent(0) == 0)
    return;

  const auto &field =
      assembly.fields.template get_simulation_field<wavefield>();
  const auto &boundaries = assembly.boundaries;

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = specfem::parallel_config::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;

  using CoupledFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template coupled_field_t<connection_tag>;
  using SelfFieldType = typename specfem::interface::attributes<
      dimension_tag, interface_tag>::template self_field_t<connection_tag>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, false>;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges, num_points);

  specfem::execution::for_each_level(
      "specfem::kokkos_kernels::impl::compute_coupling", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const int league_index = chunk_index.get_policy_index().league_rank();

        specfem::execution::for_each_level(
            chunk_index.get_iterator(),
            [&](const typename std::remove_const_t<std::remove_reference_t<
                    decltype(chunk_index)> >::iterator_type::index_type
                    &iterator_index) {
              const specfem::point::interface_index<dimension_tag> &index =
                  iterator_index.get_index();
              auto self_index = index.self_index;
              const auto coupled_index = index.coupled_index;
              self_index.iedge += league_index * parallel_config::chunk_size;

              specfem::point::coupled_interface<dimension_tag, connection_tag,
                                                interface_tag, boundary_tag>
                  point_interface_data;
              specfem::assembly::load_on_device(self_index, coupled_interfaces,
                                                point_interface_data);

              CoupledFieldType coupled_field;
              specfem::assembly::load_on_device(coupled_index, field,
                                                coupled_field);

              SelfFieldType self_field;

              specfem::medium::compute_coupling(point_interface_data,
                                                coupled_field, self_field);

              PointBoundaryType point_boundary;
              specfem::assembly::load_on_device(self_index, boundaries,
                                                point_boundary);
              if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                               acoustic_free_surface) {
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, self_field);
              }

              specfem::assembly::atomic_add_on_device(self_index, field,
                                                      self_field);
            });
      });

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          int NQuad_intersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    std::integral_constant<
        specfem::connections::type,
        specfem::connections::type::nonconforming> /*unused*/,
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr bool using_simd = false;

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto connection_tag =
      specfem::connections::type::nonconforming;
  constexpr static auto interface_tag = InterfaceTag;
  constexpr static auto boundary_tag = BoundaryTag;
  constexpr static auto wavefield = WavefieldType;
  const auto &coupled_interfaces = assembly.coupled_interfaces;
  const auto [self_edges, coupled_edges] =
      assembly.edge_types.get_edges_on_device(connection_tag, interface_tag,
                                              boundary_tag);

  if (self_edges.extent(0) == 0 && coupled_edges.extent(0) == 0)
    return;

  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const auto num_points = assembly.mesh.element_grid.ngllx;

  using parallel_config = specfem::parallel_config::default_chunk_edge_config<
      DimensionTag, Kokkos::DefaultExecutionSpace>;

  // As written, field types cannot readily be defined in attributes. Define
  // them here.
  constexpr specfem::element::medium_tag self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  constexpr specfem::element::medium_tag coupled_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::coupled_medium();
  using CoupledFieldType = std::conditional_t<
      InterfaceTag == specfem::interface::interface_tag::acoustic_elastic,
      specfem::chunk_edge::displacement<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd>,
      specfem::chunk_edge::acceleration<parallel_config::chunk_size, NGLL,
                                        dimension_tag, coupled_medium,
                                        using_simd> >;

  using SelfFieldType =
      specfem::point::acceleration<dimension_tag, self_medium, using_simd>;
  using CoupledInterfaceDataType =
      typename specfem::chunk_edge::nonconforming_coupled_interface<
          parallel_config::chunk_size, NGLL, NQuad_intersection, dimension_tag,
          connection_tag, interface_tag, boundary_tag,
          specfem::kokkos::DevScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, using_simd>;
  using CoupledTransferFunctionType =
      typename specfem::chunk_edge::nonconforming_transfer_and_normal<
          false, parallel_config::chunk_size, NGLL, NQuad_intersection,
          dimension_tag, connection_tag, interface_tag, boundary_tag,
          specfem::kokkos::DevScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, using_simd>;
  using IntersectionFactorType =
      typename specfem::chunk_edge::nonconforming_intersection_factor<
          parallel_config::chunk_size, NQuad_intersection,
          dimension_tag, connection_tag, interface_tag, boundary_tag,
          specfem::kokkos::DevScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, using_simd>;

  using InterfaceFieldViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, parallel_config::chunk_size,
      NQuad_intersection,
      specfem::element::attributes<DimensionTag, self_medium>::components,
      using_simd, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  specfem::execution::ChunkedIntersectionIterator chunk(
      parallel_config(), self_edges, coupled_edges, num_points);

  int scratch_size = CoupledFieldType::shmem_size() +
                     CoupledTransferFunctionType::shmem_size() +
                     InterfaceFieldViewType::shmem_size()
                     +IntersectionFactorType::shmem_size();

  specfem::execution::for_each_level(
      "specfem::kokkos_kernels::impl::compute_coupling",
      chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::index_type &chunk_iterator_index) {
        const auto &chunk_index = chunk_iterator_index.get_index();
        const auto &team = chunk_index.get_policy_index();
        const auto &self_chunk_iterator_index = chunk_index.get_self_index();
        const auto &coupled_chunk_iterator_index =
            chunk_index.get_coupled_index();
        const auto coupled_chunk_index =
            coupled_chunk_iterator_index.get_index();
        const auto self_chunk_index = self_chunk_iterator_index.get_index();

        CoupledFieldType coupled_field(team.team_scratch(0));
        specfem::assembly::load_on_device(coupled_chunk_index, field,
                                          coupled_field);


        // TODO add point access for mortar transfer function and replace self
        // side of this:
        CoupledTransferFunctionType coupled_transfer_function(team);

        specfem::assembly::load_on_device(self_chunk_index, coupled_interfaces,
                                          coupled_transfer_function);
        InterfaceFieldViewType interface_field(team.team_scratch(0));

        team.team_barrier();
        specfem::medium::compute_coupling(self_chunk_index,
                                          coupled_transfer_function,
                                          coupled_field, interface_field);

        IntersectionFactorType intersection_factor(team);

        specfem::assembly::load_on_device(self_chunk_index, coupled_interfaces,
                                          intersection_factor);

        team.team_barrier();

        specfem::algorithms::integrate_fieldtilde_1d(
            assembly, self_chunk_index, interface_field, intersection_factor, [&](
                const auto& self_index, SelfFieldType& self_field){

              specfem::point::boundary<boundary_tag, dimension_tag, false>
                  point_boundary;
              specfem::assembly::load_on_device(self_index, assembly.boundaries,
                                                point_boundary);
              if constexpr (BoundaryTag == specfem::element::boundary_tag::
                                               acoustic_free_surface) {
                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, self_field);
              }

              specfem::assembly::atomic_add_on_device(self_index, field,
                                                      self_field);

            }
        );
      });

  return;
}

template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          int NQuad_intersection,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_coupling(
    const specfem::assembly::assembly<DimensionTag> &assembly) {
  // Create dispatch tag for connection type
  using connection_dispatch =
      std::integral_constant<specfem::connections::type, ConnectionTag>;

  // Forward to implementation with dispatch tag
  if constexpr (ConnectionTag == specfem::connections::type::nonconforming) {
    compute_coupling<DimensionTag, WavefieldType, NGLL, NQuad_intersection,
                     InterfaceTag, BoundaryTag>(connection_dispatch(),
                                                assembly);
  } else {
    compute_coupling<DimensionTag, WavefieldType, InterfaceTag, BoundaryTag>(
        connection_dispatch(), assembly);
  }
}
