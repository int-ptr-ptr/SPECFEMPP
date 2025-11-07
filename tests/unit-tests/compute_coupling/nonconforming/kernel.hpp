#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/chunk_edge/nonconforming_transfer_and_normal.hpp"
#include <Kokkos_Core.hpp>

class ChunkEdgeIndexSimulator {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndexSimulator(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
};

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag, int chunk_size,
          int nquad_edge, int nquad_intersection>
void compute_kernel(const int &num_chunks) {

  constexpr bool using_simd = false;

  using InterfaceDataType =
      specfem::chunk_edge::nonconforming_transfer_and_normal<
          false, chunk_size, nquad_edge, nquad_intersection, DimensionTag,
          specfem::connections::type::nonconforming, InterfaceTag,
          specfem::element::boundary_tag::none,
          specfem::kokkos::DevScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  constexpr auto self_medium =
      specfem::interface::attributes<DimensionTag, InterfaceTag>::self_medium();
  constexpr auto coupled_medium =
      specfem::interface::attributes<DimensionTag,
                                     InterfaceTag>::coupled_medium();
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  static constexpr int ncomp_self =
      specfem::element::attributes<DimensionTag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<DimensionTag, coupled_medium>::components;
  using CoupledFieldType = std::conditional_t<
      InterfaceTag == specfem::interface::interface_tag::acoustic_elastic,
      specfem::chunk_edge::displacement<chunk_size, nquad_edge, DimensionTag,
                                        coupled_medium, using_simd>,
      specfem::chunk_edge::acceleration<chunk_size, nquad_edge, DimensionTag,
                                        coupled_medium, using_simd> >;

  using InterfaceFieldViewType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, DimensionTag, chunk_size, nquad_intersection,
      specfem::element::attributes<DimensionTag, self_medium>::components,
      using_simd, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  // ====================================================================
  // views for inputs / outputs
  // ====================================================================
  Kokkos::View<type_real *[chunk_size][nquad_intersection][ndim],
               Kokkos::DefaultExecutionSpace>
      normals("normals", num_chunks);
  Kokkos::View<type_real *[chunk_size][nquad_edge][ncomp_self],
               Kokkos::DefaultExecutionSpace>
      field_self("field_self", num_chunks);
  Kokkos::View<type_real *[chunk_size][nquad_edge][ncomp_coupled],
               Kokkos::DefaultExecutionSpace>
      field_coupled("field_coupled", num_chunks);
  Kokkos::View<type_real *[chunk_size][nquad_edge][nquad_intersection],
               Kokkos::DefaultExecutionSpace>
      transfer_self("transfer_self", num_chunks);
  Kokkos::View<type_real *[chunk_size][nquad_edge][nquad_intersection],
               Kokkos::DefaultExecutionSpace>
      transfer_coupled("transfer_coupled", num_chunks);
  Kokkos::View<type_real *[chunk_size][nquad_intersection][ncomp_self],
               Kokkos::DefaultExecutionSpace>
      computed_coupling("computed_coupling", num_chunks);
  typename decltype(normals)::HostMirror h_normals =
      Kokkos::create_mirror_view(normals);
  typename decltype(field_self)::HostMirror h_field_self =
      Kokkos::create_mirror_view(field_self);
  typename decltype(field_coupled)::HostMirror h_field_coupled =
      Kokkos::create_mirror_view(field_coupled);
  typename decltype(transfer_self)::HostMirror h_transfer_self =
      Kokkos::create_mirror_view(transfer_self);
  typename decltype(transfer_coupled)::HostMirror h_transfer_coupled =
      Kokkos::create_mirror_view(transfer_coupled);
  typename decltype(computed_coupling)::HostMirror h_computed_coupling =
      Kokkos::create_mirror_view(computed_coupling);

  // ====================================================================
  // initialize views
  // ====================================================================

  // TODO

  // ====================================================================
  // run kernel
  // ====================================================================
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(num_chunks, chunk_size)
          .set_scratch_size(
              0, Kokkos::PerTeam(InterfaceDataType::shmem_size() +
                                 CoupledFieldType::shmem_size() +
                                 InterfaceFieldViewType::shmem_size())),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        InterfaceDataType interface_data(team);
        CoupledFieldType coupled_field(team.team_scratch(0));
        InterfaceFieldViewType interface_field(team.team_scratch(0));

        const int ichunk = team.league_rank();

        // ====================================================================
        // initialize chunk_edge views
        // ====================================================================

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, chunk_size), [&](const auto &iedge) {
              for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                for (int iintersection = 0; iintersection < nquad_intersection;
                     iintersection++) {
                  interface_data.transfer_function_coupled(iedge, ipoint,
                                                           iintersection) =
                      transfer_self(ichunk, iedge, ipoint, iintersection);
                }

                for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
                  coupled_field(iedge, ipoint, icomp) =
                      field_coupled(ichunk, iedge, ipoint, icomp);
                }
              }

              for (int iintersection = 0; iintersection < nquad_intersection;
                   iintersection++) {
                interface_data.interface_normal(iedge, iintersection, 2) =
                    normals(ichunk, iedge, iintersection, 2);
              }
            });

        specfem::medium::compute_coupling(
            ChunkEdgeIndexSimulator(chunk_size, team), interface_data,
            coupled_field, interface_field);

        // ====================================================================
        // store output
        // ====================================================================

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, chunk_size * nquad_intersection *
                                              ncomp_self),
            [&](const auto &i) {
              const int icomp = i % ncomp_self;
              int tmp = i / ncomp_self;
              const int iintersection = i % nquad_intersection;
              const int iedge = i / nquad_intersection;
              computed_coupling(ichunk, iedge, iintersection, icomp) =
                  interface_field(iedge, iintersection, icomp);
            });
      });
}
