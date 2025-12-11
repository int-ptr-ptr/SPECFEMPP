#include "enumerations/dimension.hpp"
#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/chunk_edge.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag, int chunk_size,
          int nquad_edge, int nquad_intersection>
struct ComputeCouplingKernelStorage {
public:
  static constexpr auto self_medium =
      specfem::interface::attributes<DimensionTag, InterfaceTag>::self_medium();
  static constexpr auto coupled_medium =
      specfem::interface::attributes<DimensionTag,
                                     InterfaceTag>::coupled_medium();
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;
  static constexpr int ncomp_self =
      specfem::element::attributes<DimensionTag, self_medium>::components;
  static constexpr int ncomp_coupled =
      specfem::element::attributes<DimensionTag, coupled_medium>::components;

  const Kokkos::View<type_real *[chunk_size][nquad_intersection][ndim],
                     Kokkos::DefaultExecutionSpace>
      normals;
  const Kokkos::View<type_real *[chunk_size][nquad_edge][ncomp_self],
                     Kokkos::DefaultExecutionSpace>
      field_self;
  const Kokkos::View<type_real *[chunk_size][nquad_edge][ncomp_coupled],
                     Kokkos::DefaultExecutionSpace>
      field_coupled;
  const Kokkos::View<type_real *[chunk_size][nquad_edge][nquad_intersection],
                     Kokkos::DefaultExecutionSpace>
      transfer_self;
  const Kokkos::View<type_real *[chunk_size][nquad_edge][nquad_intersection],
                     Kokkos::DefaultExecutionSpace>
      transfer_coupled;
  const Kokkos::View<type_real *[chunk_size][nquad_intersection][ncomp_self],
                     Kokkos::DefaultExecutionSpace>
      computed_coupling;
  const Kokkos::View<type_real *[chunk_size][nquad_intersection][ncomp_self],
                     Kokkos::DefaultExecutionSpace>
      expected_coupling;
  const typename decltype(normals)::HostMirror h_normals;
  const typename decltype(field_self)::HostMirror h_field_self;
  const typename decltype(field_coupled)::HostMirror h_field_coupled;
  const typename decltype(transfer_self)::HostMirror h_transfer_self;
  const typename decltype(transfer_coupled)::HostMirror h_transfer_coupled;
  const typename decltype(computed_coupling)::HostMirror h_computed_coupling;
  const typename decltype(expected_coupling)::HostMirror h_expected_coupling;
  const int num_chunks;

  ComputeCouplingKernelStorage(const int &num_chunks)
      : num_chunks(num_chunks), normals("normals", num_chunks),
        field_self("field_self", num_chunks),
        field_coupled("field_coupled", num_chunks),
        transfer_self("transfer_self", num_chunks),
        transfer_coupled("transfer_coupled", num_chunks),
        computed_coupling("computed_coupling", num_chunks),
        expected_coupling("expected_coupling", num_chunks),
        h_normals(Kokkos::create_mirror_view(normals)),
        h_field_self(Kokkos::create_mirror_view(field_self)),
        h_field_coupled(Kokkos::create_mirror_view(field_coupled)),
        h_transfer_self(Kokkos::create_mirror_view(transfer_self)),
        h_transfer_coupled(Kokkos::create_mirror_view(transfer_coupled)),
        h_computed_coupling(Kokkos::create_mirror_view(computed_coupling)),
        h_expected_coupling(Kokkos::create_mirror_view(expected_coupling)) {}

  void sync_to_device() const {
    Kokkos::deep_copy(normals, h_normals);
    Kokkos::deep_copy(field_self, h_field_self);
    Kokkos::deep_copy(field_coupled, h_field_coupled);
    Kokkos::deep_copy(transfer_self, h_transfer_self);
    Kokkos::deep_copy(transfer_coupled, h_transfer_coupled);
    Kokkos::deep_copy(computed_coupling, h_computed_coupling);
    Kokkos::deep_copy(expected_coupling, h_expected_coupling);
  }
  void sync_to_host() const {
    Kokkos::deep_copy(h_normals, normals);
    Kokkos::deep_copy(h_field_self, field_self);
    Kokkos::deep_copy(h_field_coupled, field_coupled);
    Kokkos::deep_copy(h_transfer_self, transfer_self);
    Kokkos::deep_copy(h_transfer_coupled, transfer_coupled);
    Kokkos::deep_copy(h_computed_coupling, computed_coupling);
    Kokkos::deep_copy(h_expected_coupling, expected_coupling);
  }
};

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
void compute_kernel(
    const ComputeCouplingKernelStorage<DimensionTag, InterfaceTag, chunk_size,
                                       nquad_edge, nquad_intersection>
        &kernel_data) {

  constexpr bool using_simd = false;
  static constexpr auto self_medium =
      specfem::interface::attributes<DimensionTag, InterfaceTag>::self_medium();
  static constexpr auto coupled_medium =
      specfem::interface::attributes<DimensionTag,
                                     InterfaceTag>::coupled_medium();
  using InterfaceDataType = specfem::chunk_edge::coupling_terms_pack<
      specfem::dimension::type::dim2, InterfaceTag,
      specfem::element::boundary_tag::none, chunk_size, nquad_edge,
      nquad_intersection>;
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

  const int &num_chunks = kernel_data.num_chunks;
  const int &ncomp_coupled = kernel_data.ncomp_coupled;
  const int &ncomp_self = kernel_data.ncomp_self;

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

        // we will need to load more things for different schemes. Handle that
        // later.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, chunk_size), [&](const auto &iedge) {
              for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                for (int iintersection = 0; iintersection < nquad_intersection;
                     iintersection++) {

                  // transfer_function_coupled
                  static_cast<decltype(std::get<0>(
                      typename InterfaceDataType::packed_accessors()))>(
                      interface_data)(iedge, ipoint, iintersection) =
                      kernel_data.transfer_coupled(ichunk, iedge, ipoint,
                                                   iintersection);
                }

                for (int icomp = 0; icomp < ncomp_coupled; icomp++) {
                  coupled_field(iedge, ipoint, icomp) =
                      kernel_data.field_coupled(ichunk, iedge, ipoint, icomp);
                }
              }

              for (int iintersection = 0; iintersection < nquad_intersection;
                   iintersection++) {
                for (int icomp = 0; icomp < ndim; icomp++) {
                  // intersection_normal
                  static_cast<decltype(std::get<1>(
                      typename InterfaceDataType::packed_accessors()))>(
                      interface_data)(iedge, iintersection, icomp) =
                      kernel_data.normals(ichunk, iedge, iintersection, icomp);
                }
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
              const int iintersection = tmp % nquad_intersection;
              const int iedge = tmp / nquad_intersection;
              kernel_data.computed_coupling(ichunk, iedge, iintersection,
                                            icomp) =
                  interface_field(iedge, iintersection, icomp);
            });
      });
}
