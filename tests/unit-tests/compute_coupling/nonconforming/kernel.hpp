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

        specfem::medium::compute_coupling(
            ChunkEdgeIndexSimulator(chunk_size, team), interface_data,
            coupled_field, interface_field);
      });
}
