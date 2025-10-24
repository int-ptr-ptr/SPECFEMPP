#include "Kokkos_Environment.hpp"
#include "medium/compute_coupling.hpp"
#include "specfem/chunk_edge.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>

struct EdgeToInterfaceParamsBase {
  virtual void run_test(const std::string &testname) const {
    throw std::runtime_error(
        testname +
        std::string(": Called EdgeToInterfaceParamsBase::run_test(). No test "
                    "should be initialized with this type."));
  }
};

template <std::size_t nquad>
KOKKOS_FORCEINLINE_FUNCTION type_real
eval_lagrange(const std::array<type_real, nquad> &quadrature_points,
              const int &poly_index, const type_real &x) {
  type_real val = 1;
  for (int i = 0; i < nquad; i++) {
    if (i != poly_index) {
      val *= (x - quadrature_points[i]) /
             (quadrature_points[poly_index] - quadrature_points[i]);
    }
  }
  return val;
}

template <int num_edges_, specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag, int nquad_edge_,
          int nquad_intersection_>
struct EdgeToInterfaceParams : EdgeToInterfaceParamsBase {
  static constexpr specfem::dimension::type dimension_tag = DimensionTag;
  static constexpr specfem::interface::interface_tag interface_tag =
      InterfaceTag;
  static constexpr int num_edges = num_edges_;
  static constexpr int nquad_edge = nquad_edge_;
  static constexpr int nquad_intersection = nquad_intersection_;

  std::array<type_real, nquad_edge_> edge_quadrature_points;
  std::array<type_real, nquad_intersection_> intersection_quadrature_points;

  EdgeToInterfaceParams(
      const type_real (&edge_quadrature_points)[nquad_edge_],
      const type_real (&intersection_quadrature_points)[nquad_intersection_]) {
    std::copy(std::begin(edge_quadrature_points),
              std::end(edge_quadrature_points),
              std::begin(this->edge_quadrature_points));
    std::copy(std::begin(intersection_quadrature_points),
              std::end(intersection_quadrature_points),
              std::begin(this->intersection_quadrature_points));
  }

  type_real tolerance() const { return 1e-6 * num_edges; }

  virtual void run_test(const std::string &testname) const {
    using SelfTransferType =
        specfem::chunk_edge::nonconforming_transfer_function<
            true, num_edges, nquad_edge, nquad_intersection, dimension_tag,
            specfem::connections::type::nonconforming, interface_tag,
            specfem::element::boundary_tag::none,
            specfem::kokkos::DevScratchSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>, false>;
    using CoupledTransferType =
        specfem::chunk_edge::nonconforming_transfer_function<
            false, num_edges, nquad_edge, nquad_intersection, dimension_tag,
            specfem::connections::type::nonconforming, interface_tag,
            specfem::element::boundary_tag::none,
            specfem::kokkos::DevScratchSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>, false>;

    constexpr auto self_medium =
        specfem::interface::attributes<dimension_tag,
                                       interface_tag>::self_medium();
    constexpr auto coupled_medium =
        specfem::interface::attributes<dimension_tag,
                                       interface_tag>::coupled_medium();

    using SelfDisplacementType =
        specfem::chunk_edge::displacement<num_edges, nquad_edge, dimension_tag,
                                          self_medium, false>;
    using CoupledDisplacementType =
        specfem::chunk_edge::displacement<num_edges, nquad_edge, dimension_tag,
                                          coupled_medium, false>;

    using SelfOnInterfaceType = specfem::datatype::VectorChunkEdgeViewType<
        type_real, specfem::dimension::type::dim2, num_edges,
        nquad_intersection,
        specfem::element::attributes<DimensionTag, self_medium>::components,
        false, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    using CoupledOnInterfaceType = specfem::datatype::VectorChunkEdgeViewType<
        type_real, specfem::dimension::type::dim2, num_edges,
        nquad_intersection,
        specfem::element::attributes<DimensionTag, coupled_medium>::components,
        false, specfem::kokkos::DevScratchSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    const type_real tol = tolerance();
    // transfer 1, x, x^2, ...
    constexpr int deg_cap = std::min(nquad_edge, nquad_intersection);
    const int num_iters = 1 + (deg_cap / num_edges);

    // check for failures through this.
    // We only store the last fail in each thread.
    using CheckContainer = Kokkos::View<
        std::tuple<bool, type_real, type_real, type_real, int, int, int> **,
        Kokkos::DefaultExecutionSpace>;
    CheckContainer self_check_container("self-check container", num_iters,
                                        num_edges);
    CheckContainer coupled_check_container("coupled-check container", num_iters,
                                           num_edges);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(num_iters, num_edges),
        KOKKOS_LAMBDA(const auto &team) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges),
              KOKKOS_LAMBDA(const auto &iedge) {
                std::get<0>(self_check_container(team.league_rank(), iedge)) =
                    false;
                std::get<0>(
                    coupled_check_container(team.league_rank(), iedge)) = false;
              });
        });

    CheckContainer::HostMirror h_self_check_container =
        Kokkos::create_mirror_view(self_check_container);
    CheckContainer::HostMirror h_coupled_check_container =
        Kokkos::create_mirror_view(coupled_check_container);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(num_iters, num_edges)
            .set_scratch_size(
                0, Kokkos::PerTeam(SelfTransferType::shmem_size() +
                                   CoupledTransferType::shmem_size() +
                                   SelfDisplacementType::shmem_size() +
                                   CoupledDisplacementType::shmem_size() +
                                   SelfOnInterfaceType::shmem_size() +
                                   CoupledOnInterfaceType::shmem_size())),
        KOKKOS_LAMBDA(const auto &team) {
          const int deg_start = team.league_rank() * num_edges;
          SelfDisplacementType self_disp(team.team_scratch(0));
          CoupledDisplacementType coupled_disp(team.team_scratch(0));
          SelfTransferType self_transfer(team);
          CoupledTransferType coupled_transfer(team);
          SelfOnInterfaceType self_on_interface(team.team_scratch(0));
          CoupledOnInterfaceType coupled_on_interface(team.team_scratch(0));

          // compute transfers

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges),
              KOKKOS_LAMBDA(const auto &iedge) {
                for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                  for (int iintersection = 0;
                       iintersection < nquad_intersection; iintersection++) {
                    type_real eval = eval_lagrange(
                        edge_quadrature_points, ipoint,
                        intersection_quadrature_points[iintersection]);
                    self_transfer.transfer_function_self(iedge, ipoint,
                                                         iintersection) = eval;
                    coupled_transfer.transfer_function_coupled(
                        iedge, ipoint, iintersection) = eval;
                  }
                }
              });

          // populate fields with polynomials

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges),
              KOKKOS_LAMBDA(const auto &iedge) {
                for (int ipoint = 0; ipoint < nquad_edge; ipoint++) {
                  for (int icomp = 0; icomp < SelfDisplacementType::components;
                       icomp++)
                    self_disp(iedge, ipoint, icomp) =
                        std::pow(edge_quadrature_points[ipoint],
                                 (deg_start + iedge + icomp) % deg_cap);
                  for (int icomp = 0;
                       icomp < CoupledDisplacementType::components; icomp++)
                    coupled_disp(iedge, ipoint, icomp) =
                        std::pow(edge_quadrature_points[ipoint],
                                 (deg_start + iedge + icomp) % deg_cap);
                }
              });

          // validate

          specfem::medium::compute_coupling(self_transfer, self_disp,
                                            self_on_interface);
          specfem::medium::compute_coupling(coupled_transfer, coupled_disp,
                                            coupled_on_interface);

          // transfer should send polys to themselves, but in the new basis.
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_edges),
              KOKKOS_LAMBDA(const auto &iedge) {
                for (int iintersection = 0; iintersection < nquad_intersection;
                     iintersection++) {
                  const type_real x =
                      intersection_quadrature_points[iintersection];
                  for (int icomp = 0; icomp < SelfDisplacementType::components;
                       icomp++) {
                    const int deg = (deg_start + iedge + icomp) % deg_cap;
                    const type_real expect = std::pow(x, deg);
                    const type_real got =
                        self_on_interface(iedge, iintersection, icomp);
                    if (std::abs(expect - got) > tol) {
                      self_check_container(team.league_rank(), iedge) =
                          std::make_tuple(true, x, expect, got, deg, icomp,
                                          iintersection);
                    }
                  }
                  for (int icomp = 0;
                       icomp < CoupledDisplacementType::components; icomp++) {
                    const int deg = (deg_start + iedge + icomp) % deg_cap;
                    const type_real expect = std::pow(x, deg);
                    const type_real got =
                        coupled_on_interface(iedge, iintersection, icomp);
                    if (std::abs(expect - got) > tol) {
                      coupled_check_container(team.league_rank(), iedge) =
                          std::make_tuple(true, x, expect, got, deg, icomp,
                                          iintersection);
                    }
                  }
                }
              });
        });

    Kokkos::deep_copy(h_self_check_container, self_check_container);
    Kokkos::deep_copy(h_coupled_check_container, coupled_check_container);

    for (int ileague = 0; ileague < num_iters; ileague++) {
      for (int iedge = 0; iedge < num_edges; iedge++) {
        {
          const auto [is_fail, x, expect, got, deg, icomp, iintersection] =
              h_self_check_container(ileague, iedge);
          if (is_fail) {
            EXPECT_NEAR(got, expect, tol)
                << "Self side: Computing " << x << " ^ " << deg
                << " for iedge = " << iedge << ", icomp = " << icomp
                << " for intersection quadrature point " << iintersection;
          }
        }
        {
          const auto [is_fail, x, expect, got, deg, icomp, iintersection] =
              h_coupled_check_container(ileague, iedge);
          if (is_fail) {
            EXPECT_NEAR(got, expect, tol)
                << "Coupled side: Computing " << x << " ^ " << deg
                << " for iedge = " << iedge << ", icomp = " << icomp
                << " for intersection quadrature point " << iintersection;
          }
        }
      }
    }

    //
    //
    // for(int ideg = 0; ideg < deg_cap; ideg += num_edges){
    //   // do multiple degrees at once

    // }

    // // Create interface data
    // specfem::point::coupled_interface<
    //     specfem::dimension::type::dim2,
    //     specfem::connections::type::weakly_conforming,
    //     specfem::interface::interface_tag::acoustic_elastic,
    //     specfem::element::boundary_tag::none>
    //     interface_data(params.edge_factor,
    //                    { params.normal[0], params.normal[1] });

    // // Create coupled field (displacement from acoustic medium)
    // specfem::point::displacement<specfem::dimension::type::dim2,
    //                              specfem::element::medium_tag::elastic_psv,
    //                              false>
    //     coupled_field;
    // coupled_field(0) = params.displacement_value[0];
    // coupled_field(1) = params.displacement_value[1];

    // // Create self field (acceleration in elastic medium)
    // specfem::point::acceleration<specfem::dimension::type::dim2,
    //                              specfem::element::medium_tag::acoustic,
    //                              false>
    //     self_field;

    // // Perform coupling computation
    // specfem::medium::compute_coupling(interface_data, coupled_field,
    // self_field);

    // // Verify result
    // EXPECT_NEAR(self_field(0), params.expected_result, params.tolerance);
  }
};

struct EdgeToInterfaceCouplingTestParams {
  std::string name;
  std::shared_ptr<EdgeToInterfaceParamsBase> params;

  void run_test() const { params->run_test(name); }

  template <int num_edges, specfem::dimension::type DimensionTag,
            specfem::interface::interface_tag InterfaceTag,
            std::size_t nquad_edge, std::size_t nquad_intersection>
  static EdgeToInterfaceCouplingTestParams
  from(const std::string &name, const type_real (&edge)[nquad_edge],
       const type_real (&intersection)[nquad_intersection]) {
    return {
      name, std::make_shared<EdgeToInterfaceParams<
                ((std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>)
                     ? 1
                     : num_edges),
                DimensionTag, InterfaceTag, nquad_edge, nquad_intersection> >(
                edge, intersection)
    };
  }
};

class EdgeToInterfaceCouplingTest
    : public ::testing::TestWithParam<EdgeToInterfaceCouplingTestParams> {};

std::ostream &operator<<(std::ostream &os,
                         const EdgeToInterfaceCouplingTestParams &params) {
  os << params.name;
  return os;
}

TEST_P(EdgeToInterfaceCouplingTest, CouplingCalculation) {
  GetParam().run_test();
}

using init_type = std::tuple<std::string, EdgeToInterfaceParamsBase>;

INSTANTIATE_TEST_SUITE_P(
    NonconformingVariations, EdgeToInterfaceCouplingTest,
    ::testing::Values(EdgeToInterfaceCouplingTestParams::from<
                          1, specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::elastic_acoustic>(
                          "dim2 elastic-acoustic", { -1, 1 }, { -0.5, 0, 0.5 }),
                      EdgeToInterfaceCouplingTestParams::from<
                          8, specfem::dimension::type::dim2,
                          specfem::interface::interface_tag::acoustic_elastic>(
                          "dim2 acoustic-elastic", { -1, -0.5, 0, 0.7, 1.2 },
                          { -0.3, 0, 0.4, 0.6 })));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
