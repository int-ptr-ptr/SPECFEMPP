#include "enumerations/medium.hpp"
#include "fixture.hpp"
#include "kokkos_kernels/domain_kernels.hpp"

#include "gtest/gtest.h"
#include <boost/graph/filtered_graph.hpp>
#include <stdexcept>

/**
 * For a given field acessor, clear those values.
 */
template <template <specfem::dimension::type, specfem::element::medium_tag,
                    bool> typename PointFieldType,
          specfem::dimension::type dimension>
void clear_field(specfem::assembly::assembly<dimension> &assembly) {
  const auto &field = assembly.fields.template get_simulation_field<
      specfem::wavefield::simulation_field::forward>();
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        const auto &elements =
            assembly.element_types.get_elements_on_host(_medium_tag_);
        const int nelem = elements.extent(0);

        PointFieldType<dimension, _medium_tag_, false> clear_point;
        for (int icomp = 0; icomp < decltype(clear_point)::components;
             icomp++) {
          clear_point(icomp) = 0;
        }

        for (int i = 0; i < nelem; i++) {
          const int ispec = elements(i);

          for (int iz = 0; iz < assembly.mesh.element_grid.ngllz; iz++) {
            for (int ix = 0; ix < assembly.mesh.element_grid.ngllx; ix++) {
              const specfem::point::index<dimension> index(ispec, iz, ix);
              specfem::assembly::store_on_host(index, field, clear_point);
            }
          }
        }
      })
}

/**
 * Assuming the fields have been set by the kernels, ensures that accelerations
 * are within a small tolerance. This is the
 * foreach(DoF) { assert accell fields close }
 * section of the test.
 */
void validate_field_at_points(
    specfem::assembly::simulation_field<
        specfem::dimension::type::dim2,
        specfem::wavefield::simulation_field::forward> &expected,
    specfem::assembly::simulation_field<
        specfem::dimension::type::dim2,
        specfem::wavefield::simulation_field::forward> &tested,
    const std::unordered_map<
        specfem::element::medium_tag,
        std::unordered_map<
            int, std::pair<
                     specfem::point::index<specfem::dimension::type::dim2>,
                     specfem::point::index<specfem::dimension::type::dim2> > > >
        &nodes_to_check_per_medium,
    const int &current_mesh_iglob, const int &current_icomp,
    const type_real &acceleration_scale) {

  specfem::point::acceleration<specfem::dimension::type::dim2,
                               specfem::element::medium_tag::acoustic, false>
      test_field;
  specfem::assembly::load_on_device(
      specfem::point::index<specfem::dimension::type::dim2>(8, 0, 3), tested,
      test_field);
  test_field(0) += std::sin(std::asin(1e-10 * test_field(0)));
  expected.copy_to_host();
  tested.copy_to_host();

  // give extra information only on the first failure
  bool first_failure = true;

  const auto iglob_elaboration =
      [&nodes_to_check_per_medium](std::ostringstream &oss, const int &iglob) {
        for (const auto &[medium, nodes_to_check] : nodes_to_check_per_medium) {
          const auto &found = nodes_to_check.find(iglob);
          if (found != nodes_to_check.end()) {
            const auto &index = found->second.first;
            oss << "[ispec = " << index.ispec << " ("
                << specfem::element::to_string(medium) << "), ix = " << index.ix
                << ", iz = " << index.iz << "]";
            break;
          }
        }
      };

  // ==================================
  // begin(foreach degree of freedom j)
  // ==================================
  for (const auto &[medium, nodes_to_check] : nodes_to_check_per_medium) {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          using PointAccelerationType =
              specfem::point::acceleration<_dimension_tag_, _medium_tag_,
                                           false>;
          if (medium == _medium_tag_) {
            PointAccelerationType test_read_accel;
            PointAccelerationType expect_read_accel;
            for (const auto &[nc_iglob, inds] : nodes_to_check) {
              const auto &[nc_index, c_index] = inds;

              specfem::assembly::load_on_host(c_index, expected,
                                              expect_read_accel);
              specfem::assembly::load_on_host(nc_index, tested,
                                              test_read_accel);

              for (int icomp = 0; icomp < PointAccelerationType::components;
                   icomp++) {
                // ===================================
                // (foreach degree of freedom j) then:
                // ===================================

                if (!specfem::utilities::is_close(
                        expect_read_accel(icomp), test_read_accel(icomp),
                        type_real(1e-4) * acceleration_scale,
                        type_real(1e-6) * acceleration_scale)) {

                  // overhaul this erroring if its needed when implementing
                  // kernels.

                  std::ostringstream oss;

                  oss << "shape function (iglob = " << current_mesh_iglob
                      << " ";
                  iglob_elaboration(oss, current_mesh_iglob);
                  oss << ", icomp = " << current_icomp
                      << ") with values at (iglob = " << nc_iglob << " ";
                  iglob_elaboration(oss, nc_iglob);
                  oss << ", icomp = " << icomp << ")\n";
                  oss << "    Got " << test_read_accel(icomp)
                      << "\n    Expected " << expect_read_accel(icomp);

                  oss << "  (" << std::showpos << std::scientific;
                  if (std::abs(expect_read_accel(icomp)) /
                          std::abs(1e-10 + test_read_accel(icomp)) >
                      1e-5) {
                    oss << (test_read_accel(icomp) - expect_read_accel(icomp)) /
                               expect_read_accel(icomp)
                        << " rel, ";
                  }
                  oss << (test_read_accel(icomp) - expect_read_accel(icomp)) /
                             acceleration_scale
                      << " * est. accel scale)" << std::fixed;
                  if (first_failure) {
                    oss << "\nshape function full index";
                    first_failure = false;
                  }
                  ADD_FAILURE() << oss.str();
                }

                // ================================
                // end(foreach degree of freedom j)
                // ================================
              }
            }
          }
        })
  }
}

void nonconforming_kernel_comparison(
    specfem::testing::kernel_compare::Test &test_config) {
  static constexpr int NGLL = 5;
  auto &c_assembly = test_config.conforming_assembly;
  auto &nc_assembly = test_config.nonconforming_assembly;
  EXPECT_EQ(nc_assembly.mesh.nspec, c_assembly.mesh.nspec)
      << "Number of elements do not match for test: " << test_config.name;
  EXPECT_EQ(nc_assembly.get_total_degrees_of_freedom(),
            c_assembly.get_total_degrees_of_freedom())
      << "Number of degrees of freedom do not match for test: "
      << test_config.name;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      {
        // the number of nonconforming edges on the conforming assembly should
        // be zero and vice versa.
        int n_nc_in_c =
            std::get<0>(c_assembly.edge_types.get_edges_on_host(
                            specfem::connections::type::nonconforming,
                            _interface_tag_, _boundary_tag_))
                .extent(0);
        EXPECT_EQ(n_nc_in_c, 0)
            << test_config.name
            << ": conforming assembly should not have any nonconforming "
               "edges. (interface = "
            << specfem::interface::to_string(_interface_tag_) << ", medium = "
            << specfem::element::to_string(
                   specfem::interface::attributes<
                       _dimension_tag_, _interface_tag_>::self_medium())
            << ", boundary = " << specfem::element::to_string(_boundary_tag_)
            << ")";
        int n_c_in_nc =
            std::get<0>(nc_assembly.edge_types.get_edges_on_host(
                            specfem::connections::type::weakly_conforming,
                            _interface_tag_, _boundary_tag_))
                .extent(0);
        EXPECT_EQ(n_c_in_nc, 0)
            << test_config.name
            << ": nonconforming assembly should not have any conforming "
               "edges. (interface = "
            << specfem::interface::to_string(_interface_tag_) << ", medium = "
            << specfem::element::to_string(
                   specfem::interface::attributes<
                       _dimension_tag_, _interface_tag_>::self_medium())
            << ", boundary = " << specfem::element::to_string(_boundary_tag_)
            << ")";
        // the number of edges should match between assemblies.

        int n_c_in_c =
            std::get<0>(c_assembly.edge_types.get_edges_on_host(
                            specfem::connections::type::weakly_conforming,
                            _interface_tag_, _boundary_tag_))
                .extent(0);
        int n_nc_in_nc =
            std::get<0>(nc_assembly.edge_types.get_edges_on_host(
                            specfem::connections::type::nonconforming,
                            _interface_tag_, _boundary_tag_))
                .extent(0);
        EXPECT_EQ(n_c_in_c, n_nc_in_nc)
            << test_config.name
            << ": assembly configurations do not have the same number of "
               "edges. (interface = "
            << specfem::interface::to_string(_interface_tag_) << ", medium = "
            << specfem::element::to_string(
                   specfem::interface::attributes<
                       _dimension_tag_, _interface_tag_>::self_medium())
            << ", boundary = " << specfem::element::to_string(_boundary_tag_)
            << ")";
      })

  // get indices to iterate over. items are (nc index, c index)
  std::unordered_map<
      specfem::element::medium_tag,
      std::unordered_map<
          int,
          std::pair<specfem::point::index<specfem::dimension::type::dim2>,
                    specfem::point::index<specfem::dimension::type::dim2> > > >
      nodes_to_check_per_medium;

  const auto &graph = nc_assembly.mesh.graph();
  auto filter = [&graph](const auto &edge) {
    return graph[edge].connection == specfem::connections::type::nonconforming;
  };
  const auto &nc_graph = boost::make_filtered_graph(graph, filter);

  for (const auto &edge : boost::make_iterator_range(boost::edges(nc_graph))) {
    // we do not assume assembly indices are the same, but mesh indices should
    // be.
    const int nc_ispec = boost::source(edge, nc_graph);
    const int mesh_ispec = nc_assembly.mesh.compute_to_mesh(nc_ispec);
    const int c_ispec = c_assembly.mesh.mesh_to_compute(mesh_ispec);

    const auto medium = nc_assembly.element_types.get_medium_tag(nc_ispec);
    EXPECT_EQ(medium, c_assembly.element_types.get_medium_tag(c_ispec))
        << "Element mesh_ispec = " << mesh_ispec
        << " has unequal media in what should be the same element."
           " nonconforming (assembly ispec: "
        << nc_ispec << ", medium = " << specfem::element::to_string(medium)
        << ") vs conforming (assembly ispec: " << c_ispec << ", medium = "
        << specfem::element::to_string(
               c_assembly.element_types.get_medium_tag(c_ispec))
        << ")";

    auto &nodes_to_check = [&]()
        -> std::unordered_map<
            int,
            std::pair<specfem::point::index<specfem::dimension::type::dim2>,
                      specfem::point::index<specfem::dimension::type::dim2> > >
            & {
              auto nodelist_it = nodes_to_check_per_medium.find(medium);
              if (nodelist_it == nodes_to_check_per_medium.end()) {
                return (nodes_to_check_per_medium.insert({ medium, {} }).first)
                    ->second;
              } else {
                return nodelist_it->second;
              }
            }();

    // add each iglob that is part of this element if not already added.
    for (int iz = 0; iz < nc_assembly.mesh.element_grid.ngllz; iz++) {
      for (int ix = 0; ix < nc_assembly.mesh.element_grid.ngllx; ix++) {
        const int nc_iglob = nc_assembly.mesh.h_index_mapping(nc_ispec, iz, ix);
        if (nodes_to_check.find(nc_iglob) == nodes_to_check.end()) {
          nodes_to_check.insert(
              { nc_iglob, { { nc_ispec, iz, ix }, { c_ispec, iz, ix } } });
        }
      }
    }
  }

  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2, NGLL>
      nc_kernels(nc_assembly);
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2, NGLL>
      c_kernels(c_assembly);

  nc_kernels.initialize(0.0);
  c_kernels.initialize(0.0);

  auto nc_field = nc_assembly.fields.template get_simulation_field<
      specfem::wavefield::simulation_field::forward>();
  auto c_field = c_assembly.fields.template get_simulation_field<
      specfem::wavefield::simulation_field::forward>();

  clear_field<specfem::point::displacement>(nc_assembly);
  clear_field<specfem::point::displacement>(c_assembly);
  clear_field<specfem::point::velocity>(nc_assembly);
  clear_field<specfem::point::velocity>(c_assembly);
  clear_field<specfem::point::acceleration>(nc_assembly);
  clear_field<specfem::point::acceleration>(c_assembly);

  // ==================================
  // begin(foreach degree of freedom i)
  // ==================================
  for (const auto &[medium, nodes_to_check] : nodes_to_check_per_medium) {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                         POROELASTIC, ELASTIC_PSV_T)),
        {
          using PointDisplacementType =
              specfem::point::displacement<_dimension_tag_, _medium_tag_,
                                           false>;
          if (medium == _medium_tag_) {

            PointDisplacementType displacement;

            for (int icomp = 0; icomp < PointDisplacementType::components;
                 icomp++) {
              displacement(icomp) = 0;
            }

            for (const auto &[nc_iglob, inds] : nodes_to_check) {
              const auto &[nc_index, c_index] = inds;
              for (int icomp = 0; icomp < PointDisplacementType::components;
                   icomp++) {
                displacement(icomp) = 1;
                // ===================================
                // (foreach degree of freedom i) then:
                // ===================================

                clear_field<specfem::point::acceleration>(nc_assembly);
                clear_field<specfem::point::acceleration>(c_assembly);

                specfem::assembly::store_on_host(nc_index, nc_field,
                                                 displacement);
                specfem::assembly::store_on_host(c_index, c_field,
                                                 displacement);

                nc_field.copy_to_device();
                c_field.copy_to_device();

                // one will skip the printouts
                nc_kernels.template update_wavefields<
                    specfem::element::medium_tag::acoustic>(1);
                c_kernels.template update_wavefields<
                    specfem::element::medium_tag::acoustic>(1);

                nc_kernels.template update_wavefields<
                    specfem::element::medium_tag::elastic_psv>(1);
                c_kernels.template update_wavefields<
                    specfem::element::medium_tag::elastic_psv>(1);

                // recover the maximum expected value for a scale
                type_real acceleration_scale = 1e-6;

                constexpr specfem::element::medium_tag other_side_medium =
                    (_medium_tag_ == specfem::element::medium_tag::acoustic)
                        ? specfem::element::medium_tag::elastic_psv
                        : specfem::element::medium_tag::acoustic;
                using PointAccelType =
                    specfem::point::acceleration<_dimension_tag_,
                                                 other_side_medium, false>;
                PointAccelType accel_sample;
                for (const auto &[nc_iglob_, inds_] :
                     nodes_to_check_per_medium[other_side_medium]) {
                  const auto &[nc_index_, c_index_] = inds_;

                  specfem::assembly::load_on_host(c_index_, c_field,
                                                  accel_sample);
                  for (int icomp_ = 0; icomp_ < PointAccelType::components;
                       icomp_++) {
                    acceleration_scale = std::max(
                        acceleration_scale, std::abs(accel_sample(icomp_)));
                  }
                }

                validate_field_at_points(c_field, nc_field,
                                         nodes_to_check_per_medium, nc_iglob,
                                         icomp, acceleration_scale);
                // ================================
                // end(foreach degree of freedom i)
                // ================================
                displacement(icomp) = 0;
                specfem::assembly::store_on_host(nc_index, nc_field,
                                                 displacement);
                specfem::assembly::store_on_host(c_index, c_field,
                                                 displacement);
              }
            }
          }
        })
  }
}

using NonconformingConformingMeshes =
    specfem::testing::kernel_compare::mesh_list;

TEST_F(NonconformingConformingMeshes, Test) {
  for (auto &test_config : *this) {

    nonconforming_kernel_comparison(test_config);
  }
}
