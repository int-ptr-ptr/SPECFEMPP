#define EDGEIND_NX 0
#define EDGEIND_NZ 1
#define EDGEIND_DET 2
#define EDGEIND_DS 3
#define EDGEIND_FIELD 4
#define EDGEIND_FIELDNDERIV 6
#define EDGEIND_SPEEDPARAM 8
#define EDGEIND_SHAPENDERIV 9
#define EDGEIND_BDRY_TYPE 14

#define data_capacity 20
#define edge_storage_quad                                                      \
  specfem::enums::element::quadrature::static_quadrature_points<5>
#define edge_capacity edge_storage_quad::NGLL
#define INTERIND_FLUXTOTAL_A 0
#define INTERIND_FLUX1_A (qp5.NGLL)
#define INTERIND_FLUX2_A (qp5.NGLL * 2)
#define INTERIND_FLUX3_A (qp5.NGLL * 3)
#define INTERIND_FLUXTOTAL_B (qp5.NGLL * 4)
#define INTERIND_FLUX1_B (qp5.NGLL * 5)
#define INTERIND_FLUX2_B (qp5.NGLL * 6)
#define INTERIND_FLUX3_B (qp5.NGLL * 7)
#define INTERIND_UJMP (qp5.NGLL * 8)
#define INTERIND_DU_AVG (qp5.NGLL * 9)
#define INTERIND_ACCEL_INCLUDE_A (qp5.NGLL * 10)
#define INTERIND_ACCEL_INCLUDE_B (qp5.NGLL * 11)

#define intersect_data_capacity (qp5.NGLL * (12))

int ISTEP;
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "kernels/kernels.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"

bool FORCE_INTO_CONTINUOUS;
bool USE_DOUBLEMESH;

// #define DEFAULT_USE_DOUBLEMESH
// #define USE_DEMO_MESH
// #define SET_INITIAL_CONDITION

#define KILL_NONNEUMANN_BDRYS

#define _RELAX_PARAM_COEF_ACOUSTIC_ 40
#define _RELAX_PARAM_COEF_ELASTIC_ 40

#define _EVENT_MARCHER_DUMPS_
#define _stepwise_simfield_dump_ std::string("dump/simfield")
#define _index_change_dump_ std::string("dump/indexchange")

#define _PARAMETER_FILENAME_DOUBLE_ std::string("specfem_config_double.yaml")
#define _PARAMETER_FILENAME_ std::string("specfem_config.yaml")

#include "_util/doublemesh.hpp"
#include "_util/dump_simfield.hpp"
#include "_util/edge_storages.hpp"
#include "_util/rewrite_simfield.hpp"
#include "event_marching/event_marcher.hpp"
#include "event_marching/timescheme_wrapper.hpp"

#include <cmath>
#include <iostream>
#include <string>

#define _DUMP_INTERVAL_ 5
_util::demo_assembly::simulation_params
load_parameters(const std::string &parameter_file, specfem::MPI::MPI *mpi);

void execute(specfem::MPI::MPI *mpi) {
  // TODO sources / receivers
  // https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html

  std::vector<specfem::adjacency_graph::adjacency_pointer> edge_removals;
#ifdef USE_DEMO_MESH
#define MATERIAL_MODE 0b0100
#define GRID_MODE 0b0001
  auto params =
      _util::demo_assembly::simulation_params().dt(1e-3).tmax(5).use_demo_mesh(
          MATERIAL_MODE + (FORCE_INTO_CONTINUOUS ? 0 : GRID_MODE),
          edge_removals);
  if (FORCE_INTO_CONTINUOUS) {
    edge_removals.clear();
  }
#else
  auto params = load_parameters(
      USE_DOUBLEMESH ? _PARAMETER_FILENAME_DOUBLE_ : _PARAMETER_FILENAME_, mpi);

#endif

  std::shared_ptr<specfem::compute::assembly> assembly = params.get_assembly();

#ifdef _EVENT_MARCHER_DUMPS_
  _util::init_dirs(_stepwise_simfield_dump_);
  _util::init_dirs(_index_change_dump_);

  _util::dump_simfield(_index_change_dump_ + "/prior_remap.dat",
                       assembly->fields.forward, assembly->mesh.points);
#endif

  // convert to compute indices
  for (int i = 0; i < edge_removals.size(); i++) {
    edge_removals[i].elem =
        assembly->mesh.mapping.mesh_to_compute(edge_removals[i].elem);
  }

  if (USE_DOUBLEMESH) {
    // add all edges along z = 0.5
    for (int ispec = 0; ispec < assembly->mesh.nspec; ispec++) {
      for (int ix = 0; ix < assembly->mesh.ngllx; ix++) {
        if (fabs(assembly->mesh.points.coord(1, ispec, 0, ix) - 0.5) < 1e-6) {
          edge_removals.push_back(
              specfem::adjacency_graph::adjacency_pointer(ispec, 3, 0));
          break;
        }
        if (fabs(assembly->mesh.points.coord(1, ispec, assembly->mesh.ngllz - 1,
                                             ix) -
                 0.5) < 1e-6) {
          edge_removals.push_back(
              specfem::adjacency_graph::adjacency_pointer(ispec, 1, 0));
          break;
        }
      }
    }
  }

  if (!FORCE_INTO_CONTINUOUS) {
    remap_with_disconts(*assembly, params, edge_removals, true);
  }

#ifdef _EVENT_MARCHER_DUMPS_
  _util::dump_simfield(_index_change_dump_ + "/post_remap.dat",
                       assembly->fields.forward, assembly->mesh.points);
#endif

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  auto kernels = specfem::kernels::kernels<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2,
      specfem::enums::element::quadrature::static_quadrature_points<5> >(
      params.get_dt(), *assembly, qp5);

  auto timescheme =
      specfem::time_scheme::newmark<specfem::simulation::type::forward>(
          params.get_numsteps(), 1, params.get_dt(), params.get_t0());
  timescheme.link_assembly(*assembly);

  params.set_plotters_from_runtime_configuration();

  auto event_system = specfem::event_marching::event_system();

  auto timescheme_wrapper =
      specfem::event_marching::timescheme_wrapper(timescheme);

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  timescheme_wrapper.set_forward_predictor_event(acoustic, 0);
  timescheme_wrapper.set_forward_predictor_event(elastic, 0.01);
  timescheme_wrapper.set_wavefield_update_event<acoustic>(kernels, 1);
  timescheme_wrapper.set_forward_corrector_event(acoustic, 2);
  timescheme_wrapper.set_wavefield_update_event<elastic>(kernels, 3);
  timescheme_wrapper.set_forward_corrector_event(elastic, 4);
  timescheme_wrapper.set_seismogram_update_event<
      specfem::wavefield::simulation_field::forward>(kernels, 5);
  timescheme_wrapper.set_plotter_update_event(params.get_plotters(), 5.1);

  timescheme_wrapper.register_under_marcher(&event_system);

  specfem::event_marching::arbitrary_call_event reset_timer(
      [&]() {
        ISTEP = timescheme_wrapper.get_istep();
        if (timescheme_wrapper.get_istep() < timescheme.get_max_timestep())
          event_system.set_current_precedence(
              specfem::event_marching::PRECEDENCE_BEFORE_INIT);
        return 0;
      },
      1);
  event_system.register_event(&reset_timer);

#ifdef SET_INITIAL_CONDITION
#define _IC_SIG 0.05
#define _IC_CENTER_X 0.3
#define _IC_CENTER_Z 0.6
  // initial condition
  set_field_disp<acoustic>(
      assembly->fields.forward, assembly->mesh, [](type_real x, type_real z) {
        x -= _IC_CENTER_X;
        z -= _IC_CENTER_Z;
        return (type_real)exp(-(x * x + z * z) / (2.0 * _IC_SIG * _IC_SIG));
      });
#undef _IC_SIG
#undef _IC_CENTER_X
#undef _IC_CENTER_Z
#endif

  // just populate dg_edges with edge_removals.
  auto edge_from_id = [&](const int8_t id) {
    switch (id) {
    case 0:
      return specfem::enums::edge::type::RIGHT;
    case 1:
      return specfem::enums::edge::type::TOP;
    case 2:
      return specfem::enums::edge::type::LEFT;
    case 3:
      return specfem::enums::edge::type::BOTTOM;
    default:
      return specfem::enums::edge::type::NONE;
    }
  };
  std::vector<_util::edge_manager::edge> dg_edges(edge_removals.size());
  for (int i = 0; i < edge_removals.size(); i++) {
    dg_edges[i].id = edge_removals[i].elem;
    dg_edges[i].bdry = edge_from_id(edge_removals[i].side);
    // medium is set in the edge_storage constructor. It need not be set here.
  }
  _util::edge_manager::edge_storage<edge_storage_quad, data_capacity>
      dg_edge_storage(dg_edges, *assembly);

  specfem::event_marching::arbitrary_call_event output_fields(
      [&]() {
        int istep = timescheme_wrapper.get_istep();
        if (istep % _DUMP_INTERVAL_ == 0) {
          _util::dump_simfield_per_step(istep, _stepwise_simfield_dump_ + "/d",
                                        *assembly, dg_edge_storage);
        }
        return 0;
      },
      -0.1);
#ifdef _EVENT_MARCHER_DUMPS_
  event_system.register_event(&output_fields);
#endif
  // geometric props
  for (int i = 0;
       i < dg_edge_storage.acoustic_acoustic_interface.num_medium1_edges; i++) {
    specfem::compute::loose::compute_geometry<1, false, false>(
        *assembly, dg_edge_storage.acoustic_acoustic_interface, i);
  }
  for (int i = 0;
       i < dg_edge_storage.acoustic_acoustic_interface.num_medium2_edges; i++) {
    specfem::compute::loose::compute_geometry<2, false, false>(
        *assembly, dg_edge_storage.acoustic_acoustic_interface, i);
  }
  for (int i = 0;
       i < dg_edge_storage.acoustic_acoustic_interface.num_interfaces; i++) {
    specfem::coupled_interface::loose::flux::symmetric_flux::kernel<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::acoustic, edge_storage_quad>::
        compute_relaxation_parameter<false>(
            i, *assembly, dg_edge_storage.acoustic_acoustic_interface);
    specfem::coupled_interface::loose::flux::symmetric_flux::kernel<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::medium_tag::acoustic, edge_storage_quad>::
        compute_mortar_trans_deriv<false>(
            i, *assembly, dg_edge_storage.acoustic_acoustic_interface);
  }
  // for (int i = 0;
  //      i < dg_edge_storage.acoustic_elastic_interface.num_medium1_edges; i++)
  //      {
  //   specfem::compute::loose::compute_geometry<1, false, false>(
  //       *assembly, dg_edge_storage.acoustic_elastic_interface, i);
  // }
  for (int i = 0;
       i < dg_edge_storage.acoustic_elastic_interface.num_medium2_edges; i++) {
    specfem::compute::loose::compute_geometry<2, false, false>(
        *assembly, dg_edge_storage.acoustic_elastic_interface, i);
  }
  for (int i = 0;
       i < dg_edge_storage.elastic_elastic_interface.num_medium1_edges; i++) {
    specfem::compute::loose::compute_geometry<1, false, false>(
        *assembly, dg_edge_storage.elastic_elastic_interface, i);
  }
  for (int i = 0;
       i < dg_edge_storage.elastic_elastic_interface.num_medium2_edges; i++) {
    specfem::compute::loose::compute_geometry<2, false, false>(
        *assembly, dg_edge_storage.elastic_elastic_interface, i);
  }

  dg_edge_storage.initialize_intersection_data(intersect_data_capacity);

  specfem::event_marching::arbitrary_call_event store_boundaryvals(
      [&]() {
        dg_edge_storage.acoustic_acoustic_interface
            .compute_edge_intermediates<1, false>(*assembly);
        dg_edge_storage.acoustic_acoustic_interface
            .compute_edge_intermediates<2, false>(*assembly);
        dg_edge_storage.acoustic_elastic_interface
            .compute_edge_intermediates<2, false>(*assembly);
        return 0;
      },
      0.9);
  timescheme_wrapper.time_stepper.register_event(&store_boundaryvals);

  specfem::event_marching::arbitrary_call_event mortar_flux_acoustic(
      [&]() {
        specfem::coupled_interface::loose::flux::traction_continuity::kernel<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::elastic, edge_storage_quad>::
            elastic_to_acoustic_accel(
                *assembly, dg_edge_storage.acoustic_elastic_interface);
        specfem::coupled_interface::loose::flux::symmetric_flux::kernel<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::acoustic, edge_storage_quad>::
            compute_fluxes(*assembly,
                           dg_edge_storage.acoustic_acoustic_interface);
        assembly->fields.forward.copy_to_device();
        return 0;
      },
      0.91);
  timescheme_wrapper.time_stepper.register_event(&mortar_flux_acoustic);

  specfem::event_marching::arbitrary_call_event mortar_flux_elastic(
      [&]() {
        specfem::coupled_interface::loose::flux::traction_continuity::kernel<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::medium_tag::elastic, edge_storage_quad>::
            acoustic_to_elastic_accel(
                *assembly, dg_edge_storage.acoustic_elastic_interface);

        assembly->fields.forward.copy_to_device();
        return 0;
      },
      1.1);
  timescheme_wrapper.time_stepper.register_event(&mortar_flux_elastic);

  specfem::event_marching::arbitrary_call_event write_outputs_at_end(
      [&]() {
        params.set_writers_from_runtime_configuration();
        for (auto writer : params.get_writers()) {
          if (writer) {
            mpi->cout("Writing Output:");
            mpi->cout("-------------------------------");

            writer->write();
          }
        }
        return 0;
      },
      2);
  event_system.register_event(&write_outputs_at_end);

  // init kernels needs to happen as of time_marcher
  kernels.initialize(timescheme.get_timestep());

  // ================================RUNTIME
  // BEGIN=================================
  event_system.run();
#undef edge_capacity
#undef data_capacity
}

// includes for load_parameters from specfem2d.cpp
#include "compute/interface.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "IO/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/solver.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

_util::demo_assembly::simulation_params
load_parameters(const std::string &parameter_file, specfem::MPI::MPI *mpi) {
  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  std::shared_ptr<specfem::runtime_configuration::setup> setup_ptr =
      std::make_shared<specfem::runtime_configuration::setup>(parameter_file,
                                                              __default_file__);
#define setup (*setup_ptr)
  const auto [database_filename, source_filename] = setup.get_databases();
  mpi->cout(setup.print_header(start_time));

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();

  auto mesh = USE_DOUBLEMESH ? _util::read_mesh(database_filename,
                                                database_filename + "_", mpi)
                             : specfem::IO::read_mesh(database_filename, mpi);

#ifdef KILL_NONNEUMANN_BDRYS

  specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2> bdry_abs(0);
  specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2> bdry_afs(
      0);
  specfem::mesh::forcing_boundary<specfem::dimension::type::dim2> bdry_forcing(
      0);
  mesh.boundaries = specfem::mesh::boundaries<specfem::dimension::type::dim2>(
      bdry_abs, bdry_afs, bdry_forcing);
  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);
#endif
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] = specfem::IO::read_sources(
      source_filename, nsteps, setup.get_t0(), setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  const auto stations_filename = setup.get_stations_file();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::IO::read_receivers(stations_filename, angle);

  mpi->cout("Source Information:");
  mpi->cout("-------------------------------");
  if (mpi->main_proc()) {
    std::cout << "Number of sources : " << sources.size() << "\n" << std::endl;
  }

  for (auto &source : sources) {
    mpi->cout(source->print());
  }

  mpi->cout("Receiver Information:");
  mpi->cout("-------------------------------");

  if (mpi->main_proc()) {
    std::cout << "Number of receivers : " << receivers.size() << "\n"
              << std::endl;
  }

  for (auto &receiver : receivers) {
    mpi->cout(receiver->print());
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme();
  if (mpi->main_proc())
    std::cout << *time_scheme << std::endl;

  const int max_seismogram_time_step = time_scheme->get_max_seismogram_step();
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  mpi->cout("Generating assembly:");
  mpi->cout("-------------------------------");
  const type_real dt = setup.get_dt();
  specfem::compute::assembly assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      setup.get_simulation_type());
  time_scheme->link_assembly(assembly);

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------

  const auto wavefield_reader = setup.instantiate_wavefield_reader(assembly);
  if (wavefield_reader) {
    mpi->cout("Reading wavefield files:");
    mpi->cout("-------------------------------");

    wavefield_reader->read();
    // Transfer the buffer field to device
    assembly.fields.buffer.copy_to_device();
  }

  _util::demo_assembly::simulation_params params;

  params.dt(dt)
      .nsteps(nsteps)
      .t0(setup.get_t0())
      .simulation_type(setup.get_simulation_type())
      .mesh(mesh)
      .quadrature(quadrature)
      .sources(sources)
      .receivers(receivers)
      .seismogram_types(setup.get_seismogram_types())
      .nseismogram_steps(max_seismogram_time_step)
      .assembly(assembly)
      .runtime_configuration(setup_ptr);
#undef setup
  return params;
}
int main(int argc, char **argv) {
#ifdef DEFAULT_USE_DOUBLEMESH
  USE_DOUBLEMESH = true;
#else
  USE_DOUBLEMESH = false;
#endif

  bool continuity_state_requested = false;
  bool continuity_desired = false;
  for (int iarg = 0; iarg < argc; iarg++) {
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'N' && argv[iarg][2] == 'O' &&
        argv[iarg][3] == 'D') {
      USE_DOUBLEMESH = false;
    }
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'D') {
      USE_DOUBLEMESH = true;
    }
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'N' && argv[iarg][2] == 'O' &&
        argv[iarg][3] == 'C') {
      continuity_state_requested = true;
      continuity_desired = false;
    }
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'C') {
      continuity_state_requested = true;
      continuity_desired = true;
    }
  }
  if (continuity_state_requested) {
    FORCE_INTO_CONTINUOUS = continuity_desired;
  } else {
    FORCE_INTO_CONTINUOUS = !USE_DOUBLEMESH;
  }

  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  if (USE_DOUBLEMESH) {
    mpi->cout("Using doublemesh\n");
  } else {
    mpi->cout("Using standard single-mesh\n");
  }
  if (FORCE_INTO_CONTINUOUS) {
    mpi->cout("Forcing into continuous domain\n");
  } else {
    mpi->cout("Using discontinuous domain\n");
  }
  { execute(mpi); }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
