#define ___DEBUG___
#define EDGEIND_NX 0
#define EDGEIND_NZ 1
#define EDGEIND_DET 2
#define EDGEIND_DS 3
#define EDGEIND_FIELD 4
#define EDGEIND_FIELDNDERIV 6
#define EDGEIND_SPEEDPARAM 8
#define EDGEIND_SHAPENDERIV 9
#define EDGEIND_BDRY_TYPE 14

#include "_util/quadrature_template_type.hpp"

#define data_capacity 20
#define QP5_NGLL 5
#define edge_storage_quad _util::static_quadrature_points<QP5_NGLL>
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
#include "_util/periodify.hpp"
#include "_util/sourcebd.hpp"
#include "compute/assembly/assembly.hpp"
#include "constants.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"

bool FORCE_INTO_CONTINUOUS = false;
int DUMP_INTERVAL = -1;
bool USE_DOUBLEMESH = false;
bool LR_PERIODIC = false;
bool KILL_NONNEUMANN_BDRYS = false;
bool TOP_ABS = false;
bool BOT_ABS = false;

// #define DEFAULT_USE_DOUBLEMESH
#define DEFAULT_SET_CONTINUOUS false
// #define USE_DEMO_MESH

#define _RELAX_PARAM_COEF_ACOUSTIC_ 40
#define _RELAX_PARAM_COEF_ELASTIC_ 40

#define _EVENT_MARCHER_DUMPS_
#define _stepwise_simfield_dump_ std::string("dump/simfield")
std::string DUMP_OUTFOL = _stepwise_simfield_dump_;

#define _PARAMETER_FILENAME_DOUBLE_ std::string("specfem_config_double.yaml")
#define _PARAMETER_FILENAME_ std::string("specfem_config.yaml")

#include "_util/dump_simfield.hpp"
#include "_util/edge_storages.hpp"
// #include "_util/rewrite_simfield.hpp"
#include "event_marching/event_marcher.hpp"
#include "event_marching/timescheme_wrapper.hpp"

#include <cmath>
#include <iostream>
#include <string>

std::string param_fname = _PARAMETER_FILENAME_;

#define _DUMP_INTERVAL_ 5
_util::demo_assembly::simulation_params
load_parameters(const std::string &parameter_file, specfem::MPI::MPI *mpi);

void execute(specfem::MPI::MPI *mpi) {
  // TODO sources / receivers
  // https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html

  std::vector<specfem::adjacency_graph::adjacency_pointer> edge_removals;
  USE_DOUBLEMESH = false;
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
  auto params = load_parameters(param_fname, mpi);
  USE_DOUBLEMESH = false;
#endif

  std::shared_ptr<specfem::compute::assembly> assembly = params.get_assembly();

#ifdef _EVENT_MARCHER_DUMPS_
  if (DUMP_INTERVAL >= 0) {
    _util::init_dirs(DUMP_OUTFOL);
  }
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

  // if (!FORCE_INTO_CONTINUOUS) {
  //   remap_with_disconts(*assembly, params, edge_removals, true);
  // }

#ifdef _EVENT_MARCHER_DUMPS_
  if (DUMP_INTERVAL >= 0) {
    mpi->cout("dumping with interval " + std::to_string(DUMP_INTERVAL));
  } else {
    mpi->cout("no dumps");
  }
#endif

  edge_storage_quad qp5;
  auto kernels = specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2, QP5_NGLL>(*assembly);

  auto timescheme =
      specfem::time_scheme::newmark<specfem::simulation::type::forward>(
          params.get_numsteps(), params.get_num_steps_between_samples(),
          params.get_dt(), params.get_t0());
  const type_real dt = params.get_dt();
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
  timescheme_wrapper.set_periodic_tasks_event(params.get_periodic_tasks(), 5.1);

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
  if (!FORCE_INTO_CONTINUOUS) {
    auto &adj = assembly->mesh.points.adjacencies;
    const int nspec = assembly->mesh.nspec;
    for (int ispec = 0; ispec < nspec; ispec++) {
      for (int iedge = 0; iedge < 4; iedge++) {
        const auto edge = adj.edge_from_index(iedge);
        if (!(adj.has_conforming_adjacency<false>(ispec, edge) ||
              adj.has_boundary<false>(ispec, edge))) {
          dg_edges.push_back(_util::edge_manager::edge(ispec, edge));
        }
      }
    }
  }

  _util::edge_manager::edge_storage<edge_storage_quad, data_capacity>
      dg_edge_storage(dg_edges, *assembly);

  specfem::event_marching::arbitrary_call_event output_fields(
      [&]() {
        int istep = timescheme_wrapper.get_istep();
        if (istep % DUMP_INTERVAL == 0) {
          mpi->cout("dumping @ step " + std::to_string(istep));
          _util::dump_simfield_per_step(istep, DUMP_OUTFOL + "/d", *assembly,
                                        dg_edge_storage);
        }
        return 0;
      },
      -0.1);
#ifdef _EVENT_MARCHER_DUMPS_

  if (DUMP_INTERVAL >= 0) {
    event_system.register_event(&output_fields);
  }
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

  if (LR_PERIODIC) {
    _util::periodify_LR(*assembly);
  }
  auto wave_inject_acoustic =
      _util::sourceboundary::kernel<acoustic>(*assembly, TOP_ABS, BOT_ABS);
  auto wave_inject_elastic =
      _util::sourceboundary::kernel<elastic>(*assembly, TOP_ABS, BOT_ABS);

  specfem::event_marching::arbitrary_call_event inject_acoustic_event(
      [&]() {
        // const int istep = timescheme_wrapper.get_istep();
        // const type_real t = istep * dt;
        // constexpr type_real c = 1;
        // constexpr type_real wind_up = 0.5;
        // constexpr type_real max_amp = 20;
        // const type_real pi = std::atan(1) * 4;
        // // const type_real amp = (t <= 0)? 0:((t < wind_up)?
        // // (1-std::cos(pi*t/wind_up))*max_amp:max_amp);
        // const type_real amp = max_amp;
        // const type_real kx = 2 * pi;
        // const type_real kz = 10;
        // const type_real k = std::sqrt(kx * kx + kz * kz);
        // const type_real omega = c * k;
        // wave_inject_acoustic.force_planar_wave(0, 0, omega * t, amp);
        wave_inject_acoustic.absorb();
        wave_inject_elastic.absorb();
        return 0;
      },
      0.92);
  timescheme_wrapper.time_stepper.register_event(&inject_acoustic_event);

  specfem::event_marching::arbitrary_call_event inject_acoustic_velget_event(
      [&]() {
        // 1.5 dt since this is before corrector stage, when vel gets to current
        // step and accel gets zeroed out. We want the next step velocity
        wave_inject_acoustic.store_velocity(dt * 1.5);
        return 0;
      },
      1.5);
  timescheme_wrapper.time_stepper.register_event(&inject_acoustic_velget_event);
  specfem::event_marching::arbitrary_call_event inject_elastic_velget_event(
      [&]() {
        // 1.5 dt since this is before corrector stage, when vel gets to current
        // step and accel gets zeroed out. We want the next step velocity
        wave_inject_elastic.store_velocity(dt * 1.5);
        return 0;
      },
      3.5);
  timescheme_wrapper.time_stepper.register_event(&inject_elastic_velget_event);

  specfem::event_marching::arbitrary_call_event store_boundaryvals(
      [&]() {
        assembly->fields.forward.copy_to_host();
        dg_edge_storage.acoustic_acoustic_interface
            .compute_edge_intermediates<1, false>(*assembly);
        dg_edge_storage.acoustic_acoustic_interface
            .compute_edge_intermediates<2, false>(*assembly);
        dg_edge_storage.acoustic_elastic_interface
            .compute_edge_intermediates<2, false>(*assembly);
        assembly->fields.forward.copy_to_device();
        return 0;
      },
      0.9);
  timescheme_wrapper.time_stepper.register_event(&store_boundaryvals);

  specfem::event_marching::arbitrary_call_event mortar_flux_acoustic(
      [&]() {
        assembly->fields.forward.copy_to_host();
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
        assembly->fields.forward.copy_to_host();
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

            writer->write(*assembly);
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
  const auto database_filename = setup.get_databases();
  mpi->cout(setup.print_header(start_time));

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();

  const auto mesh_modifiers =
      setup.instantiate_mesh_modifiers<specfem::dimension::type::dim2>();
  auto mesh = specfem::IO::read_mesh(database_filename, mpi);
  mesh_modifiers->apply(mesh);
  if (KILL_NONNEUMANN_BDRYS) {

    specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2> bdry_abs(
        0);
    specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2>
        bdry_afs(0);
    specfem::mesh::forcing_boundary<specfem::dimension::type::dim2>
        bdry_forcing(0);
    mesh.boundaries = specfem::mesh::boundaries<specfem::dimension::type::dim2>(
        bdry_abs, bdry_afs, bdry_forcing);
    mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
        mesh.materials, mesh.boundaries);
  } else {
    if (LR_PERIODIC) {
      _util::kill_LR_BCs(mesh);
      mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
          mesh.materials, mesh.boundaries);
    }
    if (TOP_ABS || BOT_ABS) {
      _util::kill_TB_BCs(mesh, TOP_ABS, BOT_ABS);
      mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
          mesh.materials, mesh.boundaries);
    }
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] =
      specfem::IO::read_sources(setup.get_sources(), nsteps, setup.get_t0(),
                                setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::IO::read_receivers(stations_node, angle);

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
  const int nstep_between_samples = time_scheme->get_nstep_between_samples();
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
      nstep_between_samples, setup.get_simulation_type(),
      setup.instantiate_property_reader());
  time_scheme->link_assembly(assembly);

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------

  const auto wavefield_reader = setup.instantiate_wavefield_reader();
  if (wavefield_reader) {
    mpi->cout("Reading wavefield files:");
    mpi->cout("-------------------------------");

    wavefield_reader->read(assembly);
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
  bool continuity_state_requested = false;
  bool continuity_desired = false;
  for (int iarg = 0; iarg < argc; iarg++) {
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'N' && argv[iarg][2] == 'O' &&
        argv[iarg][3] == 'C') {
      continuity_state_requested = true;
      continuity_desired = false;
      continue;
    }
    if (argv[iarg][0] == '%' && argv[iarg][1] == 'C') {
      continuity_state_requested = true;
      continuity_desired = true;
      continue;
    }

    std::string arg(argv[iarg]);
    bool has_next = iarg + 1 < argc;
    if ((arg == "--file" || arg == "-f" || arg == "-p") && has_next) {
      param_fname = std::string(argv[iarg + 1]);
      iarg++;
      continue;
    }
    if ((arg == "--dump" || arg == "-d") && has_next) {
      DUMP_INTERVAL = atoi(argv[iarg + 1]);
      iarg++;
      continue;
    }

    if ((arg == "--dumpfolder" || arg == "-D") && has_next) {
      DUMP_OUTFOL = std::string(argv[iarg + 1]);
      iarg++;
      continue;
    }
    if (arg == "--lr_periodic") {
      LR_PERIODIC = true;
      continue;
    }
    if (arg == "--kill_boundaries") {
      KILL_NONNEUMANN_BDRYS = true;
      continue;
    }
    if (arg == "--absorb_top") {
      TOP_ABS = true;
      continue;
    }
    if (arg == "--absorb_bottom") {
      BOT_ABS = true;
      continue;
    }
  }
  if (continuity_state_requested) {
    FORCE_INTO_CONTINUOUS = continuity_desired;
  } else {
    FORCE_INTO_CONTINUOUS = false;
  }
#ifdef DEFAULT_SET_CONTINUOUS
  FORCE_INTO_CONTINUOUS = DEFAULT_SET_CONTINUOUS;
#endif

  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  if (USE_DOUBLEMESH) {
    mpi->cout("Using doublemesh");
  } else {
    mpi->cout("Using standard single-mesh");
  }
  if (FORCE_INTO_CONTINUOUS) {
    mpi->cout("Forcing into continuous domain");
  } else {
    mpi->cout("Using discontinuous domain");
  }
  if (LR_PERIODIC) {
    mpi->cout("Setting Left/Right periodic boundaries");
  }
  if (KILL_NONNEUMANN_BDRYS) {
    mpi->cout("Removing unnatural boundaries");
  }
  {
    execute(mpi);
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
