#include "fixture.hpp"
#include "io/interface.hpp"

#include <map>

specfem::mesh::mesh<specfem::dimension::type::dim2> convert_to_conforming(
    const specfem::mesh::mesh<specfem::dimension::type::dim2> &ncmesh) {
  specfem::mesh::mesh<specfem::dimension::type::dim2> mesh = ncmesh;

  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::dimension::type::dim2>::EdgeProperties;

  // remember -- copy constructor is shallow.
  mesh.adjacency_graph =
      specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>(
          mesh.nspec);

  const auto &ncgraph = ncmesh.adjacency_graph.graph();
  auto &graph = mesh.adjacency_graph.graph();

  std::unordered_map<int, std::tuple<int, specfem::mesh_entity::type, int,
                                     specfem::mesh_entity::type> >
      nc_to_c_interface;
  int num_nc_edges = 0;

  for (const auto &edge : boost::make_iterator_range(boost::edges(ncgraph))) {
    auto edge_conf = ncgraph[edge];

    const auto src = boost::source(edge, ncgraph);
    const auto tgt = boost::target(edge, ncgraph);

    // convert nonconforming connections to conforming
    if (edge_conf.connection == specfem::connections::type::nonconforming) {
      // only add edges, not corners

      const auto [inverse, found] = boost::edge(tgt, src, ncgraph);
      if (!found) {
        throw std::runtime_error(
            "kernel_compare test: Non-symmetric adjacency graph "
            "detected in `compute_intersection`.");
      }
      if ((!specfem::mesh_entity::contains(specfem::mesh_entity::edges,
                                           edge_conf.orientation)) ||
          (!specfem::mesh_entity::contains(specfem::mesh_entity::edges,
                                           ncgraph[inverse].orientation))) {
        continue;
      }

      // should this be weakly conforming instead?
      edge_conf.connection = specfem::connections::type::strongly_conforming;
      num_nc_edges++;
      if (auto search = nc_to_c_interface.find(tgt);
          search == nc_to_c_interface.end() ||
          std::get<2>(nc_to_c_interface[tgt]) != src) {
        // inverse not already inside. add ourselves
        nc_to_c_interface.insert(
            std::make_pair(src, std::make_tuple(src, edge_conf.orientation, tgt,
                                                ncgraph[inverse].orientation)));
      }
    }

    boost::add_edge(src, tgt, edge_conf, graph);
  }
  mesh.adjacency_graph.assert_symmetry();

  // fill mesh::coupled_interfaces, since assembly needs to get conforming from
  // there:
  if (nc_to_c_interface.size() != num_nc_edges / 2) {
    throw std::runtime_error(
        "Constructing conforming mesh from nonconforming mesh: the number of "
        "nonconforming edges is not twice the number of acoustic-elastic "
        "interfaces found!");
  }

  mesh.coupled_interfaces.elastic_acoustic =
      decltype(mesh.coupled_interfaces.elastic_acoustic)(num_nc_edges / 2);

  auto it = nc_to_c_interface.begin();
  for (int i = 0; i < num_nc_edges / 2; i++) {
    const auto &[src, src_edge, tgt, tgt_edge] = it->second;
    if (mesh.tags.tags_container(src).medium_tag ==
        specfem::element::medium_tag::elastic_psv) {
      mesh.coupled_interfaces.elastic_acoustic.medium1_index_mapping(i) = src;
      mesh.coupled_interfaces.elastic_acoustic.medium2_index_mapping(i) = tgt;
      mesh.coupled_interfaces.elastic_acoustic.medium1_edge_type(i) = src_edge;
      mesh.coupled_interfaces.elastic_acoustic.medium2_edge_type(i) = tgt_edge;
    } else {
      mesh.coupled_interfaces.elastic_acoustic.medium2_index_mapping(i) = src;
      mesh.coupled_interfaces.elastic_acoustic.medium1_index_mapping(i) = tgt;
      mesh.coupled_interfaces.elastic_acoustic.medium2_edge_type(i) = src_edge;
      mesh.coupled_interfaces.elastic_acoustic.medium1_edge_type(i) = tgt_edge;
    }
    it++;
  }

  return mesh;
}

specfem::testing::kernel_compare::Test::Test(const YAML::Node &Node,
                                             specfem::MPI::MPI *mpi)
    : name(Node["name"].as<std::string>()),
      description(Node["description"].as<std::string>()),
      nonconforming_database_file(Node["mesh"].as<std::string>()),
      nonconforming_mesh(specfem::io::read_2d_mesh(
          this->nonconforming_database_file, specfem::enums::elastic_wave::psv,
          specfem::enums::electromagnetic_wave::te, mpi)),
      conforming_mesh(convert_to_conforming(this->nonconforming_mesh)),
      quadrature(specfem::quadrature::gll::gll()), sources(), receivers(),
      nonconforming_assembly(
          specfem::assembly::assembly<specfem::dimension::type::dim2>(
              this->nonconforming_mesh, this->quadrature, this->sources,
              this->receivers, {}, 0.0, 0.1, 1, 1, 1,
              specfem::simulation::type::forward, false, nullptr)),
      conforming_assembly(
          specfem::assembly::assembly<specfem::dimension::type::dim2>(
              this->conforming_mesh, this->quadrature, this->sources,
              this->receivers, {}, 0.0, 0.1, 1, 1, 1,
              specfem::simulation::type::forward, false, nullptr)) {}

specfem::testing::kernel_compare::mesh_list::mesh_list() {

  std::string config_filename = "nonconforming/kernel_compare/test_config.yaml";
  const YAML::Node yaml_root = YAML::LoadFile(config_filename);
  YAML::Node all_tests = yaml_root["Tests"];
  assert(all_tests.IsSequence());
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  for (const auto &test_node : all_tests) {
    this->push_back(specfem::testing::kernel_compare::Test(test_node, mpi));
  }
}
