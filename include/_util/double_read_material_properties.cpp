#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/read_material_properties.hpp"
// #include "mesh/materials/materials.hpp"
#include "mesh/materials/materials.tpp"
#include "specfem_mpi/interface.hpp"
#include "utilities/interface.hpp"
#include <memory>
#include <vector>

namespace {
constexpr auto elastic = specfem::element::medium_tag::elastic;
constexpr auto isotropic = specfem::element::property_tag::isotropic;
constexpr auto acoustic = specfem::element::medium_tag::acoustic;

struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

std::vector<specfem::mesh::materials::material_specification> read_materials(
    std::ifstream &stream1, std::ifstream &stream2, const int numat,
    specfem::mesh::materials::material<elastic, isotropic> &elastic_isotropic,
    specfem::mesh::materials::material<acoustic, isotropic> &acoustic_isotropic,
    const int numat1, const specfem::MPI::MPI *mpi) {

  input_holder read_values;

  std::ostringstream message;

  std::vector<specfem::mesh::materials::material_specification> index_mapping(
      numat);

  message << "Material systems:\n"
          << "------------------------------";

  mpi->cout(message.str());

  if (mpi->get_rank() == 0)
    std::cout << "Number of material systems = " << numat << "\n\n";

  std::vector<specfem::material::material<elastic, isotropic> >
      l_elastic_isotropic;

  l_elastic_isotropic.reserve(numat);

  int index_elastic_isotropic = 0;

  std::vector<specfem::material::material<acoustic, isotropic> >
      l_acoustic_isotropic;

  l_acoustic_isotropic.reserve(numat);

  int index_acoustic_isotropic = 0;

  std::ifstream *stream = &stream1;
  int ifile = -1;
  int numat_file = numat1;
  for (int i = 0; i < numat; i++) {
    ifile++;
    if (i == numat1) {
      stream = &stream2;
      ifile = 0;
      numat_file = numat - numat1;
    }

    specfem::IO::fortran_read_line(
        *stream, &read_values.n, &read_values.indic, &read_values.val0,
        &read_values.val1, &read_values.val2, &read_values.val3,
        &read_values.val4, &read_values.val5, &read_values.val6,
        &read_values.val7, &read_values.val8, &read_values.val9,
        &read_values.val10, &read_values.val11, &read_values.val12);

    if (read_values.n < 1 || read_values.n > numat_file) {
      throw std::runtime_error(
          "Wrong material set number. Check database file.");
    }

    assert(read_values.n == ifile + 1);

    if (read_values.indic == 1) {
      // Acoustic Material
      if (read_values.val2 == 0) {
        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;

        specfem::material::material<acoustic, isotropic> acoustic_holder(
            density, cp, Qkappa, Qmu, compaction_grad);

        acoustic_holder.print();

        l_acoustic_isotropic.push_back(acoustic_holder);

        index_mapping[i] = specfem::mesh::materials::material_specification(
            specfem::element::medium_tag::acoustic,
            specfem::element::property_tag::isotropic,
            index_acoustic_isotropic);

        index_acoustic_isotropic++;

      } else {

        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real cs = read_values.val2;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;

        specfem::material::material<elastic, isotropic> elastic_holder(
            density, cs, cp, Qkappa, Qmu, compaction_grad);

        elastic_holder.print();

        l_elastic_isotropic.push_back(elastic_holder);

        index_mapping[i] = specfem::mesh::materials::material_specification(
            specfem::element::medium_tag::elastic,
            specfem::element::property_tag::isotropic, index_elastic_isotropic);

        index_elastic_isotropic++;
      }
    } else {
      throw std::runtime_error("Material type not supported");
    }
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() == numat);

  elastic_isotropic = specfem::mesh::materials::material<elastic, isotropic>(
      l_elastic_isotropic.size(), l_elastic_isotropic);

  acoustic_isotropic = specfem::mesh::materials::material<acoustic, isotropic>(
      l_acoustic_isotropic.size(), l_acoustic_isotropic);

  return index_mapping;
}

void read_material_indices(
    std::ifstream &stream1, std::ifstream &stream2, const int nspec,
    const int numat,
    const std::vector<specfem::mesh::materials::material_specification>
        &index_mapping,
    const specfem::kokkos::HostView1d<
        specfem::mesh::materials::material_specification>
        material_index_mapping,
    const specfem::kokkos::HostView2d<int> knods, const int numat1,
    const int nspec1, const int npgeo1, const specfem::MPI::MPI *mpi) {

  const int ngnod = knods.extent(0);

  int n, kmato_read, pml_read;

  std::vector<int> knods_read(ngnod, -1);

  bool is_stream2 = false;
  for (int ispec = 0; ispec < nspec; ispec++) {
    if (ispec == nspec1) {
      is_stream2 = true;
    }
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    specfem::IO::fortran_read_line(is_stream2 ? stream2 : stream1, &n,
                                   &kmato_read, &knods_read, &pml_read);

    if (n < 1 || n > (is_stream2 ? (nspec - nspec1) : nspec1)) {
      throw std::runtime_error("Error reading material indices");
    }

    if (kmato_read < 1 ||
        kmato_read > (is_stream2 ? (numat - numat1) : numat1)) {
      throw std::runtime_error("Error reading material indices");
    }

    int global_index = (n - 1) + (is_stream2 ? nspec1 : 0);
    for (int i = 0; i < ngnod; i++) {
      if (knods_read[i] == 0)
        throw std::runtime_error("Error reading knods (node_id) values");

      knods(i, global_index) = knods_read[i] - 1 + (is_stream2 ? npgeo1 : 0);
    }

    material_index_mapping(global_index) =
        index_mapping[kmato_read - 1 + (is_stream2 ? numat1 : 0)];
  }

  return;
}

} // namespace

namespace _util {

specfem::mesh::materials
read_material_properties(std::ifstream &stream1, std::ifstream &stream2,
                         const int numat, const int nspec,
                         const specfem::kokkos::HostView2d<int> knods,
                         const int numat1, const int nspec1, const int npgeo1,
                         const specfem::MPI::MPI *mpi) {

  // Create materials instances
  specfem::mesh::materials materials(nspec, numat);

  // Read material properties
  auto index_mapping =
      ::read_materials(stream1, stream2, numat, materials.elastic_isotropic,
                       materials.acoustic_isotropic, numat1, mpi);

  // Read material indices
  ::read_material_indices(stream1, stream2, nspec, numat, index_mapping,
                          materials.material_index_mapping, knods, numat1,
                          nspec1, npgeo1, mpi);

  return materials;
}
} // namespace _util
