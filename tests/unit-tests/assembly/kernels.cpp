#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "policies/chunk.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"
#include <gtest/gtest.h>

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd = false>
std::string get_error_message(
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  PropertyTag, false> &point_kernel,
    const type_real value);

template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trho = " << point_kernel.rho << "\n";
  message << "\t\tmu = " << point_kernel.mu << "\n";
  message << "\t\tkappa = " << point_kernel.kappa << "\n";
  message << "\t\trhop = " << point_kernel.rhop << "\n";
  message << "\t\talpha = " << point_kernel.alpha << "\n";
  message << "\t\tbeta = " << point_kernel.beta << "\n";

  return message.str();
}

template <>
std::string get_error_message(
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, false> &point_kernel,
    const type_real value) {
  std::ostringstream message;

  message << "\n\t Expected: " << value;
  message << "\n\t Got: \n";
  message << "\t\trho = " << point_kernel.rho << "\n";
  message << "\t\tkappa = " << point_kernel.kappa << "\n";
  message << "\t\trhop = " << point_kernel.rhop << "\n";
  message << "\t\talpha = " << point_kernel.alpha << "\n";

  return message.str();
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::kernels<specfem::dimension::type::dim2, MediumTag, PropertyTag,
                        false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels);

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
specfem::point::kernels<specfem::dimension::type::dim2, MediumTag, PropertyTag,
                        false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<specfem::dimension::type::dim2, MediumTag,
                                  PropertyTag, true> &point_kernel);

template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto elastic_isotropic = kernels.elastic_isotropic;

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel;

  point_kernel.rho = elastic_isotropic.h_rho(ispec_l, iz, ix);
  point_kernel.mu = elastic_isotropic.h_mu(ispec_l, iz, ix);
  point_kernel.kappa = elastic_isotropic.h_kappa(ispec_l, iz, ix);
  point_kernel.rhop = elastic_isotropic.h_rhop(ispec_l, iz, ix);
  point_kernel.alpha = elastic_isotropic.h_alpha(ispec_l, iz, ix);
  point_kernel.beta = elastic_isotropic.h_beta(ispec_l, iz, ix);

  return point_kernel;
}

template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rho = point_kernel.rho[lane];
  point_kernel_l.mu = point_kernel.mu[lane];
  point_kernel_l.kappa = point_kernel.kappa[lane];
  point_kernel_l.rhop = point_kernel.rhop[lane];
  point_kernel_l.alpha = point_kernel.alpha[lane];
  point_kernel_l.beta = point_kernel.beta[lane];

  return point_kernel_l;
}

template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(const int ispec, const int iz, const int ix,
                 const specfem::compute::kernels &kernels) {

  const auto acoustic_isotropic = kernels.acoustic_isotropic;

  const int ispec_l = kernels.h_property_index_mapping(ispec);

  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel;

  point_kernel.rho = acoustic_isotropic.h_rho(ispec_l, iz, ix);
  point_kernel.kappa = acoustic_isotropic.h_kappa(ispec_l, iz, ix);
  point_kernel.alpha = acoustic_isotropic.h_alpha(ispec_l, iz, ix);
  point_kernel.rhop = acoustic_isotropic.h_rho_prime(ispec_l, iz, ix);

  return point_kernel;
}

template <>
specfem::point::kernels<specfem::dimension::type::dim2,
                        specfem::element::medium_tag::acoustic,
                        specfem::element::property_tag::isotropic, false>
get_point_kernel(
    const int lane,
    const specfem::point::kernels<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, true> &point_kernel) {
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>
      point_kernel_l;

  point_kernel_l.rho = point_kernel.rho[lane];
  point_kernel_l.kappa = point_kernel.kappa[lane];
  point_kernel_l.alpha = point_kernel.alpha[lane];
  point_kernel_l.rhop = point_kernel.rhop[lane];

  return point_kernel_l;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd,
          typename IndexViewType, typename ValueViewType>
void check_to_value(const specfem::compute::kernels kernels,
                    const IndexViewType &ispecs,
                    const ValueViewType &values_to_store) {
  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  std::vector<int> elements;

  const auto element_types = kernels.h_element_types;
  const auto element_properties = kernels.h_element_property;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types(ispec) == MediumTag) &&
        (element_properties(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  for (int i = 0; i < ispecs.extent(0); ++i) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        const int ielement = ispecs(i);
        const int n_simd_elements = (simd_size + ielement > elements.size())
                                        ? elements.size() - ielement
                                        : simd_size;
        for (int j = 0; j < n_simd_elements; j++) {
          const auto point_kernel = get_point_kernel<MediumTag, PropertyTag>(
              ielement + j, iz, ix, kernels);
          const type_real value = values_to_store(i);
          if (point_kernel != value) {
            std::ostringstream message;

            message << "\n \t Error at ispec = " << ielement + j
                    << ", iz = " << iz << ", ix = " << ix;
            message << get_error_message(point_kernel, value);

            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool Store, bool Add,
          bool using_simd, typename IndexViewType, typename ValueViewType>
void execute_store_or_add(specfem::compute::kernels &kernels,
                          const int element_size, const IndexViewType &ispecs,
                          const ValueViewType &values_to_store) {

  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;

  const int N = ispecs.extent(0);

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  Kokkos::parallel_for(
      "check_store_on_device",
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3> >(
          { 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = ispecs(i);
        constexpr int simd_size = PointType::simd::size();
        auto &kernels_l = kernels;
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;

        const auto index =
            get_index<using_simd>(ielement, n_simd_elements, iz, ix);
        const type_real value = values_to_store(i);
        PointType point(value);
        if constexpr (Store) {
          specfem::compute::store_on_device(index, point, kernels_l);
        } else if constexpr (Add) {
          specfem::compute::add_on_device(index, point, kernels_l);
        }
      });

  Kokkos::fence();
  kernels.copy_to_host();
  return;
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_store_and_add(specfem::compute::kernels &kernels) {

  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;
  std::vector<int> elements;

  const auto element_types = kernels.h_element_types;
  const auto element_properties = kernels.h_element_property;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types(ispec) == MediumTag) &&
        (element_properties(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.size() < N) {
    return;
  }

  Kokkos::View<int[N], Kokkos::DefaultExecutionSpace> ispecs("ispecs");
  Kokkos::View<type_real[N], Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store");
  auto ispecs_h = Kokkos::create_mirror_view(ispecs);
  auto values_to_store_h = Kokkos::create_mirror_view(values_to_store);

  const int element_size = elements.size();
  const int step = element_size / N;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements[i * step];
    values_to_store_h(i) = 10.5 + i;
  }

  ispecs_h(N - 1) = elements[element_size - 5]; // check when simd is not full

  Kokkos::deep_copy(ispecs, ispecs_h);
  Kokkos::deep_copy(values_to_store, values_to_store_h);

  execute_store_or_add<MediumTag, PropertyTag, true, false, using_simd>(
      kernels, element_size, ispecs, values_to_store);

  check_to_value<MediumTag, PropertyTag, using_simd>(kernels, ispecs_h,
                                                     values_to_store_h);

  execute_store_or_add<MediumTag, PropertyTag, false, true, using_simd>(
      kernels, element_size, ispecs, values_to_store);

  for (int i = 0; i < N; i++) {
    values_to_store_h(i) *= 2;
  }

  check_to_value<MediumTag, PropertyTag, using_simd>(kernels, ispecs_h,
                                                     values_to_store_h);
}

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool using_simd>
void check_load_on_device(specfem::compute::kernels &kernels) {
  const int nspec = kernels.nspec;
  const int ngllx = kernels.ngllx;
  const int ngllz = kernels.ngllz;
  std::vector<int> elements;

  const auto element_types = kernels.h_element_types;
  const auto element_properties = kernels.h_element_property;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if ((element_types(ispec) == MediumTag) &&
        (element_properties(ispec) == PropertyTag)) {
      elements.push_back(ispec);
    }
  }

  // Evaluate at N evenly spaced points
  constexpr int N = 20;

  if (elements.size() < N) {
    return;
  }

  using PointType = specfem::point::kernels<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag, using_simd>;

  Kokkos::View<int[N], Kokkos::DefaultExecutionSpace> ispecs("ispecs");
  Kokkos::View<type_real[N], Kokkos::DefaultExecutionSpace> values_to_store(
      "values_to_store");
  auto ispecs_h = Kokkos::create_mirror_view(ispecs);
  auto values_to_store_h = Kokkos::create_mirror_view(values_to_store);

  const int element_size = elements.size();
  const int step = element_size / N;

  for (int i = 0; i < N; i++) {
    ispecs_h(i) = elements[i * step];
    values_to_store_h(i) = 2 * (10.5 + i);
  }

  ispecs_h(N - 1) = elements[element_size - 5]; // check when simd is not full

  Kokkos::deep_copy(ispecs, ispecs_h);

  Kokkos::View<PointType **[N], Kokkos::DefaultExecutionSpace> point_kernels(
      "point_kernels", ngllz, ngllx);
  auto h_point_kernels = Kokkos::create_mirror_view(point_kernels);

  Kokkos::parallel_for(
      "check_load_on_device",
      Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 }, { N, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int &i, const int &iz, const int &ix) {
        const int ielement = ispecs(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;

        const auto index =
            get_index<using_simd>(ielement, n_simd_elements, iz, ix);
        PointType point;
        specfem::compute::load_on_device(index, kernels, point);
        point_kernels(iz, ix, i) = point;
      });

  Kokkos::fence();
  Kokkos::deep_copy(h_point_kernels, point_kernels);

  for (int i = 0; i < N; i++) {
    for (int iz = 0; iz < ngllz; iz++) {
      for (int ix = 0; ix < ngllx; ix++) {
        using simd = specfem::datatype::simd<type_real, using_simd>;
        const auto &point_kernel = h_point_kernels(iz, ix, i);
        const int ielement = ispecs_h(i);
        constexpr int simd_size = PointType::simd::size();
        const int n_simd_elements = (simd_size + ielement > element_size)
                                        ? element_size - ielement
                                        : simd_size;
        const type_real value_l = values_to_store_h(i);
        if constexpr (using_simd) {
          for (int lane = 0; lane < n_simd_elements; lane++) {
            const auto point_kernel_l = get_point_kernel(lane, point_kernel);
            if (point_kernel_l != value_l) {
              std::ostringstream message;

              message << "\n \t Error in function load_on_device";

              message << "\n \t Error at ispec = " << ielement << ", iz = " << 0
                      << ", ix = " << 0;
              message << get_error_message(point_kernel_l, value_l);

              throw std::runtime_error(message.str());
            }
          }
        } else if constexpr (!using_simd) {
          if (point_kernel != value_l) {
            std::ostringstream message;
            message << "\n \t Error in function load_on_device";

            message << "\n \t Error at ispec = " << ielement << ", iz = " << 0
                    << ", ix = " << 0;
            message << get_error_message(point_kernel, value_l);

            throw std::runtime_error(message.str());
          }
        }
      }
    }
  }

  return;
}

void test_kernels(specfem::compute::assembly &assembly) {

  auto &kernels = assembly.kernels;

  check_store_and_add<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, false>(
      kernels);

  check_load_on_device<specfem::element::medium_tag::elastic,
                       specfem::element::property_tag::isotropic, false>(
      kernels);

  check_store_and_add<specfem::element::medium_tag::elastic,
                      specfem::element::property_tag::isotropic, true>(kernels);

  check_load_on_device<specfem::element::medium_tag::elastic,
                       specfem::element::property_tag::isotropic, true>(
      kernels);

  check_store_and_add<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, false>(
      kernels);

  check_load_on_device<specfem::element::medium_tag::acoustic,
                       specfem::element::property_tag::isotropic, false>(
      kernels);

  check_store_and_add<specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, true>(kernels);

  check_load_on_device<specfem::element::medium_tag::acoustic,
                       specfem::element::property_tag::isotropic, true>(
      kernels);
}

TEST_F(ASSEMBLY, kernels_device_functions) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto assembly = std::get<1>(parameters);

    try {
      test_kernels(assembly);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}