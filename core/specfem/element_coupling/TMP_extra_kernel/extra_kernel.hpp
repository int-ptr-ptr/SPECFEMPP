#pragma once

#include "specfem/algorithms/gradient.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::element_coupling::TMP_extra_kernel {
// TEMPORARY: figure out a better way of finalizing this design
template <int NGLL, typename Tags, typename = void>
struct compute_coupling_extra_kernel {
  void
  execute(const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {};
};

// ========================================================
//       acoustic-elastic specialization
// ========================================================

template <int NGLL, typename Tags>
struct compute_coupling_extra_kernel<
    NGLL, Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
        (Tags::interface_tag ==
             specfem::element_coupling::interface_tag::acoustic_elastic ||
         Tags::interface_tag ==
             specfem::element_coupling::interface_tag::elastic_acoustic)>> {
  void
  execute(const specfem::assembly::assembly<Tags::dimension_tag> &assembly);
};

template <int NGLL, typename Tags>
void execute(const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {
  compute_coupling_extra_kernel<NGLL, Tags>().execute(assembly);
}
} // namespace specfem::element_coupling::TMP_extra_kernel
