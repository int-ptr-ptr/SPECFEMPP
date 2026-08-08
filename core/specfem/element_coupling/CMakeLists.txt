file(GLOB_RECURSE ELEMENT_COUPLING_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(specfem_element_coupling ${ELEMENT_COUPLING_SOURCE_FILES})
add_library(specfem::element_coupling ALIAS specfem_element_coupling)

# TODO (Hanson: this is only here for the TMP_extra_kernel. Remove when no longer needed!)
target_link_libraries(specfem_element_coupling
    Kokkos::kokkos
    specfem::assembly
)
