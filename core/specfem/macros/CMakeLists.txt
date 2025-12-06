
add_library(
        specfem_macros INTERFACE
)

target_link_libraries(
        specfem_macros
        INTERFACE
        ${BOOST_LIBS}
)

add_library(specfem::macros ALIAS specfem_macros)

target_compile_options(
        specfem_macros
        INTERFACE
        $<$<BOOL:${__GNUC__}>:-DSPECFEMPP_GNU_BUILD>
        $<$<BOOL:${__APPLE__}>:-DSPECFEMPP_APPLE_BUILD>
        $<$<BOOL:${__MSVC__}>:-DSPECFEMPP_MSVC_BUILD>
        $<$<BOOL:${__INTEL_LLVM__}>:-DSPECFEMPP_INTEL_LLVM_BUILD>
)
