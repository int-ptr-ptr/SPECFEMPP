file(GLOB_RECURSE COMPUTE_SOURCE_FILES "./*.cpp" "./**/*.cpp")

add_library(assembly ${COMPUTE_SOURCE_FILES})
add_library(specfem::assembly ALIAS assembly)

target_link_libraries(
        assembly
        algorithms
        enumerations
        quadrature
        mesh
        source_class
        jacobian
        point
        edge
        specfem::receivers
        shape_functions
        Kokkos::kokkos
        enumerations
        ${BOOST_LIBS}
)

set_target_properties(assembly PROPERTIES
    # UNITY_BUILD $<NOT:$<BOOL:${__INTEL_LLVM__}>>
    UNITY_BUILD_BATCH_SIZE $<IF:$<BOOL:${__INTEL_LLVM__}>,2,8>
)


if (NOT SPECFEM_ENABLE_VTK)
        target_compile_definitions(
                assembly
                PUBLIC -DNO_VTK
        )
else ()
        target_link_libraries(
                assembly
                ${VTK_LIBRARIES}
                )

        # Only define the __APPLE__ compile definition if it is defined
        if (__APPLE__)
        target_compile_definitions(assembly PRIVATE __APPLE__)
        endif(__APPLE__)
endif()

target_include_directories(
        assembly
        INTERFACE
        ${HDF5_INCLUDE_DIRS}
)
