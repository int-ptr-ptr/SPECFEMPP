file(GLOB_RECURSE SOURCE_FILES "./*.cpp" "**/*.cpp")

add_library(timescheme ${SOURCE_FILES})

target_link_libraries(
        timescheme
        Kokkos::kokkos
        assembly
)
