file(GLOB POINT_SOURCE_FILES "./*.cpp")

add_library(
        periodic_tasks
        ${POINT_SOURCE_FILES}
)

target_link_libraries(
        periodic_tasks
        PUBLIC
        $<$<BOOL:${SPECFEM_ENABLE_VTK}>:${VTK_LIBRARIES}>
        PRIVATE
        reader
        writer
        io
        assembly
)

target_compile_definitions(
        periodic_tasks
        PUBLIC
        $<$<NOT:$<BOOL:${SPECFEM_ENABLE_VTK}>>:-DNO_VTK>
        $<$<BOOL:${__APPLE__}>:__APPLE__>
)
