CMAKE_MINIMUM_REQUIRED(VERSION 3.17.5)


if (SPECFEMPP_CONFIGURE_TEST)
    configure_file(CMakeFiles/Snakefile.in ${CMAKE_CURRENT_SOURCE_DIR}/Snakefile)
    configure_file(CMakeFiles/Par_file.in ${CMAKE_CURRENT_SOURCE_DIR}/Par_file)
endif()
