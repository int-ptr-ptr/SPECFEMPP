#pragma once

#include <Kokkos_Core.hpp>
#include <boost/preprocessor/stringize.hpp>

#define KOKKOS_ABORT_WITH_LOCATION(message)                                    \
  Kokkos::abort(__FILE__ ":" BOOST_PP_STRINGIZE(__LINE__) " - " message);
