#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "dim3/test_fixture.hpp"
#include "gtest/gtest.h"

INSTANTIATE_TEST_SUITE_P(Assembly3DTests, Assembly3DTest,
                         ::testing::Values("EightNodeElastic"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
