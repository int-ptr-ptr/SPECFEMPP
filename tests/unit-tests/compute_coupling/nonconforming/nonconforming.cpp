#include "../../SPECFEM_Environment.hpp"
#include "algorithms/transfer.hpp"
#include "medium/compute_coupling.hpp"
#include "parallel_configuration/chunk_edge_config.hpp"
#include "specfem/chunk_edge.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>

// We need to simulate a chunk_edge iteration:
template <specfem::dimension::type DimensionTag> class ChunkEdgeIndexSimulator {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndexSimulator(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
};

/**
 * @brief Test fixture for 2D .
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct TransferFunctionTest2D : public ::testing::Test {
  using TransferFunctionInitializer = std::tuple_element_t<0, TestingTypes>;
  using FunctionInitializer = std::tuple_element_t<1, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        function(FunctionInitializer()) {}

  specfem::test_fixture::TransferFunction2D<TransferFunctionInitializer>
      transfer_function;                                               /**< Test
                                                  transfer
                                                  function
                                                */
  specfem::test_fixture::EdgeFunction2D<FunctionInitializer> function; /**<
                                                                           Test
                                                                           field
                                                                         */
};

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
