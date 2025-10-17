#include <type_traits>

namespace specfem::assembly::coupled_interfaces_impl {
template <typename T, typename = void>
struct stores_transfer_function_self : std::false_type {};

template <typename T>
struct stores_transfer_function_self<T, decltype(
                                            (void)(T().transfer_function_self))>
    : std::true_type {};

template <typename T, typename = void>
struct stores_transfer_function_coupled : std::false_type {};

template <typename T>
struct stores_transfer_function_coupled<
    T, decltype((void)(T().transfer_function_coupled))> : std::true_type {};

template <typename T, typename = void>
struct stores_intersection_factor : std::false_type {};

template <typename T>
struct stores_intersection_factor<T, decltype((void)(T().intersection_factor))>
    : std::true_type {};

template <typename T, typename = void>
struct stores_intersection_normal : std::false_type {};

template <typename T>
struct stores_intersection_normal<T, decltype((void)(T().intersection_normal))>
    : std::true_type {};
} // namespace specfem::assembly::coupled_interfaces_impl
