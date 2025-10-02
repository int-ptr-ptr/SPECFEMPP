#include "enumerations/coupled_interface.hpp"

namespace specfem::interface {
std::string to_string(const interface_tag &interface_tag) {

  std::string interface_string;

  switch (interface_tag) {
  case specfem::interface::interface_tag::acoustic_elastic:
    interface_string = "acoustic_elastic";
    break;
  case specfem::interface::interface_tag::elastic_acoustic:
    interface_string = "elastic_acoustic";
    break;
  default:
    interface_string = "unknown";
    break;
  }

  return interface_string;
}
} // namespace specfem::interface
