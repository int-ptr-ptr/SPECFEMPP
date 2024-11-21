#ifndef _NAMESPACE_NAMESPACE_HPP
#define _NAMESPACE_NAMESPACE_HPP

#include "delphi.hpp"
#include <string>
#include <unordered_map>

namespace specfem {
namespace delphi {
namespace namespace {

class variable {
public:
private:
  std::string name;
};

class environment {
public:
private:
  // https://en.cppreference.com/w/cpp/container/unordered_map
  std::unordered_map<std::string, variable> variables;
};

} // namespace namespace
} // namespace delphi
} // namespace specfem
#endif
