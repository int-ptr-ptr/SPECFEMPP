#pragma once
#include "specfem_setup.hpp"
#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
namespace specfem::test::tabulate_impl {

enum class EntryType {
  string,
  real,
  vector,
  integer,
  error,

  num_types
};

struct EntryFormatBase {
  virtual bool is_union() const { return false; }
  virtual std::vector<std::string> format(const std::any &data) const = 0;
  virtual std::unique_ptr<EntryFormatBase> copy() const = 0;
};

template <EntryType> struct TypedEntryFormatter;

template <>
struct TypedEntryFormatter<EntryType::string> : public EntryFormatBase {
  std::vector<std::string> format(const std::any &data) const {
    return { std::any_cast<std::string>(data) };
  }
  std::unique_ptr<EntryFormatBase> copy() const {
    return std::make_unique<TypedEntryFormatter>();
  }
};
template <>
struct TypedEntryFormatter<EntryType::real> : public EntryFormatBase {
  std::vector<std::string> format(const std::any &data) const {
    return { std::to_string(std::any_cast<type_real>(data)) };
  }
  std::unique_ptr<EntryFormatBase> copy() const {
    return std::make_unique<TypedEntryFormatter>();
  }
};
template <>
struct TypedEntryFormatter<EntryType::vector> : public EntryFormatBase {
  std::vector<std::string> format(const std::any &data) const {
    const auto &vec = std::any_cast<std::vector<type_real> >(data);
    std::vector<std::string> repr;
    for (const float &val : vec) {
      repr.push_back(std::to_string(val));
    }
    return repr;
  }
  std::unique_ptr<EntryFormatBase> copy() const {
    return std::make_unique<TypedEntryFormatter>();
  }
};
template <>
struct TypedEntryFormatter<EntryType::integer> : public EntryFormatBase {
  std::vector<std::string> format(const std::any &data) const {
    return { std::to_string(std::any_cast<int>(data)) };
  }
  std::unique_ptr<EntryFormatBase> copy() const {
    return std::make_unique<TypedEntryFormatter>();
  }
};
template <>
struct TypedEntryFormatter<EntryType::error> : public EntryFormatBase {
  std::vector<std::string> format(const std::any &data) const {
    return { std::to_string(
        std::get<0>(std::any_cast<std::tuple<type_real, type_real> >(data))) };
  }
  std::unique_ptr<EntryFormatBase> copy() const {
    return std::make_unique<TypedEntryFormatter>();
  }
};

struct EntryFormatUnion : public EntryFormatBase {
  bool is_union() const { return true; }
  std::vector<std::unique_ptr<const EntryFormatBase> > formats;

  template <typename... EntryFormats>
  EntryFormatUnion(const EntryFormats &...format_pack) {
    ([&]() -> void { append(format_pack); }(), ...);
  }

  void append(const EntryFormatBase &fmt) {
    if (fmt.is_union()) {
      for (const auto &fmt : static_cast<EntryFormatUnion>(fmt).formats) {
        formats.push_back(fmt->copy());
      }
    } else {
      formats.push_back(fmt.copy());
    }
  }

  std::vector<std::string> format(const std::any &data) const {
    throw std::runtime_error("CombinedFormat not yet implemented!");
  };
  std::unique_ptr<EntryFormatBase> copy() const {
    auto fmt = std::make_unique<EntryFormatUnion>();
    fmt->append(*this);
    return fmt;
  };
};

struct EntryFormat {
private:
  std::unique_ptr<const EntryFormatBase> impl;

public:
  EntryFormat(std::unique_ptr<EntryFormatBase> formatter)
      : impl(std::move(formatter)) {}

  EntryFormat() = default;
  std::vector<std::string> format(const std::any &data) const {
    if (impl == nullptr) {
      return { "" };
    } else {
      return impl->format(data);
    }
  }
  EntryFormat(const EntryFormat &other)
      : impl((other.impl == nullptr) ? nullptr : other.impl->copy()) {}

  // EntryFormat(EntryFormat &&other) : impl(std::move(other.impl)) {}

  void operator=(EntryFormat &&other) { impl = std::move(other.impl); }

  /**
   * @brief Combines two formatters, if they are compatible.
   *
   * @param other The format to combine with this one.
   * @return EntryFormat The combined format.
   */
  EntryFormat operator|(const EntryFormat &other) const {
    if (impl == nullptr) {
      return other;
    } else {
      if (other.impl == nullptr) {
        return *this;
      } else {

        return EntryFormat(
            std::make_unique<EntryFormatUnion>(*impl, *other.impl));
      }
    }
  }
};

template <std::underlying_type_t<EntryType>... types>
static EntryFormat formatter_from_type(
    const EntryType &type,
    std::integer_sequence<std::underlying_type_t<EntryType>, types...>) {
  std::unique_ptr<EntryFormatBase> data;

  (
      [&]() {
        constexpr EntryType type_to_check = static_cast<EntryType>(types);
        static_assert(
            std::is_base_of_v<EntryFormatBase,
                              TypedEntryFormatter<type_to_check> >,
            "TypedEntryFormatter<EntryType::...> not properly implemented for "
            "some "
            "EntryType.");
        if (type_to_check == type) {
          data = std::make_unique<TypedEntryFormatter<type_to_check> >();
        }
      }(),
      ...);

  return EntryFormat(std::move(data));
}

static EntryFormat formatter_from_type(const EntryType &type) {
  return formatter_from_type(
      type, std::make_integer_sequence<
                std::underlying_type_t<EntryType>,
                static_cast<std::underlying_type_t<EntryType> >(
                    EntryType::num_types)>{});
}

} // namespace specfem::test::tabulate_impl
