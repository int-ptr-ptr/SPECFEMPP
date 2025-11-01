#pragma once

#include <type_traits>

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#if __MSVC__
// mscv (microsoft)
// this is untested.

// https://learn.microsoft.com/en-us/cpp/preprocessor/warning?view=msvc-170#push-and-pop
#define _SUPPRESS_WARNING_START(GCC_CLANG_CODE, WINDOWS_CODE)                  \
  {                                                                            \
    _Pragma("warning( push )");                                                \
    _Pragma(STRINGIFY("warning( disable : " #WINDOWS_CODE " )"));              \
  }

#define _SUPPRESS_WARNING_END()                                                \
  {                                                                            \
    _Pragma("warning( pop )");                                                 \
  }
#else
// gcc
// https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html

// clang should also be reading these (first paragraph of
// https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas
//)
#define _SUPPRESS_WARNING_START(GCC_CLANG_CODE, WINDOWS_CODE)                  \
  {                                                                            \
    _Pragma("GCC diagnostic push");                                            \
    _Pragma(STRINGIFY("GCC diagnostic ignored " #GCC_CLANG_CODE));             \
  }
#define SUPPRESS_WARNING_END()                                                 \
  {                                                                            \
    _Pragma("GCC diagnostic pop");                                             \
  }

#endif

#define _SUPPRESS_WARNING(GCC_CLANG_CODE, WINDOWS_CODE, CODEBLOCK)             \
  {                                                                            \
    _SUPPRESS_WARNING_START(GCC_CLANG_CODE, WINDOWS_CODE)                      \
    CODEBLOCK;                                                                 \
    SUPPRESS_WARNING_END()                                                     \
  }

#define SUPPRESS_WARNING_START_TEMPORARY_REF()                                 \
  { _SUPPRESS_WARNING_START("-Wreturn-local-addr", C4172) }

#define SUPPRESS_WARNING_TEMPORARY_REF(CODEBLOCK)                              \
  {                                                                            \
    SUPPRESS_WARNING_START_TEMPORARY_REF();                                    \
    CODEBLOCK;                                                                 \
    SUPPRESS_WARNING_END()                                                     \
  }

namespace specfem::__future__ {

template <typename ConstructedType>
inline ConstructedType SUPPRESS_WARNING_TEMPORARY_REF_BLANK_CONSTRUCTION() {
#if __MSVC__
#pragma warning(push)
#pragma warning(disable : C4172)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#endif
  return {};
#if __MSVC__
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
}

} // namespace specfem::__future__
