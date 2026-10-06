#pragma once

#include <cstddef>

// Compiler / platform portability macros. Keep this file tiny.

#if defined(_MSC_VER) && !defined(__clang__)
#define ENGINE_FORCE_INLINE __forceinline
#define ENGINE_NOINLINE __declspec(noinline)
#define ENGINE_RESTRICT __restrict
#else
#define ENGINE_FORCE_INLINE inline __attribute__((always_inline))
#define ENGINE_NOINLINE __attribute__((noinline))
#define ENGINE_RESTRICT __restrict__
#endif

#if defined(_WIN32)
#define ENGINE_OS_WINDOWS 1
#elif defined(__linux__)
#define ENGINE_OS_LINUX 1
#elif defined(__APPLE__)
#define ENGINE_OS_MACOS 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define ENGINE_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ENGINE_ARCH_ARM64 1
#endif

namespace dynacore {

// Destructive-interference size used to pad hot shared counters. Hard-coded
// rather than std::hardware_destructive_interference_size, which is not
// ABI-stable across compilers and warns under GCC.
inline constexpr std::size_t kCacheLineSize = 64;

}  // namespace dynacore
