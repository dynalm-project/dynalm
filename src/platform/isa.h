#pragma once

// CPU instruction-set tiers for kernel dispatch.
//
// A tier is usable when it is (a) compiled into this binary and (b) supported
// by the running CPU/OS. The kernel dispatcher picks the best usable tier once
// at startup; kernels never re-check features per call.

#include <string_view>
#include <vector>

#include "platform/cpu_info.h"

namespace engine {

enum class CpuIsa : int {
  kGeneric = 0,  // portable scalar C++ (auto-vectorized to SSE2 on x86-64)
  kAvx2,         // AVX2 + FMA + F16C
  kAvx512,       // AVX-512 F/BW/VL/DQ (+VNNI if present)
  kAmx,          // AVX-512 + AMX tiles
  kNeon,         // AArch64 NEON
};

std::string_view isa_name(CpuIsa isa);

// Whether the running CPU/OS can execute this tier.
bool isa_supported(CpuIsa isa, const CpuFeatures& f);

// Whether kernels for this tier were compiled into the binary.
bool isa_compiled(CpuIsa isa);

// Best tier that is both compiled and supported, ordered from most to least
// capable. Always returns at least kGeneric.
CpuIsa select_best_isa(const CpuFeatures& f);

// Parses "generic|avx2|avx512|amx|neon" (for --kernel overrides).
bool parse_isa(std::string_view s, CpuIsa& out);

std::vector<CpuIsa> usable_isas(const CpuFeatures& f);

}  // namespace engine
