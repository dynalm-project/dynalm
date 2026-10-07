#include "dynacore/hardware/isa.h"

#include <cstdlib>

#include "dynacore/base/platform.h"

namespace dynacore {

std::string_view isa_name(CpuIsa isa) {
  switch (isa) {
    case CpuIsa::kGeneric: return "generic";
    case CpuIsa::kAvx2: return "avx2";
    case CpuIsa::kAvx512: return "avx512";
    case CpuIsa::kAmx: return "amx";
    case CpuIsa::kNeon: return "neon";
  }
  return "?";
}

bool isa_supported(CpuIsa isa, const CpuFeatures& f) {
  switch (isa) {
    case CpuIsa::kGeneric: return true;
    case CpuIsa::kAvx2: return f.avx2 && f.fma && f.f16c;
    case CpuIsa::kAvx512: return f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq;
    case CpuIsa::kAmx:
      return isa_supported(CpuIsa::kAvx512, f) && f.amx_tile && f.amx_int8 && f.amx_bf16;
    case CpuIsa::kNeon: return f.neon;
  }
  return false;
}

bool isa_compiled(CpuIsa isa) {
  switch (isa) {
    case CpuIsa::kGeneric: return true;
#if ENGINE_ARCH_X86_64
    case CpuIsa::kAvx2: return ENGINE_HAS_AVX2;
    case CpuIsa::kAvx512: return ENGINE_HAS_AVX512;
    case CpuIsa::kAmx: return ENGINE_HAS_AMX;
    case CpuIsa::kNeon: return false;
#elif ENGINE_ARCH_ARM64
    case CpuIsa::kNeon: return ENGINE_HAS_NEON;
    default: return false;
#else
    default: return false;
#endif
  }
  return false;
}

std::vector<CpuIsa> usable_isas(const CpuFeatures& f) {
  std::vector<CpuIsa> out;
  for (CpuIsa isa : {CpuIsa::kAmx, CpuIsa::kAvx512, CpuIsa::kAvx2, CpuIsa::kNeon, CpuIsa::kGeneric}) {
    if (isa_compiled(isa) && isa_supported(isa, f)) out.push_back(isa);
  }
  return out;
}

CpuIsa select_best_isa(const CpuFeatures& f) { return usable_isas(f).front(); }

CpuIsa select_isa(const CpuFeatures& f, std::string* note) {
  const CpuIsa best = select_best_isa(f);
  const char* env = std::getenv("DYNACORE_ISA");
  if (env == nullptr || *env == '\0') {
    if (note) note->clear();
    return best;
  }
  CpuIsa want;
  if (!parse_isa(env, want)) {
    if (note) *note = "DYNACORE_ISA=" + std::string(env) + " is unknown; using " + std::string(isa_name(best));
    return best;
  }
  if (!isa_compiled(want) || !isa_supported(want, f)) {
    if (note) {
      *note = "DYNACORE_ISA=" + std::string(env) + " is not available on this CPU/build; using " +
              std::string(isa_name(best));
    }
    return best;
  }
  if (note) *note = "forced by DYNACORE_ISA";
  return want;
}

bool parse_isa(std::string_view s, CpuIsa& out) {
  for (CpuIsa isa : {CpuIsa::kGeneric, CpuIsa::kAvx2, CpuIsa::kAvx512, CpuIsa::kAmx, CpuIsa::kNeon}) {
    if (s == isa_name(isa)) {
      out = isa;
      return true;
    }
  }
  return false;
}

}  // namespace dynacore
