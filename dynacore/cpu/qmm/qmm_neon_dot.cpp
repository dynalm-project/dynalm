// qmm NEON tier with the dot-product instructions (FEAT_DotProd): int8
// activations, gemv. Compiled with dot-product flags (engine_set_isa
// NEONDOT) and used only where cpu_info reports the feature.

#include "dynacore/cpu/qmm.h"

#if ENGINE_HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_DOTPROD)

#define QMM_I8MM 0
#define QMM_NS neon_dot
#include "qmm_neon_impl.h"

namespace dynacore::qmm {

const Kernels* neon_dot_kernels() {
  static const Kernels k = {"neon_dot", /*gemm_matrix=*/false, /*int16=*/false, neon_dot::quantize_row, neon_dot::gemv,
                            nullptr, neon_dot::gemv_packed, nullptr};
  return &k;
}

}  // namespace dynacore::qmm

#else

namespace dynacore::qmm {
const Kernels* neon_dot_kernels() { return nullptr; }
}  // namespace dynacore::qmm

#endif
