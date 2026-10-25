// qmm NEON tier with the int8 matrix instructions (FEAT_I8MM: SMMLA, USMMLA,
// USDOT; Apple M2+, Neoverse V1/N2+, Cortex-A510/A710+): int8 and int16
// activations, gemv and the matrix gemm. Compiled with i8mm flags
// (engine_set_isa NEONI8MM) and used only where cpu_info reports both
// FEAT_DotProd and FEAT_I8MM.

#include "dynacore/cpu/qmm.h"

#if ENGINE_HAS_NEON && (defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)

#define QMM_I8MM 1
#define QMM_NS neon_i8mm
#include "qmm_neon_impl.h"

namespace dynacore::qmm {

const Kernels* neon_i8mm_kernels() {
  static const Kernels k = {"neon_i8mm", /*gemm_matrix=*/true, /*int16=*/true, neon_i8mm::quantize_row,
                            neon_i8mm::gemv, neon_i8mm::gemm, neon_i8mm::gemv_packed, neon_i8mm::gemm_packed};
  return &k;
}

}  // namespace dynacore::qmm

#else

namespace dynacore::qmm {
const Kernels* neon_i8mm_kernels() { return nullptr; }
}  // namespace dynacore::qmm

#endif
