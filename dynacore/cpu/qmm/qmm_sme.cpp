// qmm SME tier (qmm.h, "SME fp32 panel gemm"): fp32 outer products (FMOPA)
// into the four 32-bit ZA tiles, for prefill-sized batches. Compiled with SME
// flags (engine_set_isa SME) and used only where cpu_info reports FEAT_SME
// (qmm::sme_gemm). Only SME base instructions: no SVE outside streaming mode
// (Apple M4 has SME but no non-streaming SVE).

#include "dynacore/cpu/qmm.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_SME)

#include <arm_sme.h>

namespace dynacore::qmm {
namespace {

// The streaming vector holds 16 fp32 lanes (SVL 512 bits) on every SME CPU
// this tier is enabled for; sme_kernels() checks it at run time.
constexpr int kLanes = 16;
static_assert(kSmePanel == 2 * kLanes);

void pack_x(const float* x, int64_t x_stride, int64_t m, int64_t k, int64_t m0, float* panel) {
  const int64_t rows = m - m0 < kSmePanel ? m - m0 : kSmePanel;
  for (int64_t i = 0; i < rows; ++i) {
    const float* xr = x + (m0 + i) * x_stride;
    float* p = panel + i;
    for (int64_t kk = 0; kk < k; ++kk) p[kk * kSmePanel] = xr[kk];
  }
  for (int64_t i = rows; i < kSmePanel; ++i) {
    for (int64_t kk = 0; kk < k; ++kk) panel[kk * kSmePanel + i] = 0.0f;
  }
}

// za0 = x[0:16] (x) w[0:16], za1 = x[0:16] (x) w[16:32], za2 = x[16:32] (x) w[0:16], za3 = x[16:32] (x) w[16:32].
__arm_new("za") __arm_locally_streaming void block(const float* xp, const float* wp, int64_t k, float* y,
                                                    int64_t y_stride, int rows, int cols) {
  const svbool_t all = svptrue_b32();
  svzero_za();
  for (int64_t kk = 0; kk < k; ++kk, xp += kSmePanel, wp += kSmePanel) {
    const svfloat32_t x0 = svld1_f32(all, xp), x1 = svld1_f32(all, xp + kLanes);
    const svfloat32_t w0 = svld1_f32(all, wp), w1 = svld1_f32(all, wp + kLanes);
    svmopa_za32_f32_m(0, all, all, x0, w0);
    svmopa_za32_f32_m(1, all, all, x0, w1);
    svmopa_za32_f32_m(2, all, all, x1, w0);
    svmopa_za32_f32_m(3, all, all, x1, w1);
  }
  const svbool_t c0 = svwhilelt_b32_s32(0, cols), c1 = svwhilelt_b32_s32(kLanes, cols);
  const int top = rows < kLanes ? rows : kLanes;
  for (int r = 0; r < top; ++r) {
    svst1_hor_za32(0, static_cast<uint32_t>(r), c0, y + r * y_stride);
    svst1_hor_za32(1, static_cast<uint32_t>(r), c1, y + r * y_stride + kLanes);
  }
  for (int r = kLanes; r < rows; ++r) {
    svst1_hor_za32(2, static_cast<uint32_t>(r - kLanes), c0, y + r * y_stride);
    svst1_hor_za32(3, static_cast<uint32_t>(r - kLanes), c1, y + r * y_stride + kLanes);
  }
}

__arm_new("za") __arm_locally_streaming void block_acc(const float* xp, const float* wp, int64_t k, float* y,
                                                        int64_t y_stride, int rows, int cols) {
  const svbool_t all = svptrue_b32();
  const svbool_t c0 = svwhilelt_b32_s32(0, cols), c1 = svwhilelt_b32_s32(kLanes, cols);
  svzero_za();  // rows past `rows` stay zero (never stored)
  const int top = rows < kLanes ? rows : kLanes;
  for (int r = 0; r < top; ++r) {
    svld1_hor_za32(0, static_cast<uint32_t>(r), c0, y + r * y_stride);
    svld1_hor_za32(1, static_cast<uint32_t>(r), c1, y + r * y_stride + kLanes);
  }
  for (int r = kLanes; r < rows; ++r) {
    svld1_hor_za32(2, static_cast<uint32_t>(r - kLanes), c0, y + r * y_stride);
    svld1_hor_za32(3, static_cast<uint32_t>(r - kLanes), c1, y + r * y_stride + kLanes);
  }
  for (int64_t kk = 0; kk < k; ++kk, xp += kSmePanel, wp += kSmePanel) {
    const svfloat32_t x0 = svld1_f32(all, xp), x1 = svld1_f32(all, xp + kLanes);
    const svfloat32_t w0 = svld1_f32(all, wp), w1 = svld1_f32(all, wp + kLanes);
    svmopa_za32_f32_m(0, all, all, x0, w0);
    svmopa_za32_f32_m(1, all, all, x0, w1);
    svmopa_za32_f32_m(2, all, all, x1, w0);
    svmopa_za32_f32_m(3, all, all, x1, w1);
  }
  for (int r = 0; r < top; ++r) {
    svst1_hor_za32(0, static_cast<uint32_t>(r), c0, y + r * y_stride);
    svst1_hor_za32(1, static_cast<uint32_t>(r), c1, y + r * y_stride + kLanes);
  }
  for (int r = kLanes; r < rows; ++r) {
    svst1_hor_za32(2, static_cast<uint32_t>(r - kLanes), c0, y + r * y_stride);
    svst1_hor_za32(3, static_cast<uint32_t>(r - kLanes), c1, y + r * y_stride + kLanes);
  }
}

__arm_locally_streaming uint64_t streaming_words() { return svcntw(); }

}  // namespace

const SmeGemm* sme_kernels() {
  static const SmeGemm g = {"sme_f32", pack_x, block, block_acc};
  // The tiles above assume 16 fp32 lanes per streaming vector.
  static const bool ok = streaming_words() == kLanes;
  return ok ? &g : nullptr;
}

}  // namespace dynacore::qmm

#else

namespace dynacore::qmm {
const SmeGemm* sme_kernels() { return nullptr; }
}  // namespace dynacore::qmm

#endif
