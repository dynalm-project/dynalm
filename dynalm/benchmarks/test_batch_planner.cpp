// BatchPlanner test (model-free): the wide-decode switch
// (KernelPlan::decode_wide_max_rows, DYNACORE_DECODE_WIDE_ROWS) applies only to
// decode-only steps of at most that many rows, and the kernels it then selects
// are the int8 / int16 decode kernels (checked through the matmul registry).
//
//   test_batch_planner

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/kernel/matmul_registry.h"
#include "execution/batch_planner.h"

using namespace dynalm;
using dynacore::KernelPlan;

namespace {

int g_checks = 0;
#define CHECK(cond)                                                                \
  do {                                                                             \
    ++g_checks;                                                                    \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

ModelConfig config() {
  ModelConfig c;
  c.num_layers = 2;
  c.num_heads = 8;
  c.num_kv_heads = 2;
  return c;
}

// `decode` one-token sequences plus one prompt chunk of `prefill` rows (0 = none).
ExecutionPlan plan(const BatchPlanner& p, int decode, int prefill) {
  static std::vector<TokenId> one(1, 1), chunk(64, 1);
  std::vector<SeqBatch> seqs;
  for (int i = 0; i < decode; ++i) {
    SeqBatch s;
    s.tokens = std::span<const TokenId>(one.data(), 1);
    s.start_pos = 100 + i;
    seqs.push_back(s);
  }
  if (prefill > 0) {
    SeqBatch s;
    s.tokens = std::span<const TokenId>(chunk.data(), static_cast<size_t>(prefill));
    seqs.push_back(s);
  }
  return p.plan(seqs);
}

// The kernel an FFN-up-sized Q4_K matmul of `rows` rows gets under `kp`
// (AVX2 tier with an interleaved copy).
dynacore::MatmulKernelId ffn_kernel(const KernelPlan& kp, int rows) {
  dynacore::MatmulCaps c;
  c.int8_accelerated = true;
  c.packed_q4 = true;
  c.quantize_act = c.quantize_act_sx = c.quantize_act16 = true;
  c.dot_q8_rows = c.dot_q8_sx_rows = true;
  c.dot_q16_x8_q4 = true;
  return dynacore::select_matmul({rows, 8960, 1536, dynacore::DType::kQ4_K, true}, kp, c).id;
}

}  // namespace

int main() {
  const dynacore::HardwareProfile hw = dynacore::HardwareProfile::detect(4, "avx2");
  KernelPlan base = KernelPlan::defaults();
  base.decode_wide_max_rows = 0;
  KernelPlan wide = base;
  wide.decode_wide_max_rows = 16;
  const BatchPlanner off(config(), hw, base), on(config(), hw, wide);
  using dynacore::MatmulKernelId;

  for (int rows : {1, 4, 8, 9, 12, 16, 17, 24}) {
    const ExecutionPlan a = plan(off, rows, 0), b = plan(on, rows, 0);
    CHECK(a.phase == StepPhase::kDecode && b.phase == StepPhase::kDecode && b.decode_rows == rows);
    // Off: the plan is the base plan. On: decode-only steps of <= 16 rows raise the decode kernels' row cap.
    CHECK(a.kernels.int8_superblock_max_rows == base.int8_superblock_max_rows);
    const bool widened = rows <= 16;
    CHECK(b.kernels.int8_superblock_max_rows == (widened ? 16 : base.int8_superblock_max_rows));
    // What runs: above 8 rows the default falls back to fp32; wide keeps the int8 interleaved kernel.
    const MatmulKernelId ka = ffn_kernel(a.kernels, rows), kb = ffn_kernel(b.kernels, rows);
    if (rows == 1) {
      CHECK(ka == MatmulKernelId::kInt8Per32 && kb == MatmulKernelId::kInt8Per32);
    } else if (rows <= 8) {
      CHECK(ka == MatmulKernelId::kInt8SubscaledX8 && kb == ka);  // wide changes nothing at <= 8 rows
    } else if (rows <= 16) {
      CHECK(ka == MatmulKernelId::kF32ExpandGemm && kb == MatmulKernelId::kInt8SubscaledX8);
    } else {
      CHECK(ka == MatmulKernelId::kF32ExpandGemm && kb == MatmulKernelId::kF32ExpandGemm);
    }
  }
  // Mixed and prefill-only steps are never widened (prefill rows keep their own kernels).
  for (int decode : {0, 4, 12}) {
    const ExecutionPlan m = plan(on, decode, 32);
    CHECK(m.phase == (decode == 0 ? StepPhase::kPrefill : StepPhase::kMixed));
    CHECK(m.kernels.int8_superblock_max_rows == base.int8_superblock_max_rows);
    CHECK(ffn_kernel(m.kernels, m.rows) != MatmulKernelId::kInt8SubscaledX8);
  }
  // A smaller limit widens only up to it.
  KernelPlan w12 = base;
  w12.decode_wide_max_rows = 12;
  const BatchPlanner on12(config(), hw, w12);
  CHECK(plan(on12, 12, 0).kernels.int8_superblock_max_rows == 12);
  CHECK(plan(on12, 13, 0).kernels.int8_superblock_max_rows == base.int8_superblock_max_rows);
  // The FFN-down plan (int8 off, int16 on) also widens: int16 on the interleaved copy.
  KernelPlan down = plan(on, 12, 0).kernels;
  down.int16_decode_max_rows = down.int8_decode_max_rows;
  down.int8_decode_max_rows = 0;
  CHECK(ffn_kernel(down, 12) == MatmulKernelId::kInt16X8);
  std::printf("all batch planner tests passed (%d checks)\n", g_checks);
  return 0;
}
