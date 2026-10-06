#include "backends/kernel_plan.h"

#include <algorithm>
#include <cstdlib>

namespace engine {

std::string_view attention_strategy_name(AttentionStrategy s) {
  switch (s) {
    case AttentionStrategy::kAuto: return "auto";
    case AttentionStrategy::kPerPair: return "per_pair";
    case AttentionStrategy::kSplitK: return "split_k";
  }
  return "auto";
}

const KernelPlan& KernelPlan::defaults() {
  static const KernelPlan plan = [] {
    KernelPlan p;
    // Tuning overrides for experiments and the autotuner.
    if (const char* em = std::getenv("DYNALM_MATMUL_EXPAND_MIN")) {
      p.expand_min_rows = static_cast<int32_t>(std::max<long>(1, std::strtol(em, nullptr, 10)));
    }
    if (const char* r = std::getenv("DYNALM_INT8_DECODE_ROWS")) {
      p.int8_decode_max_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(r, nullptr, 10)));
    }
    if (const char* c = std::getenv("DYNALM_MATMUL_CHUNKS")) {
      p.matmul_chunks_per_thread = static_cast<int32_t>(std::max<long>(1, std::strtol(c, nullptr, 10)));
    }
    if (const char* g = std::getenv("DYNALM_ATTN_GROUPED")) p.grouped_attention = std::strtol(g, nullptr, 10) != 0;
    if (const char* kc = std::getenv("DYNALM_GEMM_KC")) {
      const long v = std::strtol(kc, nullptr, 10);
      p.gemm_k_block = v > 0 ? static_cast<int32_t>(v / 256 * 256) : 0;
    }
    return p;
  }();
  return plan;
}

}  // namespace engine
