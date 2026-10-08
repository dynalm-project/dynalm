#include "dynacore/kernel/kernel_plan.h"

#include <algorithm>
#include <cstdlib>

#include "dynacore/hardware/cpu_info.h"

namespace dynacore {

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
    if (const char* em = std::getenv("DYNACORE_MATMUL_EXPAND_MIN")) {
      p.expand_min_rows = static_cast<int32_t>(std::max<long>(1, std::strtol(em, nullptr, 10)));
    }
    if (const char* r = std::getenv("DYNACORE_INT8_DECODE_ROWS")) {
      p.int8_decode_max_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(r, nullptr, 10)));
    }
    if (const char* c = std::getenv("DYNACORE_MATMUL_CHUNKS")) {
      p.matmul_chunks_per_thread = static_cast<int32_t>(std::max<long>(1, std::strtol(c, nullptr, 10)));
    }
    if (const char* sb = std::getenv("DYNACORE_INT8_SUPERBLOCK")) {
      p.int8_superblock_min_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(sb, nullptr, 10)));
    }
    if (const char* sx = std::getenv("DYNACORE_INT8_SUBSCALE")) {
      p.int8_subscale_min_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(sx, nullptr, 10)));
    }
    if (const char* rp = std::getenv("DYNACORE_Q4_REPACK_ROWS")) {
      p.q4_repack_min_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(rp, nullptr, 10)));
    }
    if (const char* rm = std::getenv("DYNACORE_Q4_REPACK_MAX_ROWS")) {
      p.q4_repack_max_rows = static_cast<int32_t>(std::max<long>(0, std::strtol(rm, nullptr, 10)));
    }
    if (const char* d = std::getenv("DYNACORE_INT16_FFN_DOWN")) p.int16_ffn_down = std::strtol(d, nullptr, 10) != 0;
    if (const char* g = std::getenv("DYNACORE_ATTN_GROUPED")) p.grouped_attention = std::strtol(g, nullptr, 10) != 0;
    if (const char* kc = std::getenv("DYNACORE_GEMM_KC")) {
      const long v = std::strtol(kc, nullptr, 10);
      p.gemm_k_block = v > 0 ? static_cast<int32_t>(v / 256 * 256) : 0;
    }
    return p;
  }();
  return plan;
}

HardwareProfile HardwareProfile::detect(int32_t threads, std::string_view isa) {
  const CpuInfo& c = cpu_info();
  HardwareProfile h;
  h.threads = std::max(1, threads);
  h.physical_cores = std::max(1, c.physical_cores);
  h.performance_cores = c.performance_cores;
  h.efficiency_cores = c.efficiency_cores;
  h.l2_bytes = c.l2_bytes;
  h.llc_bytes = c.l3_bytes;
  h.isa = isa;
  return h;
}

KernelPlan plan_kernels(const StepShape& shape, const HardwareProfile& hw, const KernelPlan& base) {
  KernelPlan k = base;
  // Attention runs one task per (row, KV head), covering that head's whole
  // query group (DD-054). Split each task's context only when the tasks
  // cannot occupy the threads (DD-032).
  const int64_t pairs = static_cast<int64_t>(shape.rows) * shape.num_kv_heads;
  auto choose = [&](int64_t longest) {
    return attention_should_split(pairs, longest, hw.threads, base.attention_chunk) ? AttentionStrategy::kSplitK
                                                                                    : AttentionStrategy::kPerPair;
  };
  k.attention_full = choose(shape.max_context);
  k.attention_window = shape.has_window_layers ? choose(std::min<int64_t>(shape.max_context, shape.sliding_window))
                                               : k.attention_full;

  // Dynamic split-K partition (DD-056): keep chunks at least the base size,
  // but round the number of chunks up so (row, KV head) units x chunks fills
  // whole waves of the thread pool. With a fixed 256-token chunk, one 4K
  // decode row was 2 x 16 = 32 tasks on 10 threads: a fourth wave of 2.
  if (k.attention_full == AttentionStrategy::kSplitK && pairs > 0) {
    const int64_t chunk = base.attention_chunk;
    const int64_t chunks = (shape.max_context + chunk - 1) / chunk;
    const int64_t waves = (pairs * chunks + hw.threads - 1) / hw.threads;
    const int64_t balanced = (waves * hw.threads + pairs - 1) / pairs;  // chunks per unit
    // Chunk length for that many chunks, rounded to whole 16-token runs.
    const int64_t len = ((shape.max_context + balanced - 1) / balanced + 15) / 16 * 16;
    k.attention_chunk = static_cast<int32_t>(std::max<int64_t>(16, len));
  }
  return k;
}

}  // namespace dynacore
