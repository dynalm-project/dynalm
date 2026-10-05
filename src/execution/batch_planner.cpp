#include "execution/batch_planner.h"

#include <algorithm>

#include "platform/cpu_info.h"

namespace engine {

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

std::string_view step_phase_name(StepPhase p) {
  switch (p) {
    case StepPhase::kDecode: return "decode";
    case StepPhase::kPrefill: return "prefill";
    case StepPhase::kMixed: return "mixed";
  }
  return "decode";
}

BatchPlanner::BatchPlanner(const ModelConfig& config, HardwareProfile hw, KernelPlan base)
    : num_kv_heads_(config.num_kv_heads > 0 ? config.num_kv_heads : config.num_heads),
      sliding_window_(config.sliding_window),
      hw_(hw),
      base_(base) {
  has_window_layers_ = false;
  for (int l = 0; l < config.num_layers; ++l) has_window_layers_ = has_window_layers_ || config.layer_uses_sliding_window(l);
}

ExecutionPlan BatchPlanner::plan(std::span<const SeqBatch> seqs) const {
  ExecutionPlan p;
  p.kernels = base_;
  p.sequences = static_cast<int32_t>(seqs.size());
  for (const SeqBatch& s : seqs) {
    const auto n = static_cast<int32_t>(s.tokens.size());
    p.rows += n;
    (n == 1 ? p.decode_rows : p.prefill_rows) += n;
    if (s.want_logits) p.logit_rows += s.logits_last;
    // Row i attends to positions [0, start + i]: spans start+1 .. start+n.
    p.max_context = std::max(p.max_context, s.start_pos + n);
    p.attention_work += static_cast<int64_t>(n) * s.start_pos + static_cast<int64_t>(n) * (n + 1) / 2;
  }
  p.phase = p.prefill_rows == 0 ? StepPhase::kDecode : p.decode_rows == 0 ? StepPhase::kPrefill : StepPhase::kMixed;

  // Attention runs one task per (row, KV head), covering that head's whole
  // query group (DD-054). Split each task's context only when the tasks
  // cannot occupy the threads (DD-032).
  const int64_t pairs = static_cast<int64_t>(p.rows) * num_kv_heads_;
  auto choose = [&](int64_t longest) {
    return attention_should_split(pairs, longest, hw_.threads, base_.attention_chunk) ? AttentionStrategy::kSplitK
                                                                                      : AttentionStrategy::kPerPair;
  };
  p.kernels.attention_full = choose(p.max_context);
  p.kernels.attention_window =
      has_window_layers_ ? choose(std::min<int64_t>(p.max_context, sliding_window_)) : p.kernels.attention_full;
  return p;
}

}  // namespace engine
