#include "execution/batch_planner.h"

#include <algorithm>
#include "common/core.h"

namespace dynalm {

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

  StepShape shape;
  shape.rows = p.rows;
  shape.max_context = p.max_context;
  shape.num_kv_heads = num_kv_heads_;
  shape.sliding_window = sliding_window_;
  shape.has_window_layers = has_window_layers_;
  p.kernels = plan_kernels(shape, hw_, base_);
  return p;
}

}  // namespace dynalm
