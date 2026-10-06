#pragma once

// Execution planner (DD-051).
//
// The scheduler decides WHICH sequences run in a step; the planner decides HOW
// that exact batch runs on this hardware: its phase, its shape, and the kernel
// strategies the backend should use. Every per-step execution heuristic lives
// here, in one testable place, instead of being re-derived inside kernels.
// The scheduler never sees ISA details and the backend never sees sequences or
// model families; they meet through ExecutionPlan / KernelPlan.

#include <cstdint>
#include <span>
#include <string_view>

#include "dynacore/kernel/kernel_plan.h"
#include "model/seq_batch.h"
#include "model_ir/model_config.h"
#include "common/core.h"

namespace dynalm {

enum class StepPhase : uint8_t {
  kDecode,   // every sequence contributes one row
  kPrefill,  // every sequence contributes a prompt chunk
  kMixed,    // both
};
std::string_view step_phase_name(StepPhase p);

struct ExecutionPlan {
  StepPhase phase = StepPhase::kDecode;
  int32_t rows = 0;          // tokens through the model
  int32_t sequences = 0;
  int32_t decode_rows = 0;   // single-row sequences
  int32_t prefill_rows = 0;  // rows of multi-row chunks
  int32_t logit_rows = 0;
  int32_t max_context = 0;   // longest attention span of any row (tokens)
  int64_t attention_work = 0;  // sum over rows of attention span (tokens): KV the step reads per layer and KV head
  KernelPlan kernels;
};

class BatchPlanner {
 public:
  BatchPlanner(const ModelConfig& config, HardwareProfile hw, KernelPlan base = KernelPlan::defaults());

  // Cheap (one pass over the sequences); called once per scheduler step.
  ExecutionPlan plan(std::span<const SeqBatch> seqs) const;

  const HardwareProfile& hardware() const { return hw_; }

 private:
  int32_t num_kv_heads_;
  int32_t sliding_window_;
  bool has_window_layers_;
  HardwareProfile hw_;
  KernelPlan base_;
};

}  // namespace dynalm
