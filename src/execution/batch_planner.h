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

#include "backends/kernel_plan.h"
#include "model/seq_batch.h"
#include "model_ir/model_config.h"

namespace engine {

// What the planner knows about the machine. Filled once at engine creation.
struct HardwareProfile {
  int32_t threads = 1;         // compute threads available to kernels
  int32_t physical_cores = 1;
  int32_t performance_cores = 0;  // 0 = not a hybrid CPU
  int32_t efficiency_cores = 0;
  int64_t l2_bytes = 0;        // per core
  int64_t llc_bytes = 0;
  std::string_view isa;        // e.g. "avx2" (informational)

  // From platform/cpu_info, with `threads` compute threads.
  static HardwareProfile detect(int32_t threads, std::string_view isa);
};

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
  int32_t num_heads_;
  int32_t sliding_window_;
  bool has_window_layers_;
  HardwareProfile hw_;
  KernelPlan base_;
};

}  // namespace engine
