#pragma once

// DynaCore compiler passes over a recorded or built IR graph (DD-071, DD-072).
//
//   verify -> canonicalize -> analyze_memory -> select_kernels -> plan_execution
//
// Each pass is a plain function, testable on its own. compile_segment() runs
// the pipeline that the compiled execution mode applies at every sync point.
// Every decision here was measured first (see the design decisions);
// passes that do not pay for themselves stay off by default.

#include <cstdint>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/ir/ir.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/kernel/kernel_plan.h"

namespace dynacore::ir {

// Uses of every value (op indices), in op order.
std::vector<std::vector<int32_t>> value_uses(const Graph& g);

// --- canonicalization ----------------------------------------------------------
// matmul with a block-quantized weight -> qmatmul; attention with
// heads != kv_heads -> gqa (and back); removes ops whose results are never
// used and have no side effect. Returns the number of rewrites.
int canonicalize(Graph& g);

// --- memory analysis -----------------------------------------------------------
struct MemoryReport {
  int64_t activation_bytes = 0;  // sum over activation values
  int64_t peak_live_bytes = 0;   // max over ops of bytes of live activation values
  int64_t buffer_bytes = 0;      // sum over distinct activation buffers (what is allocated today)
  int32_t values = 0;
  int32_t buffers = 0;
  // A buffer assignment that reuses storage once a value is dead: the bytes it
  // would need. Analysis only; DynaLM owns its scratch today.
  int64_t planned_bytes = 0;
};
// Symbolic dimensions (M) count as `symbol_size`.
MemoryReport analyze_memory(const Graph& g, int64_t symbol_size = 1);

// --- cost model and kernel selection -------------------------------------------
// A first, deliberately small cost model: memory-bound ops cost bytes moved
// over bandwidth, plus a fixed price per parallel region (the dispatch +
// barrier cost measured by bench_thread_pool / bench_op_trace). Fitted to
// i7-1255U measurements; the point is ranking alternatives, not prediction.
struct CostModel {
  double dram_gbps = 16.0;      // sustained weight streaming (read_bw: 18.4 peak)
  double cache_gbps = 100.0;    // activations that stay in L2/L3
  double region_us = 20.0;      // one parallel region: dispatch, tail wait, coherence misses
  int32_t threads = 10;
  int32_t int8_max_rows = 4;    // KernelPlan::int8_decode_max_rows
  int32_t expand_min_rows = 2;  // KernelPlan::expand_min_rows

  static CostModel from(const KernelPlan& plan, int32_t threads);
};

// Estimated microseconds of one op on its own (one parallel region).
double estimate_op_us(const Graph& g, const Op& op, const CostModel& cm);

// Annotates Op::kernel with the implementation the CPU device will run for
// this shape (e.g. "cpu.int8_dot.q4_K", "cpu.panel_gemm.q6_K", "cpu.gqa_grouped").
void select_kernels(Graph& g, const CostModel& cm);

// --- fusion and execution planning ----------------------------------------------
struct FusionOptions {
  // Consecutive matmuls reading the same activations (Q/K/V) run as one
  // matmul_many region.
  bool group_shared_input = true;
  // gate + up matmuls followed by act_mul -> one matmul_gated call (no
  // intermediate gate/up tensors written; the activation is applied in the
  // dot-product epilogue).
  bool gated_mlp = true;
};

struct FusionReport {
  int32_t groups = 0;        // matmul groups formed
  int32_t grouped_ops = 0;   // matmuls inside them
  int32_t gated = 0;         // gated MLP fusions
  double est_saved_us = 0;   // cost-model estimate of the savings
};

// Builds the execution plan of a recorded segment and tags fused ops with
// Op::fusion_group. Never reorders across a data dependence.
ExecPlan plan_execution(Graph& g, const FusionOptions& opts, const CostModel& cm, FusionReport* report = nullptr);

// --- pipeline ------------------------------------------------------------------
struct CompileOptions {
  bool verify = false;  // verify before and after (tests, --verify); costs time per step
  FusionOptions fusion;
  CostModel cost;
};

struct CompileReport {
  int32_t canonicalized = 0;
  MemoryReport memory;
  FusionReport fusion;
};

Result<ExecPlan> compile_segment(Graph& g, const CompileOptions& opts, CompileReport* report = nullptr);

// The ExecutionPlanner a RecordingDevice in deferred mode uses.
ExecutionPlanner make_planner(CompileOptions opts);

}  // namespace dynacore::ir
