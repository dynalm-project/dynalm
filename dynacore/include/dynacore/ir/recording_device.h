#pragma once

// RecordingDevice: a Device that turns the op stream of a forward pass into
// DynaCore IR (DD-071) and, in deferred mode, compiles and executes it
// (DD-072). The model code above it does not change: whatever calls Device
// ops (DynaLM's Transformer today) is captured.
//
// Modes:
//   kTrace     build IR for each op and run it on the inner device at once.
//              With timing on, every IR op gets its wall time: the op-level
//              profile that drives compiler decisions.
//   kDeferred  record calls (with private copies of every host argument) and
//              run nothing until a sync point (synchronize, download, copy,
//              allocate, upload). At the sync point the segment is compiled
//              and executed on the inner device:
//                1. its structural signature (op kinds, buffers, shapes, the
//                   kernel plan of each call; not positions, token ids or
//                   block tables) is looked up in the plan cache;
//                2. on a miss the IR graph is built, compiled by the
//                   ExecutionPlanner (canonicalize, kernel selection, fusion)
//                   and the resulting call-level plan is cached;
//                3. the plan runs on the inner device.
//              Decode steps repeat one structure, so after the first step a
//              segment costs one signature hash, no IR construction and no
//              planning. Planner errors and invalid plans fall back to running
//              the calls in recorded order: compilation can never break
//              inference.
//
// set_kernel_plan is not a sync point: each call keeps the plan in effect and
// execution re-applies it, so per-op plans (int8 off for the FFN down
// projection, DD-053) survive fusion. The recording device reports
// host_accessible() == false, so callers fetch results through download(),
// which is a sync point.

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/ir/ir.h"

namespace dynacore::ir {

// How a recorded segment runs, in IR op indices: a sequence of steps, each
// one op or a group executed by one inner-device call:
//   kOp           the ops in order, one call each
//   kMatmulGroup  matmuls -> one matmul_many (shared-input projections)
//   kGatedMatmul  {gate matmul, up matmul, act_mul} -> one matmul_gated
struct ExecStep {
  enum class Kind : uint8_t { kOp, kMatmulGroup, kGatedMatmul };
  Kind kind = Kind::kOp;
  std::vector<int32_t> ops;  // IR op indices
};
struct ExecPlan {
  std::vector<ExecStep> steps;
};
// Turns a recorded graph into an ExecPlan. It may annotate the graph (kernel
// choices, fusion groups). Returning an error means "run unplanned".
using ExecutionPlanner = std::function<Result<ExecPlan>(Graph&)>;

struct RecordingStats {
  int64_t segments = 0;       // flushed segments (deferred)
  int64_t ops = 0;            // recorded device calls
  int64_t planned_steps = 0;  // inner-device calls after planning
  int64_t fallbacks = 0;      // segments run unplanned after a planner error / invalid plan
  int64_t cache_hits = 0;     // segments that reused a cached plan
  int64_t cache_misses = 0;   // segments compiled from IR
  int64_t record_ns = 0;      // time spent recording calls
  int64_t plan_ns = 0;        // time spent building IR + planning (misses only)
};

class RecordingDevice final : public Device {
 public:
  enum class Mode : uint8_t { kTrace, kDeferred };

  RecordingDevice(Device& inner, Mode mode);
  ~RecordingDevice() override;

  // Debug names for weights / caches (shown by the printer).
  void set_name(const void* data, std::string name);
  // Trace mode: per-op wall time in the graph (op_ns()).
  void set_timing(bool on) { timing_ = on; }
  // Deferred mode: the planner applied on plan-cache misses (none = in order).
  void set_planner(ExecutionPlanner planner);
  // Deferred mode: reuse plans of structurally identical segments (default on).
  void set_plan_cache(bool on) { cache_on_ = on; }
  // Called with each compiled segment (cache misses) after planning.
  void set_segment_observer(std::function<void(const Graph&, const ExecPlan&)> f) { observer_ = std::move(f); }

  // Starts a new graph (trace mode: one per forward pass, caller-driven).
  void begin_graph(std::string name);
  // Trace mode: everything since begin_graph. Deferred mode: the last
  // compiled segment.
  const Graph& graph() const { return graph_; }
  // Trace mode: wall time of each graph op (0 for leaves), aligned with ops().
  const std::vector<int64_t>& op_ns() const { return op_ns_; }
  const RecordingStats& stats() const { return stats_; }
  Device& inner() { return inner_; }
  void prepack_weight(const TensorView& w, bool int16_activations = false) override {
    inner_.prepack_weight(w, int16_activations);
  }
  PrepackStats prepack_stats() const override { return inner_.prepack_stats(); }

  // Runs everything recorded so far (deferred mode). Call it (or any sync
  // point) before destroying buffers the recorded calls use: the destructor
  // discards pending calls.
  Status flush();

  // --- Device ---
  std::string_view name() const override { return name_; }
  DeviceLoc device() const override { return inner_.device(); }
  bool host_accessible() const override { return false; }
  Result<std::shared_ptr<Storage>> allocate(size_t bytes) override;
  void copy(void* dst, const void* src, size_t bytes) override;
  void synchronize() override;
  Result<Tensor> upload(const Tensor& host) override;
  void download(const TensorView& src, std::span<float> dst) override;
  void set_kernel_plan(const KernelPlan& plan) override;
  int32_t parallelism() const override { return inner_.parallelism(); }
  bool supports_weight_type(DType type) const override { return inner_.supports_weight_type(type); }

  void embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) override;
  void matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) override;
  void matmul_many(std::span<const MatmulJob> jobs) override;
  void rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) override;
  void layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
                  const TensorView& y) override;
  void rope(const TensorView& x, int32_t num_heads, int32_t head_dim, std::span<const int32_t> positions,
            const RopeConfig& rope, const float* freq_factors) override;
  void kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) override;
  void attention(const AttentionParams& p) override;
  void act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) override;
  void activation(Activation act, const TensorView& x, const TensorView& out) override;
  void add(const TensorView& a, const TensorView& b, const TensorView& y) override;
  void scale(const TensorView& x, float s) override;
  void softcap(const TensorView& x, float cap) override;
  void fill(const TensorView& x, float value) override;
  void gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) override;
  void scatter_add_rows(const TensorView& src, std::span<const int32_t> rows, std::span<const float> weights,
                        const TensorView& dst) override;

  // A recorded call with private copies of its host arguments.
  struct Call;

 private:
  enum class Role : uint8_t { kActivation, kWeight };
  // Execution steps in call indices (what the plan cache stores).
  struct CallStep {
    ExecStep::Kind kind = ExecStep::Kind::kOp;
    std::vector<int32_t> calls;
  };
  struct CacheEntry {
    std::vector<uint64_t> signature;
    std::vector<CallStep> steps;
  };

  Call& next_call(int kind);
  void submit(Call& c);
  void sign(const Call& c);
  // IR for one call; returns the index of the first op it added.
  int32_t build_ir(const Call& c);
  ValueId use(const TensorView& t, Role role);
  ValueId index_const(std::span<const int32_t> data, std::string_view name);
  ValueId kv_value(const KvLayerView& kv);
  ValueId record_op(const Call* c, OpKind kind, std::vector<ValueId> inputs, Attrs attrs, const TensorView& out,
                    bool in_place);
  void apply_plan(const Call& c);
  void run_call(const Call& c);
  std::vector<CallStep> compile_pending();
  void execute(const std::vector<CallStep>& steps);

  Device& inner_;
  Mode mode_;
  std::string name_;
  bool timing_ = false;
  bool cache_on_ = true;
  Graph graph_;
  std::vector<int64_t> op_ns_;
  ExecutionPlanner planner_;
  std::function<void(const Graph&, const ExecPlan&)> observer_;
  // IR construction state (graph values per buffer view, names, caches).
  std::map<std::tuple<const void*, int64_t, int64_t, int64_t>, ValueId> live_;
  std::map<const void*, int32_t> buffers_;
  std::map<const void*, std::string> names_;
  std::map<const void*, ValueId> kv_values_;
  // Recorded calls of the current segment; Call objects are reused.
  std::vector<std::unique_ptr<Call>> calls_;
  size_t ncalls_ = 0;
  std::vector<uint64_t> signature_;
  std::unordered_map<uint64_t, CacheEntry> cache_;
  std::shared_ptr<const KernelPlan> plan_;     // in effect for new calls
  std::shared_ptr<const KernelPlan> applied_;  // last plan given to the inner device
  RecordingStats stats_;
};

}  // namespace dynacore::ir
