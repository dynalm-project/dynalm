#pragma once

// Whole-segment execution (docs/gpu-backend.md, section 4).
//
// A DeviceCall is one recorded Device call with private copies of everything
// that is not device memory (spans, block tables, parameter structs). The
// RecordingDevice (dynacore/ir) records a forward step as DeviceCalls; at a
// sync point it hands the whole step to the inner device when that device is
// a SegmentExecutor, instead of replaying the calls one by one. A GPU device
// turns the segment into one backend graph per step (no per-op CPU/GPU round
// trips); the CPU device is not a SegmentExecutor and is unaffected.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/device/device.h"
#include "dynacore/kernel/kernel_plan.h"

namespace dynacore {

struct DeviceCall {
  enum class M : uint8_t {
    kEmbedding, kMatmul, kMatmulMany, kRmsNorm, kLayerNorm, kRope, kKvStore, kAttention,
    kActMul, kActivation, kAdd, kScale, kSoftcap, kFill, kGather, kScatterAdd, kMoeFfn,
  };
  // Operands by method (see RecordingDevice):
  //   kEmbedding    a = table, b = out, ints0 = ids
  //   kMatmul       a = x, b = w, c = bias (has_c), d = y
  //   kMatmulMany   jobs (biases point into `biases`)
  //   kRmsNorm      a = x, b = weight, f = eps, d = y
  //   kLayerNorm    a = x, b = weight, c = bias (has_c), f = eps, d = y
  //   kRope         a = x (in place), i0 = heads, i1 = head_dim, ints0 = positions, rope, freq
  //   kKvStore      a = k, b = v, ints0 = positions, ints1 = row_seq, kv
  //   kAttention    ap (positions/row_seq/kv point into ints0/ints1/kv)
  //   kActMul       act, a = gate, b = up, d = out
  //   kActivation   act, a = x, d = out
  //   kAdd          a, b, d
  //   kScale        a (in place), f
  //   kSoftcap      a (in place), f
  //   kFill         a, f
  //   kGather       a = src, d = dst, ints0 = rows
  //   kScatterAdd   a = src, d = dst, ints0 = rows, floats = weights
  //   kMoeFfn       moe (segment executors only; no IR form)
  M m{};
  TensorView a, b, c, d;
  bool has_c = false;
  float f = 0;
  int32_t i0 = 0, i1 = 0;
  Activation act{};
  RopeConfig rope{};
  const float* freq = nullptr;
  std::vector<int32_t> ints0, ints1;
  std::vector<float> floats;
  std::vector<std::vector<int32_t>> tables;
  std::vector<KvLayerView> kv;
  std::vector<Device::MatmulJob> jobs;
  std::vector<TensorView> biases;  // matmul_many biases (jobs point here)
  AttentionParams ap{};
  MoeFfnParams moe{};
  std::shared_ptr<const KernelPlan> plan;  // plan in effect when recorded

  void reset(M kind) {
    m = kind;
    a = b = c = d = TensorView();
    has_c = false;
    f = 0;
    i0 = i1 = 0;
    freq = nullptr;
    ints0.clear();
    ints1.clear();
    floats.clear();
    kv.clear();
    jobs.clear();
    biases.clear();
    ap = AttentionParams{};
    moe = MoeFfnParams{};
  }

  void copy_kv(std::span<const KvLayerView> src) {
    if (tables.size() < src.size()) tables.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) tables[i].assign(src[i].block_table.begin(), src[i].block_table.end());
    kv.assign(src.begin(), src.end());
    for (size_t i = 0; i < kv.size(); ++i) kv[i].block_table = tables[i];
  }
};

// Counters of a segment executor (observability; docs/gpu-backend.md section 6).
struct SegmentStats {
  int64_t segments = 0;        // segments run
  int64_t graph_builds = 0;    // prepared graphs built (cache misses)
  int64_t graph_reuses = 0;    // segments that reused a prepared graph
  int64_t graph_evictions = 0;
  int64_t graph_nodes = 0;     // nodes of the last graph run
  int64_t build_ns = 0;        // building + allocating graphs (misses)
  int64_t input_ns = 0;        // uploading dynamic inputs
  int64_t compute_ns = 0;      // submit + wait for the graph
  int64_t input_bytes = 0;     // dynamic inputs uploaded
  int64_t download_bytes = 0;  // Device::download
  int64_t download_ns = 0;
  int64_t errors = 0;
};

// Implemented by devices that run a recorded segment as one unit.
class SegmentExecutor {
 public:
  virtual ~SegmentExecutor() = default;
  // Runs `calls` (in recorded order). `signature` is the recorder's structural
  // signature of the segment: op kinds, buffers, shapes, strides, dtypes and
  // parameters, but not token ids, positions or block tables, which the
  // executor reads from the calls on every run. The executor adds whatever
  // else determines its graph (for example per-sequence row splits) to its own
  // cache key. On error nothing is partially applied that the caller can
  // observe as success: the step must be treated as failed.
  virtual Status run_segment(std::span<const DeviceCall* const> calls, std::span<const uint64_t> signature) = 0;
  virtual SegmentStats segment_stats() const = 0;
};

}  // namespace dynacore
