#pragma once

// DynaKernel backends: where a verified program meets a target.
//
// The program (program.h) says what each kernel must compute and under which
// numerical contract. A backend says what it can run here: check() rejects
// ops, dtypes or contracts it does not implement, before any data is touched;
// run() executes over caller-owned buffers. Target capabilities, kernel and
// tile choices belong to the backend, never to the program.
//
// ReferenceBackend lowers every op to the existing DynaCore CPU kernels
// through its own CpuDevice (Device::matmul). It is the correctness baseline
// for future backends (generated code, tuned kernels). It generates no code.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/dynakernel/program.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/tensor/tensor.h"

namespace dynacore {
class CpuDevice;
}

namespace dynacore::dk {

// What a backend targets (informational in v0; a future planner keys
// measured costs on it).
struct TargetDesc {
  std::string backend;    // "reference"
  std::string isa;        // "avx2", "generic", ...
  int32_t threads = 1;
  bool int8_dot = false;  // integer-dot kernels are accelerated on this tier
  // ... including the sub-scaled family and the interleaved Q4_K layout
  // (DD-077 / DD-078: AVX2 today); without it, int8 means per-32 only (NEON).
  bool int8_subscaled = false;
};

// Caller memory for a program's values. Bindings never own memory: every
// bound buffer must stay valid until run() returns, and results are written
// in place (nothing is allocated or resized). Views must match the declared
// layout exactly and meet the declared and natural alignment; results must
// not overlap any other bound buffer.
//
// A TensorView carries no capacity. Bind with a capacity (or a Tensor, whose
// Storage knows its size) to have run() check that the layout's byte span
// fits the buffer; a bare view is trusted to cover it.
class Bindings {
 public:
  struct Entry {
    TensorView view;
    int64_t capacity = -1;  // bytes available from view.data(); -1 = unknown
  };
  void bind(ValueId v, const TensorView& view) { entries_[v] = {view, -1}; }
  void bind(ValueId v, const TensorView& view, int64_t capacity_bytes) { entries_[v] = {view, capacity_bytes}; }
  // Capacity from the tensor's storage (bytes from the view to the storage end).
  void bind(ValueId v, const Tensor& t);
  const TensorView* find(ValueId v) const;
  const Entry* entry(ValueId v) const;

 private:
  std::map<ValueId, Entry> entries_;
};

// Checks every value of a verified program against its binding: bound,
// non-null, exactly the declared dtype/shape/strides, the declared and
// natural alignment, the declared capacity, and no result overlapping any
// other bound buffer. Shared by all backends; nothing runs unless it passes.
Status validate_bindings(const Program& p, const Bindings& b);

// FNV-1a 64 of `bytes` bytes: the prepacked-source fingerprint.
uint64_t fingerprint_bytes(const void* data, int64_t bytes);

class Backend {
 public:
  virtual ~Backend() = default;
  virtual TargetDesc target() const = 0;
  // verify(), then OK when every op can run here with its contract; else
  // kUnsupported naming the op and the reason.
  virtual Status check(const Program& p) const = 0;
  // verify + check + binding validation, then executes ops in order. Nothing
  // runs unless every check passes.
  virtual Status run(const Program& p, const Bindings& b) = 0;
};

struct RunOptions {
  // Runs maximal runs of consecutive, mutually independent matmuls with the
  // same activation contract as one Device::matmul_many call (DD-060), as
  // GraphExecutor does for a planned matmul group. Off: one Device::matmul
  // per op. The two are NOT bit-identical in general: matmul_many's panel
  // path does not K-block (K > gemm_k_block sums in another order), f32
  // weights take the per-row dot instead of the panel GEMM, and strided x
  // goes through the panel GEMM instead of per-row dots. Int8 paths, the
  // single-row fused dot and quantized panels with K <= gemm_k_block match
  // exactly. A group of one is a plain matmul.
  bool group_independent = false;
  // Before running, re-fingerprint the source bytes of every bound weight
  // that was prepacked (prepack_weight) and fail if they changed: the
  // interleaved copy is a snapshot, and the device would otherwise use it
  // silently. Costs one pass over those weights per run.
  bool check_packed = true;
  // EXPERIMENTAL: run with KernelPlan::packed_int8_8rows on (8-row tiles on
  // interleaved Q4_K copies for 5..8 rows). Off: the plan's default (off
  // unless DYNACORE_PACKED_8ROWS=1).
  bool experimental_packed_8rows = false;
  // EXPERIMENTAL (GeneratedInt8Backend, hence LlvmInt8Backend): quantize the
  // activations in a parallel region of (row, super-block range) chunks
  // instead of serially, when the pool has more than one thread and the op
  // has at least parallel_quant_min_superblocks super-blocks (M * K / 256).
  // Same quantizer per super-block, so codes and outputs are identical. Off:
  // the serial loop. ReferenceBackend ignores it.
  bool experimental_parallel_quant = false;
  int64_t parallel_quant_min_superblocks = 96;
};

// An interleaved copy made by ReferenceBackend::prepack_weight. The copy is
// owned by the backend's device and lives until the backend is destroyed;
// it is used for a bound weight only when the binding has exactly this data
// pointer, dtype and shape (so an ordinary GGUF weight is never treated as
// packed). The source bytes must stay alive and unchanged while the copy is
// in use (checked when RunOptions::check_packed).
struct PackedWeight {
  const void* data = nullptr;
  DType dtype = DType::kQ4_K;
  int64_t n = 0, k = 0;
  int64_t source_bytes = 0;
  int64_t packed_bytes = 0;
  uint64_t fingerprint = 0;  // FNV-1a of the source bytes at packing time
};

class ReferenceBackend final : public Backend {
 public:
  // `pool` must outlive the backend. The backend owns a CpuDevice of its own,
  // so its kernel plans never reach a model's device. run() is not
  // thread-safe (the device keeps per-call scratch), and the pool must not be
  // running another device's work at the same time: ThreadPool::parallel_for
  // has one calling thread and is not reentrant (checked only by assert, so a
  // Release build would race silently). Do not share a live engine's pool.
  ReferenceBackend(ThreadPool& pool, CpuIsa isa);
  ~ReferenceBackend() override;

  TargetDesc target() const override;
  Status check(const Program& p) const override;
  Status run(const Program& p, const Bindings& b) override;
  Status run(const Program& p, const Bindings& b, const RunOptions& opts);

  // Opt-in: makes the interleaved 8-row Q4_K copy (DD-078) of a weight on this
  // backend's device, so matmuls of 2..8 rows under the auto contract reach
  // the packed sub-scaled kernel. Requires a dense Q4_K [N, K] view with
  // N % 8 == 0 (K is whole super-blocks by construction), natural alignment,
  // and, when given, capacity_bytes >= its span. kUnsupported when this tier
  // has no interleaved kernel or the device declined (memory budget,
  // DYNACORE_Q4_REPACK=off, KernelPlan q4_repack_min_rows = 0). Nothing is
  // packed unless this is called; production devices are not affected.
  Result<PackedWeight> prepack_weight(const TensorView& w, int64_t capacity_bytes = -1);
  // Ends the registration of a packed source before its memory is freed or
  // reused. The device keeps its copy (CpuDevice has no release; the bytes
  // return when the backend is destroyed), but run() never lets the device use
  // a copy for a weight that is not registered: ops on unregistered weights
  // run with packed copies disabled, so an address reused by another weight
  // cannot pick up a stale copy. A released address cannot be packed again
  // on this backend (the device would keep the old copy). false if `data`
  // was not registered.
  bool release_packed(const void* data);
  const std::vector<PackedWeight>& packed_weights() const { return packed_; }

  // How run() partitions a verified program's ops: each inner vector is one
  // device call (size 1: matmul; size > 1: matmul_many), in program order.
  static std::vector<std::vector<int32_t>> plan_calls(const Program& p, const RunOptions& opts);

  // The kernel plan a matmul runs under: KernelPlan::defaults() for kAuto;
  // with every reduced-precision activation path off for kF32.
  static KernelPlan plan_for(const MatMulAttrs& a);

 private:
  Status validate_bindings(const Program& p, const Bindings& b, const RunOptions& opts) const;

  ThreadPool& pool_;
  std::unique_ptr<CpuDevice> dev_;
  std::vector<PackedWeight> packed_;
  std::vector<const void*> retired_;  // released sources: the device still holds their copies
  std::string isa_;
  int32_t threads_ = 1;
};

}  // namespace dynacore::dk
