#pragma once

// Compute backend interface.
//
// The model runtime expresses a forward pass as a sequence of these ops on
// TensorViews; it never touches SIMD or device APIs. CpuBackend implements it
// today; a CUDA/HIP/Metal backend implements the same interface with device
// memory. Ops are coarse (a whole matmul / attention over the batch), so a
// virtual call per op is negligible.
//
// Conventions: activations are fp32 row-major [rows, cols]; weights may be any
// dtype the backend supports; ops returning void cannot fail on valid shapes
// (the runtime validates shapes once at model build time).

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/base/status.h"
#include "dynacore/attention/paged_kv.h"
#include "dynacore/device/ops.h"
#include "dynacore/memory/storage.h"
#include "dynacore/tensor/tensor.h"

namespace engine {

// Rows of a batch may belong to different sequences: row r reads/writes the
// KV of sequence row_seq[r] through kv[row_seq[r]].
struct AttentionParams {
  TensorView q;    // [m, n_heads * head_dim] fp32
  TensorView out;  // [m, n_heads * head_dim_v] fp32
  std::span<const int32_t> positions;  // absolute position of each query row
  std::span<const int32_t> row_seq;    // sequence index of each row
  std::span<const KvLayerView> kv;     // one view per sequence
  int32_t num_heads = 0;
  float scale = 1.0f;
  float softcap = 0.0f;         // 0 = off
  int32_t sliding_window = 0;   // 0 = full causal attention
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string_view name() const = 0;
  virtual Device device() const = 0;

  // --- memory (DD-045) ---
  // Device memory; the runtime never dereferences it directly unless
  // host_accessible().
  virtual Result<std::shared_ptr<Storage>> allocate(size_t bytes) = 0;
  // Device-to-device copy within this backend's memory.
  virtual void copy(void* dst, const void* src, size_t bytes) = 0;
  virtual void synchronize() = 0;
  // True if device memory is ordinary host memory (CPU): the runtime may then
  // skip uploads/downloads (zero-copy weights, logits written in place).
  virtual bool host_accessible() const { return device().type == DeviceType::kCpu; }
  // A device copy of a host tensor (weights, converted vectors). CPU returns
  // the tensor itself (zero-copy, e.g. straight from the mmapped file).
  virtual Result<Tensor> upload(const Tensor& host) = 0;
  // Copies fp32 rows of a device tensor [rows, cols] to dense host memory.
  virtual void download(const TensorView& src, std::span<float> dst) = 0;

  // Execution decisions for the next forward pass, from the planner (DD-051).
  // Backends without per-step choices ignore it.
  virtual void set_kernel_plan(const KernelPlan& plan) { (void)plan; }
  // Independent workers kernels can use (CPU: thread-pool size).
  virtual int32_t parallelism() const { return 1; }

  // Whether matmul/embedding accept this weight dtype.
  virtual bool supports_weight_type(DType type) const = 0;

  // --- ops ---
  // out[i, :] = table[ids[i], :]
  virtual void embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) = 0;
  // y[m, n] = x[m, k] · w[n, k]ᵀ (+ bias[n])
  virtual void matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) = 0;
  // Several independent matmuls (no bias), e.g. the active experts of an MoE
  // layer. Backends may run them in one parallel region; the default runs
  // them in order.
  struct MatmulJob {
    TensorView x, w, y;
  };
  virtual void matmul_many(std::span<const MatmulJob> jobs) {
    for (const MatmulJob& j : jobs) matmul(j.x, j.w, nullptr, j.y);
  }
  // Row-wise RMSNorm: y = x / rms(x) * weight
  virtual void rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) = 0;
  // Row-wise LayerNorm: y = (x - mean) / std * weight + bias (bias optional)
  virtual void layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
                          const TensorView& y) = 0;
  // In-place RoPE on x [m, heads * head_dim]. freq_factors optional (per rotated pair).
  virtual void rope(const TensorView& x, int32_t num_heads, int32_t head_dim,
                    std::span<const int32_t> positions, const RopeConfig& rope,
                    const float* freq_factors) = 0;
  // Writes k [m, n_kv * head_dim] and v [m, n_kv * head_dim_v] into the cache
  // of each row's sequence.
  virtual void kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                        std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) = 0;
  virtual void attention(const AttentionParams& p) = 0;
  // out = act(gate) * up  (elementwise)
  virtual void act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) = 0;
  // out = act(x)
  virtual void activation(Activation act, const TensorView& x, const TensorView& out) = 0;
  // y = a + b (same shapes); may alias a or b
  virtual void add(const TensorView& a, const TensorView& b, const TensorView& y) = 0;
  // x *= s
  virtual void scale(const TensorView& x, float s) = 0;
  // x = cap * tanh(x / cap)
  virtual void softcap(const TensorView& x, float cap) = 0;
  // x[:] = value
  virtual void fill(const TensorView& x, float value) = 0;
  // dst[i, :] = src[rows[i], :]
  virtual void gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) = 0;
  // dst[rows[i], :] += weights[i] * src[i, :]   (rows may repeat; applied in order)
  virtual void scatter_add_rows(const TensorView& src, std::span<const int32_t> rows, std::span<const float> weights,
                                const TensorView& dst) = 0;
};

}  // namespace engine
