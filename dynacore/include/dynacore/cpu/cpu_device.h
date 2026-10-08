#pragma once

// CPU backend: ops over TensorViews, with the innermost primitives taken
// from a CpuKernels table selected once for the running CPU (generic / AVX2).

#include <memory>
#include <unordered_map>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/quantization/repack.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/execution/thread_pool.h"

namespace dynacore {


class CpuDevice final : public Device {
 public:
  // `pool` must outlive the backend.
  CpuDevice(ThreadPool& pool, CpuIsa isa);

  std::string_view name() const override { return name_; }
  DeviceLoc device() const override { return DeviceLoc{}; }
  const CpuKernels& kernels() const { return k_; }

  Result<std::shared_ptr<Storage>> allocate(size_t bytes) override;
  void copy(void* dst, const void* src, size_t bytes) override;
  void synchronize() override {}
  bool supports_weight_type(DType type) const override;
  void prepack_weight(const TensorView& w) override;
  PrepackStats prepack_stats() const override { return prepack_stats_; }
  void set_kernel_plan(const KernelPlan& plan) override { plan_ = plan; }
  int32_t parallelism() const override { return pool_.size(); }
  const KernelPlan& kernel_plan() const { return plan_; }

  void embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) override;
  void matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) override;
  void matmul_many(std::span<const MatmulJob> jobs) override;
  Result<Tensor> upload(const Tensor& host) override { return host; }  // zero-copy
  void download(const TensorView& src, std::span<float> dst) override;
  void fill(const TensorView& x, float value) override;
  void gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) override;
  void scatter_add_rows(const TensorView& src, std::span<const int32_t> rows, std::span<const float> weights,
                        const TensorView& dst) override;
  void rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) override;
  void layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
                  const TensorView& y) override;
  void rope(const TensorView& x, int32_t num_heads, int32_t head_dim, std::span<const int32_t> positions,
            const RopeConfig& rope, const float* freq_factors) override;
  void kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) override;
  void attention(const AttentionParams& p) override;
  void act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) override;
  void matmul_gated(Activation act, const TensorView& x, const TensorView& w_gate, const TensorView& w_up,
                    const TensorView& out, const TensorView& gate_scratch, const TensorView& up_scratch) override;
  void activation(Activation act, const TensorView& x, const TensorView& out) override;
  void add(const TensorView& a, const TensorView& b, const TensorView& y) override;
  void scale(const TensorView& x, float s) override;
  void softcap(const TensorView& x, float cap) override;

 private:
  ThreadPool& pool_;
  CpuKernels k_;
  // Matmul path thresholds, GEMM K-blocking and attention strategy for the
  // current forward pass (DD-051). Defaults reproduce the pre-planner rules.
  KernelPlan plan_ = KernelPlan::defaults();
  std::vector<float> split_scratch_;
  // int8 activations of the current decode matmul (scheduler thread only).
  std::vector<ActBlockQ8> act_q8_;
  std::vector<ActBlockQ16> act_q16_;  // int16 activations (DD-076)
  // True when this tier has SIMD integer-dot kernels; the generic tier's are
  // reference implementations, slower than its fused fp32 path.
  bool int8_accelerated_ = false;  // split-K attention partials (scheduler thread only)

  // The int8 decode kernel for an m x k activation against weight type wt, or
  // dot == nullptr when the int8 path does not apply. `format` names the
  // activation layout `quantize` produces: per-32 scales (DD-053), super-block
  // (DD-075) or sub-scaled (DD-077); jobs share quantized activations only
  // within one format.
  enum class ActFormat : uint8_t { kPer32, kSuperblock, kSubscaled };
  struct Int8Kernel {
    QuantizeActFn quantize = nullptr;
    DotQ8RowsFn dot = nullptr;
    ActFormat format = ActFormat::kPer32;
    // Interleaved copy of this weight (DD-078), when the plan allows it.
    const BlockQ4_Kx8* packed = nullptr;
  };
  // `w` identifies the weight (its data pointer) for the packed-copy lookup.
  Int8Kernel int8_kernel(DType wt, int64_t m, int64_t k, const void* w = nullptr, int64_t n = 0) const;
  // y[0..m)[n0..n1) = x . w^T for int8 activations `act` (m rows, k / 32
  // blocks each); with ik.packed, n0 and n1 must be multiples of 8.
  void int8_rows(const Int8Kernel& ik, const std::byte* wbase, int64_t w_row_bytes, const ActBlockQ8* act,
                 int64_t m, int64_t k, int64_t n0, int64_t n1, const TensorView& y, const float* bias) const;

  // Interleaved Q4_K copies made by prepack_weight, by weight data pointer.
  struct Packed {
    std::vector<BlockQ4_Kx8> blocks;  // (n / 8) groups x (k / 256) blocks
    int64_t n = 0, k = 0;             // shape it was packed from
  };
  // The interleaved copy of the weight at `w` with exactly n x k elements.
  const BlockQ4_Kx8* find_packed(const void* w, int64_t n, int64_t k) const;
  std::unordered_map<const void*, Packed> packed_;
  PrepackStats prepack_stats_;

  // One query row, one KV head, query heads [kvh*group + h0, + count) of the
  // group sharing it (DD-054).
  void attend_group(const AttentionParams& p, size_t r, int32_t kvh, int32_t group, int32_t h0, int32_t count,
                    int64_t t0, int64_t t1, float* acc, float* mx_out, double* sum_out);
  std::string name_;
};

}  // namespace dynacore
