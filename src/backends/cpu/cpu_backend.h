#pragma once

// CPU backend: ops over TensorViews, with the innermost primitives taken
// from a CpuKernels table selected once for the running CPU (generic / AVX2).

#include <memory>
#include <vector>

#include "backends/backend.h"
#include "backends/cpu/cpu_kernels.h"
#include "platform/isa.h"
#include "runtime/thread_pool.h"

namespace engine {


class CpuBackend final : public Backend {
 public:
  // `pool` must outlive the backend.
  CpuBackend(ThreadPool& pool, CpuIsa isa);

  std::string_view name() const override { return name_; }
  Device device() const override { return Device{}; }
  const CpuKernels& kernels() const { return k_; }

  Result<std::shared_ptr<Storage>> allocate(size_t bytes) override;
  void copy(void* dst, const void* src, size_t bytes) override;
  void synchronize() override {}
  bool supports_weight_type(DType type) const override;
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
  // True when this tier has SIMD integer-dot kernels; the generic tier's are
  // reference implementations, slower than its fused fp32 path.
  bool int8_accelerated_ = false;  // split-K attention partials (scheduler thread only)

  void attend_range(const AttentionParams& p, size_t r, int32_t h, int64_t t0, int64_t t1, float* acc, float& mx_out,
                    double& sum_out);
  std::string name_;
};

}  // namespace engine
