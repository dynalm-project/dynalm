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

 private:
  ThreadPool& pool_;
  CpuKernels k_;
  int64_t gemm_kc_ = 1024;  // GEMM K-slice width (multiple of 256; 0 = no K-blocking)
  // Rows at or above which matmul expands weight panels and runs the GEMM
  // kernel instead of per-row fused dequantize-dot, which re-reads the
  // weights for every row (DD-036). ENGINE_MATMUL_EXPAND_MIN overrides.
  int64_t expand_min_rows_ = 2;
  std::vector<float> split_scratch_;  // split-K attention partials (scheduler thread only)

  void attend_range(const AttentionParams& p, size_t r, int32_t h, int64_t t0, int64_t t1, float* acc, float& mx_out,
                    double& sum_out);
  std::string name_;
};

}  // namespace engine
