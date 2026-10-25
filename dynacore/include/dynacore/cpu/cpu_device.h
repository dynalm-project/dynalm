#pragma once

// CPU backend: ops over TensorViews, with the innermost primitives taken
// from a CpuKernels table selected once for the running CPU (generic / AVX2).

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/cpu/qmm.h"
#include "dynacore/kernel/matmul_registry.h"
#include "dynacore/quantization/repack.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/execution/thread_pool.h"

namespace dynacore {

// A matmul kernel supplied at run time (DynaKernel extension point,
// matmul_registry.h): consulted before the built-in kernels only while
// KernelPlan::external_matmul is set. `eligible` must not touch tensor data;
// `run` computes y = x . w^T (+ bias) for an eligible call.
struct ExternalMatmulKernel {
  std::string name;
  std::function<bool(const MatmulShape& shape, const void* weight)> eligible;
  std::function<void(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y)> run;
};

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
  void prepack_weight(const TensorView& w, bool int16_activations = false) override;
  PrepackStats prepack_stats() const override { return prepack_stats_; }
  void set_kernel_plan(const KernelPlan& plan) override { plan_ = plan; }
  int32_t parallelism() const override { return pool_.size(); }
  // Process-wide count of int8_rows calls that used the experimental 8-row
  // packed Q4_K path (KernelPlan::packed_int8_8rows); 0 while it is off.
  static uint64_t experimental_rows8_dispatches();
  // Matmuls run by the experimental int16 prefill path (KernelPlan::int16_prefill_max_rows).
  static uint64_t experimental_int16_prefill_calls();
  const KernelPlan& kernel_plan() const { return plan_; }

  // --- matmul registry (matmul_registry.h) ---
  // The kernel CpuDevice::matmul would run for x . w^T under the current plan
  // (no execution; explain = list the passed-over kernels).
  MatmulSelection plan_matmul(const TensorView& x, const TensorView& w, bool explain = false) const;
  // Process-wide per-kernel counters: calls, rows (sum of M), MACs (M*N*K);
  // ns while timing is on. One entry per MatmulKernelId.
  static std::vector<MatmulKernelStats> matmul_stats();
  // Per (kernel, weight type, N, K, M) calls and ns, recorded while timing is on.
  static std::vector<MatmulShapeStats> matmul_shape_stats();
  static void reset_matmul_stats();
  // Wall-clock timing per matmul (two clock reads per call). Off by default;
  // DYNACORE_MATMUL_TIMING=1 turns it on at start-up.
  static void set_matmul_timing(bool on);
  static bool matmul_timing();
  // Registers an external kernel (before serving: not synchronised with
  // matmul). Fails on an empty or duplicate name or missing functions.
  Status register_external_matmul(ExternalMatmulKernel kernel);
  size_t external_matmul_count() const { return external_.size(); }

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
  std::vector<ActBlockQ16S> act_q16s_;  // split-byte int16 activations (int16_split_prefill_x8)
  std::vector<ExternalMatmulKernel> external_;
  // Native quantized matmul tier (qmm.h); null where only the scalar
  // reference exists (it is slower than the int8 / fp32 paths here).
  const qmm::Kernels* qmm_ = nullptr;
  std::vector<qmm::ActBuffer> qmm_acts_;  // quantized activations (scheduler thread only)
  // SME fp32 panel gemm (qmm.h) for prefill-sized batches; null without SME.
  const qmm::SmeGemm* sme_ = nullptr;
  std::vector<float> sme_x_;  // activation panels of the current sme_matmul (scheduler thread only)
  // Every job through the SME panel gemm in one parallel region: activation
  // panels packed once per distinct x, each weight panel dequantized once and
  // multiplied with all of them.
  void sme_matmul(std::span<const MatmulJob> jobs);
  // Prompt-chunk attention on the SME tiles (flash attention: 32 query rows x
  // the group's heads against 128-key blocks, online softmax), for runs of
  // consecutive positions of one sequence [begin, end) of the batch rows.
  // K/V are converted into fp32 panels once per call (per slab of KV heads,
  // at most kSmeAttnPanelBytes at a time) and shared by every query block.
  struct SmeRun {
    size_t begin, end;
  };
  void attention_sme(const AttentionParams& p, std::span<const SmeRun> runs);
  std::vector<float> sme_kv_panels_;  // scheduler thread only
  // The per-row attention path for the rows in `sel` (empty: every row).
  void attention_rows(const AttentionParams& p, std::span<const size_t> sel);
  // Packed (4-row interleaved) copies made by prepack_weight for the qmm
  // kernels, by weight data pointer; once a weight is packed every qmm call
  // reads the copy (rows past n / 4 * 4 the original), never the original.
  struct QmmPacked {
    std::vector<std::byte> data;
    int64_t n = 0, k = 0;
  };
  std::unordered_map<const void*, QmmPacked> qmm_packed_;
  const std::byte* find_qmm_packed(const void* w, int64_t n, int64_t k) const;
  // Quantizes x (rows may be strided) into buf for the qmm kernels.
  void qmm_quantize(qmm::ActBuffer& buf, const TensorView& x, qmm::ActMode mode, bool paired);
  // y[.., n0..n1) for one weight (no threading; the caller partitions n).
  // `packed`: the weight's packed copy (find_qmm_packed), or null. Resolved
  // once per matmul by the caller, never per work item.
  void qmm_columns(const qmm::Acts& a, bool gemm, const TensorView& w, const std::byte* packed, const TensorView& y,
                   const float* bias, int64_t n0, int64_t n1) const;
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
    // 8-row tiles on the interleaved copy (int8_subscaled_x8_rows8).
    bool rows8 = false;
  };
  // `w` identifies the weight (its data pointer) for the packed-copy lookup.
  Int8Kernel int8_kernel(DType wt, int64_t m, int64_t k, const void* w = nullptr, int64_t n = 0) const;
  // The int8 kernel a selection names (dot == nullptr when it is not int8).
  Int8Kernel int8_from(const MatmulSelection& sel, DType wt, const void* w, int64_t n, int64_t k, int64_t m) const;
  // What this tier offers for one call (registry input).
  MatmulCaps matmul_caps(DType wt, const void* w, int64_t n, int64_t k, int64_t m) const;
  // Index of the first eligible external kernel, or -1.
  int external_for(const MatmulShape& s, const void* w) const;
  // Executes the selected kernel (ext: external kernel index for kExternal).
  void run_matmul(const MatmulSelection& sel, int ext, const TensorView& x, const TensorView& w, const TensorView* bias,
                  const TensorView& y);
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
  // Interleaved Q6_K copies for the int16 path (DD-082).
  struct Packed6 {
    std::vector<BlockQ6_Kx8> blocks;
    int64_t n = 0, k = 0;
  };
  std::unordered_map<const void*, Packed6> packed6_;
  const BlockQ6_Kx8* find_packed6(const void* w, int64_t n, int64_t k) const;
  // The DD-078 memory guard: false (and counted as skipped) when packing
  // `bytes` more would leave too little RAM, or packing is off.
  bool prepack_allowed(int64_t bytes);
  std::unordered_map<const void*, Packed> packed_;
  PrepackStats prepack_stats_;

  // One query row, one KV head, query heads [kvh*group + h0, + count) of the
  // group sharing it (DD-054).
  void attend_group(const AttentionParams& p, size_t r, int32_t kvh, int32_t group, int32_t h0, int32_t count,
                    int64_t t0, int64_t t1, float* acc, float* mx_out, double* sum_out);
  std::string name_;
};

}  // namespace dynacore
