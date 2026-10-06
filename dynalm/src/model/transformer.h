#pragma once

// Generic decoder-only transformer.
//
// One implementation for every supported family: all variation (RoPE style,
// norms, biases, fused QKV / gate-up, QK-norm, sandwich norms, soft-capping,
// sliding windows, tied embeddings) comes from ModelConfig. Weights are
// resolved into per-layer views once at construction, and all activation
// scratch is preallocated for `max_batch_tokens`, so forward() does no weight
// lookups and no allocation.

#include <array>
#include <memory>
#include <optional>
#include <string_view>
#include <span>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/base/status.h"
#include "execution/batch_planner.h"
#include "kv_cache/kv_cache.h"
#include "model_ir/model_config.h"
#include "model/seq_batch.h"
#include "model_ir/tensor_registry.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

// Optional per-op timing of forward passes (off by default; profiling adds
// one timestamp per op).
enum class ForwardOp : uint8_t {
  kEmbed, kNorm, kQkv, kRope, kKvStore, kAttention, kAttnOut, kMlpUp, kAct, kMlpDown, kMoeRoute, kMoeExperts,
  kMoeScatter, kLmHead, kCount
};
std::string_view forward_op_name(ForwardOp op);

struct ForwardProfile {
  std::array<int64_t, static_cast<size_t>(ForwardOp::kCount)> ns{};
  uint64_t calls = 0;
  uint64_t rows = 0;
  int64_t total_ns() const {
    int64_t t = 0;
    for (int64_t v : ns) t += v;
    return t;
  }
};

class Transformer {
 public:
  static Result<std::unique_ptr<Transformer>> create(const ModelConfig& config, const TensorRegistry& weights,
                                                     Device& backend, int32_t max_batch_tokens);

  const ModelConfig& config() const { return config_; }
  int32_t max_batch_tokens() const { return max_batch_; }

  // Runs tokens[i] at positions[i] (1 <= size <= max_batch_tokens) for one
  // sequence whose KV lives at `block_table`, writing K/V for every token and
  // the logits of the LAST token into `logits` (size vocab_size).
  Status forward(std::span<const TokenId> tokens, std::span<const int32_t> positions, KvBlockPool& cache,
                 std::span<const int32_t> block_table, std::span<float> logits);

  // Batched forward over several sequences (total tokens <= max_batch_tokens).
  // Every weight is read once for the whole batch. Logits for the last
  // `logits_last` tokens of each sequence with want_logits are written to
  // `logits` in order, one row of vocab_size floats each.
  // `plan` comes from the scheduler's planner call; null = plan here with
  // planner() (direct callers: generator, speculative decoding, tests).
  Status forward_batch(std::span<const SeqBatch> seqs, KvBlockPool& cache, std::span<float> logits,
                       const ExecutionPlan* plan = nullptr);

  // Plans batches for this model on this backend (DD-051).
  const BatchPlanner& planner() const { return *planner_; }
  // Replaces the planner's base kernel settings (before serving starts).
  void set_kernel_base(const KernelPlan& base);

  void set_profiling(bool on) { profiling_ = on; }
  const ForwardProfile& profile() const { return profile_; }
  void reset_profile() { profile_ = ForwardProfile{}; }

 private:
  struct Layer {
    TensorView attn_norm, attn_norm_b;
    TensorView wq, wk, wv, wqkv, wo;
    TensorView bq, bk, bv, bqkv, bo;
    TensorView q_norm, k_norm, post_attn_norm;
    TensorView ffn_norm, ffn_norm_b;
    TensorView w_gate, w_up, w_gate_up, w_down;
    TensorView b_up, b_down;
    TensorView post_ffn_norm;
    bool fused_qkv = false, fused_gate_up = false;
    // MoE: router [experts, hidden]; per-expert 2-D views of the 3-D expert
    // tensors (gate, up, down); optional shared expert with sigmoid gate.
    TensorView router;
    std::vector<std::array<TensorView, 3>> experts;
    TensorView sh_router, sh_gate, sh_up, sh_down;
  };
  // Routed MoE MLP for rows of `xn`, accumulated into `out` (zeroed first).
  void moe_mlp(const Layer& L, const TensorView& xn, const TensorView& out, int64_t m);

  Transformer(const ModelConfig& c, Device& b) : config_(c), backend_(b) {}
  Status init(const TensorRegistry& weights, int32_t max_batch);
  // fp32 copy of a small vector weight (norms, biases), owned by the model.
  Result<TensorView> f32_vector(const Tensor& t);
  // Weights and scratch in backend memory (DD-045).
  Result<TensorView> device_weight(const Tensor& host);
  Result<Tensor> device_scratch(int64_t rows, int64_t cols);
  void norm(const TensorView& x, const TensorView& w, const TensorView& b, const TensorView& y);

  ModelConfig config_;
  Device& backend_;
  int32_t max_batch_ = 0;
  std::unique_ptr<BatchPlanner> planner_;
  KernelPlan kernels_;  // applied for the current forward pass

  TensorView tok_embd_, output_norm_, output_norm_b_, lm_head_;
  std::vector<float> rope_freq_factors_;
  std::vector<Layer> layers_;
  std::vector<Tensor> owned_;  // fp32 conversions + scratch

  // Scratch [max_batch, ...]
  Tensor x_, xn_, qkv_, attn_, o_, ff_a_, ff_b_;
  // MoE scratch: router logits, gathered rows, expert intermediates/outputs.
  Tensor router_, moe_x_, moe_a_, moe_b_, moe_y_, sh_gate_;
  std::vector<std::vector<std::pair<int32_t, float>>> expert_rows_;  // per expert: (row, weight)
  std::vector<std::pair<float, int32_t>> route_scratch_;
  std::vector<Device::MatmulJob> up_jobs_, down_jobs_;
  std::vector<int32_t> gather_idx_;  // host-side row indices for gather/scatter ops
  std::vector<float> scatter_w_, router_host_;
  Tensor logits_dev_;                // device logits rows (non-host-accessible backends)
  // Per-call batch metadata (reused, capacity max_batch).
  std::vector<TokenId> batch_tokens_;
  std::vector<int32_t> batch_pos_, batch_seq_, logit_rows_;
  std::vector<KvLayerView> kv_views_;

  void mark(ForwardOp op);
  bool profiling_ = false;
  int64_t mark_ns_ = 0;
  ForwardProfile profile_;
};

}  // namespace dynalm
