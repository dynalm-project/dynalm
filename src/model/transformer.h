#pragma once

// Generic decoder-only transformer.
//
// One implementation for every supported family: all variation (RoPE style,
// norms, biases, fused QKV / gate-up, QK-norm, sandwich norms, soft-capping,
// sliding windows, tied embeddings) comes from ModelConfig. Weights are
// resolved into per-layer views once at construction, and all activation
// scratch is preallocated for `max_batch_tokens`, so forward() does no weight
// lookups and no allocation.

#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "backends/backend.h"
#include "common/status.h"
#include "kv_cache/kv_cache.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"
#include "tokenizer/tokenizer.h"

namespace engine {

// One sequence's slice of a batched forward pass: `tokens` occupy positions
// [start_pos, start_pos + tokens.size()) of the sequence whose KV lives at
// `block_table`.
struct SeqBatch {
  std::span<const TokenId> tokens;
  int32_t start_pos = 0;
  std::span<const int32_t> block_table;
  bool want_logits = true;  // compute logits for this sequence's last token
};

class Transformer {
 public:
  static Result<std::unique_ptr<Transformer>> create(const ModelConfig& config, const TensorRegistry& weights,
                                                     Backend& backend, int32_t max_batch_tokens);

  const ModelConfig& config() const { return config_; }
  int32_t max_batch_tokens() const { return max_batch_; }

  // Runs tokens[i] at positions[i] (1 <= size <= max_batch_tokens) for one
  // sequence whose KV lives at `block_table`, writing K/V for every token and
  // the logits of the LAST token into `logits` (size vocab_size).
  Status forward(std::span<const TokenId> tokens, std::span<const int32_t> positions, KvBlockPool& cache,
                 std::span<const int32_t> block_table, std::span<float> logits);

  // Batched forward over several sequences (total tokens <= max_batch_tokens).
  // Every weight is read once for the whole batch. Logits for the last token
  // of each sequence with want_logits are written to `logits` in order, one
  // row of vocab_size floats each.
  Status forward_batch(std::span<const SeqBatch> seqs, KvBlockPool& cache, std::span<float> logits);

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
  };

  Transformer(const ModelConfig& c, Backend& b) : config_(c), backend_(b) {}
  Status init(const TensorRegistry& weights, int32_t max_batch);
  // fp32 copy of a small vector weight (norms, biases), owned by the model.
  Result<TensorView> f32_vector(const Tensor& t);
  void norm(const TensorView& x, const TensorView& w, const TensorView& b, const TensorView& y);

  ModelConfig config_;
  Backend& backend_;
  int32_t max_batch_ = 0;

  TensorView tok_embd_, output_norm_, output_norm_b_, lm_head_;
  std::vector<float> rope_freq_factors_;
  std::vector<Layer> layers_;
  std::vector<Tensor> owned_;  // fp32 conversions + scratch

  // Scratch [max_batch, ...]
  Tensor x_, xn_, qkv_, attn_, o_, ff_a_, ff_b_;
  // Per-call batch metadata (reused, capacity max_batch).
  std::vector<TokenId> batch_tokens_;
  std::vector<int32_t> batch_pos_, batch_seq_, logit_rows_;
  std::vector<KvLayerView> kv_views_;
};

}  // namespace engine
