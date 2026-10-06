#include "model_ir/model_config.h"

#include <algorithm>
#include <string>
#include "common/core.h"

namespace dynalm {

int64_t ModelConfig::kv_bytes_per_token(DType kv) const {
  const int64_t per_layer = static_cast<int64_t>(num_kv_heads) * (head_dim + head_dim_v);
  return per_layer * num_layers * dtype_block_bytes(kv) / dtype_block_elems(kv);
}

Status ModelConfig::validate() const {
  auto bad = [](const std::string& m) { return InvalidArgument("model config: " + m); };
  if (architecture.empty()) return bad("architecture not set");
  if (vocab_size <= 0) return bad("vocab_size must be > 0");
  if (hidden_size <= 0) return bad("hidden_size must be > 0");
  if (intermediate_size <= 0 && moe.num_experts == 0) return bad("intermediate_size must be > 0");
  if (num_layers <= 0) return bad("num_layers must be > 0");
  if (num_heads <= 0) return bad("num_heads must be > 0");
  if (num_kv_heads <= 0) return bad("num_kv_heads must be > 0");
  if (num_heads % num_kv_heads != 0) {
    return bad("num_heads (" + std::to_string(num_heads) + ") not divisible by num_kv_heads (" +
               std::to_string(num_kv_heads) + ")");
  }
  if (head_dim <= 0 || head_dim_v <= 0) return bad("head_dim must be > 0");
  if (rope.style != RopeStyle::kNone) {
    if (rope.dim <= 0 || rope.dim > head_dim || rope.dim % 2 != 0) {
      return bad("rope dim " + std::to_string(rope.dim) + " invalid for head_dim " +
                 std::to_string(head_dim));
    }
    if (rope.freq_base <= 0) return bad("rope freq_base must be > 0");
  }
  if (!(norm_eps > 0)) return bad("norm_eps must be > 0");
  if (context_length <= 0) return bad("context_length must be > 0");
  if (!sliding_layers.empty() && sliding_layers.size() != static_cast<size_t>(num_layers)) {
    return bad("sliding_layers size != num_layers");
  }
  if (moe.num_experts > 0 &&
      (moe.experts_per_token <= 0 || moe.experts_per_token > moe.num_experts)) {
    return bad("invalid MoE experts_per_token");
  }
  return Status::Ok();
}

MemoryEstimate estimate_memory(const ModelConfig& c, int64_t weight_bytes, int64_t context_tokens,
                               DType kv_dtype, int64_t max_batch_tokens) {
  MemoryEstimate m;
  m.weight_bytes = weight_bytes;
  m.kv_bytes = c.kv_bytes_per_token(kv_dtype) * context_tokens;
  // fp32 activations for one step: residual + attention/MLP intermediates for
  // the batch, plus logits for up to one row per sequence (bounded by batch).
  const int64_t widest = std::max<int64_t>(
      {c.hidden_size, c.intermediate_size * 2, static_cast<int64_t>(c.num_heads) * c.head_dim * 3});
  m.activation_bytes = max_batch_tokens * (c.hidden_size * 2 + widest) * 4 +
                       std::min<int64_t>(max_batch_tokens, 64) * c.vocab_size * 4;
  return m;
}

std::string_view norm_type_name(NormType n) {
  return n == NormType::kRmsNorm ? "rmsnorm" : "layernorm";
}

std::string_view activation_name(Activation a) {
  switch (a) {
    case Activation::kSilu: return "silu";
    case Activation::kGelu: return "gelu";
    case Activation::kGeluTanh: return "gelu_tanh";
  }
  return "?";
}

std::string_view rope_style_name(RopeStyle r) {
  switch (r) {
    case RopeStyle::kNone: return "none";
    case RopeStyle::kInterleaved: return "interleaved";
    case RopeStyle::kHalfSplit: return "half-split";
  }
  return "?";
}

}  // namespace dynalm
