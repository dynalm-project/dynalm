#pragma once

// Model IR: a format- and family-independent description of a decoder-only
// transformer. The runtime, scheduler, KV cache and memory planner consume
// only this, never GGUF keys or HF config.json fields.
//
// Filled in two steps:
//   1. The format loader copies raw hyperparameters (sizes, eps, rope base).
//   2. The ModelArchitecture adapter sets family semantics (RoPE style, norm,
//      activation, biases, soft-capping) and validates the result.

#include <cstdint>
#include <string>
#include <vector>

#include "common/status.h"
#include "dtype/dtype.h"

namespace engine {

enum class NormType : uint8_t { kRmsNorm, kLayerNorm };
enum class Activation : uint8_t { kSilu, kGelu, kGeluTanh };

enum class MlpType : uint8_t {
  kGated,  // down(act(gate(x)) * up(x))   — SwiGLU / GeGLU
  kPlain,  // down(act(up(x)))
};

enum class RopeStyle : uint8_t {
  kNone,
  kInterleaved,  // rotate adjacent pairs (x0,x1),(x2,x3)...  (GGUF "llama" after Q/K permute)
  kHalfSplit,    // rotate (x_i, x_{i+d/2})                   (NeoX / HF default)
};

enum class RopeScaling : uint8_t { kNone, kLinear, kYarn, kLlama3 };

struct RopeConfig {
  RopeStyle style = RopeStyle::kHalfSplit;
  int32_t dim = 0;              // rotated dims per head (<= head_dim; partial RoPE when smaller)
  float freq_base = 10000.0f;
  RopeScaling scaling = RopeScaling::kNone;
  float scaling_factor = 1.0f;
  int64_t original_context = 0;  // pre-scaling training context (YaRN / Llama3)
  // YaRN / Llama3 parameters (used only when the matching scaling is set).
  float yarn_beta_fast = 32.0f;
  float yarn_beta_slow = 1.0f;
  float llama3_low_freq_factor = 1.0f;
  float llama3_high_freq_factor = 4.0f;
};

struct MoeConfig {
  int32_t num_experts = 0;  // 0 = dense model
  int32_t experts_per_token = 0;
  int32_t num_shared_experts = 0;
  int64_t expert_intermediate_size = 0;
};

// Precision is tracked per concept, never conflated. Weight dtypes are per
// tensor (see TensorRegistry); these are the runtime-chosen compute types.
struct PrecisionConfig {
  DType activation = DType::kF32;
  DType accumulator = DType::kF32;
  DType kv_cache = DType::kF16;
};

struct ModelConfig {
  std::string architecture;  // canonical family id: "llama", "qwen2", "gemma2", ...
  std::string name;

  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int32_t num_layers = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;         // per-head Q/K dim
  int32_t head_dim_v = 0;       // per-head V dim (usually == head_dim)
  int64_t context_length = 0;   // trained maximum

  NormType norm = NormType::kRmsNorm;
  float norm_eps = 1e-5f;
  Activation activation = Activation::kSilu;
  MlpType mlp = MlpType::kGated;
  RopeConfig rope;
  // Sliding-window layers use `rope_local` when set (Gemma 3: base 10k local,
  // 1M global).
  bool has_rope_local = false;
  RopeConfig rope_local;

  bool attn_qkv_bias = false;     // Qwen2
  bool attn_output_bias = false;
  bool attn_qk_norm = false;      // Qwen3 / Gemma3: RMSNorm on per-head Q and K
  bool post_attn_norm = false;    // Gemma2/3 sandwich norms
  bool post_ffn_norm = false;
  bool tied_embeddings = false;   // lm_head shares token embedding

  float embedding_scale = 1.0f;       // Gemma: sqrt(hidden_size)
  float attn_logit_softcap = 0.0f;    // 0 = off (Gemma2)
  float final_logit_softcap = 0.0f;   // 0 = off (Gemma2)
  float attn_scale = 0.0f;            // 0 = 1/sqrt(head_dim)

  // Sliding-window attention. `sliding_window` = 0 means full attention on all
  // layers; otherwise `sliding_layers[i]` says whether layer i uses the window.
  int32_t sliding_window = 0;
  std::vector<bool> sliding_layers;

  MoeConfig moe;

  int32_t gqa_group() const { return num_kv_heads > 0 ? num_heads / num_kv_heads : 0; }
  const RopeConfig& layer_rope(int layer) const {
    return has_rope_local && layer_uses_sliding_window(layer) ? rope_local : rope;
  }
  bool layer_uses_sliding_window(int layer) const {
    return sliding_window > 0 &&
           (sliding_layers.empty() || sliding_layers[static_cast<size_t>(layer)]);
  }

  // Bytes of K+V per token across all layers at the given KV dtype.
  int64_t kv_bytes_per_token(DType kv) const;

  // Structural consistency (positive sizes, head divisibility, rope dim fits).
  Status validate() const;
};

// Rough resident-memory estimate for serving `context_tokens` total KV tokens.
struct MemoryEstimate {
  int64_t weight_bytes = 0;
  int64_t kv_bytes = 0;
  int64_t activation_bytes = 0;  // per-step scratch (logits + largest intermediates)
  int64_t total() const { return weight_bytes + kv_bytes + activation_bytes; }
};
MemoryEstimate estimate_memory(const ModelConfig& c, int64_t weight_bytes, int64_t context_tokens,
                               DType kv_dtype, int64_t max_batch_tokens = 512);

std::string_view norm_type_name(NormType n);
std::string_view activation_name(Activation a);
std::string_view rope_style_name(RopeStyle r);

}  // namespace engine
