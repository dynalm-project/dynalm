// Gemma family: Gemma 1, Gemma 2, Gemma 3 (text).
//
// GGUF conversion already folds Gemma's (1 + weight) RMSNorm scaling into the
// stored norm weights, so the standard RMSNorm applies.

#include <array>
#include <cmath>

#include "model/architectures.h"
#include "common/core.h"

namespace dynalm {
namespace {

// Layer l is sliding-window unless it is the last of each group of `pattern`
// layers (Gemma 2: pattern 2 → even layers local; Gemma 3: 5 local, 1 global).
std::vector<bool> swa_pattern(int32_t num_layers, int32_t pattern) {
  std::vector<bool> v(static_cast<size_t>(num_layers));
  for (int32_t l = 0; l < num_layers; ++l) v[static_cast<size_t>(l)] = (l % pattern) < pattern - 1;
  return v;
}

class GemmaArchitecture final : public ModelArchitecture {
 public:
  std::string_view name() const override { return "Gemma"; }

  std::span<const std::string_view> ids() const override {
    static constexpr std::array<std::string_view, 3> kIds = {"gemma", "gemma2", "gemma3"};
    return kIds;
  }

  Status configure(ModelConfig& c, const TensorRegistry& w) const override {
    const bool g2 = c.architecture == "gemma2";
    const bool g3 = c.architecture == "gemma3";
    c.norm = NormType::kRmsNorm;
    c.activation = Activation::kGeluTanh;
    c.mlp = MlpType::kGated;
    c.rope.style = RopeStyle::kHalfSplit;
    c.embedding_scale = std::sqrt(static_cast<float>(c.hidden_size));
    c.tied_embeddings = !w.has(TensorRole::kOutput);
    c.post_attn_norm = g2 || g3;
    c.post_ffn_norm = g2 || g3;
    c.attn_qk_norm = g3;

    // Query scaling: 1/sqrt(head_dim), except the 27B models which scale by
    // 1/sqrt(hidden / heads) (query_pre_attn_scalar in the HF config).
    const bool is_27b = (g2 && c.num_layers == 46) || (g3 && c.num_layers == 62);
    c.attn_scale = is_27b ? 1.0f / std::sqrt(static_cast<float>(c.hidden_size / c.num_heads))
                          : 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    if (g2) {
      if (c.sliding_window > 0) c.sliding_layers = swa_pattern(c.num_layers, 2);
      // Soft-caps come from metadata (attn 50, final 30 on released models).
    }
    if (g3) {
      if (c.sliding_window > 0) c.sliding_layers = swa_pattern(c.num_layers, 6);
      // Local layers: RoPE base 10k without scaling; global layers keep the
      // metadata base (1M) and any linear scaling.
      c.has_rope_local = true;
      c.rope_local = c.rope;
      c.rope_local.freq_base = 10000.0f;
      c.rope_local.scaling = RopeScaling::kNone;
      c.rope_local.scaling_factor = 1.0f;
      c.attn_logit_softcap = 0.0f;
      c.final_logit_softcap = 0.0f;
    }
    if (c.rope.scaling == RopeScaling::kYarn) return Unsupported("Gemma with YaRN RoPE scaling");
    return Status::Ok();
  }
};

}  // namespace

const ModelArchitecture& gemma_architecture() {
  static const GemmaArchitecture a;
  return a;
}

}  // namespace dynalm
