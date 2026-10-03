// Llama family: Llama 1/2/3.x, Mistral, Mixtral (MoE: expert tensors present),
// DeepSeek-LLM (dense), SmolLM, TinyLlama, and IBM Granite / Granite-MoE
// (Llama layout plus embedding / attention / residual / logit multipliers,
// which the loader puts in ModelConfig).

#include <array>

#include "model/architectures.h"

namespace engine {
namespace {

class LlamaArchitecture final : public ModelArchitecture {
 public:
  std::string_view name() const override { return "Llama"; }

  std::span<const std::string_view> ids() const override {
    static constexpr std::array<std::string_view, 3> kIds = {"llama", "granite", "granitemoe"};
    return kIds;
  }

  Status configure(ModelConfig& c, const TensorRegistry& w) const override {
    c.norm = NormType::kRmsNorm;
    c.activation = Activation::kSilu;
    c.mlp = MlpType::kGated;
    // GGUF conversion permutes Llama Q/K rows so RoPE rotates adjacent pairs;
    // unpermuted (Hugging Face) weights rotate halves.
    c.rope.style = c.qk_rows_interleaved ? RopeStyle::kInterleaved : RopeStyle::kHalfSplit;
    c.attn_qkv_bias = w.has(TensorRole::kAttnQBias, 0);
    c.attn_output_bias = w.has(TensorRole::kAttnOutputBias, 0);
    c.tied_embeddings = !w.has(TensorRole::kOutput);
    if (c.rope.scaling == RopeScaling::kYarn) {
      return Unsupported("Llama with YaRN RoPE scaling is not supported yet");
    }
    // Mixtral and Granite-MoE renormalize the top-k routing weights.
    c.moe.normalize_topk = true;
    return Status::Ok();
  }
};

}  // namespace

const ModelArchitecture& llama_architecture() {
  static const LlamaArchitecture a;
  return a;
}

}  // namespace engine
