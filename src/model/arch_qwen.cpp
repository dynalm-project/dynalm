// Qwen family (dense): Qwen2 / Qwen2.5 (QKV biases) and Qwen3 (QK-norm).
// Also runs DeepSeek-R1-Distill-Qwen and other qwen2-architecture GGUFs.

#include <array>

#include "model/architectures.h"

namespace engine {
namespace {

class QwenArchitecture final : public ModelArchitecture {
 public:
  std::string_view name() const override { return "Qwen"; }

  std::span<const std::string_view> ids() const override {
    static constexpr std::array<std::string_view, 2> kIds = {"qwen2", "qwen3"};
    return kIds;
  }

  Status configure(ModelConfig& c, const TensorRegistry& w) const override {
    c.norm = NormType::kRmsNorm;
    c.activation = Activation::kSilu;
    c.mlp = MlpType::kGated;
    c.rope.style = RopeStyle::kHalfSplit;
    c.attn_qkv_bias = w.has(TensorRole::kAttnQBias, 0);  // Qwen2 yes, Qwen3 no
    c.attn_qk_norm = c.architecture == "qwen3";
    c.tied_embeddings = !w.has(TensorRole::kOutput);
    if (c.rope.scaling == RopeScaling::kYarn) {
      return Unsupported("Qwen with YaRN RoPE scaling is not supported yet");
    }
    return Status::Ok();
  }
};

}  // namespace

const ModelArchitecture& qwen_architecture() {
  static const QwenArchitecture a;
  return a;
}

}  // namespace engine
