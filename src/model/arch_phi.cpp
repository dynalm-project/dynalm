// Phi-3 / Phi-3.5 / Phi-4-mini (dense). Fused QKV projection, and a fused
// gate|up projection that GGUF stores under the ffn_up name.

#include <array>

#include "model/architectures.h"

namespace engine {
namespace {

class PhiArchitecture final : public ModelArchitecture {
 public:
  std::string_view name() const override { return "Phi"; }

  std::span<const std::string_view> ids() const override {
    static constexpr std::array<std::string_view, 1> kIds = {"phi3"};
    return kIds;
  }

  Status configure(ModelConfig& c, const TensorRegistry& w) const override {
    c.norm = NormType::kRmsNorm;
    c.activation = Activation::kSilu;
    c.mlp = MlpType::kGated;
    c.rope.style = RopeStyle::kHalfSplit;
    c.attn_qkv_bias = w.has(TensorRole::kAttnQkvBias, 0);
    c.tied_embeddings = !w.has(TensorRole::kOutput);
    return Status::Ok();
  }

  Status prepare_weights(const ModelConfig& c, TensorRegistry& w) const override {
    for (int l = 0; l < c.num_layers; ++l) {
      const Tensor* up = w.find(TensorRole::kFfnUp, l);
      if (up && !w.has(TensorRole::kFfnGate, l) && up->shape()[0] == 2 * c.intermediate_size) {
        Tensor fused = *w.take(TensorRole::kFfnUp, l);
        ENGINE_RETURN_IF_ERROR(w.add(TensorRole::kFfnGateUp, l, std::move(fused)));
      }
    }
    return Status::Ok();
  }
};

}  // namespace

const ModelArchitecture& phi_architecture() {
  static const PhiArchitecture a;
  return a;
}

}  // namespace engine
