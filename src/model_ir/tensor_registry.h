#pragma once

// TensorRegistry: model weights addressed by semantic role, not file names.
//
// Format loaders translate their naming scheme ("blk.3.attn_q.weight" in GGUF,
// "model.layers.3.self_attn.q_proj.weight" in HF SafeTensors) into
// (TensorRole, layer). Adapters and the runtime look weights up by role.
// Lookups happen at model build time, never per token.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/status.h"
#include "tensor/tensor.h"

namespace engine {

enum class TensorRole : uint8_t {
  // Global
  kTokenEmbedding,
  kOutputNorm,
  kOutputNormBias,
  kOutput,  // lm_head (absent when embeddings are tied)
  kRopeFreqs,  // precomputed per-dim RoPE frequency factors (Llama3 / long-rope)

  // Per layer: attention
  kAttnNorm,
  kAttnNormBias,
  kAttnQ,
  kAttnK,
  kAttnV,
  kAttnQkv,  // fused Q|K|V (Phi-3)
  kAttnQBias,
  kAttnKBias,
  kAttnVBias,
  kAttnQkvBias,
  kAttnOutput,
  kAttnOutputBias,
  kAttnQNorm,
  kAttnKNorm,
  kPostAttnNorm,

  // Per layer: MLP
  kFfnNorm,
  kFfnNormBias,
  kFfnGate,
  kFfnUp,
  kFfnGateUp,  // fused gate|up (Phi-3)
  kFfnDown,
  kFfnUpBias,
  kFfnDownBias,
  kPostFfnNorm,

  // Per layer: MoE (Phase 25)
  kFfnRouter,
  kFfnGateExperts,
  kFfnUpExperts,
  kFfnDownExperts,

  kCount,
};

std::string_view tensor_role_name(TensorRole r);
bool tensor_role_is_per_layer(TensorRole r);

class TensorRegistry {
 public:
  explicit TensorRegistry(int num_layers = 0);

  int num_layers() const { return num_layers_; }

  // Fails on duplicates or out-of-range layers. layer = -1 for global roles.
  Status add(TensorRole role, int layer, Tensor tensor);

  const Tensor* find(TensorRole role, int layer = -1) const;
  Result<Tensor> get(TensorRole role, int layer = -1) const;  // kNotFound if absent
  bool has(TensorRole role, int layer = -1) const { return find(role, layer) != nullptr; }

  size_t size() const { return count_; }
  int64_t total_bytes() const;

 private:
  static constexpr size_t kRoles = static_cast<size_t>(TensorRole::kCount);
  using Slots = std::array<std::optional<Tensor>, kRoles>;

  std::optional<Tensor>* slot(TensorRole role, int layer);
  const std::optional<Tensor>* slot(TensorRole role, int layer) const;

  int num_layers_;
  Slots global_;
  std::vector<Slots> layers_;
  size_t count_ = 0;
};

}  // namespace engine
