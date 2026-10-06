#include "model/architecture.h"

#include <array>
#include <string>

#include "model/architectures.h"
#include "common/core.h"

namespace dynalm {
namespace {

Status expect_shape(const TensorRegistry& w, TensorRole role, int layer, const TensorShape& want,
                    bool required = true) {
  const Tensor* t = w.find(role, layer);
  if (!t) {
    if (!required) return Status::Ok();
    std::string where = layer >= 0 ? " in layer " + std::to_string(layer) : "";
    return NotFound("missing weight " + std::string(tensor_role_name(role)) + where);
  }
  if (!(t->shape() == want)) {
    std::string where = layer >= 0 ? " (layer " + std::to_string(layer) + ")" : "";
    return InvalidArgument("weight " + std::string(tensor_role_name(role)) + where + " has shape " +
                           t->shape().to_string() + ", expected " + want.to_string());
  }
  return Status::Ok();
}

}  // namespace

Status validate_standard_decoder(const ModelConfig& c, const TensorRegistry& w) {
  ENGINE_RETURN_IF_ERROR(c.validate());
  const int64_t d = c.hidden_size;
  const int64_t q_dim = static_cast<int64_t>(c.num_heads) * c.head_dim;
  const int64_t k_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim;
  const int64_t v_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim_v;
  const int64_t o_dim = static_cast<int64_t>(c.num_heads) * c.head_dim_v;
  const int64_t ff = c.intermediate_size;
  const bool layernorm = c.norm == NormType::kLayerNorm;

  ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kTokenEmbedding, -1, {c.vocab_size, d}));
  ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kOutputNorm, -1, {d}));
  if (layernorm) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kOutputNormBias, -1, {d}, false));
  if (!c.tied_embeddings) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kOutput, -1, {c.vocab_size, d}));

  for (int l = 0; l < c.num_layers; ++l) {
    ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnNorm, l, {d}));
    if (w.has(TensorRole::kAttnQkv, l)) {
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnQkv, l, {q_dim + k_dim + v_dim, d}));
      ENGINE_RETURN_IF_ERROR(
          expect_shape(w, TensorRole::kAttnQkvBias, l, {q_dim + k_dim + v_dim}, c.attn_qkv_bias));
    } else {
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnQ, l, {q_dim, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnK, l, {k_dim, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnV, l, {v_dim, d}));
      if (c.attn_qkv_bias) {
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnQBias, l, {q_dim}));
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnKBias, l, {k_dim}));
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnVBias, l, {v_dim}));
      }
    }
    ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnOutput, l, {d, o_dim}));
    if (c.attn_output_bias) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnOutputBias, l, {d}));
    if (c.attn_qk_norm) {
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnQNorm, l, {c.head_dim}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kAttnKNorm, l, {c.head_dim}));
    }
    if (c.post_attn_norm) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kPostAttnNorm, l, {d}));

    ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnNorm, l, {d}));
    if (c.moe.num_experts > 0) {
      // Routed experts (gated MLP each) + optional shared expert.
      const int64_t ne = c.moe.num_experts, fe = c.moe.expert_intermediate_size;
      if (fe <= 0) return InvalidArgument("MoE model without expert_feed_forward_length");
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnRouter, l, {ne, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnGateExperts, l, {ne, fe, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnUpExperts, l, {ne, fe, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnDownExperts, l, {ne, d, fe}));
      if (w.has(TensorRole::kFfnUpShared, l)) {
        const int64_t fs = c.moe.shared_intermediate_size;
        if (fs <= 0) return InvalidArgument("shared expert without expert_shared_feed_forward_length");
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnGateShared, l, {fs, d}));
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnUpShared, l, {fs, d}));
        ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnDownShared, l, {d, fs}));
        if (w.has(TensorRole::kFfnSharedRouter, l)) {
          ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnSharedRouter, l, {d}));
        }
      }
      if (c.post_ffn_norm) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kPostFfnNorm, l, {d}));
      continue;
    }
    if (w.has(TensorRole::kFfnGateUp, l)) {
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnGateUp, l, {2 * ff, d}));
    } else {
      if (c.mlp == MlpType::kGated) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnGate, l, {ff, d}));
      ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnUp, l, {ff, d}));
    }
    ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kFfnDown, l, {d, ff}));
    if (c.post_ffn_norm) ENGINE_RETURN_IF_ERROR(expect_shape(w, TensorRole::kPostFfnNorm, l, {d}));
  }
  if (const Tensor* f = w.find(TensorRole::kRopeFreqs)) {
    if (!(f->shape() == TensorShape{c.rope.dim / 2})) {
      return InvalidArgument("rope_freqs has shape " + f->shape().to_string());
    }
  }
  return Status::Ok();
}

Status ModelArchitecture::validate(const ModelConfig& config, const TensorRegistry& weights) const {
  return validate_standard_decoder(config, weights);
}

std::span<const ModelArchitecture* const> registered_architectures() {
  static const std::array<const ModelArchitecture*, 4> kAll = {&llama_architecture(), &qwen_architecture(),
                                                                &gemma_architecture(), &phi_architecture()};
  return kAll;
}

const ModelArchitecture* find_architecture(std::string_view arch_id) {
  for (const ModelArchitecture* a : registered_architectures()) {
    for (std::string_view id : a->ids()) {
      if (id == arch_id) return a;
    }
  }
  return nullptr;
}

std::vector<std::string_view> supported_architecture_ids() {
  std::vector<std::string_view> ids;
  for (const ModelArchitecture* a : registered_architectures()) {
    for (std::string_view id : a->ids()) ids.push_back(id);
  }
  return ids;
}

}  // namespace dynalm
