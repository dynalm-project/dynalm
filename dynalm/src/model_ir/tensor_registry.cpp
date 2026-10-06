#include "model_ir/tensor_registry.h"

#include <utility>
#include "common/core.h"

namespace dynalm {

std::string_view tensor_role_name(TensorRole r) {
  static constexpr std::string_view kNames[] = {
      "token_embedding", "output_norm", "output_norm_bias", "output", "rope_freqs",
      "attn_norm", "attn_norm_bias", "attn_q", "attn_k", "attn_v", "attn_qkv",
      "attn_q_bias", "attn_k_bias", "attn_v_bias", "attn_qkv_bias", "attn_output",
      "attn_output_bias", "attn_q_norm", "attn_k_norm", "post_attn_norm",
      "ffn_norm", "ffn_norm_bias", "ffn_gate", "ffn_up", "ffn_gate_up", "ffn_down",
      "ffn_up_bias", "ffn_down_bias", "post_ffn_norm",
      "ffn_router", "ffn_gate_experts", "ffn_up_experts", "ffn_down_experts",
      "ffn_shared_router", "ffn_gate_shared", "ffn_up_shared", "ffn_down_shared",
  };
  static_assert(std::size(kNames) == static_cast<size_t>(TensorRole::kCount));
  return kNames[static_cast<size_t>(r)];
}

bool tensor_role_is_per_layer(TensorRole r) {
  return static_cast<int>(r) >= static_cast<int>(TensorRole::kAttnNorm);
}

TensorRegistry::TensorRegistry(int num_layers)
    : num_layers_(num_layers), layers_(static_cast<size_t>(num_layers)) {}

std::optional<Tensor>* TensorRegistry::slot(TensorRole role, int layer) {
  return const_cast<std::optional<Tensor>*>(std::as_const(*this).slot(role, layer));
}

const std::optional<Tensor>* TensorRegistry::slot(TensorRole role, int layer) const {
  const auto idx = static_cast<size_t>(role);
  if (idx >= kRoles) return nullptr;
  if (!tensor_role_is_per_layer(role)) return layer == -1 ? &global_[idx] : nullptr;
  if (layer < 0 || layer >= num_layers_) return nullptr;
  return &layers_[static_cast<size_t>(layer)][idx];
}

Status TensorRegistry::add(TensorRole role, int layer, Tensor tensor) {
  std::optional<Tensor>* s = slot(role, layer);
  if (!s) {
    return InvalidArgument("tensor role " + std::string(tensor_role_name(role)) + " with layer " +
                           std::to_string(layer) + " is out of range");
  }
  if (s->has_value()) {
    return Status(StatusCode::kAlreadyExists, "duplicate tensor for role " +
                                                  std::string(tensor_role_name(role)) + " layer " +
                                                  std::to_string(layer));
  }
  *s = std::move(tensor);
  ++count_;
  return Status::Ok();
}

const Tensor* TensorRegistry::find(TensorRole role, int layer) const {
  const std::optional<Tensor>* s = slot(role, layer);
  return (s && s->has_value()) ? &**s : nullptr;
}

std::optional<Tensor> TensorRegistry::take(TensorRole role, int layer) {
  std::optional<Tensor>* s = slot(role, layer);
  if (!s || !s->has_value()) return std::nullopt;
  std::optional<Tensor> out = std::move(*s);
  s->reset();
  --count_;
  return out;
}

Result<Tensor> TensorRegistry::get(TensorRole role, int layer) const {
  if (const Tensor* t = find(role, layer)) return *t;
  std::string where = layer >= 0 ? " (layer " + std::to_string(layer) + ")" : "";
  return NotFound("missing tensor: " + std::string(tensor_role_name(role)) + where);
}

void TensorRegistry::for_each(FunctionRef<void(const Tensor&)> fn) const {
  for (const auto& s : global_) {
    if (s) fn(*s);
  }
  for (const auto& l : layers_) {
    for (const auto& s : l) {
      if (s) fn(*s);
    }
  }
}

int64_t TensorRegistry::total_bytes() const {
  int64_t total = 0;
  auto add_slots = [&](const Slots& slots) {
    for (const auto& s : slots) {
      if (s) total += s->view().span_bytes();
    }
  };
  add_slots(global_);
  for (const auto& l : layers_) add_slots(l);
  return total;
}

}  // namespace dynalm
