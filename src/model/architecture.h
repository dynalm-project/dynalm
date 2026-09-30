#pragma once

// Model architecture adapters.
//
// An adapter turns raw hyperparameters into family semantics and checks that
// the weights match. It contains no execution code: every supported family
// runs on the same generic Transformer, parameterized by ModelConfig. Adding
// a family = one adapter (configure + optional extra validation) and a
// registry entry.

#include <span>
#include <string_view>
#include <vector>

#include "common/status.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"

namespace engine {

class ModelArchitecture {
 public:
  virtual ~ModelArchitecture() = default;

  virtual std::string_view name() const = 0;
  // Architecture ids (GGUF `general.architecture`) this adapter handles.
  virtual std::span<const std::string_view> ids() const = 0;

  // Sets family semantics on a config holding raw hyperparameters. May read
  // the registry for optional features (e.g. biases present).
  virtual Status configure(ModelConfig& config, const TensorRegistry& weights) const = 0;

  // Checks required tensors and their shapes. The default validates the
  // standard pre-norm decoder described by the config.
  virtual Status validate(const ModelConfig& config, const TensorRegistry& weights) const;
};

// Adapter for an architecture id, or nullptr if unsupported.
const ModelArchitecture* find_architecture(std::string_view arch_id);
std::vector<std::string_view> supported_architecture_ids();

// Shared validation of a standard decoder (exposed for adapters and tests).
Status validate_standard_decoder(const ModelConfig& config, const TensorRegistry& weights);

}  // namespace engine
