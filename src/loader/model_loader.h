#pragma once

// ModelLoader: file → validated, runnable model description.
//
//   detect format → read metadata → config → architecture adapter → map
//   tensors → configure → validate → tokenizer → chat template
//
// Weights stay memory-mapped; nothing is copied or dequantized here.

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chat_template/chat_template.h"
#include "common/status.h"
#include "model/architecture.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"
#include "tokenizer/tokenizer.h"

namespace engine {

struct LoadedModel {
  std::string path;
  std::string format;        // "gguf"
  std::string quantization;  // e.g. "Q4_K_M", "F16"
  ModelConfig config;
  TensorRegistry weights;
  const ModelArchitecture* architecture = nullptr;
  std::unique_ptr<Tokenizer> tokenizer;
  std::optional<ChatTemplate> chat_template;
  std::vector<std::string> unmapped_tensors;  // present in the file, unused by the engine
  int64_t weight_bytes = 0;
};

Result<std::unique_ptr<LoadedModel>> load_model(const std::string& path);

}  // namespace engine
