#pragma once

// GGUF → Model IR translation. The only code that knows GGUF metadata key
// names and GGUF tensor naming.

#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "loader/gguf/gguf.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"
#include "common/core.h"

namespace dynalm::gguf {

// Raw hyperparameters from `<arch>.*` keys. Family semantics (RoPE style, MLP
// type, norms, biases) are left at defaults for the architecture adapter.
Result<ModelConfig> read_model_config(const GgufFile& file);

struct TensorMapping {
  TensorRegistry registry;
  std::vector<std::string> unmapped;  // names with no known role (adapter decides if fatal)
};

// Zero-copy: every registry tensor views the file mapping.
Result<TensorMapping> map_tensors(const GgufFile& file, int num_layers);

// llama.cpp `general.file_type` label, e.g. 15 -> "Q4_K_M". "unknown" if unrecognized.
std::string_view file_type_name(uint32_t file_type);

// Parses a GGUF tensor name into (role, layer). Exposed for tests.
bool parse_tensor_name(std::string_view name, TensorRole& role, int& layer);

}  // namespace dynalm::gguf
