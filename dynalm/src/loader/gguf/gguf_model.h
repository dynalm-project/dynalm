#pragma once

// GGUF → Model IR translation. The only code that knows GGUF metadata key
// names and GGUF tensor naming.

#include <memory>
#include <span>
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
// The same over the parts of a split model (each tensor views its part).
Result<TensorMapping> map_tensors(std::span<const GgufFile* const> parts, int num_layers);

// A model split into parts by llama.cpp's gguf-split (NAME-00001-of-00003.gguf,
// ...: keys split.no, split.count, split.tensors.count; the first part holds
// the metadata): opens every part, given the first one's path. A file that
// is not split comes back alone. Splitting never changes a weight: the
// tensors are the same bytes as in one merged file.
Result<std::vector<std::unique_ptr<GgufFile>>> open_parts(const std::string& path);

// llama.cpp `general.file_type` label, e.g. 15 -> "Q4_K_M". "unknown" if unrecognized.
std::string_view file_type_name(uint32_t file_type);

// Parses a GGUF tensor name into (role, layer). Exposed for tests.
bool parse_tensor_name(std::string_view name, TensorRole& role, int& layer);

}  // namespace dynalm::gguf
