#pragma once

// Hugging Face model directories: config.json + *.safetensors (+ tokenizer
// files). Translates HF conventions to the same ModelConfig / TensorRegistry
// the GGUF loader produces, so adapters and the runtime never see the
// difference (spec §5):
//
//   GGUF loader ────────┐
//   SafeTensors loader ─┴─► TensorRegistry + ModelConfig ─► adapters ─► runtime
//
// Conventions the GGUF converter applies offline, applied here at load:
// Gemma RMSNorm weights are stored as w and used as (1 + w) — folded into an
// F32 copy; Llama 3 RoPE scaling becomes per-dimension frequency factors.
// Llama Q/K rows stay unpermuted (config.qk_rows_interleaved = false).

#include <string>
#include <string_view>
#include <vector>

#include "api/json.h"
#include "common/status.h"
#include "model_ir/model_config.h"
#include "model_ir/tensor_registry.h"

namespace engine::hf {

// Files of a model given as a directory or as one of its .safetensors files.
struct ModelFiles {
  std::string dir;
  std::vector<std::string> weights;  // one or more shards
};
Result<ModelFiles> locate(const std::string& path);
// True if `path` looks like a Hugging Face model (directory with config.json,
// or a .safetensors file).
bool is_hf_model(const std::string& path);

Result<json::Value> read_json_file(const std::string& path, size_t max_bytes = 64 << 20);

// config.json -> ModelConfig with raw hyperparameters; `architecture` is the
// adapter id ("llama", "qwen2", "gemma3", ...).
Result<ModelConfig> read_config(const json::Value& config);

// HF parameter name -> (role, layer); false if the engine does not use it.
bool parse_tensor_name(std::string_view name, std::string_view arch, TensorRole& role, int& layer);

// Llama 3 "rope_scaling" -> rope_freqs factors (head_dim / 2 values), the same
// values llama.cpp's converter stores. Empty if the config has no such scaling.
Result<std::vector<float>> llama3_rope_factors(const json::Value& config, int32_t rope_dim, float base);

// Applies HF-specific weight conventions after mapping (Gemma norm fold,
// Llama 3 rope factors).
Status apply_conventions(const json::Value& config, const ModelConfig& c, TensorRegistry& weights);

}  // namespace engine::hf
