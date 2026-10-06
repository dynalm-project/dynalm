#pragma once

// Model references and the local model store.
//
// A model reference is what a user types after `dynalm run`/`serve`/...:
//   ./models/qwen3-4b.gguf     a file or Hugging Face model directory
//   qwen3:4b                   a registry name (family:size)
//   gemma:270m, llama:3b       aliases of registry names
//   qwen3                      a family alone: its default size
// Registry names map to one verified GGUF download each. The registry is a
// DynaLM concept; DynaCore never sees names, only the loaded tensors.
//
// Store: $DYNALM_MODELS_DIR, else ~/.dynalm/models (%USERPROFILE%\.dynalm\models).
// ./models is searched too, for checkouts that keep models next to the source.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/core.h"
#include "dynacore/base/status.h"

namespace dynalm {

struct RegistryEntry {
  std::string_view name;   // "qwen3:4b"
  std::string_view arch;   // display only: "qwen3"
  std::string_view quant;  // "Q4_K_M"
  std::string_view url;    // direct download link
  int64_t size_bytes;      // approximate download size, for messages before the transfer
  std::string_view note;   // one line: "Qwen3 4B (thinking model)"

  // File name in the store (the last path component of the URL).
  std::string_view file() const { return url.substr(url.rfind('/') + 1); }
};

std::span<const RegistryEntry> registry_entries();

// Exact name, alias or family default (case-insensitive, ":latest" ignored);
// nullptr when unknown.
const RegistryEntry* find_registry_entry(std::string_view name);

// The entry whose download has this file name, if any (names local files).
const RegistryEntry* registry_entry_for_file(std::string_view file_name);

// True when `ref` reads as a name rather than a path: no path separators and
// no .gguf/.safetensors extension.
bool is_model_name(std::string_view ref);

// $DYNALM_MODELS_DIR, else dynalm_home()/models.
std::string models_dir();
// Directories searched for models: the store, then ./models; existing ones only.
std::vector<std::string> model_search_dirs();

struct ResolvedModel {
  std::string path;                      // file or directory to load (store target if not downloaded)
  const RegistryEntry* entry = nullptr;  // set for registry names and known files
  bool downloaded = false;
};

// Resolves a reference. Paths must exist. Registry names resolve whether or
// not they are downloaded (see `downloaded`); unknown names are kNotFound
// with the list of known names.
Result<ResolvedModel> resolve_model(std::string_view ref);

}  // namespace dynalm
