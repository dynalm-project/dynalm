#include "registry/model_registry.h"

#include "config/config.h"

#include <cstdlib>
#include <filesystem>
#include <utility>

namespace dynalm {
namespace {

namespace fs = std::filesystem;

constexpr int64_t kMB = 1000 * 1000;

// Every URL resolved (HTTP 200) when it was added. Sizes are approximate, for
// messages only. Supported architectures only (docs/dynalm.md).
constexpr RegistryEntry kEntries[] = {
    {"qwen3:0.6b", "qwen3", "Q8_0", "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
     639 * kMB, "Qwen3 0.6B (thinking model)"},
    {"qwen3:1.7b", "qwen3", "Q8_0", "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
     1830 * kMB, "Qwen3 1.7B (thinking model)"},
    {"qwen3:4b", "qwen3", "Q4_K_M", "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
     2497 * kMB, "Qwen3 4B (thinking model)"},
    {"qwen2.5:0.5b", "qwen2", "Q4_K_M",
     "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
     491 * kMB, "Qwen2.5 0.5B Instruct"},
    {"qwen2.5:1.5b", "qwen2", "Q4_K_M",
     "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
     1117 * kMB, "Qwen2.5 1.5B Instruct"},
    {"llama3.2:1b", "llama", "Q4_K_M",
     "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     808 * kMB, "Llama 3.2 1B Instruct (Llama 3.2 license)"},
    {"llama3.2:3b", "llama", "Q4_K_M",
     "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     2020 * kMB, "Llama 3.2 3B Instruct (Llama 3.2 license)"},
    {"gemma3:270m", "gemma3", "Q8_0",
     "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf", 292 * kMB,
     "Gemma 3 270M instruct (Gemma license)"},
    {"gemma3:1b", "gemma3", "Q4_K_M",
     "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf", 806 * kMB,
     "Gemma 3 1B instruct (Gemma license)"},
    {"smollm2:135m", "llama", "Q8_0",
     "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
     145 * kMB, "SmolLM2 135M Instruct (tiny, for testing)"},
    {"granite3.1-moe:1b", "granitemoe", "Q4_K_M",
     "https://huggingface.co/bartowski/granite-3.1-1b-a400m-instruct-GGUF/resolve/main/"
     "granite-3.1-1b-a400m-instruct-Q4_K_M.gguf",
     822 * kMB, "Granite 3.1 1B-A400M instruct (MoE, 32 experts, 8 active)"},
    {"phi3.5:3.8b", "phi3", "Q4_K_M",
     "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
     2390 * kMB, "Phi-3.5 mini instruct"},
};

// Short names. A family alone picks its default size.
constexpr std::pair<std::string_view, std::string_view> kAliases[] = {
    {"qwen3", "qwen3:4b"},
    {"qwen:4b", "qwen3:4b"},
    {"qwen2.5", "qwen2.5:1.5b"},
    {"llama", "llama3.2:3b"},
    {"llama:1b", "llama3.2:1b"},
    {"llama:3b", "llama3.2:3b"},
    {"llama3.2", "llama3.2:3b"},
    {"gemma", "gemma3:1b"},
    {"gemma:270m", "gemma3:270m"},
    {"gemma:1b", "gemma3:1b"},
    {"gemma3", "gemma3:1b"},
    {"smollm2", "smollm2:135m"},
    {"granite3.1-moe", "granite3.1-moe:1b"},
    {"phi3.5", "phi3.5:3.8b"},
    {"phi", "phi3.5:3.8b"},
};

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
  return out;
}

}  // namespace

std::span<const RegistryEntry> registry_entries() { return kEntries; }

const RegistryEntry* find_registry_entry(std::string_view name) {
  std::string n = lower(name);
  if (n.ends_with(":latest")) n.resize(n.size() - 7);
  for (const auto& [alias, target] : kAliases) {
    if (n == alias) {
      n = target;
      break;
    }
  }
  for (const RegistryEntry& e : kEntries) {
    if (e.name == n) return &e;
  }
  return nullptr;
}

const RegistryEntry* registry_entry_for_file(std::string_view file_name) {
  for (const RegistryEntry& e : kEntries) {
    if (e.file() == file_name) return &e;
  }
  return nullptr;
}

bool is_model_name(std::string_view ref) {
  if (ref.empty() || ref.find_first_of("/\\") != std::string_view::npos) return false;
  const std::string l = lower(ref);
  return !l.ends_with(".gguf") && !l.ends_with(".safetensors");
}

std::string models_dir() {
  if (const char* env = std::getenv("DYNALM_MODELS_DIR"); env != nullptr && *env != '\0') return env;
  return (fs::path(dynalm_home()) / "models").string();
}

std::vector<std::string> model_search_dirs() {
  std::vector<std::string> dirs;
  std::error_code ec;
  const std::string store = models_dir();
  if (fs::is_directory(store, ec)) dirs.push_back(store);
  if (fs::is_directory("models", ec) && !(fs::exists(store, ec) && fs::equivalent("models", store, ec))) {
    dirs.emplace_back("models");
  }
  return dirs;
}

Result<ResolvedModel> resolve_model(std::string_view ref) {
  std::error_code ec;
  ResolvedModel r;
  const fs::path as_path{std::string(ref)};
  if (fs::exists(as_path, ec)) {
    r.path = std::string(ref);
    r.downloaded = true;
    r.entry = registry_entry_for_file(as_path.filename().string());
    return r;
  }
  const bool bare = !ref.empty() && ref.find_first_of("/\\") == std::string_view::npos;
  if (bare && !is_model_name(ref)) {
    // A bare file name with its extension, e.g. the name `dynalm models` prints.
    for (const std::string& dir : model_search_dirs()) {
      const fs::path c = fs::path(dir) / std::string(ref);
      if (fs::exists(c, ec)) {
        r.path = c.string();
        r.downloaded = true;
        r.entry = registry_entry_for_file(c.filename().string());
        return r;
      }
    }
  }
  if (!is_model_name(ref)) return NotFound("model file not found: " + std::string(ref));
  r.entry = find_registry_entry(ref);
  if (r.entry == nullptr) {
    // A bare file name in a model directory, with or without .gguf.
    for (const std::string& dir : model_search_dirs()) {
      for (const fs::path& c : {fs::path(dir) / std::string(ref), fs::path(dir) / (std::string(ref) + ".gguf")}) {
        if (fs::exists(c, ec)) {
          r.path = c.string();
          r.downloaded = true;
          return r;
        }
      }
    }
    std::string known;
    for (const RegistryEntry& e : kEntries) known += (known.empty() ? "" : ", ") + std::string(e.name);
    return NotFound("unknown model '" + std::string(ref) + "'. Known names: " + known +
                    ". Or pass a path to a .gguf file or a Hugging Face model directory.");
  }
  for (const std::string& dir : model_search_dirs()) {
    const fs::path p = fs::path(dir) / std::string(r.entry->file());
    if (fs::is_regular_file(p, ec)) {
      r.path = p.string();
      r.downloaded = true;
      return r;
    }
  }
  r.path = (fs::path(models_dir()) / std::string(r.entry->file())).string();
  return r;
}

}  // namespace dynalm
