// `dynalm inspect <model> [--metadata] [--tensors]`: a GGUF file, or a Hugging
// Face model directory / .safetensors file.

#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "cli/commands.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_model.h"
#include "loader/hf/hf_model.h"
#include "loader/safetensors/safetensors.h"
#include "model/architecture.h"
#include "model_ir/model_config.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

namespace engine::cli {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

std::string format_value(const gguf::Value& v) {
  using gguf::ValueType;
  switch (v.type) {
    case ValueType::kString: {
      std::string s(v.str.substr(0, 80));
      for (char& c : s) {
        if (c == '\n' || c == '\r') c = ' ';
      }
      return "\"" + s + (v.str.size() > 80 ? "...\"" : "\"");
    }
    case ValueType::kArray:
      return "[" + std::string(gguf::value_type_name(v.arr.elem_type)) + " x " +
             std::to_string(v.arr.count) + "]";
    case ValueType::kBool: return v.scalar.b ? "true" : "false";
    case ValueType::kF32:
    case ValueType::kF64: return std::to_string(v.scalar.f);
    case ValueType::kI8:
    case ValueType::kI16:
    case ValueType::kI32:
    case ValueType::kI64: return std::to_string(v.scalar.i);
    default: return std::to_string(v.scalar.u);
  }
}

}  // namespace

void print_model(const ModelConfig& c, int64_t weight_bytes);
int inspect_hf(const std::string& path, bool show_metadata, bool show_tensors);

int cmd_inspect(std::span<const std::string_view> args) {
  std::string path;
  bool show_metadata = false, show_tensors = false;
  for (std::string_view a : args) {
    if (a == "--metadata") {
      show_metadata = true;
    } else if (a == "--tensors") {
      show_tensors = true;
    } else if (path.empty() && !a.starts_with("--")) {
      path = a;
    } else {
      std::fprintf(stderr, "inspect: unexpected argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      return 1;
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "usage: dynalm inspect <model> [--metadata] [--tensors]\n");
    return 1;
  }

  if (hf::is_hf_model(path)) return inspect_hf(path, show_metadata, show_tensors);

  auto file = gguf::GgufFile::open(path);
  if (!file.ok()) {
    std::fprintf(stderr, "inspect: %s\n", file.status().to_string().c_str());
    return 1;
  }
  const gguf::GgufFile& g = **file;

  auto str_or = [&](std::string_view key, std::string_view fallback) {
    auto r = g.get_string(key);
    return std::string(r.ok() ? *r : fallback);
  };

  std::printf("File:          %s (%.1f MiB)\n", path.c_str(), g.file().size() / kMiB);
  std::printf("Format:        GGUF v%u, alignment %llu\n", g.version(),
              static_cast<unsigned long long>(g.alignment()));
  std::printf("Name:          %s\n", str_or("general.name", "(unnamed)").c_str());
  std::printf("Architecture:  %s\n", str_or("general.architecture", "(missing)").c_str());
  std::printf("Metadata keys: %zu\n", g.metadata().size());
  std::printf("Tensors:       %zu (%.1f MiB)\n", g.tensors().size(), g.total_tensor_bytes() / kMiB);

  struct TypeStat {
    int count = 0;
    uint64_t bytes = 0;
    bool executable = true;
  };
  std::map<std::string_view, TypeStat> by_type;
  for (const auto& t : g.tensors()) {
    TypeStat& s = by_type[gguf::ggml_type_name(t.ggml_type)];
    ++s.count;
    s.bytes += t.nbytes;
    s.executable = t.dtype.has_value();
  }
  for (const auto& [name, s] : by_type) {
    std::printf("  %-8.*s %5d tensors %10.1f MiB%s\n", static_cast<int>(name.size()), name.data(),
                s.count, s.bytes / kMiB, s.executable ? "" : "  (unsupported type)");
  }

  if (auto ft = g.get_uint("general.file_type"); ft.ok()) {
    std::printf("Quantization:  %s\n", std::string(gguf::file_type_name(static_cast<uint32_t>(*ft))).c_str());
  }

  auto cfg = gguf::read_model_config(g);
  if (!cfg.ok()) {
    std::printf("\nModel config:  unreadable (%s)\n", cfg.status().to_string().c_str());
  } else {
    print_model(*cfg, static_cast<int64_t>(g.total_tensor_bytes()));
  }

  if (show_metadata) {
    std::printf("\nMetadata:\n");
    for (const auto& kv : g.metadata()) {
      std::printf("  %-45.*s %s\n", static_cast<int>(kv.key.size()), kv.key.data(),
                  format_value(kv.value).c_str());
    }
  }
  if (show_tensors) {
    std::printf("\nTensors:\n");
    for (const auto& t : g.tensors()) {
      std::printf("  %-40.*s %-8.*s %-22s %10.2f MiB\n", static_cast<int>(t.name.size()), t.name.data(),
                  static_cast<int>(gguf::ggml_type_name(t.ggml_type).size()),
                  gguf::ggml_type_name(t.ggml_type).data(), t.shape.to_string().c_str(), t.nbytes / kMiB);
    }
  }
  return 0;
}

void print_model(const ModelConfig& c, int64_t weight_bytes) {
  {
    std::printf("\nModel:\n");
    std::printf("  Layers:        %d\n", c.num_layers);
    std::printf("  Hidden:        %lld\n", static_cast<long long>(c.hidden_size));
    std::printf("  FFN:           %lld\n", static_cast<long long>(c.intermediate_size));
    std::printf("  Heads:         %d (head_dim %d)\n", c.num_heads, c.head_dim);
    std::printf("  KV heads:      %d (GQA group %d)\n", c.num_kv_heads, c.gqa_group());
    std::printf("  Vocab:         %lld\n", static_cast<long long>(c.vocab_size));
    std::printf("  Context:       %lld\n", static_cast<long long>(c.context_length));
    std::printf("  RoPE:          dim %d, base %.0f%s\n", c.rope.dim, c.rope.freq_base,
                c.rope.scaling == RopeScaling::kNone ? "" : ", scaled");
    std::printf("  Norm:          %s, eps %g\n", std::string(norm_type_name(c.norm)).c_str(), c.norm_eps);
    std::printf("  Embeddings:    %s\n", c.tied_embeddings ? "tied" : "separate lm_head");
    if (c.sliding_window > 0) std::printf("  Sliding win:   %d\n", c.sliding_window);
    if (c.moe.num_experts > 0) {
      std::printf("  MoE:           %d experts, %d active\n", c.moe.num_experts, c.moe.experts_per_token);
    }
    const Status valid = c.validate();
    std::printf("  Valid:         %s\n", valid.ok() ? "yes" : valid.to_string().c_str());

    const auto est = estimate_memory(c, weight_bytes, c.context_length, DType::kF16);
    std::printf("\nEstimated RAM (full %lld-token context, f16 KV):\n",
                static_cast<long long>(c.context_length));
    std::printf("  Weights:       %8.1f MiB (memory-mapped)\n", est.weight_bytes / kMiB);
    std::printf("  KV cache:      %8.1f MiB (%lld bytes/token)\n", est.kv_bytes / kMiB,
                static_cast<long long>(c.kv_bytes_per_token(DType::kF16)));
    std::printf("  Scratch:       %8.1f MiB\n", est.activation_bytes / kMiB);
    std::printf("  Total:         %8.1f MiB\n", est.total() / kMiB);
    std::printf("\nSupported:     %s\n",
                find_architecture(c.architecture) ? "YES" : "NO (no adapter for this architecture)");
    std::printf("Backend:       CPU/%s\n", std::string(isa_name(select_best_isa(cpu_info().features))).c_str());
  }
}

int inspect_hf(const std::string& path, bool show_metadata, bool show_tensors) {
  auto files = hf::locate(path);
  if (!files.ok()) {
    std::fprintf(stderr, "inspect: %s\n", files.status().to_string().c_str());
    return 1;
  }
  auto cfg_json = hf::read_json_file(files->dir + "/config.json", 4 << 20);
  if (!cfg_json.ok()) {
    std::fprintf(stderr, "inspect: %s\n", cfg_json.status().to_string().c_str());
    return 1;
  }
  int64_t bytes = 0;
  size_t count = 0;
  std::map<std::string, double> by_dtype;
  std::vector<std::unique_ptr<safetensors::SafeTensorsFile>> shards;
  for (const std::string& w : files->weights) {
    auto st = safetensors::SafeTensorsFile::open(w);
    if (!st.ok()) {
      std::fprintf(stderr, "inspect: %s: %s\n", w.c_str(), st.status().to_string().c_str());
      return 1;
    }
    for (const auto& t : (*st)->tensors()) {
      bytes += static_cast<int64_t>(t.nbytes);
      by_dtype[t.dtype_name] += static_cast<double>(t.nbytes);
      ++count;
    }
    shards.push_back(std::move(*st));
  }
  std::printf("Directory:     %s\n", files->dir.c_str());
  std::printf("Format:        SafeTensors (%zu shard%s)\n", files->weights.size(),
              files->weights.size() == 1 ? "" : "s");
  if (const json::Value* mt = cfg_json->find("model_type"); mt && mt->is_string()) {
    std::printf("model_type:    %s\n", mt->as_string().c_str());
  }
  if (auto q = hf::read_quantization(*cfg_json); !q.ok()) {
    std::printf("Quantization:  unsupported (%s)\n", q.status().message().c_str());
  } else if (q->has_value()) {
    std::printf("Quantization:  %s (repacked at load to Q4_0/Q4_1/Q8_0/F16, DD-041)\n", (*q)->describe().c_str());
  }
  std::printf("Tensors:       %zu (%.1f MiB)\n", count, static_cast<double>(bytes) / kMiB);
  for (const auto& [name, b] : by_dtype) std::printf("  %-8s %10.1f MiB\n", name.c_str(), b / kMiB);
  auto cfg = hf::read_config(*cfg_json);
  if (!cfg.ok()) {
    std::printf("\nModel config:  unsupported (%s)\n", cfg.status().to_string().c_str());
  } else {
    print_model(*cfg, bytes);
  }
  if (show_metadata) std::printf("\nconfig.json:\n%s\n", json::dump(*cfg_json).c_str());
  if (show_tensors) {
    std::printf("\nTensors:\n");
    for (const auto& sh : shards) {
      for (const auto& t : sh->tensors()) {
        std::printf("  %-55s %-6s %-22s %10.2f MiB\n", t.name.c_str(), t.dtype_name.c_str(),
                    t.shape.to_string().c_str(), static_cast<double>(t.nbytes) / kMiB);
      }
    }
  }
  return 0;
}

}  // namespace engine::cli
