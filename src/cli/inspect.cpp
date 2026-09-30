// `engine inspect <model.gguf> [--metadata] [--tensors]`

#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "cli/commands.h"
#include "loader/gguf/gguf.h"

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
    std::fprintf(stderr, "usage: engine inspect <model.gguf> [--metadata] [--tensors]\n");
    return 1;
  }

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

}  // namespace engine::cli
