#include "loader/model_source.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "model/architecture.h"
#include "common/core.h"

namespace dynalm {
namespace {

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

std::vector<std::string_view> split_path(std::string_view s) {
  std::vector<std::string_view> parts;
  while (!s.empty()) {
    const size_t slash = s.find('/');
    const std::string_view part = s.substr(0, slash);
    if (!part.empty()) parts.push_back(part);
    if (slash == std::string_view::npos) break;
    s.remove_prefix(slash + 1);
  }
  return parts;
}

// Bounds-checked little-endian reader over a prefix of a file.
struct Peek {
  std::span<const std::byte> b;
  size_t pos = 0;
  bool take(void* out, size_t n) {
    if (b.size() - pos < n) return false;
    std::memcpy(out, b.data() + pos, n);
    pos += n;
    return true;
  }
  bool skip(uint64_t n) {
    if (b.size() - pos < n) return false;
    pos += static_cast<size_t>(n);
    return true;
  }
  bool string(std::string_view* out) {
    uint64_t n = 0;
    if (!take(&n, 8) || b.size() - pos < n) return false;
    *out = {reinterpret_cast<const char*>(b.data() + pos), static_cast<size_t>(n)};
    pos += static_cast<size_t>(n);
    return true;
  }
  // Skips one value of GGUF value type `t`.
  bool skip_value(uint32_t t, int depth = 0) {
    switch (t) {
      case 0: case 1: case 7: return skip(1);    // u8 i8 bool
      case 2: case 3: return skip(2);            // u16 i16
      case 4: case 5: case 6: return skip(4);    // u32 i32 f32
      case 10: case 11: case 12: return skip(8); // u64 i64 f64
      case 8: { std::string_view s; return string(&s); }
      case 9: {
        uint32_t et = 0;
        uint64_t n = 0;
        if (depth > 2 || !take(&et, 4) || !take(&n, 8)) return false;
        for (uint64_t i = 0; i < n; ++i) {
          if (!skip_value(et, depth + 1)) return false;
        }
        return true;
      }
      default: return false;
    }
  }
};

}  // namespace

Result<std::string> resolve_model_url(std::string_view ref) {
  if (ref.empty()) return InvalidArgument("empty model link");
  for (char c : ref) {
    if (static_cast<unsigned char>(c) <= ' ' || c == '"') return InvalidArgument("model link contains a space or quote");
  }
  constexpr std::string_view kHf = "https://huggingface.co/";
  if (starts_with(ref, "http://") || starts_with(ref, "https://")) {
    if (!starts_with(ref, kHf)) return std::string(ref);
    std::string_view rest = ref.substr(kHf.size());
    rest = rest.substr(0, rest.find_first_of("?#"));
    const std::vector<std::string_view> p = split_path(rest);
    // <owner>/<repo>/{blob,resolve}/<rev>/<path...>
    if (p.size() >= 5 && (p[2] == "blob" || p[2] == "resolve")) {
      std::string url = std::string(kHf) + std::string(p[0]) + "/" + std::string(p[1]) + "/resolve";
      for (size_t i = 3; i < p.size(); ++i) url += "/" + std::string(p[i]);
      return url;
    }
    if (p.size() == 2 || (p.size() >= 3 && p[2] == "tree")) {
      return InvalidArgument("that is a Hugging Face repository, not a file: open \"Files and versions\", "
                             "pick one .gguf (Q4_K_M is a good default) and use its link");
    }
    return InvalidArgument("unrecognized Hugging Face link (expected .../resolve/<rev>/<file> or .../blob/<rev>/<file>)");
  }
  // Short form <owner>/<repo>/<path...>
  const std::vector<std::string_view> p = split_path(ref);
  if (p.size() < 3) {
    return InvalidArgument("expected a URL or <owner>/<repo>/<file>.gguf, e.g. "
                           "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  }
  std::string url = std::string(kHf) + std::string(p[0]) + "/" + std::string(p[1]) + "/resolve/main";
  for (size_t i = 2; i < p.size(); ++i) url += "/" + std::string(p[i]);
  return url;
}

std::string url_file_name(std::string_view url) {
  url = url.substr(0, url.find_first_of("?#"));
  const size_t slash = url.rfind('/');
  return std::string(slash == std::string_view::npos ? url : url.substr(slash + 1));
}

Result<std::string> peek_gguf_architecture(std::span<const std::byte> head) {
  Peek r{head};
  char magic[4];
  uint32_t version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!r.take(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    return Corrupt("not a GGUF file (the link may point to a web page or need a login)");
  }
  if (!r.take(&version, 4) || version < 2 || version > 3) {
    return Unsupported("GGUF version " + std::to_string(version) + " (supported: 2, 3)");
  }
  if (!r.take(&n_tensors, 8) || !r.take(&n_kv, 8)) return NotFound("GGUF header is truncated");
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string_view key;
    uint32_t type = 0;
    if (!r.string(&key) || !r.take(&type, 4)) break;
    if (key == "general.architecture" && type == 8) {
      std::string_view arch;
      if (!r.string(&arch)) break;
      return std::string(arch);
    }
    if (!r.skip_value(type)) break;
  }
  return NotFound("general.architecture not found in the first " + std::to_string(head.size()) + " bytes");
}

std::string gguf_support_status(const gguf::GgufFile& f) {
  auto arch = f.get_string("general.architecture");
  if (!arch.ok()) return "missing general.architecture";
  if (find_architecture(*arch) == nullptr) return "unsupported architecture '" + std::string(*arch) + "'";
  std::vector<std::string_view> bad;
  for (const gguf::TensorInfo& t : f.tensors()) {
    if (t.dtype) continue;
    const std::string_view name = gguf::ggml_type_name(t.ggml_type);
    bool seen = false;
    for (std::string_view b : bad) seen = seen || b == name;
    if (!seen) bad.push_back(name);
  }
  if (bad.empty()) return "ok";
  std::string s = "unsupported tensor types:";
  for (size_t i = 0; i < bad.size(); ++i) s += (i ? ", " : " ") + std::string(bad[i]);
  return s;
}

}  // namespace dynalm
