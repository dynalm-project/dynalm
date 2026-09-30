#pragma once

// Test helper: builds GGUF files in memory so loader tests need no real model.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace engine::testing {

class GgufBuilder {
 public:
  explicit GgufBuilder(uint32_t version = 3) : version_(version) {}

  GgufBuilder& kv_u32(const std::string& k, uint32_t v) { key(k, 4); pod(v); ++n_kv_; return *this; }
  GgufBuilder& kv_i32(const std::string& k, int32_t v) { key(k, 5); pod(v); ++n_kv_; return *this; }
  GgufBuilder& kv_f32(const std::string& k, float v) { key(k, 6); pod(v); ++n_kv_; return *this; }
  GgufBuilder& kv_bool(const std::string& k, bool v) { key(k, 7); pod(uint8_t{v}); ++n_kv_; return *this; }
  GgufBuilder& kv_u64(const std::string& k, uint64_t v) { key(k, 10); pod(v); ++n_kv_; return *this; }
  GgufBuilder& kv_str(const std::string& k, const std::string& v) { key(k, 8); str(kv_, v); ++n_kv_; return *this; }
  GgufBuilder& kv_str_array(const std::string& k, const std::vector<std::string>& v) {
    key(k, 9);
    pod(uint32_t{8});
    pod(uint64_t{v.size()});
    for (const auto& s : v) str(kv_, s);
    ++n_kv_;
    return *this;
  }
  GgufBuilder& kv_f32_array(const std::string& k, const std::vector<float>& v) {
    key(k, 9);
    pod(uint32_t{6});
    pod(uint64_t{v.size()});
    for (float f : v) pod(f);
    ++n_kv_;
    return *this;
  }
  GgufBuilder& kv_i32_array(const std::string& k, const std::vector<int32_t>& v) {
    key(k, 9);
    pod(uint32_t{5});
    pod(uint64_t{v.size()});
    for (int32_t x : v) pod(x);
    ++n_kv_;
    return *this;
  }

  // dims in GGUF order (ne[0] = innermost). `data` is written verbatim.
  GgufBuilder& tensor(const std::string& name, std::vector<uint64_t> ne, uint32_t ggml_type,
                      const std::vector<uint8_t>& data) {
    pending_.push_back({name, std::move(ne), ggml_type, data});
    return *this;
  }

  std::vector<uint8_t> build(uint64_t alignment = 32) const {
    std::vector<uint8_t> out;
    append_pod(out, uint32_t{0x46554747});
    append_pod(out, version_);
    append_pod(out, uint64_t{pending_.size()});
    append_pod(out, n_kv_);
    out.insert(out.end(), kv_.begin(), kv_.end());
    uint64_t off = 0;
    std::vector<uint64_t> offsets;
    for (const auto& t : pending_) {
      str(out, t.name);
      append_pod(out, static_cast<uint32_t>(t.ne.size()));
      for (uint64_t d : t.ne) append_pod(out, d);
      append_pod(out, t.type);
      append_pod(out, off);
      offsets.push_back(off);
      off += (t.data.size() + alignment - 1) / alignment * alignment;
    }
    out.resize((out.size() + alignment - 1) / alignment * alignment, 0);
    const size_t base = out.size();
    for (size_t i = 0; i < pending_.size(); ++i) {
      out.resize(base + offsets[i], 0);
      out.insert(out.end(), pending_[i].data.begin(), pending_[i].data.end());
    }
    return out;
  }

  static std::string write_temp(const std::vector<uint8_t>& bytes, const std::string& name) {
    // Unique per call: on Windows a still-mapped file cannot be overwritten.
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("engine_test_" + name + "_" + std::to_string(counter++) + ".gguf");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) throw std::runtime_error("cannot write " + path.string());
    return path.string();
  }

 private:
  struct PendingTensor {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type;
    std::vector<uint8_t> data;
  };

  template <typename T>
  static void append_pod(std::vector<uint8_t>& out, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
  }
  template <typename T>
  void pod(T v) { append_pod(kv_, v); }
  static void str(std::vector<uint8_t>& out, const std::string& s) {
    append_pod(out, uint64_t{s.size()});
    out.insert(out.end(), s.begin(), s.end());
  }
  void key(const std::string& k, uint32_t type) {
    str(kv_, k);
    pod(type);
  }

  uint32_t version_;
  uint64_t n_kv_ = 0;
  std::vector<uint8_t> kv_;
  std::vector<PendingTensor> pending_;
};

}  // namespace engine::testing
