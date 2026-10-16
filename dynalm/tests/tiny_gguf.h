#pragma once

// A tiny synthetic GGUF model for model-free tests: a 2-layer qwen2-style
// decoder (hidden 128, 4 query heads / 2 KV heads of 32, FFN 256, Q/K/V
// biases, tied embeddings) with deterministic random Q8_0 matrices and f32
// norms, and a GPT-2 byte-level BPE tokenizer (256 byte tokens plus ChatML
// specials, no merges) with a ChatML chat template. Its outputs are
// meaningless text; what matters is that every runtime path (loader,
// tokenizer, chat template, Transformer, KV cache, scheduler, engine, server,
// CPU and GPU devices) runs on real GGUF bytes without a downloaded model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace dynalm::test {

struct TinyModelSpec {
  int32_t layers = 2, hidden = 128, heads = 4, kv_heads = 2, ffn = 256, context = 1024;
  uint32_t seed = 7;
  bool qwen3 = false;  // qwen3: per-head Q/K norm, no QKV bias (runs on the CPU, not on the GPU)
};

class GgufWriter {
 public:
  // GGUF value types.
  enum : uint32_t { kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7, kStr = 8, kArr = 9 };

  void kv_u32(const std::string& k, uint32_t v) { key(k, kU32); put(v); ++n_kv_; }
  void kv_f32(const std::string& k, float v) { key(k, kF32); put(v); ++n_kv_; }
  void kv_bool(const std::string& k, bool v) { key(k, kBool); put(static_cast<uint8_t>(v)); ++n_kv_; }
  void kv_str(const std::string& k, const std::string& v) { key(k, kStr); str(meta_, v); ++n_kv_; }
  void kv_strs(const std::string& k, const std::vector<std::string>& v) {
    key(k, kArr);
    put(kStr);
    put(static_cast<uint64_t>(v.size()));
    for (const auto& s : v) str(meta_, s);
    ++n_kv_;
  }
  void kv_i32s(const std::string& k, const std::vector<int32_t>& v) {
    key(k, kArr);
    put(kI32);
    put(static_cast<uint64_t>(v.size()));
    for (int32_t x : v) put(x);
    ++n_kv_;
  }
  // ggml type ids: 0 = f32, 8 = q8_0. dims innermost first.
  void tensor(const std::string& name, uint32_t type, std::vector<uint64_t> dims, const std::vector<uint8_t>& data) {
    tensors_.push_back({name, type, std::move(dims), data});
  }

  std::vector<uint8_t> bytes() const {
    std::vector<uint8_t> out;
    auto w = [&](const void* p, size_t n) { out.insert(out.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); };
    w("GGUF", 4);
    const uint32_t version = 3;
    w(&version, 4);
    const uint64_t nt = tensors_.size(), nkv = n_kv_;
    w(&nt, 8);
    w(&nkv, 8);
    out.insert(out.end(), meta_.begin(), meta_.end());
    uint64_t offset = 0;
    for (const auto& t : tensors_) {
      std::vector<uint8_t> info;
      str(info, t.name);
      out.insert(out.end(), info.begin(), info.end());
      const auto nd = static_cast<uint32_t>(t.dims.size());
      w(&nd, 4);
      for (uint64_t d : t.dims) w(&d, 8);
      w(&t.type, 4);
      w(&offset, 8);
      offset += (t.data.size() + 31) / 32 * 32;
    }
    out.resize((out.size() + 31) / 32 * 32, 0);
    for (const auto& t : tensors_) {
      out.insert(out.end(), t.data.begin(), t.data.end());
      out.resize((out.size() + 31) / 32 * 32, 0);
    }
    return out;
  }

 private:
  struct T {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> data;
  };
  template <class V>
  void put(V v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    meta_.insert(meta_.end(), p, p + sizeof(V));
  }
  static void str(std::vector<uint8_t>& o, const std::string& s) {
    const uint64_t n = s.size();
    const auto* p = reinterpret_cast<const uint8_t*>(&n);
    o.insert(o.end(), p, p + 8);
    o.insert(o.end(), s.begin(), s.end());
  }
  void key(const std::string& k, uint32_t type) {
    str(meta_, k);
    put(type);
  }
  std::vector<uint8_t> meta_;
  uint64_t n_kv_ = 0;
  std::vector<T> tensors_;
};

// IEEE half from float (round to nearest even; finite inputs).
inline uint16_t to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
  uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1))) ++h;
  return static_cast<uint16_t>(h);
}

struct Rng {
  uint64_t s;
  float next() {  // uniform [-1, 1)
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>((s >> 40) & 0xffffff) / 8388608.0f - 1.0f;
  }
};

inline std::vector<uint8_t> f32_bytes(const std::vector<float>& v) {
  std::vector<uint8_t> o(v.size() * 4);
  std::memcpy(o.data(), v.data(), o.size());
  return o;
}

// Q8_0 rows: blocks of 32 values, fp16 scale + 32 int8.
inline std::vector<uint8_t> q8_0_bytes(const std::vector<float>& v) {
  std::vector<uint8_t> o;
  for (size_t b = 0; b < v.size(); b += 32) {
    float amax = 0;
    for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[b + i]));
    const float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
    const uint16_t h = to_half(d);
    o.push_back(static_cast<uint8_t>(h & 0xff));
    o.push_back(static_cast<uint8_t>(h >> 8));
    for (size_t i = 0; i < 32; ++i) o.push_back(static_cast<uint8_t>(static_cast<int8_t>(std::lround(v[b + i] * id))));
  }
  return o;
}

// GPT-2 bytes_to_unicode as UTF-8 strings, in byte order.
inline std::vector<std::string> byte_tokens() {
  std::vector<int> cps(256, -1);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    cps[static_cast<size_t>(b)] = printable ? b : 256 + n++;
  }
  std::vector<std::string> out;
  for (int cp : cps) {
    std::string s;
    if (cp < 0x80) {
      s += static_cast<char>(cp);
    } else {
      s += static_cast<char>(0xC0 | (cp >> 6));
      s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    out.push_back(s);
  }
  return out;
}

inline std::vector<uint8_t> tiny_model_gguf(const TinyModelSpec& m = {}) {
  GgufWriter w;
  const std::string a = m.qwen3 ? "qwen3" : "qwen2";
  w.kv_str("general.architecture", a);
  w.kv_str("general.name", "dynalm-tiny-test");
  w.kv_u32(a + ".block_count", static_cast<uint32_t>(m.layers));
  w.kv_u32(a + ".context_length", static_cast<uint32_t>(m.context));
  w.kv_u32(a + ".embedding_length", static_cast<uint32_t>(m.hidden));
  w.kv_u32(a + ".feed_forward_length", static_cast<uint32_t>(m.ffn));
  w.kv_u32(a + ".attention.head_count", static_cast<uint32_t>(m.heads));
  w.kv_u32(a + ".attention.head_count_kv", static_cast<uint32_t>(m.kv_heads));
  w.kv_f32(a + ".rope.freq_base", 10000.0f);
  w.kv_f32(a + ".attention.layer_norm_rms_epsilon", 1e-6f);
  std::vector<std::string> toks = byte_tokens();
  std::vector<int32_t> types(toks.size(), 1);
  for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) {
    toks.emplace_back(special);
    types.push_back(3);
  }
  const auto vocab = static_cast<int32_t>(toks.size());
  w.kv_str("tokenizer.ggml.model", "gpt2");
  w.kv_str("tokenizer.ggml.pre", "qwen2");
  w.kv_strs("tokenizer.ggml.tokens", toks);
  w.kv_i32s("tokenizer.ggml.token_type", types);
  w.kv_strs("tokenizer.ggml.merges", {});
  w.kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(vocab - 1));  // <|im_end|>
  w.kv_u32("tokenizer.ggml.padding_token_id", static_cast<uint32_t>(vocab - 3));
  w.kv_bool("tokenizer.ggml.add_bos_token", false);
  w.kv_str("tokenizer.chat_template",
           "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
           "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}");

  Rng rng{m.seed};
  auto rnd = [&](size_t n, float scale) {
    std::vector<float> v(n);
    for (float& x : v) x = rng.next() * scale;
    return v;
  };
  auto ones = [&](size_t n) {
    std::vector<float> v(n);
    for (float& x : v) x = 1.0f + 0.1f * rng.next();
    return v;
  };
  const auto H = static_cast<uint64_t>(m.hidden), F = static_cast<uint64_t>(m.ffn);
  const uint64_t hd = H / static_cast<uint64_t>(m.heads), kvd = hd * static_cast<uint64_t>(m.kv_heads);
  w.tensor("token_embd.weight", 8, {H, static_cast<uint64_t>(vocab)}, q8_0_bytes(rnd(H * static_cast<uint64_t>(vocab), 1.0f)));
  w.tensor("output_norm.weight", 0, {H}, f32_bytes(ones(H)));
  for (int l = 0; l < m.layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    w.tensor(p + "attn_norm.weight", 0, {H}, f32_bytes(ones(H)));
    w.tensor(p + "attn_q.weight", 8, {H, H}, q8_0_bytes(rnd(H * H, 0.15f)));
    w.tensor(p + "attn_k.weight", 8, {H, kvd}, q8_0_bytes(rnd(H * kvd, 0.15f)));
    w.tensor(p + "attn_v.weight", 8, {H, kvd}, q8_0_bytes(rnd(H * kvd, 0.15f)));
    if (m.qwen3) {
      w.tensor(p + "attn_q_norm.weight", 0, {hd}, f32_bytes(ones(hd)));
      w.tensor(p + "attn_k_norm.weight", 0, {hd}, f32_bytes(ones(hd)));
    } else {
      w.tensor(p + "attn_q.bias", 0, {H}, f32_bytes(rnd(H, 0.05f)));
      w.tensor(p + "attn_k.bias", 0, {kvd}, f32_bytes(rnd(kvd, 0.05f)));
      w.tensor(p + "attn_v.bias", 0, {kvd}, f32_bytes(rnd(kvd, 0.05f)));
    }
    w.tensor(p + "attn_output.weight", 8, {H, H}, q8_0_bytes(rnd(H * H, 0.1f)));
    w.tensor(p + "ffn_norm.weight", 0, {H}, f32_bytes(ones(H)));
    w.tensor(p + "ffn_gate.weight", 8, {H, F}, q8_0_bytes(rnd(H * F, 0.12f)));
    w.tensor(p + "ffn_up.weight", 8, {H, F}, q8_0_bytes(rnd(H * F, 0.12f)));
    w.tensor(p + "ffn_down.weight", 8, {F, H}, q8_0_bytes(rnd(F * H, 0.08f)));
  }
  return w.bytes();
}

// A unique suffix for per-test temporary directories (portable: no getpid).
inline std::string unique_suffix() {
  std::random_device rd;
  const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::to_string(rd()) + "-" + std::to_string(t);
}

// Writes the tiny model to `path`; false on I/O failure.
inline bool write_file(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return false;
  const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  return std::fclose(f) == 0 && ok;
}

}  // namespace dynalm::test
