// Bounded, deterministic mutation fuzzing of the parsers that read untrusted
// bytes (docs/security.md): GGUF model files (load_model), SafeTensors files,
// and request JSON (json::parse). Every mutated input must give a clean error
// or a valid object, never a crash, hang or undefined behaviour; the sanitizer
// CI jobs (ASan + UBSan) run this program. A mutated GGUF that still loads is
// also tokenized and run for one forward step on the CPU.
//
//   test_fuzz_parsers [iterations]   (default 1500 per target)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include "api/json.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "loader/safetensors/safetensors.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "sampling/json_constraint.h"
#include "tests/tiny_gguf.h"

using namespace dynalm;

namespace {

struct Mutator {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  size_t below(size_t n) { return n == 0 ? 0 : static_cast<size_t>(next() % n); }

  // One of: bit flips, an extreme integer written at a random offset (biased
  // to the header), truncation, a duplicated or zeroed range.
  std::vector<uint8_t> mutate(std::vector<uint8_t> b, size_t header_bias) {
    const int kind = static_cast<int>(below(5));
    const size_t focus = std::min(b.size(), header_bias);
    auto pos = [&] { return below(4) != 0 ? below(focus) : below(b.size()); };
    switch (kind) {
      case 0:
        for (int i = 0, n = 1 + static_cast<int>(below(8)); i < n; ++i) b[pos()] ^= static_cast<uint8_t>(1u << below(8));
        break;
      case 1: {
        static const uint64_t kExtreme[] = {0,          1,          0x7f,       0xff,       0xffff,     0x7fffffff,
                                            0x80000000, 0xffffffff, uint64_t{1} << 32, uint64_t{1} << 40, ~uint64_t{0}, ~uint64_t{0} >> 1};
        const uint64_t v = kExtreme[below(std::size(kExtreme))];
        const size_t p = pos();
        const size_t n = std::min<size_t>(below(2) ? 8 : 4, b.size() - p);
        std::memcpy(b.data() + p, &v, n);
        break;
      }
      case 2: b.resize(below(b.size())); break;
      case 3: {
        const size_t p = pos(), n = std::min(below(64) + 1, b.size() - p);
        b.insert(b.begin() + static_cast<long>(p), b.begin() + static_cast<long>(p), b.begin() + static_cast<long>(p + n));
        break;
      }
      default: {
        const size_t p = pos(), n = std::min(below(256) + 1, b.size() - p);
        std::memset(b.data() + p, 0, n);
        break;
      }
    }
    return b;
  }
};

bool write(const std::string& path, const std::vector<uint8_t>& b) { return test::write_file(path, b); }

int fuzz_gguf(const std::string& dir, int iters) {
  const std::vector<uint8_t> seed = test::tiny_model_gguf();
  const std::string path = dir + "/fuzz.gguf";
  Mutator m{0x9E3779B97F4A7C15ull};
  dynacore::ThreadPool pool(2);
  dynacore::CpuDevice cpu(pool, dynacore::select_best_isa(dynacore::cpu_info().features));
  int loaded = 0, ran = 0;
  for (int i = 0; i < iters; ++i) {
    if (std::getenv("FUZZ_TRACE")) std::fprintf(stderr, "gguf iteration %d\n", i);
    const std::vector<uint8_t> input = m.mutate(seed, 8192);
    // FUZZ_DUMP=<i>: write iteration i's input to fuzz-<i>.gguf and stop (reproduction).
    if (const char* dump = std::getenv("FUZZ_DUMP"); dump != nullptr && std::atoi(dump) == i) {
      write("fuzz-" + std::to_string(i) + ".gguf", input);
      std::printf("wrote fuzz-%d.gguf\n", i);
      return 0;
    }
    if (!write(path, input)) return -1;
    auto lm = load_model(path);
    if (!lm.ok()) continue;
    ++loaded;
    std::vector<TokenId> toks;
    if (!(*lm)->tokenizer->encode("hello fuzz", true, false, toks).ok() || toks.empty()) continue;
    auto tf = Transformer::create((*lm)->config, (*lm)->weights, cpu, 16);
    if (!tf.ok()) continue;
    auto kv = KvBlockPool::create(kv_geometry_for((*lm)->config, DType::kF16, 16, 64), cpu);
    if (!kv.ok()) continue;
    KvBlockTable t(**kv);
    if (!t.reserve(static_cast<int64_t>(toks.size())).ok()) continue;
    std::vector<float> logits(static_cast<size_t>((*lm)->config.vocab_size));
    const SeqBatch b{toks, 0, t.block_table(), true};
    (void)(*tf)->forward_batch({&b, 1}, **kv, logits, nullptr);  // any status; must not crash
    ++ran;
  }
  std::printf("  gguf: %d mutations, %d still loaded, %d ran a forward step\n", iters, loaded, ran);
  return 0;
}

std::vector<uint8_t> tiny_safetensors() {
  const std::string header =
      R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("b":{"dtype":"F16","shape":[4],"data_offsets":[24,32]}})";
  std::vector<uint8_t> out(8);
  const uint64_t n = header.size();
  std::memcpy(out.data(), &n, 8);
  out.insert(out.end(), header.begin(), header.end());
  out.resize(out.size() + 32, 0x3c);
  return out;
}

int fuzz_safetensors(const std::string& dir, int iters) {
  const std::vector<uint8_t> seed = tiny_safetensors();
  const std::string path = dir + "/fuzz.safetensors";
  Mutator m{0xD1B54A32D192ED03ull};
  int ok = 0;
  for (int i = 0; i < iters; ++i) {
    if (!write(path, m.mutate(seed, seed.size()))) return -1;
    auto f = safetensors::SafeTensorsFile::open(path);
    if (!f.ok()) continue;
    ++ok;
    for (const auto& t : (*f)->tensors()) (void)(*f)->load_tensor(t);
  }
  std::printf("  safetensors: %d mutations, %d parsed\n", iters, ok);
  return 0;
}

int fuzz_json(int iters) {
  const std::string seeds[] = {
      R"({"model":"m","messages":[{"role":"user","content":"hi é 😀"}],"max_tokens":16,"stream":true})",
      R"({"prompt":["a","b"],"temperature":0.7,"top_p":0.9,"stop":["\n"],"n":1,"logit_bias":{"1":-100}})",
      R"([1,2.5e10,-0,true,false,null,{"a":[[[[[]]]]]}])",
  };
  Mutator m{0x94D049BB133111EBull};
  int ok = 0;
  for (int i = 0; i < iters; ++i) {
    const std::string& s = seeds[static_cast<size_t>(i) % std::size(seeds)];
    const std::vector<uint8_t> b = m.mutate(std::vector<uint8_t>(s.begin(), s.end()), s.size());
    ok += json::parse(std::string(b.begin(), b.end())).ok() ? 1 : 0;
  }
  // Deep nesting and huge numbers fail cleanly (no stack exhaustion).
  if (json::parse(std::string(100000, '[')).ok()) return -1;
  (void)json::parse("1e999999");  // out-of-range number: any status, no UB
  std::printf("  json: %d mutations, %d parsed; 100000-deep nesting refused\n", iters, ok);
  return 0;
}

// Structured output: mutated JSON schemas compile or are refused cleanly, and
// the matcher runs over mutated documents without crashing (ASan/UBSan in CI).
int fuzz_json_schema(int iters) {
  const std::string schemas[] = {
      R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":["string","null"]},"c":{"type":"array","items":{"enum":["x",1,true]}}},"required":["a"],"additionalProperties":false})",
      R"({"anyOf":[{"type":"object","properties":{"k\"q":{"const":-0.5e3}}},{"type":"array","items":{"type":"number"}}]})",
      R"({"type":"object"})",
  };
  const std::string docs[] = {
      R"({"a": 12, "b": null, "c": ["x", 1, true]})",
      R"({"k\"q": -500.0, "z": [1, 2.5e-3, -0]})",
      R"({"s": "\u00e9\n", "t": [{}, [], {"u": false}]})",
  };
  Mutator m{0xD6E8FEB86659FD93ull};
  auto vocab = VocabBytes::from_bytes({"", "{", "}", "\"", "a"});
  int compiled = 0, accepted = 0;
  for (int i = 0; i < iters; ++i) {
    const std::string& s = schemas[static_cast<size_t>(i) % std::size(schemas)];
    // Half the schemas are mutated (compile or refuse cleanly); the other half
    // stay valid so the matcher meets many mutated documents.
    const std::vector<uint8_t> b = i % 2 == 0 ? m.mutate(std::vector<uint8_t>(s.begin(), s.end()), s.size())
                                              : std::vector<uint8_t>(s.begin(), s.end());
    auto v = json::parse(std::string(b.begin(), b.end()));
    if (!v.ok()) continue;
    auto g = JsonGrammar::compile(&*v);
    if (!g.ok()) continue;
    ++compiled;
    JsonConstraint c(*g, vocab, {0});
    JsonConstraint::State st = c.state();
    const std::string& d = docs[static_cast<size_t>(i / 3) % std::size(docs)];
    const std::vector<uint8_t> doc = i % 4 == 1 ? std::vector<uint8_t>(d.begin(), d.end())
                                                : m.mutate(std::vector<uint8_t>(d.begin(), d.end()), d.size());
    bool ok = true;
    for (uint8_t ch : doc) {
      if (!(ok = c.feed(st, ch))) break;
    }
    accepted += ok && c.can_finish(st) ? 1 : 0;
  }
  std::printf("  json schema: %d mutations, %d compiled, %d documents matched\n", iters, compiled, accepted);
  return compiled > 0 && accepted > 0 ? 0 : -1;
}

// Regression (S10, found by this fuzzer): metadata that would size the
// runtime far beyond the file (a block_count of 67,108,866 with 2 layers of
// tensors, a huge embedding_length) is refused before any allocation.
int metadata_limits(const std::string& dir) {
  struct Case {
    const char* what;
    test::TinyModelSpec spec;
  };
  test::TinyModelSpec ok;
  const std::string path = dir + "/limits.gguf";
  if (!write(path, test::tiny_model_gguf(ok)) || !load_model(path).ok()) return -1;
  // Patch block_count in the generated file to a huge value.
  std::vector<uint8_t> b = test::tiny_model_gguf(ok);
  const std::string key = "qwen2.block_count";
  auto it = std::search(b.begin(), b.end(), key.begin(), key.end());
  if (it == b.end()) return -1;
  const size_t at = static_cast<size_t>(it - b.begin()) + key.size() + 4;  // after the u32 type tag
  const uint32_t huge = 67108866;
  std::memcpy(b.data() + at, &huge, 4);
  if (!write(path, b)) return -1;
  auto r = load_model(path);
  if (r.ok()) return -1;
  std::printf("  metadata limits: block_count 67108866 refused (%s)\n", r.status().message().c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 1500;
  const auto dir = std::filesystem::temp_directory_path() / ("dynalm-fuzz-" + test::unique_suffix());
  std::filesystem::create_directories(dir);
  int rc = metadata_limits(dir.string());
  rc |= fuzz_gguf(dir.string(), iters);
  rc |= fuzz_safetensors(dir.string(), iters);
  rc |= fuzz_json(iters * 4);
  rc |= fuzz_json_schema(iters * 4);
  std::filesystem::remove_all(dir);
  if (rc != 0) {
    std::fprintf(stderr, "fuzz harness error\n");
    return 1;
  }
  std::printf("all parser fuzz checks passed\n");
  return 0;
}
