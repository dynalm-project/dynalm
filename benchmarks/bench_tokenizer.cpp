// Phase 4 benchmark: tokenizer throughput on a real vocabulary.
// Usage: bench_tokenizer <model.gguf>

#include <cstdio>
#include <string>

#include "bench_harness.h"
#include "loader/gguf/gguf.h"
#include "loader/gguf/gguf_tokenizer.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_tokenizer <model.gguf>\n");
    return 1;
  }
  auto g = gguf::GgufFile::open(argv[1]);
  if (!g.ok()) return std::fprintf(stderr, "%s\n", g.status().to_string().c_str()), 1;
  auto data = gguf::read_tokenizer_data(**g);
  if (!data.ok()) return std::fprintf(stderr, "%s\n", data.status().to_string().c_str()), 1;

  Stopwatch build_timer;
  auto tok = Tokenizer::create(std::move(*data));
  if (!tok.ok()) return std::fprintf(stderr, "%s\n", tok.status().to_string().c_str()), 1;
  std::printf("model: %s  vocab %d  tokenizer build %.1f ms\n\n", argv[1], (*tok)->vocab_size(),
              build_timer.elapsed_ms());

  // ~32 KiB of mixed prose, code, numbers and non-Latin text.
  const std::string unit =
      "The quick brown fox jumps over the lazy dog. It's 2024-09-30 and we've got 12345 items!\n"
      "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n"
      "Unicode: café, naïve, 中文测试，日本語, emoji 😀🎉 and math ∑ x² ≤ ∞.\n\n";
  std::string text;
  while (text.size() < 32 * 1024) text += unit;

  std::vector<TokenId> ids;
  const bench::Options opt{.warmup_samples = 3, .samples = 30, .batch = 1};
  auto s = bench::run([&] {
    ids.clear();
    (void)(*tok)->encode(text, false, false, ids);
  }, opt);
  bench::print_header();
  bench::print_row("encode 32 KiB", s);
  std::printf("  = %zu tokens, %.2f MB/s, %.2f M tokens/s (p50)\n", ids.size(),
              text.size() / (s.p50 * 1e-9) / 1e6, ids.size() / (s.p50 * 1e-9) / 1e6);

  std::string out;
  s = bench::run([&] {
    out.clear();
    for (TokenId id : ids) (*tok)->decode_token(id, out);
  }, opt);
  bench::print_row("decode all tokens", s);
  std::printf("  = %.1f ns/token (p50)\n", s.p50 / ids.size());
  return 0;
}
