// Phase 2 benchmark: GGUF open+parse latency (metadata only; weights are
// mmapped and not touched). Usage: bench_loader <model.gguf>

#include <cstdio>

#include "bench_harness.h"
#include "loader/gguf/gguf.h"
#include "common/core.h"

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_loader <model.gguf>\n");
    return 1;
  }
  auto probe = gguf::GgufFile::open(argv[1]);
  if (!probe.ok()) {
    std::fprintf(stderr, "%s\n", probe.status().to_string().c_str());
    return 1;
  }
  std::printf("model: %s (%zu tensors, %zu kv)\n\n", argv[1], (*probe)->tensors().size(),
              (*probe)->metadata().size());
  bench::print_header();

  const bench::Options opt{.warmup_samples = 2, .samples = 50, .batch = 1};
  auto s = bench::run([&] {
    auto g = gguf::GgufFile::open(argv[1]);
    bench::do_not_optimize(g);
  }, opt);
  bench::print_row("GgufFile::open (warm cache)", s);

  s = bench::run([&] {
    auto toks = gguf::GgufFile::array_strings(*(*probe)->get_array("tokenizer.ggml.tokens"));
    bench::do_not_optimize(toks);
  }, opt);
  bench::print_row("decode tokenizer.ggml.tokens", s);

  s = bench::run([&] {
    for (const auto& t : (*probe)->tensors()) {
      auto tensor = (*probe)->load_tensor(t);
      bench::do_not_optimize(tensor);
    }
  }, opt);
  bench::print_row("load_tensor x all (zero-copy)", s);
  return 0;
}
