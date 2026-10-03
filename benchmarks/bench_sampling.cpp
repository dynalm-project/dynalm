// Phase 26: cost of choosing one token from a 151,936-entry logits row
// (Qwen2.5 vocabulary) for each sampling configuration. Includes copying the
// row, as the scheduler's in-place pipeline would overwrite it.

#include <cstdio>
#include <random>
#include <vector>

#include "bench_harness.h"
#include "sampling/sampler.h"

int main() {
  using namespace engine;
  constexpr size_t kVocab = 151936;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  std::vector<float> logits(kVocab), row(kVocab);
  for (float& v : logits) v = nd(rng);
  std::vector<TokenId> context(512);
  for (TokenId& t : context) t = static_cast<TokenId>(rng() % kVocab);

  struct Case {
    const char* name;
    SamplingParams p;
  };
  auto make = [](float t, int k, float top_p, float min_p, float rep) {
    SamplingParams p;
    p.temperature = t;
    p.top_k = k;
    p.top_p = top_p;
    p.min_p = min_p;
    p.repetition_penalty = rep;
    p.seed = 1;
    p.has_seed = true;
    return p;
  };
  const Case cases[] = {
      {"greedy", make(0, 0, 1, 0, 1)},
      {"temperature 1", make(1, 0, 1, 0, 1)},
      {"top_k 40", make(0.8f, 40, 1, 0, 1)},
      {"top_p 0.9", make(0.8f, 0, 0.9f, 0, 1)},
      {"min_p 0.05", make(0.8f, 0, 1, 0.05f, 1)},
      {"top_k 40 + top_p 0.95 + rep 1.1", make(0.8f, 40, 0.95f, 0, 1.1f)},
  };
  std::printf("vocab %zu, one sample per call (row copy included)\n\n%-34s %10s %10s\n", kVocab, "config",
              "p50 us", "p99 us");
  {
    const auto copy = bench::run([&] { row = logits; bench::do_not_optimize(row[0]); }, {.warmup_samples = 5, .samples = 100, .batch = 10});
    std::printf("%-34s %10.1f %10.1f\n", "(row copy only)", copy.p50 * 1e-3, copy.p99 * 1e-3);
  }
  for (const Case& c : cases) {
    Sampler s(c.p);
    TokenId sink = 0;
    const auto st = bench::run(
        [&] {
          row = logits;
          sink += s.sample(row, context);
        },
        {.warmup_samples = 5, .samples = 100, .batch = 10});
    bench::do_not_optimize(sink);
    std::printf("%-34s %10.1f %10.1f\n", c.name, st.p50 * 1e-3, st.p99 * 1e-3);
  }
  return 0;
}
