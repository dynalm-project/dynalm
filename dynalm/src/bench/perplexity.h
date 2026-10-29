#pragma once

// Perplexity over a text, scored exactly as llama.cpp's llama-perplexity does
// (same file, same context), and optionally compared position by position with
// a llama-perplexity --kl-divergence-base file: top-1 agreement, top-5
// overlap, KL divergence (the DD-053 accuracy check). Used by bench_ppl and
// by `dynalm cpu-report`.
//
// The text is tokenized once (with BOS when the model adds one) and cut into
// chunks of `ctx` tokens; each chunk starts with BOS (when the model uses one)
// and runs from position 0; the tokens of its second half are scored (logits
// at j predict token j + 1, j from ctx/2 to ctx - 2). The device runs its
// default plan, as the engine does.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "common/core.h"

namespace dynalm {

struct PerplexityOptions {
  std::string model_path;
  std::string text_path;
  int32_t ctx = 512;
  int max_chunks = 0;  // 0 = all
  int threads = 0;     // 0 = physical cores
  // native (DynaCore's CPU device) | ggml-cpu | metal | vulkan | cuda
  std::string device = "native";
  int32_t ubatch = 0;      // score each chunk in batches of n tokens (llama-perplexity -ub n); 0 = ctx
  bool fresh_kv = true;    // a new KV pool per chunk, as llama-perplexity clears its cache
  int32_t int8_rows = -1;  // native CPU int8 activation row limit; -1 = the plan's default
  std::string kld_ref;     // llama-perplexity --kl-divergence-base file to compare with
  std::string kld_base;    // write DynaLM's log-probabilities in that format
  std::FILE* log = nullptr;  // progress and results, bench_ppl's format; nullptr = silent
};

struct PerplexityParity {
  int64_t positions = 0, agree = 0, top5 = 0, near_ties = 0;
  double kl_mean = 0, kl_max = 0, dp_max = 0, dlogp_top1_max = 0, median_margin = 0;
  int64_t first_chunk = -1, first_pos = -1, first_ref = -1, first_dyn = -1;
  double first_margin = 0;
  int64_t tokenizer_first_mismatch = -1;
  double top1_pct() const { return positions > 0 ? 100.0 * static_cast<double>(agree) / static_cast<double>(positions) : 0; }
  double top5_pct() const {
    return positions > 0 ? 100.0 * static_cast<double>(top5) / (5.0 * static_cast<double>(positions)) : 0;
  }
};

struct PerplexityResult {
  double ppl = 0, ppl_se = 0;
  int64_t scored = 0;
  int chunks = 0;
  std::string device_name;
  std::string quantization;
  bool has_parity = false;
  PerplexityParity parity;
};

Result<PerplexityResult> run_perplexity(const PerplexityOptions& opts);

// bench_ppl's PARITY line payload: {"positions": ..., "top1": ..., ...}.
std::string parity_json(const PerplexityResult& r);

}  // namespace dynalm
