// Perplexity over a text file, scored exactly as llama.cpp's llama-perplexity
// does, so the two numbers compare directly (same file, same context):
//
//   bench_ppl <model.gguf> <text file> [ctx 512] [chunks 0 = all] [threads]
//   BENCH_DEVICE=native (default) | ggml-cpu | metal | vulkan | cuda
//   BENCH_UBATCH=n: score each chunk in batches of n tokens (llama-perplexity -ub n)
//   BENCH_FRESH_KV=0: reuse one KV pool for all chunks (default: a new pool per
//     chunk, as llama-perplexity clears its cache; a reused pool hands blocks back
//     newest first, so attention adds the same cells in another order)
//   BENCH_TOKENS=1: print the text's token ids instead (tokenizer comparisons)
//   BENCH_INT8_ROWS=n: native CPU int8 activation row limit (0 = fp32 reference)
//   BENCH_KLD_BASE=<file>: also write DynaLM's scored log-probabilities in
//     llama-perplexity's --kl-divergence-base format, so that
//       llama-perplexity -m <model.gguf> -c <ctx> --kl-divergence-base <file> --kl-divergence
//     runs llama.cpp on the same token ids (read from the file) and reports
//     its KL divergence, "Same top p" (top-1 agreement) and delta-p against DynaLM.
//
//   llama-perplexity -m <model.gguf> -f <text file> -c <ctx> [--chunks N]
//
// The text is tokenized once (with BOS when the model adds one) and cut into
// chunks of `ctx` tokens; each chunk starts with BOS (when the model uses one)
// and runs as one batch from position 0; the tokens of its second half are
// scored (logits at j predict token j + 1, j from ctx/2 to ctx - 2). The
// device runs its default plan, as the engine does (on the native CPU device:
// DynaCore's DD-053 kernels). Prints the running estimate per chunk, as
// llama-perplexity does, and the final estimate with its standard error.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "bench/perplexity.h"
#include "common/core.h"
#include "loader/model_loader.h"

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_ppl <model.gguf> <text file> [ctx 512] [chunks 0 = all] [threads]\n");
    return 1;
  }
  // BENCH_TOKENS=1: print the file's token ids (no BOS) and exit, to compare
  // tokenizers (llama-server: POST /tokenize {"content": ..., "add_special": false}).
  if (const char* t = std::getenv("BENCH_TOKENS"); t != nullptr && t[0] == '1') {
    std::ifstream f(argv[2], std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "cannot read %s\n", argv[2]);
      return 1;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    auto lm = load_model(argv[1]);
    if (!lm.ok()) {
      std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
      return 1;
    }
    for (TokenId id : (*lm)->tokenizer->encode(ss.str(), /*add_special=*/false, false)) std::printf("%d ", id);
    std::printf("\n");
    return 0;
  }
  PerplexityOptions o;
  o.model_path = argv[1];
  o.text_path = argv[2];
  o.ctx = argc > 3 ? std::atoi(argv[3]) : 512;
  o.max_chunks = argc > 4 ? std::atoi(argv[4]) : 0;
  o.threads = argc > 5 ? std::atoi(argv[5]) : 0;
  if (const char* v = std::getenv("BENCH_DEVICE"); v != nullptr) o.device = v;
  if (const char* v = std::getenv("BENCH_UBATCH"); v != nullptr) o.ubatch = std::atoi(v);
  if (const char* v = std::getenv("BENCH_FRESH_KV"); v != nullptr) o.fresh_kv = std::string_view(v) != "0";
  if (const char* v = std::getenv("BENCH_INT8_ROWS"); v != nullptr && *v != '\0') o.int8_rows = static_cast<int32_t>(std::atol(v));
  if (const char* v = std::getenv("BENCH_KLD_REF"); v != nullptr) o.kld_ref = v;
  if (const char* v = std::getenv("BENCH_KLD_BASE"); v != nullptr) o.kld_base = v;
  o.log = stdout;
  auto r = run_perplexity(o);
  if (!r.ok()) {
    std::fprintf(stderr, "%s\n", r.status().message().c_str());
    return 1;
  }
  return 0;
}
