// `engine run <model> -p <prompt> [options]`: single-sequence generation.

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <string>

#include "backends/cpu/cpu_backend.h"
#include "cli/commands.h"
#include "common/timer.h"
#include "loader/model_loader.h"
#include "logging/log.h"
#include "model/transformer.h"
#include "platform/cpu_info.h"
#include "runtime/generator.h"
#include "runtime/sequence.h"
#include "runtime/thread_pool.h"

namespace engine::cli {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

void usage() {
  std::fprintf(stderr,
               "usage: engine run <model.gguf> -p <prompt> [options]\n"
               "  -p, --prompt TEXT     prompt text\n"
               "  -n, --max-tokens N    tokens to generate (default 128)\n"
               "  --chat                wrap the prompt in the model's chat template (default)\n"
               "  --raw                 use the prompt as-is (no chat template)\n"
               "  --system TEXT         system message (chat mode)\n"
               "  -t, --threads N       worker threads (default: physical cores)\n"
               "  -c, --ctx N           KV cache capacity in tokens (default 4096)\n"
               "  --batch N             max tokens per forward pass (default 256)\n"
               "  --kv f16|f32          KV cache dtype (default f16)\n"
               "  --kernel ISA          force kernel tier (generic, avx2, ...)\n"
               "  --no-stream           print only the final text\n");
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

}  // namespace

int cmd_run(std::span<const std::string_view> args) {
  std::string path, prompt, system;
  int max_tokens = 128, threads = cpu_info().physical_cores, ctx = 4096, batch = 256;
  bool chat = true, stream = true;
  DType kv_dtype = DType::kF16;
  CpuIsa isa = select_best_isa(cpu_info().features);

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "-p" || a == "--prompt") prompt = value();
    else if (a == "--system") system = value();
    else if (a == "-n" || a == "--max-tokens") ok = parse_int(value(), max_tokens);
    else if (a == "-t" || a == "--threads") ok = parse_int(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int(value(), ctx);
    else if (a == "--batch") ok = parse_int(value(), batch);
    else if (a == "--chat") chat = true;
    else if (a == "--raw") chat = false;
    else if (a == "--no-stream") stream = false;
    else if (a == "--kv") {
      const std::string_view v = value();
      ok = v == "f16" || v == "f32";
      kv_dtype = v == "f32" ? DType::kF32 : DType::kF16;
    } else if (a == "--kernel") ok = parse_isa(value(), isa);
    else if (path.empty() && !a.starts_with("-")) path = a;
    else ok = false;
    if (!ok) {
      std::fprintf(stderr, "run: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (path.empty() || prompt.empty() || max_tokens <= 0 || threads <= 0 || ctx <= 0 || batch <= 0) {
    usage();
    return 1;
  }

  const Stopwatch load_timer;
  auto model = load_model(path);
  if (!model.ok()) {
    std::fprintf(stderr, "run: %s\n", model.status().to_string().c_str());
    return 1;
  }
  LoadedModel& lm = **model;
  const ModelConfig& cfg = lm.config;
  ctx = static_cast<int>(std::min<int64_t>(ctx, cfg.context_length));

  ThreadPool pool(threads);
  CpuBackend backend(pool, isa);

  const KvGeometry geom = kv_geometry_for(cfg, kv_dtype, /*block_size=*/16, ctx);
  auto cache = KvCache::create(geom, backend);
  if (!cache.ok()) {
    std::fprintf(stderr, "run: %s\n", cache.status().to_string().c_str());
    return 1;
  }
  auto transformer = Transformer::create(cfg, lm.weights, backend, batch);
  if (!transformer.ok()) {
    std::fprintf(stderr, "run: %s\n", transformer.status().to_string().c_str());
    return 1;
  }

  LOG_INFO("Model: {} ({}, {})", cfg.name.empty() ? cfg.architecture : cfg.name, lm.architecture->name(),
           lm.quantization);
  LOG_INFO("Backend: {}, threads: {}", backend.name(), pool.size());
  LOG_INFO("Weights: {:.1f} MiB mmapped, KV cache: {:.1f} MiB ({} tokens, {})", lm.weight_bytes / kMiB,
           geom.total_bytes() / kMiB, ctx, dtype_name(kv_dtype));
  LOG_INFO("Load time: {:.1f} ms", load_timer.elapsed_ms());

  // Build the prompt.
  std::vector<TokenId> tokens;
  if (chat) {
    if (!lm.chat_template) {
      std::fprintf(stderr, "run: model has no recognized chat template; use --raw\n");
      return 1;
    }
    std::vector<ChatMessage> msgs;
    if (!system.empty()) msgs.push_back({"system", system});
    msgs.push_back({"user", prompt});
    auto text = lm.chat_template->apply(msgs, true);
    if (!text.ok()) {
      std::fprintf(stderr, "run: %s\n", text.status().to_string().c_str());
      return 1;
    }
    tokens = lm.tokenizer->encode(*text, /*add_special=*/true, /*parse_special=*/true);
  } else {
    tokens = lm.tokenizer->encode(prompt, /*add_special=*/true, /*parse_special=*/false);
  }

  Generator gen(**transformer, **cache, *lm.tokenizer);
  GenerateOptions opts;
  opts.max_new_tokens = max_tokens;
  GenerationStats stats;
  Utf8Buffer utf8;
  std::string piece, text_out, printable;
  const Status st = gen.generate(tokens, opts, [&](TokenId id) {
    if (lm.tokenizer->is_eog(id)) return true;
    piece.clear();
    lm.tokenizer->decode_token(id, piece);
    printable.clear();
    utf8.push(piece, printable);
    text_out += printable;
    if (stream) {
      std::fwrite(printable.data(), 1, printable.size(), stdout);
      std::fflush(stdout);
    }
    return true;
  }, &stats);
  printable.clear();
  utf8.flush(printable);
  text_out += printable;
  if (stream) {
    std::fwrite(printable.data(), 1, printable.size(), stdout);
    std::printf("\n");
  } else {
    std::printf("%s\n", text_out.c_str());
  }
  if (!st.ok()) {
    std::fprintf(stderr, "run: %s\n", st.to_string().c_str());
    return 1;
  }

  std::vector<double> itl = stats.itl_ms;
  std::sort(itl.begin(), itl.end());
  auto pct = [&](double p) {
    return itl.empty() ? 0.0 : itl[std::min(itl.size() - 1, static_cast<size_t>(p / 100.0 * itl.size()))];
  };
  std::fprintf(stderr,
               "\n[stats] prompt %d tok, %.1f ms (%.1f tok/s) | TTFT %.1f ms | generated %d tok, %.1f tok/s | "
               "ITL p50 %.1f p90 %.1f p99 %.1f ms\n",
               stats.prompt_tokens, stats.prefill_ms, stats.prefill_tok_per_s(), stats.ttft_ms,
               stats.generated_tokens, stats.decode_tok_per_s(), pct(50), pct(90), pct(99));
  return 0;
}

}  // namespace engine::cli
