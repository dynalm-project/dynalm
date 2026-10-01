// `engine run <model> -p <prompt> [options]`: streamed generation through the
// Engine (the same path the HTTP server uses).

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <string>

#include "cli/commands.h"
#include "common/timer.h"
#include "logging/log.h"
#include "platform/cpu_info.h"
#include "runtime/engine.h"

namespace engine::cli {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

void usage() {
  std::fprintf(stderr,
               "usage: engine run <model> -p <prompt> [options]\n"
               "  -p, --prompt TEXT     prompt text\n"
               "  -n, --max-tokens N    tokens to generate (default 128)\n"
               "  --chat                wrap the prompt in the model's chat template (default)\n"
               "  --raw                 use the prompt as-is (no chat template)\n"
               "  --system TEXT         system message (chat mode)\n"
               "  --stop TEXT           stop string (repeatable; never printed)\n"
               "  -t, --threads N       worker threads (default: physical cores)\n"
               "  -c, --ctx N           KV cache capacity in tokens (default 4096)\n"
               "  --batch N             max tokens per forward pass (default 256)\n"
               "  --kv f16|f32          KV cache dtype (default f16)\n"
               "  --no-stream           print only the final text\n");
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

}  // namespace

int cmd_run(std::span<const std::string_view> args) {
  std::string path, prompt, system;
  int max_tokens = 128, threads = 0, ctx = 4096, batch = 256;
  bool chat = true, stream = true;
  EngineOptions opts;
  GenerateParams params;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "-p" || a == "--prompt") prompt = value();
    else if (a == "--system") system = value();
    else if (a == "--stop") params.stop.emplace_back(value());
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
      opts.kv_dtype = v == "f32" ? DType::kF32 : DType::kF16;
    } else if (path.empty() && !a.starts_with("-")) path = a;
    else ok = false;
    if (!ok) {
      std::fprintf(stderr, "run: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (path.empty() || prompt.empty() || max_tokens <= 0 || threads < 0 || ctx <= 0 || batch <= 0) {
    usage();
    return 1;
  }

  const Stopwatch load_timer;
  opts.model_path = path;
  opts.threads = threads;
  opts.kv_tokens = ctx;
  opts.max_batch_tokens = batch;
  auto eng = Engine::create(opts);
  if (!eng.ok()) {
    std::fprintf(stderr, "run: %s\n", eng.status().to_string().c_str());
    return 1;
  }
  Engine& e = **eng;
  const LoadedModel& lm = e.model();
  const ModelConfig& cfg = lm.config;
  LOG_INFO("Model: {} ({}, {})", cfg.name.empty() ? cfg.architecture : cfg.name, lm.architecture->name(),
           lm.quantization);
  LOG_INFO("Backend: {}, threads: {}", e.backend_name(), e.threads());
  LOG_INFO("Weights: {:.1f} MiB mmapped, KV cache: {:.1f} MiB ({} tokens, {})", lm.weight_bytes / kMiB,
           e.kv_geometry().total_bytes() / kMiB, ctx, dtype_name(opts.kv_dtype));
  LOG_INFO("Load time: {:.1f} ms", load_timer.elapsed_ms());

  params.max_tokens = max_tokens;
  const Stopwatch request_timer;
  Result<std::shared_ptr<RequestStream>> s = InvalidArgument("unreachable");
  if (chat) {
    std::vector<ChatMessage> msgs;
    if (!system.empty()) msgs.push_back({"system", system});
    msgs.push_back({"user", prompt});
    s = e.generate_chat(msgs, params);
  } else {
    s = e.generate_text(prompt, /*parse_special=*/false, params);
  }
  if (!s.ok()) {
    std::fprintf(stderr, "run: %s\n", s.status().to_string().c_str());
    return 1;
  }

  std::string text;
  StreamEvent ev;
  double ttft_ms = -1, last_ms = 0;
  std::vector<double> gaps;
  while ((*s)->next(ev)) {
    if (!ev.text.empty()) {
      const double now = request_timer.elapsed_ms();
      if (ttft_ms < 0) {
        ttft_ms = now;
      } else {
        gaps.push_back(now - last_ms);
      }
      last_ms = now;
      if (stream) {
        std::fwrite(ev.text.data(), 1, ev.text.size(), stdout);
        std::fflush(stdout);
      } else {
        text += ev.text;
      }
    }
    if (ev.done) break;
  }
  if (stream) {
    std::printf("\n");
  } else {
    std::printf("%s\n", text.c_str());
  }
  if (ev.finish == StreamFinish::kError) {
    std::fprintf(stderr, "run: %s\n", ev.error.to_string().c_str());
    return 1;
  }

  const double total_ms = request_timer.elapsed_ms();
  std::sort(gaps.begin(), gaps.end());
  auto pct = [&](double p) {
    return gaps.empty() ? 0.0 : gaps[std::min(gaps.size() - 1, static_cast<size_t>(p / 100.0 * gaps.size()))];
  };
  const double decode_s = (total_ms - std::max(ttft_ms, 0.0)) / 1e3;
  std::fprintf(stderr,
               "\n[stats] prompt %d tok | TTFT %.1f ms | generated %d tok, %.1f tok/s | finish %s | "
               "inter-delta p50 %.1f p90 %.1f p99 %.1f ms\n",
               ev.prompt_tokens, ttft_ms, ev.completion_tokens,
               decode_s > 0 ? (ev.completion_tokens - 1) / decode_s : 0.0,
               std::string(stream_finish_name(ev.finish)).c_str(), pct(50), pct(90), pct(99));
  return 0;
}

}  // namespace engine::cli
