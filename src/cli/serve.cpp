// `engine serve <model> [options]`: OpenAI-compatible HTTP server.

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

#include "cli/commands.h"
#include "logging/log.h"
#include "runtime/engine.h"
#include "server/server.h"

namespace engine::cli {
namespace {

std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop.store(true); }

void usage() {
  std::fprintf(stderr,
               "usage: engine serve <model.gguf> [options]\n"
               "  --host ADDR           bind address (default 127.0.0.1)\n"
               "  --port N              port (default 8000)\n"
               "  --model-id NAME       id reported by /v1/models (default: file name)\n"
               "  -t, --threads N       compute threads (default: physical cores)\n"
               "  -c, --ctx N           KV cache capacity in tokens (default: min(context, 16384))\n"
               "  --batch N             max tokens per forward pass (default 256)\n"
               "  --kv f16|f32          KV cache dtype (default f16)\n"
               "  --http-threads N      HTTP worker threads (default 16)\n"
               "  --max-tokens N        default max_tokens per request (default 1024)\n");
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

}  // namespace

int cmd_serve(std::span<const std::string_view> args) {
  EngineOptions eo;
  ServerOptions so;
  int threads = 0, ctx = 0, batch = 256, http_threads = 16, max_tokens = 1024;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "--host") so.host = value();
    else if (a == "--port") ok = parse_int(value(), so.port);
    else if (a == "--model-id") so.model_id = value();
    else if (a == "-t" || a == "--threads") ok = parse_int(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int(value(), ctx);
    else if (a == "--batch") ok = parse_int(value(), batch);
    else if (a == "--http-threads") ok = parse_int(value(), http_threads);
    else if (a == "--max-tokens") ok = parse_int(value(), max_tokens);
    else if (a == "--kv") {
      const std::string_view v = value();
      ok = v == "f16" || v == "f32";
      eo.kv_dtype = v == "f32" ? DType::kF32 : DType::kF16;
    } else if (eo.model_path.empty() && !a.starts_with("-")) eo.model_path = a;
    else ok = false;
    if (!ok) {
      std::fprintf(stderr, "serve: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (eo.model_path.empty() || threads < 0 || ctx < 0 || batch <= 0 || http_threads <= 0 || max_tokens <= 0) {
    usage();
    return 1;
  }
  eo.threads = threads;
  eo.kv_tokens = ctx;
  eo.max_batch_tokens = batch;
  so.http_threads = http_threads;
  so.default_max_tokens = max_tokens;

  auto eng = Engine::create(eo);
  if (!eng.ok()) {
    std::fprintf(stderr, "serve: %s\n", eng.status().to_string().c_str());
    return 1;
  }
  const LoadedModel& lm = (*eng)->model();
  LOG_INFO("Model: {} ({}, {})", lm.config.name.empty() ? lm.config.architecture : lm.config.name,
           lm.architecture->name(), lm.quantization);
  LOG_INFO("Backend: {}, threads: {}", (*eng)->backend_name(), (*eng)->threads());
  LOG_INFO("KV cache: {} tokens, prefix cache: radix", (*eng)->kv_geometry().num_blocks * (*eng)->kv_geometry().block_size);
  LOG_INFO("Scheduler: continuous batching (prefill budget {}, decode budget {})", eo.scheduler.prefill_token_budget,
           eo.scheduler.decode_token_budget);

  Server server(**eng, so);
  if (Status st = server.start(); !st.ok()) {
    std::fprintf(stderr, "serve: %s\n", st.to_string().c_str());
    return 1;
  }
  LOG_INFO("Listening on http://{}:{} (OpenAI-compatible: /v1/chat/completions, /v1/completions)", so.host,
           server.port());

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
  LOG_INFO("Shutting down");
  server.stop();
  return 0;
}

}  // namespace engine::cli
