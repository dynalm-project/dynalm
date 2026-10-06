// `dynalm serve <model> [options]`: OpenAI-compatible HTTP server.

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "dynacore/device/device_registry.h"
#include "cli/commands.h"
#include "config/config.h"
#include "registry/model_registry.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"
#include "runtime/engine.h"
#include "server/server.h"
#include "common/core.h"

namespace dynalm::cli {
namespace {

std::atomic<int> g_signals{0};
std::atomic<bool> g_shutdown_requested{false};
extern "C" void on_signal(int) { g_signals.fetch_add(1); }

constexpr OptionSpec kServeOptions[] = {
    {"model"},        {"host"},        {"port"},           {"model-id"},         {"threads"},
    {"ctx"},          {"batch"},       {"kv"},             {"http-threads"},     {"max-tokens"},
    {"max-active"},   {"request-timeout"}, {"temperature"}, {"backend"}, {"shutdown-timeout"}, {"disable-admin", false}, {"int8-decode"}, {"policy"},
};

void usage() {
  std::fprintf(stderr,
               "usage: dynalm serve <model> [options]\n"
               "  --config FILE           read options from FILE (key = value lines)\n"
               "  --host ADDR             bind address (default 127.0.0.1)\n"
               "  --port N                port (default 8000)\n"
               "  --model-id NAME         id reported by /v1/models (default: file name)\n"
               "  -t, --threads N|auto    compute threads (auto: physical cores)\n"
               "  -c, --ctx N|auto        KV cache capacity in tokens (auto: from free RAM)\n"
               "  --batch N|auto          max tokens per forward pass (auto: 256)\n"
               "  --kv f16|f32            KV cache dtype (default f16)\n"
               "  --int8-decode N         int8 activations for matmuls of <= N rows (default 4, 0 = off;\n"
               "                          off makes outputs independent of batching, DD-053)\n"
               "  --policy P              balanced (default) | latency | throughput: first-token wait vs\n"
               "                          smooth streaming under load (DD-061)\n"
               "  --backend cpu           compute backend (GPU backends are not built yet)\n"
               "  --max-active N          concurrent requests before 503 (default 64)\n"
               "  --http-threads N|auto   HTTP workers (auto: max-active + 8)\n"
               "  --max-tokens N          default max_tokens per request (default 1024)\n"
               "  --temperature T         default temperature when a request omits it (default 1.0)\n"
               "  --request-timeout S     per-request timeout in seconds, 0 = none (default 600)\n"
               "  --shutdown-timeout S    drain time for in-flight requests on stop (default 30)\n"
               "  --disable-admin         turn off POST /admin/shutdown (used by `dynalm stop`)\n"
               "Every option can also be set as DYNALM_<OPTION> (e.g. DYNALM_HTTP_THREADS=32)\n"
               "or in the config file; precedence: command line > environment > file.\n");
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

std::string gib(int64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

}  // namespace

int cmd_serve(std::span<const std::string_view> raw_args) {
  auto merged = merge_config(raw_args, kServeOptions);
  if (!merged.ok()) {
    std::fprintf(stderr, "serve: %s\n", merged.status().to_string().c_str());
    return 1;
  }
  const std::vector<std::string>& args = *merged;

  EngineOptions eo;
  ServerOptions so;
  int threads = 0, ctx = 0, batch = 0, http_threads = 0, max_tokens = 1024, max_active = 64;
  int request_timeout_s = 600, shutdown_timeout_s = 30;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? std::string_view(args[++i]) : std::string_view(); };
    bool ok = true;
    if (a == "--model") eo.model_path = value();
    else if (a == "--host") so.host = value();
    else if (a == "--port") ok = parse_int(value(), so.port);
    else if (a == "--model-id") so.model_id = value();
    else if (a == "-t" || a == "--threads") ok = parse_int_or_auto(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int_or_auto(value(), ctx);
    else if (a == "--batch") ok = parse_int_or_auto(value(), batch);
    else if (a == "--int8-decode") ok = parse_int(value(), eo.int8_decode_rows) && eo.int8_decode_rows >= 0;
    else if (a == "--policy") ok = parse_scheduler_policy(value(), eo.scheduler.policy);
    else if (a == "--http-threads") ok = parse_int_or_auto(value(), http_threads);
    else if (a == "--max-tokens") ok = parse_int(value(), max_tokens);
    else if (a == "--max-active") ok = parse_int(value(), max_active);
    else if (a == "--temperature") {
      const std::string v(value());
      char* end = nullptr;
      so.default_temperature = std::strtof(v.c_str(), &end);
      ok = !v.empty() && end == v.c_str() + v.size() && so.default_temperature >= 0.0f && so.default_temperature <= 2.0f;
    }
    else if (a == "--request-timeout") ok = parse_int(value(), request_timeout_s);
    else if (a == "--shutdown-timeout") ok = parse_int(value(), shutdown_timeout_s);
    else if (a == "--disable-admin") so.enable_admin = false;
    else if (a == "--backend") {
      auto k = parse_device_kind(value());
      ok = k.ok();
      if (ok) eo.backend = *k;
    } else if (a == "--kv") {
      const std::string_view v = value();
      ok = v == "f16" || v == "f32";
      eo.kv_dtype = v == "f32" ? DType::kF32 : DType::kF16;
    } else if (!a.starts_with("-")) eo.model_path = a;  // positional wins over file/env
    else ok = false;
    if (!ok) {
      std::fprintf(stderr, "serve: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (eo.model_path.empty() || max_tokens <= 0 || max_active <= 0 || request_timeout_s < 0 || shutdown_timeout_s < 0 ||
      so.port < 0 || so.port > 65535) {
    usage();
    return 1;
  }
  // A registry name is also the id clients see in /v1/models.
  if (const RegistryEntry* entry = is_model_name(eo.model_path) ? find_registry_entry(eo.model_path) : nullptr;
      entry != nullptr && so.model_id.empty()) {
    so.model_id = entry->name;
  }
  if (auto resolved = ensure_model(eo.model_path, true); resolved.ok()) {
    eo.model_path = *resolved;
  } else {
    std::fprintf(stderr, "serve: %s\n", resolved.status().message().c_str());
    return 1;
  }
  eo.threads = threads;
  eo.kv_tokens = ctx;
  eo.max_batch_tokens = batch > 0 ? batch : 256;
  so.http_threads = http_threads;
  so.max_active = max_active;
  so.default_max_tokens = max_tokens;
  so.request_timeout_ms = static_cast<int64_t>(request_timeout_s) * 1000;
  so.on_shutdown_request = [] { g_shutdown_requested.store(true); };

  auto eng = Engine::create(eo);
  if (!eng.ok()) {
    std::fprintf(stderr, "serve: %s\n", eng.status().to_string().c_str());
    return 1;
  }
  Engine& e = **eng;
  const LoadedModel& lm = e.model();
  const MemoryInfo mem = memory_info();
  LOG_INFO("Model: {} ({})", lm.config.name.empty() ? lm.config.architecture : lm.config.name, lm.architecture->name());
  LOG_INFO("Quantization: {}", lm.quantization);
  LOG_INFO("Device: {}", e.backend_name());
  LOG_INFO("Threads: {}", e.threads());
  LOG_INFO("RAM required: {} (weights {}, KV cache {}); available {}", gib(e.weight_bytes() + e.kv_bytes()),
           gib(e.weight_bytes()), gib(e.kv_bytes()), gib(mem.available_bytes));
  LOG_INFO("KV cache: {} tokens ({}{})", e.kv_capacity_tokens(), eo.kv_dtype == DType::kF32 ? "f32" : "f16",
           ctx == 0 ? ", auto" : "");
  LOG_INFO("Scheduler: continuous batching, {} policy (prefill budget {}, decode budget {}, max batch {})",
           scheduler_policy_name(eo.scheduler.policy), eo.scheduler.prefill_token_budget,
           eo.scheduler.decode_token_budget, eo.max_batch_tokens);
  LOG_INFO("Prefix cache: {}", eo.scheduler.enable_prefix_cache ? "enabled (radix)" : "disabled");

  Server server(e, so);
  if (Status st = server.start(); !st.ok()) {
    std::fprintf(stderr, "serve: %s\n", st.to_string().c_str());
    return 1;
  }
  LOG_INFO("Listening on http://{}:{} (max {} concurrent requests, timeout {} s)", so.host, server.port(), max_active,
           request_timeout_s);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  while (g_signals.load() == 0 && !g_shutdown_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Graceful drain: refuse new work, let in-flight requests finish, then stop.
  // A second signal skips the wait (remaining requests end as cancelled).
  server.begin_drain();
  const int signals_at_drain = g_signals.load();
  LOG_INFO("Draining {} in-flight request(s) (up to {} s; signal again to stop now)", server.active_requests(),
           shutdown_timeout_s);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(shutdown_timeout_s);
  while (server.active_requests() > 0 && std::chrono::steady_clock::now() < deadline &&
         g_signals.load() == signals_at_drain) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (server.active_requests() > 0) LOG_WARN("Stopping with {} request(s) still active", server.active_requests());
  LOG_INFO("Shutting down");
  server.stop();
  return 0;
}

}  // namespace dynalm::cli
