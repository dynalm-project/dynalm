#include "server/server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

// cpp-httplib (pinned single header, fetched by CMake). Kept private to this TU.
#include "httplib.h"

#include "api/json.h"
#include "api/openai.h"
#include "common/timer.h"
#include "common/version.h"
#include "logging/log.h"
#include "metrics/metrics.h"

namespace engine {
namespace {

int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string random_id(const char* prefix) {
  static std::atomic<uint64_t> counter{0};
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%s-%llx%04llx", prefix, static_cast<unsigned long long>(now_ns() & 0xffffffffffull),
                static_cast<unsigned long long>(counter.fetch_add(1) & 0xffff));
  return buf;
}

std::string_view error_type(int status) {
  switch (status) {
    case 400:
    case 403:
    case 404: return "invalid_request_error";
    case 503: return "overloaded_error";
    case 504: return "timeout_error";
    default: return "server_error";
  }
}

void send_error(httplib::Response& res, int status, std::string_view message) {
  res.status = status;
  res.set_content(api::error_json(message, error_type(status)), "application/json");
}

bool is_loopback(const std::string& addr) {
  return addr == "127.0.0.1" || addr == "::1" || addr == "::ffff:127.0.0.1" || addr.starts_with("127.");
}

}  // namespace

int http_status_for(StatusCode code) {
  switch (code) {
    case StatusCode::kInvalidArgument:
    case StatusCode::kUnsupported: return 400;
    case StatusCode::kResourceExhausted:
    case StatusCode::kCancelled: return 503;
    case StatusCode::kDeadlineExceeded: return 504;
    default: return 500;
  }
}

Status request_server_shutdown(const std::string& host, int port) {
  httplib::Client cli(host, port);
  cli.set_connection_timeout(5, 0);
  cli.set_read_timeout(10, 0);
  auto res = cli.Post("/admin/shutdown");
  if (!res) return IoError("no server at " + host + ":" + std::to_string(port) + " (" + httplib::to_string(res.error()) + ")");
  if (res->status != 202) {
    return InvalidArgument("server refused shutdown (HTTP " + std::to_string(res->status) + "): " + res->body);
  }
  return Status::Ok();
}

struct Server::Impl {
  Engine& engine;
  ServerOptions opts;
  httplib::Server http;
  std::thread listener;
  int bound_port = 0;
  int64_t created = unix_now();
  std::atomic<bool> draining{false};
  std::atomic<int> active{0};

  // Metrics (spec §44).
  metrics::Counter requests_total, requests_failed, requests_cancelled, requests_rejected;
  metrics::Histogram ttft_ms{metrics::latency_buckets_ms()};
  metrics::Histogram itl_ms{metrics::latency_buckets_ms()};
  metrics::Histogram tpot_ms{metrics::latency_buckets_ms()};
  metrics::Histogram e2e_ms{metrics::latency_buckets_ms()};
  metrics::Counter prompt_tokens, completion_tokens;

  Impl(Engine& e, ServerOptions o) : engine(e), opts(std::move(o)) {}

  // One admitted completion request. Its destructor runs when the response
  // (and any streaming provider holding it) is released, whatever path the
  // request took — so the admission slot and the engine request are never
  // leaked, even if the client vanished before the provider ever ran.
  struct Inflight {
    Impl& srv;
    std::shared_ptr<RequestStream> stream;
    int64_t t0 = now_ns();
    int64_t first_ns = 0;
    bool finished = false;

    Inflight(Impl& s, std::shared_ptr<RequestStream> st) : srv(s), stream(std::move(st)) {}
    ~Inflight() {
      if (!finished) {
        stream->cancel();  // frees its KV on the next scheduler step
        srv.requests_cancelled.inc();
      }
      srv.active.fetch_sub(1);
    }
    void on_token(int64_t now) {
      if (first_ns == 0) {
        first_ns = now;
        srv.ttft_ms.observe(static_cast<double>(now - t0) * 1e-6);
      }
    }
    void finish(const StreamEvent& last) {
      finished = true;
      const int64_t now = now_ns();
      srv.e2e_ms.observe(static_cast<double>(now - t0) * 1e-6);
      if (first_ns != 0 && last.completion_tokens > 1) {
        srv.tpot_ms.observe(static_cast<double>(now - first_ns) * 1e-6 / (last.completion_tokens - 1));
      }
      srv.prompt_tokens.inc(static_cast<uint64_t>(last.prompt_tokens));
      srv.completion_tokens.inc(static_cast<uint64_t>(last.completion_tokens));
      if (last.finish == StreamFinish::kError) srv.requests_failed.inc();
      if (last.finish == StreamFinish::kCancelled) srv.requests_cancelled.inc();
    }
  };

  void setup();
  void completions(const httplib::Request& req, httplib::Response& res, bool chat);
  std::string render_metrics() const;
};

void Server::Impl::setup() {
  const int workers = opts.http_threads > 0 ? opts.http_threads : opts.max_active + 8;
  http.new_task_queue = [workers] { return new httplib::ThreadPool(static_cast<size_t>(workers)); };
  http.set_payload_max_length(opts.max_body_bytes);
  http.set_read_timeout(30, 0);
  http.set_write_timeout(30, 0);

  http.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    if (draining.load()) {
      res.status = 503;
      res.set_content(R"({"status":"draining"})", "application/json");
      return;
    }
    res.set_content(R"({"status":"ok"})", "application/json");
  });
  http.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(api::models_json(opts.model_id, created), "application/json");
  });
  http.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(render_metrics(), "text/plain; version=0.0.4");
  });
  http.Post("/v1/chat/completions",
            [this](const httplib::Request& req, httplib::Response& res) { completions(req, res, true); });
  http.Post("/v1/completions",
            [this](const httplib::Request& req, httplib::Response& res) { completions(req, res, false); });
  http.Post("/admin/shutdown", [this](const httplib::Request& req, httplib::Response& res) {
    if (!opts.enable_admin) return send_error(res, 404, "not found");
    if (!is_loopback(req.remote_addr)) return send_error(res, 403, "admin endpoints are loopback-only");
    LOG_INFO("shutdown requested via /admin/shutdown");
    draining.store(true);
    if (opts.on_shutdown_request) opts.on_shutdown_request();
    res.status = 202;
    res.set_content(R"({"status":"draining"})", "application/json");
  });
  http.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.body.empty()) {
      res.set_content(api::error_json(res.status == 404 ? "not found" : "request error", "invalid_request_error"),
                      "application/json");
    }
  });
  http.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
    send_error(res, 500, "internal server error");
  });
}

void Server::Impl::completions(const httplib::Request& req, httplib::Response& res, bool chat) {
  requests_total.inc();
  if (draining.load()) {
    requests_rejected.inc();
    res.set_header("Retry-After", "5");
    return send_error(res, 503, "server is shutting down");
  }
  if (active.fetch_add(1) >= opts.max_active) {
    active.fetch_sub(1);
    requests_rejected.inc();
    res.set_header("Retry-After", "1");
    return send_error(res, 503, "server overloaded: too many concurrent requests");
  }
  bool admitted = false;  // until an Inflight owns the slot
  struct SlotGuard {
    std::atomic<int>& a;
    bool& owned;
    ~SlotGuard() {
      if (!owned) a.fetch_sub(1);
    }
  } guard{active, admitted};

  auto body = json::parse(req.body);
  if (!body.ok()) {
    requests_failed.inc();
    return send_error(res, 400, body.status().message());
  }
  auto parsed = chat ? api::parse_chat_request(*body, opts.default_max_tokens, opts.default_temperature)
                     : api::parse_completion_request(*body, opts.default_max_tokens, opts.default_temperature);
  if (!parsed.ok()) {
    requests_failed.inc();
    return send_error(res, 400, parsed.status().message());
  }
  api::CompletionRequest r = std::move(*parsed);
  if (r.params.timeout_ms <= 0) r.params.timeout_ms = opts.request_timeout_ms;

  auto stream = chat ? engine.generate_chat(r.messages, r.params)
                     : engine.generate_text(r.prompt, /*parse_special=*/false, r.params);
  if (!stream.ok()) {
    requests_failed.inc();
    return send_error(res, http_status_for(stream.status().code()), stream.status().message());
  }
  auto inflight = std::make_shared<Inflight>(*this, std::move(*stream));
  admitted = true;
  const std::string id = random_id(chat ? "chatcmpl" : "cmpl");
  const std::string model = r.model.empty() ? opts.model_id : r.model;
  const int64_t created_at = unix_now();

  if (!r.stream) {
    std::string text;
    StreamEvent ev;
    // Bounded wait: the request's own timeout ends it; this only guards
    // against a wedged engine.
    while (inflight->stream->next(ev, std::chrono::milliseconds(600000))) {
      if (!ev.text.empty()) inflight->on_token(now_ns());
      text += ev.text;
      if (ev.done) break;
    }
    if (!ev.done) return send_error(res, 504, "generation timed out");
    inflight->finish(ev);
    if (ev.finish == StreamFinish::kError) {
      return send_error(res, http_status_for(ev.error.code()), ev.error.message());
    }
    if (ev.finish == StreamFinish::kCancelled) return send_error(res, 503, "request cancelled");
    const api::Usage usage{ev.prompt_tokens, ev.completion_tokens};
    res.set_content(chat ? api::chat_completion_json(id, model, created_at, text, ev.finish, usage)
                         : api::text_completion_json(id, model, created_at, text, ev.finish, usage),
                    "application/json");
    return;
  }

  // Streaming: Server-Sent Events, one chunk per text delta.
  res.set_header("Cache-Control", "no-cache");
  res.set_header("X-Accel-Buffering", "no");
  res.set_chunked_content_provider(
      "text/event-stream",
      [this, inflight, id, model, created_at, chat, usage_wanted = r.stream_usage](size_t, httplib::DataSink& sink) {
        auto send = [&sink](const std::string& payload) {
          const std::string frame = "data: " + payload + "\n\n";
          return sink.write(frame.data(), frame.size());
        };
        RequestStream& s = *inflight->stream;
        bool first = true;
        int64_t last_ns = 0;
        StreamEvent ev;
        for (;;) {
          if (!s.next(ev, std::chrono::milliseconds(200))) {
            if (!sink.is_writable()) return false;  // client went away; ~Inflight cancels
            continue;
          }
          if (!ev.text.empty() || (first && chat)) {
            const int64_t now = now_ns();
            if (!ev.text.empty()) {
              if (inflight->first_ns != 0) itl_ms.observe(static_cast<double>(now - last_ns) * 1e-6);
              inflight->on_token(now);
              last_ns = now;
            }
            const std::string chunk = chat ? api::chat_chunk_json(id, model, created_at, ev.text, first, StreamFinish::kNone)
                                           : api::text_chunk_json(id, model, created_at, ev.text, StreamFinish::kNone);
            first = false;
            if (!send(chunk)) return false;  // client disconnected; ~Inflight cancels, KV freed
          }
          if (ev.done) break;
        }
        inflight->finish(ev);
        if (ev.finish == StreamFinish::kError || ev.finish == StreamFinish::kCancelled) {
          const int status = ev.finish == StreamFinish::kError ? http_status_for(ev.error.code()) : 503;
          send(api::error_json(ev.finish == StreamFinish::kError ? ev.error.message() : "request cancelled",
                               error_type(status)));
        } else {
          send(chat ? api::chat_chunk_json(id, model, created_at, "", false, ev.finish)
                    : api::text_chunk_json(id, model, created_at, "", ev.finish));
          if (usage_wanted) {
            send(api::usage_chunk_json(id, model, created_at, chat, {ev.prompt_tokens, ev.completion_tokens}));
          }
        }
        send("[DONE]");
        sink.done();
        return true;
      });
}

std::string Server::Impl::render_metrics() const {
  const EngineStats st = engine.stats();
  std::string out;
  using metrics::render_counter;
  using metrics::render_gauge;
  auto counter = [&](std::string_view name, std::string_view help, uint64_t v) {
    render_counter(name, help, static_cast<double>(v), out);
  };
  counter("engine_requests_total", "HTTP completion requests received", requests_total.value());
  counter("engine_requests_failed_total", "Requests that ended in an error", requests_failed.value());
  counter("engine_requests_cancelled_total", "Requests cancelled (e.g. client disconnect)", requests_cancelled.value());
  counter("engine_requests_rejected_total", "Requests refused with 503 (overload or draining)", requests_rejected.value());
  counter("engine_requests_timed_out_total", "Requests that hit their timeout", st.scheduler.timed_out);
  render_gauge("engine_requests_active", "Requests currently being served", active.load(), out);
  counter("engine_prompt_tokens_total", "Prompt tokens of completed requests", prompt_tokens.value());
  counter("engine_generation_tokens_total", "Tokens generated", st.scheduler.tokens_generated);
  counter("engine_prefill_tokens_total", "Prompt tokens computed (excludes prefix-cache hits)", st.prefill_tokens);
  counter("engine_tokens_processed_total", "Token rows computed by the model (prefill + decode)",
          st.scheduler.tokens_computed);
  render_gauge("engine_generation_tokens_per_second", "Generation throughput over the last busy second",
               st.generation_tok_s, out);
  render_gauge("engine_prefill_tokens_per_second", "Prefill throughput over the last busy second", st.prefill_tok_s,
               out);
  render_gauge("engine_scheduler_running", "Sequences admitted", st.scheduler.running, out);
  render_gauge("engine_scheduler_waiting", "Sequences queued", st.scheduler.waiting, out);
  counter("engine_preemptions_total", "Sequences preempted under KV pressure", st.scheduler.preemptions);
  render_gauge("engine_queue_latency_ms_avg", "Mean submission-to-admission latency",
               st.scheduler.admitted ? st.scheduler.queue_ms_total / static_cast<double>(st.scheduler.admitted) : 0.0, out);
  render_gauge("engine_queue_latency_ms_max", "Max submission-to-admission latency", st.scheduler.queue_ms_max, out);
  render_gauge("engine_kv_cache_used_blocks", "KV blocks in use", st.kv_blocks_used, out);
  render_gauge("engine_kv_cache_capacity_blocks", "KV blocks total", st.kv_blocks_total, out);
  render_gauge("engine_kv_cache_capacity_tokens", "KV capacity in tokens", static_cast<double>(engine.kv_capacity_tokens()),
               out);
  render_gauge("engine_kv_cache_hit_rate", "Prefix-cache hit rate (reused / eligible prompt tokens)", st.prefix.hit_rate(),
               out);
  render_gauge("engine_prefix_cache_blocks", "KV blocks held by the prefix cache", static_cast<double>(st.prefix.cached_blocks),
               out);
  engine.step_ms().render("engine_scheduler_step_ms", "Scheduler step (batched forward pass) latency (ms)", out);
  ttft_ms.render("engine_ttft_ms", "Time to first token (ms)", out);
  itl_ms.render("engine_itl_ms", "Inter-token latency of streamed responses (ms)", out);
  tpot_ms.render("engine_tpot_ms", "Time per output token after the first, per request (ms)", out);
  e2e_ms.render("engine_e2e_latency_ms", "End-to-end request latency (ms)", out);
  return out;
}

Server::Server(Engine& engine, ServerOptions options) : impl_(std::make_unique<Impl>(engine, std::move(options))) {
  if (impl_->opts.model_id.empty()) {
    impl_->opts.model_id = std::filesystem::path(engine.model().path).stem().string();
  }
  impl_->setup();
}

Server::~Server() { stop(); }

Status Server::start() {
  Impl& s = *impl_;
  if (s.opts.port == 0) {
    s.bound_port = s.http.bind_to_any_port(s.opts.host);
  } else {
    s.bound_port = s.http.bind_to_port(s.opts.host, s.opts.port) ? s.opts.port : -1;
  }
  if (s.bound_port <= 0) return IoError("cannot bind " + s.opts.host + ":" + std::to_string(s.opts.port));
  s.listener = std::thread([&s] { s.http.listen_after_bind(); });
  return Status::Ok();
}

int Server::port() const { return impl_->bound_port; }

void Server::begin_drain() { impl_->draining.store(true); }
bool Server::draining() const { return impl_->draining.load(); }
int Server::active_requests() const { return impl_->active.load(); }

void Server::stop() {
  if (!impl_) return;
  impl_->http.stop();
  if (impl_->listener.joinable()) impl_->listener.join();
}

}  // namespace engine
