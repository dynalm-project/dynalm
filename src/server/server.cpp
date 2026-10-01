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

void send_error(httplib::Response& res, int status, std::string_view message, std::string_view type) {
  res.status = status;
  res.set_content(api::error_json(message, type), "application/json");
}

}  // namespace

struct Server::Impl {
  Engine& engine;
  ServerOptions opts;
  httplib::Server http;
  std::thread listener;
  int bound_port = 0;
  int64_t created = unix_now();
  std::atomic<bool> warned_sampling{false};

  // Metrics (spec §44).
  metrics::Counter requests_total, requests_failed, requests_cancelled;
  metrics::Gauge requests_active;
  metrics::Histogram ttft_ms{metrics::latency_buckets_ms()};
  metrics::Histogram itl_ms{metrics::latency_buckets_ms()};
  metrics::Histogram e2e_ms{metrics::latency_buckets_ms()};
  metrics::Counter prompt_tokens, completion_tokens;

  Impl(Engine& e, ServerOptions o) : engine(e), opts(std::move(o)) {}

  void setup();
  void completions(const httplib::Request& req, httplib::Response& res, bool chat);
  std::string render_metrics() const;
};

void Server::Impl::setup() {
  http.new_task_queue = [n = opts.http_threads] { return new httplib::ThreadPool(static_cast<size_t>(n)); };
  http.set_payload_max_length(opts.max_body_bytes);
  http.set_read_timeout(30, 0);
  http.set_write_timeout(30, 0);

  http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
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
  http.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (res.body.empty()) {
      res.set_content(api::error_json(res.status == 404 ? "not found" : "request error", "invalid_request_error"),
                      "application/json");
    }
  });
  http.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
    send_error(res, 500, "internal server error", "server_error");
  });
}

void Server::Impl::completions(const httplib::Request& req, httplib::Response& res, bool chat) {
  requests_total.inc();
  auto body = json::parse(req.body);
  if (!body.ok()) {
    requests_failed.inc();
    return send_error(res, 400, body.status().message(), "invalid_request_error");
  }
  auto parsed = chat ? api::parse_chat_request(*body, opts.default_max_tokens)
                     : api::parse_completion_request(*body, opts.default_max_tokens);
  if (!parsed.ok()) {
    requests_failed.inc();
    return send_error(res, 400, parsed.status().message(), "invalid_request_error");
  }
  api::CompletionRequest r = std::move(*parsed);
  if (r.sampling_requested && !warned_sampling.exchange(true)) {
    LOG_WARN("sampling parameters (temperature/top_p/...) are accepted but decoding is greedy in this version");
  }

  auto stream = chat ? engine.generate_chat(r.messages, r.params)
                     : engine.generate_text(r.prompt, /*parse_special=*/false, r.params);
  if (!stream.ok()) {
    requests_failed.inc();
    const int code = stream.status().code() == StatusCode::kInvalidArgument ||
                             stream.status().code() == StatusCode::kUnsupported
                         ? 400
                         : 500;
    return send_error(res, code, stream.status().message(), code == 400 ? "invalid_request_error" : "server_error");
  }
  std::shared_ptr<RequestStream> s = std::move(*stream);
  const std::string id = random_id(chat ? "chatcmpl" : "cmpl");
  const std::string model = r.model.empty() ? opts.model_id : r.model;
  const int64_t created_at = unix_now();
  const int64_t t0 = now_ns();
  requests_active.add(1);

  // Records one finished request's metrics.
  auto finish_metrics = [this, t0](const StreamEvent& last) {
    requests_active.add(-1);
    e2e_ms.observe(static_cast<double>(now_ns() - t0) * 1e-6);
    prompt_tokens.inc(static_cast<uint64_t>(last.prompt_tokens));
    completion_tokens.inc(static_cast<uint64_t>(last.completion_tokens));
    if (last.finish == StreamFinish::kError) requests_failed.inc();
    if (last.finish == StreamFinish::kCancelled) requests_cancelled.inc();
  };

  if (!r.stream) {
    std::string text;
    StreamEvent ev;
    bool first = true;
    while (s->next(ev)) {
      if (!ev.text.empty() && first) {
        ttft_ms.observe(static_cast<double>(now_ns() - t0) * 1e-6);
        first = false;
      }
      text += ev.text;
      if (ev.done) break;
    }
    finish_metrics(ev);
    if (!ev.done || ev.finish == StreamFinish::kError) {
      const bool client_error = ev.error.code() == StatusCode::kInvalidArgument;
      return send_error(res, client_error ? 400 : 500, ev.done ? ev.error.message() : "generation timed out",
                        client_error ? "invalid_request_error" : "server_error");
    }
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
      [this, s, id, model, created_at, chat, t0, usage_wanted = r.stream_usage, finish_metrics](
          size_t, httplib::DataSink& sink) {
        auto send = [&sink](const std::string& payload) {
          const std::string frame = "data: " + payload + "\n\n";
          return sink.write(frame.data(), frame.size());
        };
        bool first = true;
        int64_t last_ns = 0;
        StreamEvent ev;
        for (;;) {
          if (!s->next(ev, std::chrono::milliseconds(200))) {
            if (!sink.is_writable()) {  // client went away while we waited
              s->cancel();
              continue;
            }
            continue;
          }
          if (!ev.text.empty() || (first && chat)) {
            const int64_t now = now_ns();
            if (first) {
              ttft_ms.observe(static_cast<double>(now - t0) * 1e-6);
            } else {
              itl_ms.observe(static_cast<double>(now - last_ns) * 1e-6);
            }
            last_ns = now;
            const std::string chunk = chat ? api::chat_chunk_json(id, model, created_at, ev.text, first, StreamFinish::kNone)
                                           : api::text_chunk_json(id, model, created_at, ev.text, StreamFinish::kNone);
            first = false;
            if (!send(chunk)) {
              s->cancel();  // client disconnected: stop generating, free KV
              while (!ev.done && s->next(ev)) {
              }
              finish_metrics(ev);
              return false;
            }
          }
          if (ev.done) break;
        }
        finish_metrics(ev);
        if (ev.finish == StreamFinish::kError || ev.finish == StreamFinish::kCancelled) {
          send(api::error_json(ev.finish == StreamFinish::kError ? ev.error.message() : "request cancelled",
                               "server_error"));
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
  render_counter("engine_requests_total", "HTTP completion requests received", static_cast<double>(requests_total.value()), out);
  render_counter("engine_requests_failed_total", "Requests that ended in an error",
                 static_cast<double>(requests_failed.value()), out);
  render_counter("engine_requests_cancelled_total", "Requests cancelled (e.g. client disconnect)",
                 static_cast<double>(requests_cancelled.value()), out);
  render_gauge("engine_requests_active", "Requests currently being served", requests_active.value(), out);
  render_counter("engine_prompt_tokens_total", "Prompt tokens of completed requests",
                 static_cast<double>(prompt_tokens.value()), out);
  render_counter("engine_generation_tokens_total", "Tokens generated", static_cast<double>(st.scheduler.tokens_generated),
                 out);
  render_counter("engine_tokens_processed_total", "Token rows computed by the model (prefill + decode)",
                 static_cast<double>(st.scheduler.tokens_computed), out);
  render_gauge("engine_scheduler_running", "Sequences admitted", st.scheduler.running, out);
  render_gauge("engine_scheduler_waiting", "Sequences queued", st.scheduler.waiting, out);
  render_counter("engine_preemptions_total", "Sequences preempted under KV pressure",
                 static_cast<double>(st.scheduler.preemptions), out);
  render_gauge("engine_queue_latency_ms_avg", "Mean submission-to-admission latency",
               st.scheduler.admitted ? st.scheduler.queue_ms_total / static_cast<double>(st.scheduler.admitted) : 0.0, out);
  render_gauge("engine_kv_cache_used_blocks", "KV blocks in use", st.kv_blocks_used, out);
  render_gauge("engine_kv_cache_capacity_blocks", "KV blocks total", st.kv_blocks_total, out);
  render_gauge("engine_kv_cache_hit_rate", "Prefix-cache hit rate (reused / eligible prompt tokens)", st.prefix.hit_rate(),
               out);
  render_gauge("engine_prefix_cache_blocks", "KV blocks held by the prefix cache", static_cast<double>(st.prefix.cached_blocks),
               out);
  ttft_ms.render("engine_ttft_ms", "Time to first token (ms)", out);
  itl_ms.render("engine_itl_ms", "Inter-token latency of streamed responses (ms)", out);
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

void Server::stop() {
  if (!impl_) return;
  impl_->http.stop();
  if (impl_->listener.joinable()) impl_->listener.join();
}

}  // namespace engine
