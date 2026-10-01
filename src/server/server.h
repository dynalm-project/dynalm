#pragma once

// OpenAI-compatible HTTP server over an Engine.
//
// Endpoints: GET /health, GET /v1/models, GET /metrics (Prometheus),
// POST /v1/chat/completions, POST /v1/completions (both with stream=true SSE).
//
// HTTP worker threads only parse requests, submit them to the Engine and
// relay their RequestStream; they never run model code or touch the
// scheduler, so slow or stalled clients cannot block inference. A client that
// disconnects mid-stream cancels its request (KV freed).
//
// Hardening (DD-039): at most `max_active` completion requests are served at
// once (more get 503 + Retry-After); the HTTP pool always has spare workers
// beyond that, so /health and /metrics answer under full load. Requests get a
// default timeout. begin_drain() stops admitting work (503, /health reports
// "draining") while in-flight requests finish. POST /admin/shutdown (loopback
// clients only, unless disabled) asks the process to drain and exit.

#include <functional>
#include <memory>
#include <string>

#include "common/status.h"
#include "runtime/engine.h"

namespace engine {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8000;                 // 0 = pick a free port (tests)
  std::string model_id;            // reported by /v1/models; default: model file stem
  int max_active = 64;             // concurrent completion requests; beyond -> 503
  int http_threads = 0;            // 0 = max_active + 8 (each streaming request holds a worker)
  size_t max_body_bytes = 8 << 20;
  int32_t default_max_tokens = 1024;
  int64_t request_timeout_ms = 600000;  // per request, queued + generating (0 = none)
  bool enable_admin = true;             // POST /admin/shutdown from loopback
  std::function<void()> on_shutdown_request;  // invoked by /admin/shutdown
};

// HTTP status for an engine error.
int http_status_for(StatusCode code);

// Client side of `engine stop`: POST /admin/shutdown to a local server.
Status request_server_shutdown(const std::string& host, int port);

class Server {
 public:
  Server(Engine& engine, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Binds and starts serving on a background thread.
  Status start();
  // Port actually bound (after start).
  int port() const;
  // Stops admitting new completion requests; in-flight ones continue.
  void begin_drain();
  bool draining() const;
  // Completion requests currently being served.
  int active_requests() const;
  // Stops accepting and waits for the listener to exit.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engine
