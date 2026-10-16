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
//
// Access control (docs/security.md):
//   - api_key set: every endpoint except GET /health requires
//     "Authorization: Bearer <key>" (constant-time comparison; never logged).
//   - Bound to a loopback address: requests must carry a loopback Host header
//     (localhost, 127.x, [::1]), which defeats DNS-rebinding pages.
//   - POST bodies must be application/json: browsers cannot send that
//     cross-origin without a CORS preflight, which this server never grants,
//     so a web page cannot drive the API or /admin/shutdown on localhost.
//   - Errors raised while generating are reported to the client by category
//     only; details go to the log.

#include <functional>
#include <memory>
#include <string>

#include "dynacore/base/status.h"
#include "runtime/engine.h"
#include "common/core.h"

namespace dynalm {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8000;                 // 0 = pick a free port (tests)
  std::string model_id;            // reported by /v1/models; default: model file stem
  int max_active = 64;             // concurrent completion requests; beyond -> 503
  int http_threads = 0;            // 0 = max_active + 8 (each streaming request holds a worker)
  size_t max_body_bytes = 8 << 20;
  int32_t default_max_tokens = 1024;
  float default_temperature = 1.0f;  // when a request omits it (OpenAI default; 0 = greedy)
  int64_t request_timeout_ms = 600000;  // per request, queued + generating (0 = none)
  bool enable_admin = true;             // POST /admin/shutdown from loopback
  std::string api_key;                  // empty = no authentication (loopback deployments)
  std::function<void()> on_shutdown_request;  // invoked by /admin/shutdown
};

// HTTP status for an engine error.
int http_status_for(StatusCode code);

// Client side of `engine stop`: POST /admin/shutdown to a local server
// (`api_key`: the server's key, when it has one).
Status request_server_shutdown(const std::string& host, int port, const std::string& api_key = {});

// The API key for `serve` / `stop`: the first line of `file` when given, else
// $DYNALM_API_KEY, else empty (no authentication). Keys shorter than 16
// characters are refused. The key is never logged.
Result<std::string> load_api_key(const std::string& file);

// Whether `host` (a bind address or Host header without port) is loopback.
bool is_loopback_host(std::string_view host);

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

}  // namespace dynalm
