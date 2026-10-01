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

#include <memory>
#include <string>

#include "common/status.h"
#include "runtime/engine.h"

namespace engine {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8000;                 // 0 = pick a free port (tests)
  std::string model_id;            // reported by /v1/models; default: model file stem
  int http_threads = 16;
  size_t max_body_bytes = 8 << 20;
  int32_t default_max_tokens = 1024;
};

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
  // Stops accepting and waits for the listener to exit.
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace engine
