// HTTP server security and robustness on the tiny synthetic model, over real
// sockets on 127.0.0.1 (docs/security.md):
//   access control: Host allowlist on a loopback bind (DNS rebinding), API key
//   (401 without / with a wrong key, /health open), JSON-only POST (415), the
//   admin endpoint; limits: body size (413), malformed JSON and invalid
//   parameters (400), overload (503 + Retry-After), client disconnect frees
//   the request, drain (503, /health draining); error bodies never echo local
//   paths; API key loading (file, length rule).

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>

#include "httplib.h"
#include "runtime/engine.h"
#include "server/server.h"
#include "tests/tiny_gguf.h"

using namespace dynalm;

namespace {

int g_checks = 0;
#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::_Exit(1);                                                                   \
    }                                                                                  \
  } while (0)

const char* kJson = "application/json";
const std::string kBody = R"({"prompt":"hello there","max_tokens":4,"temperature":0})";

// POST with a streaming body receiver (return false to abort the transfer).
void stream_post(int port, const std::string& body, const std::function<bool(size_t)>& on_chunk) {
  httplib::Client c("127.0.0.1", port);
  c.set_read_timeout(120, 0);
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/completions";
  req.body = body;
  req.set_header("Content-Type", "application/json");
  req.content_receiver = [&](const char*, size_t n, uint64_t, uint64_t) { return on_chunk(n); };
  (void)c.send(req);
}

struct Fixture {
  std::unique_ptr<Engine> engine;
  std::unique_ptr<Server> server;
  std::atomic<bool> shutdown_requested{false};
  int port = 0;

  Fixture(const std::string& model, ServerOptions so) {
    EngineOptions o;
    o.model_path = model;
    o.threads = 2;
    o.kv_tokens = 1024;
    auto e = Engine::create(o);
    CHECK(e.ok());
    engine = std::move(*e);
    so.host = so.host.empty() ? "127.0.0.1" : so.host;
    so.port = 0;
    so.on_shutdown_request = [this] { shutdown_requested.store(true); };
    server = std::make_unique<Server>(*engine, so);
    CHECK(server->start().ok());
    port = server->port();
  }
  ~Fixture() {
    server->stop();
    server.reset();
  }
  httplib::Client client() const {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(120, 0);
    return c;
  }
};

void test_open_server(const std::string& model) {
  ServerOptions so;
  so.max_body_bytes = 4096;
  so.max_active = 1;
  Fixture f(model, so);
  auto c = f.client();

  auto r = c.Get("/health");
  CHECK(r && r->status == 200);
  r = c.Post("/v1/completions", kBody, kJson);
  CHECK(r && r->status == 200 && r->body.find("\"choices\"") != std::string::npos);

  // DNS rebinding: a non-loopback Host on a loopback bind.
  r = c.Post("/v1/completions", httplib::Headers{{"Host", "evil.example.com"}}, kBody, kJson);
  CHECK(r && r->status == 403);
  r = c.Get("/v1/models", httplib::Headers{{"Host", "attacker.test:8000"}});
  CHECK(r && r->status == 403);
  r = c.Get("/health", httplib::Headers{{"Host", "localhost:1234"}});
  CHECK(r && r->status == 200);
  r = c.Get("/health", httplib::Headers{{"Host", "[::1]:8000"}});
  CHECK(r && r->status == 200);

  // Browser "simple" requests (text/plain, forms) are refused, admin included.
  r = c.Post("/v1/completions", kBody, "text/plain");
  CHECK(r && r->status == 415);
  r = c.Post("/admin/shutdown", "", "application/x-www-form-urlencoded");
  CHECK(r && r->status == 415);
  CHECK(!f.shutdown_requested.load());

  // Limits and validation.
  r = c.Post("/v1/completions", std::string(8192, 'a'), kJson);
  CHECK(r && r->status == 413);
  r = c.Post("/v1/completions", "{not json", kJson);
  CHECK(r && r->status == 400);
  r = c.Post("/v1/completions", R"({"prompt":"x","max_tokens":-5})", kJson);
  CHECK(r && r->status == 400);
  r = c.Post("/v1/completions", R"({"prompt":"x","temperature":"hot"})", kJson);
  CHECK(r && r->status == 400);
  r = c.Get("/nope");
  CHECK(r && r->status == 404);
  // Error bodies never echo the model path.
  CHECK(r->body.find(model) == std::string::npos);

  // Overload: max_active = 1, a long streaming request holds the slot. The second request is sent
  // from the holder's first chunk, while that stream is certainly open, and the holder ignores EOS
  // so it cannot end early (a fast runner used to finish the holder first and get a 200).
  bool overload_sent = false;
  httplib::Result overload;
  std::thread holder([&] {
    stream_post(f.port, R"({"prompt":"hold","max_tokens":900,"ignore_eos":true,"stream":true,"temperature":0})",
                [&](size_t) {
                  if (!overload_sent) {
                    overload_sent = true;
                    auto c2 = f.client();
                    overload = c2.Post("/v1/completions", kBody, kJson);
                  }
                  return true;
                });
  });
  holder.join();
  CHECK(overload_sent);
  CHECK(overload && overload->status == 503 && overload->has_header("Retry-After"));

  // Disconnect mid-stream frees the request (slot back to 0).
  {
    int chunks = 0;
    stream_post(f.port, R"({"prompt":"bye","max_tokens":600,"stream":true,"temperature":0})",
                [&](size_t) { return ++chunks < 3; });  // abort the transfer
  }
  bool freed = false;
  for (int i = 0; i < 2000 && !freed; ++i) {
    freed = f.server->active_requests() == 0;
    if (!freed) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  CHECK(freed);
  r = c.Post("/v1/completions", kBody, kJson);  // and the server still serves
  CHECK(r && r->status == 200);

  // Admin from loopback with JSON: accepted; then draining.
  r = c.Post("/admin/shutdown", "{}", kJson);
  CHECK(r && r->status == 202 && f.shutdown_requested.load());
  f.server->begin_drain();
  r = c.Post("/v1/completions", kBody, kJson);
  CHECK(r && r->status == 503);
  r = c.Get("/health");
  CHECK(r && r->status == 503);
  std::printf("  open loopback server: Host allowlist, JSON-only POST, limits, overload, disconnect, admin, drain\n");
}

void test_api_key(const std::string& model) {
  ServerOptions so;
  so.api_keys = {"test-key-0123456789abcdef", "other-key-0123456789abcdef"};
  Fixture f(model, so);
  auto c = f.client();
  auto r = c.Get("/health");
  CHECK(r && r->status == 200);  // liveness stays open
  r = c.Post("/v1/completions", kBody, kJson);
  CHECK(r && r->status == 401 && r->has_header("WWW-Authenticate"));
  r = c.Get("/metrics");
  CHECK(r && r->status == 401);
  r = c.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer wrong-key-0123456789abc"}});
  CHECK(r && r->status == 401);
  r = c.Get("/v1/models", httplib::Headers{{"Authorization", "Basic dGVzdA=="}});
  CHECK(r && r->status == 401);
  r = c.Post("/admin/shutdown", "{}", kJson);
  CHECK(r && r->status == 401 && !f.shutdown_requested.load());
  r = c.Get("/v1/models", httplib::Headers{{"Authorization", "Bearer " + so.api_keys[1]}});
  CHECK(r && r->status == 200);  // any of the keys
  c.set_bearer_token_auth(so.api_keys[0]);
  r = c.Post("/v1/completions", kBody, kJson);
  CHECK(r && r->status == 200);
  r = c.Get("/metrics");
  CHECK(r && r->status == 200);
  for (const std::string& k : so.api_keys) CHECK(r->body.find(k) == std::string::npos);
  CHECK(request_server_shutdown("127.0.0.1", f.port, so.api_keys[1]).ok());
  CHECK(f.shutdown_requested.load());
  std::printf("  API keys: 401 without / wrong / non-bearer, any key accepted, /health open, admin needs a key, keys never echoed\n");
}

void test_metrics_disabled(const std::string& model) {
  ServerOptions so;
  so.enable_metrics = false;
  Fixture f(model, so);
  auto c = f.client();
  auto r = c.Get("/metrics");
  CHECK(r && r->status == 404);
  r = c.Get("/health");
  CHECK(r && r->status == 200);
  std::printf("  --disable-metrics: /metrics 404, /health still answers\n");
}

void test_key_loading(const std::filesystem::path& dir) {
  const auto file = dir / "key.txt";
  CHECK(test::write_file(file.string(), std::vector<uint8_t>{'s', 'h', 'o', 'r', 't', '\n'}));
  CHECK(!load_api_key(file.string()).ok());  // < 16 characters
  const std::string good = "0123456789abcdef-good\nsecond line ignored\n";
  CHECK(test::write_file(file.string(), std::vector<uint8_t>(good.begin(), good.end())));
  auto k = load_api_key(file.string());
  CHECK(k.ok() && *k == "0123456789abcdef-good");
  CHECK(!load_api_key((dir / "missing").string()).ok());
  const std::string multi = "# team keys\n0123456789abcdef-alice\r\n\n  0123456789abcdef-bob  \n";
  CHECK(test::write_file(file.string(), std::vector<uint8_t>(multi.begin(), multi.end())));
  auto ks = load_api_keys(file.string());
  CHECK(ks.ok() && ks->size() == 2 && (*ks)[0] == "0123456789abcdef-alice" && (*ks)[1] == "0123456789abcdef-bob");
  const std::string bad = "0123456789abcdef-alice\nshort\n";
  CHECK(test::write_file(file.string(), std::vector<uint8_t>(bad.begin(), bad.end())));
  CHECK(!load_api_keys(file.string()).ok());  // every key >= 16 characters
  const std::string none = "# no keys\n\n";
  CHECK(test::write_file(file.string(), std::vector<uint8_t>(none.begin(), none.end())));
  CHECK(!load_api_keys(file.string()).ok());
  CHECK(!load_api_keys((dir / "missing").string()).ok());
  CHECK(is_loopback_host("127.0.0.1") && is_loopback_host("localhost") && is_loopback_host("::1"));
  CHECK(!is_loopback_host("0.0.0.0") && !is_loopback_host("192.168.1.2") && !is_loopback_host("localhost.evil.com"));
  std::printf("  API key loading (file, first line, >= 16 chars; multi-key file), loopback host classification\n");
}

}  // namespace

int main() {
  const auto dir = std::filesystem::temp_directory_path() / ("dynalm-srv-" + test::unique_suffix());
  std::filesystem::create_directories(dir);
  const std::string model = (dir / "tiny.gguf").string();
  CHECK(test::write_file(model, test::tiny_model_gguf()));
  test_open_server(model);
  test_api_key(model);
  test_metrics_disabled(model);
  test_key_loading(dir);
  std::filesystem::remove_all(dir);
  std::printf("all server tests passed (%d checks)\n", g_checks);
  return 0;
}
