// HTTP benchmark target: any OpenAI-compatible server (this engine's server,
// llama.cpp's llama-server, Ollama's /v1 API). Built with the server.

#include <cstdio>

#include "api/json.h"
#include "bench/loadgen.h"
#include "common/timer.h"
#include "httplib.h"

namespace engine::bench {
namespace {

class HttpTarget final : public Target {
 public:
  HttpTarget(std::string host, int port, std::string model)
      : host_(std::move(host)), port_(port), model_(std::move(model)) {}

  std::string name() const override { return "http://" + host_ + ":" + std::to_string(port_); }

  RequestResult run(const std::string& prompt, int32_t max_tokens) override {
    RequestResult r;
    json::Object body;
    body["model"] = model_;
    body["prompt"] = prompt;
    body["max_tokens"] = max_tokens;
    body["stream"] = true;
    body["ignore_eos"] = true;  // fixed output length (llama.cpp + this engine)
    body["temperature"] = 0.0;
    json::Object so;
    so["include_usage"] = true;
    body["stream_options"] = std::move(so);

    httplib::Client client(host_, port_);
    client.set_read_timeout(600, 0);
    client.set_connection_timeout(10, 0);
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/completions";
    req.body = json::dump(body);
    req.set_header("Content-Type", "application/json");

    std::string buf;
    int32_t deltas = 0;
    bool first = true, done = false;
    int64_t last = 0;
    const int64_t t0 = now_ns();
    // Parses complete SSE frames as bytes arrive (timing per delta).
    req.content_receiver = [&](const char* data, size_t len, uint64_t, uint64_t) {
      buf.append(data, len);
      size_t end;
      while ((end = buf.find("\n\n")) != std::string::npos) {
        std::string frame = buf.substr(0, end);
        buf.erase(0, end + 2);
        if (frame.rfind("data: ", 0) != 0) continue;
        const std::string payload = frame.substr(6);
        if (payload == "[DONE]") {
          done = true;
          continue;
        }
        auto v = json::parse(payload);
        if (!v.ok()) continue;
        if (const json::Value* u = v->find("usage"); u && u->is_object()) {
          if (const json::Value* p = u->find("prompt_tokens"); p && p->is_number()) r.prompt_tokens = static_cast<int32_t>(p->as_number());
          if (const json::Value* c = u->find("completion_tokens"); c && c->is_number()) {
            r.completion_tokens = static_cast<int32_t>(c->as_number());
          }
        }
        const json::Value* ch = v->find("choices");
        if (!ch || !ch->is_array() || ch->as_array().empty()) continue;
        const json::Value* text = ch->as_array()[0].find("text");
        if (!text || !text->is_string() || text->as_string().empty()) continue;
        const int64_t now = now_ns();
        if (first) {
          r.ttft_ms = static_cast<double>(now - t0) * 1e-6;
          first = false;
        } else {
          r.gaps_ms.push_back(static_cast<double>(now - last) * 1e-6);
        }
        last = now;
        ++deltas;
      }
      return true;
    };
    auto res = client.send(req);
    r.e2e_ms = static_cast<double>(now_ns() - t0) * 1e-6;
    if (!res) {
      r.error = "HTTP error: " + httplib::to_string(res.error());
      return r;
    }
    if (res->status != 200) {
      r.error = "HTTP " + std::to_string(res->status) + ": " + res->body.substr(0, 200);
      return r;
    }
    if (r.completion_tokens == 0) r.completion_tokens = deltas;  // no usage chunk: one delta per token
    if (first) r.ttft_ms = r.e2e_ms;
    r.ok = done || deltas > 0;
    if (!r.ok) r.error = "no tokens streamed";
    return r;
  }

 private:
  std::string host_;
  int port_;
  std::string model_;
};

}  // namespace

Result<std::unique_ptr<Target>> make_http_target(const std::string& url, const std::string& model) {
  // http://host:port only (benchmarks run against local servers).
  constexpr std::string_view kScheme = "http://";
  if (url.rfind(kScheme, 0) != 0) return InvalidArgument("benchmark URL must start with http://");
  const std::string rest = url.substr(kScheme.size());
  const auto colon = rest.rfind(':');
  if (colon == std::string::npos) return InvalidArgument("benchmark URL needs a port: " + url);
  const std::string host = rest.substr(0, colon);
  const int port = std::atoi(rest.substr(colon + 1).c_str());
  if (host.empty() || port <= 0) return InvalidArgument("invalid benchmark URL: " + url);
  return std::unique_ptr<Target>(std::make_unique<HttpTarget>(host, port, model));
}

}  // namespace engine::bench
