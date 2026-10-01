// Phase 20: OpenAI-compatible API — request parsing (no sockets) and an
// end-to-end HTTP server test against the tiny model.

#include <gtest/gtest.h>

#include <thread>

#include "api/json.h"
#include "api/openai.h"
#include "httplib.h"
#include "server/server.h"

namespace engine {
namespace {

json::Value J(const char* s) {
  auto v = json::parse(s);
  EXPECT_TRUE(v.ok()) << s;
  return *v;
}

// --- parsing ---

TEST(OpenAiParse, ChatRequest) {
  auto r = api::parse_chat_request(J(R"({"model":"x","messages":[{"role":"system","content":"S"},
      {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],
      "max_tokens":7,"stop":"END","stream":true,"stream_options":{"include_usage":true},"temperature":0.7})"),
                                    100);
  ASSERT_TRUE(r.ok()) << r.status().to_string();
  EXPECT_EQ(r->messages.size(), 2u);
  EXPECT_EQ(r->messages[1].content, "ab");
  EXPECT_EQ(r->params.max_tokens, 7);
  EXPECT_EQ(r->params.stop, std::vector<std::string>{"END"});
  EXPECT_TRUE(r->stream);
  EXPECT_TRUE(r->stream_usage);
  EXPECT_TRUE(r->sampling_requested);
  EXPECT_EQ(r->model, "x");
}

TEST(OpenAiParse, Defaults) {
  auto r = api::parse_chat_request(J(R"({"messages":[{"role":"developer","content":"d"},{"role":"user","content":"u"}]})"), 64);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r->params.max_tokens, 64);
  EXPECT_EQ(r->messages[0].role, "system");  // developer -> system
  EXPECT_FALSE(r->stream);
  auto c = api::parse_completion_request(J(R"({"prompt":"hello","max_completion_tokens":3})"), 64);
  ASSERT_TRUE(c.ok());
  EXPECT_EQ(c->prompt, "hello");
  EXPECT_EQ(c->params.max_tokens, 3);
}

TEST(OpenAiParse, RejectsInvalidAndUnsupported) {
  for (const char* bad : {R"([])", R"({})", R"({"messages":[]})", R"({"messages":[{"content":"x"}]})",
                          R"({"messages":[{"role":"tool","content":"x"}]})",
                          R"({"messages":[{"role":"user","content":[{"type":"image_url"}]}]})",
                          R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})",
                          R"({"messages":[{"role":"user","content":"x"}],"max_tokens":1.5})",
                          R"({"messages":[{"role":"user","content":"x"}],"n":2})",
                          R"({"messages":[{"role":"user","content":"x"}],"tools":[{}]})",
                          R"({"messages":[{"role":"user","content":"x"}],"stop":["a","b","c","d","e"]})",
                          R"({"messages":[{"role":"user","content":"x"}],"stop":[1]})",
                          R"({"messages":[{"role":"user","content":"x"}],"stream":"yes"})"}) {
    EXPECT_FALSE(api::parse_chat_request(J(bad), 16).ok()) << bad;
  }
  EXPECT_FALSE(api::parse_completion_request(J(R"({"prompt":["a","b"]})"), 16).ok());
  EXPECT_FALSE(api::parse_completion_request(J(R"({"prompt":""})"), 16).ok());
}

TEST(OpenAiFormat, ResponseShapes) {
  auto v = J(api::chat_completion_json("id1", "m", 5, "hi", StreamFinish::kLength, {3, 2}).c_str());
  EXPECT_EQ(v.find("object")->as_string(), "chat.completion");
  const json::Value& ch = v.find("choices")->as_array()[0];
  EXPECT_EQ(ch.find("message")->find("content")->as_string(), "hi");
  EXPECT_EQ(ch.find("finish_reason")->as_string(), "length");
  EXPECT_EQ(v.find("usage")->find("total_tokens")->as_number(), 5);
  auto chunk = J(api::chat_chunk_json("id1", "m", 5, "x", true, StreamFinish::kNone).c_str());
  const json::Value& d = chunk.find("choices")->as_array()[0];
  EXPECT_EQ(d.find("delta")->find("role")->as_string(), "assistant");
  EXPECT_TRUE(d.find("finish_reason")->is_null());
  auto err = J(api::error_json("bad", "invalid_request_error").c_str());
  EXPECT_EQ(err.find("error")->find("message")->as_string(), "bad");
}

// --- end to end over HTTP ---

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EngineOptions o;
    o.model_path = std::string(ENGINE_TEST_DATA_DIR) + "/tiny_llama.gguf";
    o.threads = 2;
    o.kv_tokens = 2048;
    o.max_batch_tokens = 64;
    auto e = Engine::create(o);
    ASSERT_TRUE(e.ok()) << e.status().to_string();
    eng = std::move(*e);
    ServerOptions so;
    so.port = 0;
    so.model_id = "tiny";
    so.default_max_tokens = 16;
    so.http_threads = 4;
    server = std::make_unique<Server>(*eng, so);
    ASSERT_TRUE(server->start().ok());
    client = std::make_unique<httplib::Client>("127.0.0.1", server->port());
    client->set_read_timeout(60, 0);
  }
  void TearDown() override {
    server.reset();
    eng.reset();
  }
  std::unique_ptr<Engine> eng;
  std::unique_ptr<Server> server;
  std::unique_ptr<httplib::Client> client;
};

TEST_F(ServerTest, HealthModelsMetrics) {
  auto h = client->Get("/health");
  ASSERT_TRUE(h);
  EXPECT_EQ(h->status, 200);
  auto m = client->Get("/v1/models");
  ASSERT_TRUE(m);
  EXPECT_EQ(J(m->body.c_str()).find("data")->as_array()[0].find("id")->as_string(), "tiny");
  auto met = client->Get("/metrics");
  ASSERT_TRUE(met);
  EXPECT_NE(met->body.find("engine_kv_cache_capacity_blocks"), std::string::npos);
  EXPECT_EQ(client->Get("/nope")->status, 404);
}

TEST_F(ServerTest, ChatCompletionNonStreaming) {
  auto res = client->Post("/v1/chat/completions",
                          R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":8})", "application/json");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200) << res->body;
  auto v = J(res->body.c_str());
  EXPECT_EQ(v.find("object")->as_string(), "chat.completion");
  EXPECT_EQ(v.find("usage")->find("completion_tokens")->as_number(), 8);
  EXPECT_EQ(v.find("choices")->as_array()[0].find("finish_reason")->as_string(), "length");
}

TEST_F(ServerTest, CompletionStreamingSse) {
  std::string body;
  auto res = client->Post("/v1/completions", R"({"prompt":"abc","max_tokens":10,"stream":true,
                                                 "stream_options":{"include_usage":true}})",
                          "application/json");
  ASSERT_TRUE(res);
  ASSERT_EQ(res->status, 200);
  EXPECT_NE(res->get_header_value("Content-Type").find("text/event-stream"), std::string::npos);
  // Frames: data: {chunk}\n\n ... data: [DONE]\n\n
  std::vector<std::string> frames;
  size_t pos = 0;
  while ((pos = res->body.find("data: ", pos)) != std::string::npos) {
    const size_t end = res->body.find("\n\n", pos);
    ASSERT_NE(end, std::string::npos);
    frames.push_back(res->body.substr(pos + 6, end - pos - 6));
    pos = end + 2;
  }
  // At least: final chunk (finish_reason), usage chunk, [DONE]. Text chunks
  // vary: the tiny model emits random bytes and incomplete UTF-8 is held back.
  ASSERT_GE(frames.size(), 3u);
  EXPECT_EQ(frames.back(), "[DONE]");
  std::string text;
  double streamed_tokens = -1;
  for (size_t i = 0; i + 1 < frames.size(); ++i) {
    auto v = J(frames[i].c_str());
    if (const json::Value* u = v.find("usage"); u && !u->is_null()) {
      streamed_tokens = u->find("completion_tokens")->as_number();
      continue;
    }
    text += v.find("choices")->as_array()[0].find("text")->as_string();
  }
  // Streamed text and usage equal the non-streamed result (generation may end
  // before max_tokens if the random tiny model emits EOS).
  auto ns = client->Post("/v1/completions", R"({"prompt":"abc","max_tokens":10})", "application/json");
  ASSERT_TRUE(ns);
  const json::Value full = J(ns->body.c_str());
  EXPECT_EQ(text, full.find("choices")->as_array()[0].find("text")->as_string());
  EXPECT_EQ(streamed_tokens, full.find("usage")->find("completion_tokens")->as_number());
  EXPECT_GE(streamed_tokens, 1);
  EXPECT_LE(streamed_tokens, 10);
}

TEST_F(ServerTest, ErrorsAreJsonAndIsolated) {
  auto bad = client->Post("/v1/chat/completions", "{not json", "application/json");
  ASSERT_TRUE(bad);
  EXPECT_EQ(bad->status, 400);
  EXPECT_EQ(J(bad->body.c_str()).find("error")->find("type")->as_string(), "invalid_request_error");
  auto too_long = client->Post("/v1/completions", R"({"prompt":"x","max_tokens":1000000})", "application/json");
  ASSERT_TRUE(too_long);
  EXPECT_EQ(too_long->status, 400);
  // The server keeps serving after bad requests.
  auto ok = client->Post("/v1/completions", R"({"prompt":"x","max_tokens":2})", "application/json");
  ASSERT_TRUE(ok);
  EXPECT_EQ(ok->status, 200);
}

TEST_F(ServerTest, ConcurrentClients) {
  std::vector<std::thread> ts;
  std::atomic<int> ok{0};
  for (int i = 0; i < 6; ++i) {
    ts.emplace_back([&, i] {
      httplib::Client c("127.0.0.1", server->port());
      c.set_read_timeout(60, 0);
      const std::string body = R"({"prompt":"client )" + std::to_string(i) + R"(","max_tokens":6})";
      auto res = c.Post("/v1/completions", body, "application/json");
      if (res && res->status == 200) ok.fetch_add(1);
    });
  }
  for (auto& t : ts) t.join();
  EXPECT_EQ(ok.load(), 6);
}

TEST_F(ServerTest, ClientDisconnectCancelsGeneration) {
  int frames = 0;
  httplib::Client c("127.0.0.1", server->port());
  c.set_read_timeout(60, 0);
  // Abort after the first few streamed chunks.
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/completions";
  req.body = R"({"prompt":"long","max_tokens":100,"stream":true})";
  req.set_header("Content-Type", "application/json");
  req.content_receiver = [&](const char*, size_t, uint64_t, uint64_t) { return ++frames < 3; };
  auto res = c.send(req);
  EXPECT_FALSE(res);  // aborted by the receiver
  // Whether the server sees the disconnect before generation ends is a race
  // with the tiny model (cancellation itself is tested in test_streaming), so
  // assert what must always hold: nothing stays running and the server keeps
  // serving.
  for (int i = 0; i < 100 && eng->stats().scheduler.running > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_EQ(eng->stats().scheduler.running, 0);
  auto ok = client->Post("/v1/completions", R"({"prompt":"after","max_tokens":3})", "application/json");
  ASSERT_TRUE(ok);
  EXPECT_EQ(ok->status, 200);
}

TEST_F(ServerTest, ChatWithoutTemplateIsClearError) {
  // The tiny Qwen model ships no chat template: chat requests must fail
  // clearly (400), completions still work.
  EngineOptions o;
  o.model_path = std::string(ENGINE_TEST_DATA_DIR) + "/tiny_qwen2.gguf";
  o.threads = 1;
  o.kv_tokens = 256;
  auto e = Engine::create(o);
  ASSERT_TRUE(e.ok());
  ServerOptions so;
  so.port = 0;
  Server s(**e, so);
  ASSERT_TRUE(s.start().ok());
  httplib::Client c("127.0.0.1", s.port());
  auto res = c.Post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})", "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 400);
  EXPECT_NE(res->body.find("chat template"), std::string::npos);
}

}  // namespace
}  // namespace engine
