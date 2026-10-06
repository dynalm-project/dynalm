#include "api/openai.h"

#include <cmath>

namespace engine::api {
namespace {

Status bad(const std::string& msg) { return InvalidArgument(msg); }

// Reads a positive integer field (absent → keep `out`).
Status read_positive_int(const json::Value& body, std::string_view key, int32_t& out) {
  const json::Value* v = body.find(key);
  if (!v || v->is_null()) return Status::Ok();
  if (!v->is_number() || v->as_number() != std::floor(v->as_number()) || v->as_number() < 1 ||
      v->as_number() > 1e7) {
    return bad("'" + std::string(key) + "' must be a positive integer");
  }
  out = static_cast<int32_t>(v->as_number());
  return Status::Ok();
}

// Reads an optional number in [lo, hi] (absent or null -> keep `out`).
Status read_number(const json::Value& body, std::string_view key, double lo, double hi, float& out) {
  const json::Value* v = body.find(key);
  if (!v || v->is_null()) return Status::Ok();
  if (!v->is_number() || !(v->as_number() >= lo) || !(v->as_number() <= hi)) {
    return bad("'" + std::string(key) + "' must be a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  }
  out = static_cast<float>(v->as_number());
  return Status::Ok();
}

// OpenAI sampling fields plus common extensions (top_k, min_p,
// repetition_penalty, repeat_last_n) as llama.cpp / vLLM accept them.
Status read_sampling(const json::Value& body, float default_temperature, SamplingParams& sp) {
  sp.temperature = default_temperature;
  ENGINE_RETURN_IF_ERROR(read_number(body, "temperature", 0.0, 2.0, sp.temperature));
  ENGINE_RETURN_IF_ERROR(read_number(body, "top_p", 0.0, 1.0, sp.top_p));
  if (sp.top_p <= 0.0f) return bad("'top_p' must be > 0");
  ENGINE_RETURN_IF_ERROR(read_number(body, "min_p", 0.0, 1.0, sp.min_p));
  ENGINE_RETURN_IF_ERROR(read_number(body, "presence_penalty", -2.0, 2.0, sp.presence_penalty));
  ENGINE_RETURN_IF_ERROR(read_number(body, "frequency_penalty", -2.0, 2.0, sp.frequency_penalty));
  ENGINE_RETURN_IF_ERROR(read_number(body, "repetition_penalty", 0.01, 10.0, sp.repetition_penalty));
  float top_k = 0, last_n = static_cast<float>(sp.penalty_last_n);
  ENGINE_RETURN_IF_ERROR(read_number(body, "top_k", 0.0, 1e6, top_k));
  ENGINE_RETURN_IF_ERROR(read_number(body, "repeat_last_n", -1.0, 1e6, last_n));
  if (top_k != std::floor(top_k) || last_n != std::floor(last_n)) return bad("'top_k' and 'repeat_last_n' must be integers");
  sp.top_k = static_cast<int32_t>(top_k);
  sp.penalty_last_n = static_cast<int32_t>(last_n);
  if (const json::Value* seed = body.find("seed"); seed && !seed->is_null()) {
    if (!seed->is_number() || seed->as_number() != std::floor(seed->as_number()) || std::abs(seed->as_number()) > 9e15) {
      return bad("'seed' must be an integer");
    }
    sp.seed = static_cast<uint64_t>(static_cast<int64_t>(seed->as_number()));
    sp.has_seed = true;
  }
  return sp.validate();
}

Status read_common(const json::Value& body, int32_t default_max_tokens, float default_temperature,
                   CompletionRequest& r) {
  if (!body.is_object()) return bad("request body must be a JSON object");
  if (const json::Value* m = body.find("model"); m && m->is_string()) r.model = m->as_string();

  r.params.max_tokens = default_max_tokens;
  ENGINE_RETURN_IF_ERROR(read_positive_int(body, "max_tokens", r.params.max_tokens));
  ENGINE_RETURN_IF_ERROR(read_positive_int(body, "max_completion_tokens", r.params.max_tokens));

  if (const json::Value* s = body.find("stop"); s && !s->is_null()) {
    if (s->is_string()) {
      r.params.stop.push_back(s->as_string());
    } else if (s->is_array()) {
      if (s->as_array().size() > 4) return bad("'stop' accepts at most 4 sequences");
      for (const json::Value& e : s->as_array()) {
        if (!e.is_string()) return bad("'stop' entries must be strings");
        r.params.stop.push_back(e.as_string());
      }
    } else {
      return bad("'stop' must be a string or an array of strings");
    }
  }

  // llama.cpp-compatible extension used by benchmarks: keep generating
  // through end-of-generation tokens up to max_tokens.
  if (const json::Value* ie = body.find("ignore_eos"); ie && ie->is_bool()) r.params.stop_at_eog = !ie->as_bool();

  if (const json::Value* s = body.find("stream"); s && !s->is_null()) {
    if (!s->is_bool()) return bad("'stream' must be a boolean");
    r.stream = s->as_bool();
  }
  if (const json::Value* so = body.find("stream_options"); so && so->is_object()) {
    if (const json::Value* u = so->find("include_usage"); u && u->is_bool()) r.stream_usage = u->as_bool();
  }

  if (const json::Value* n = body.find("n"); n && !n->is_null()) {
    if (!n->is_number() || n->as_number() != 1) return bad("only n = 1 is supported");
  }
  for (std::string_view unsupported : {"tools", "functions", "logprobs", "top_logprobs", "response_format"}) {
    const json::Value* v = body.find(unsupported);
    if (v && !v->is_null() && !(v->is_bool() && !v->as_bool())) {
      return bad("'" + std::string(unsupported) + "' is not supported");
    }
  }
  return read_sampling(body, default_temperature, r.params.sampling);
}

// Message content: a string, or an array of {"type": "text", "text": ...} parts.
Status read_content(const json::Value& c, std::string& out) {
  if (c.is_string()) {
    out = c.as_string();
    return Status::Ok();
  }
  if (c.is_null()) return Status::Ok();
  if (!c.is_array()) return bad("message 'content' must be a string or an array of parts");
  for (const json::Value& part : c.as_array()) {
    const json::Value* type = part.find("type");
    const json::Value* text = part.find("text");
    if (!type || !type->is_string() || type->as_string() != "text" || !text || !text->is_string()) {
      return bad("only text content parts are supported");
    }
    out += text->as_string();
  }
  return Status::Ok();
}

std::string_view finish_or_null(StreamFinish f) { return f == StreamFinish::kNone ? "" : finish_reason(f); }

json::Value usage_json(const Usage& u) {
  json::Object o;
  o["prompt_tokens"] = u.prompt_tokens;
  o["completion_tokens"] = u.completion_tokens;
  o["total_tokens"] = u.prompt_tokens + u.completion_tokens;
  return o;
}

json::Value finish_value(StreamFinish f) {
  const std::string_view r = finish_or_null(f);
  return r.empty() ? json::Value() : json::Value(std::string(r));
}

json::Object base(const std::string& id, const char* object, int64_t created, const std::string& model) {
  json::Object o;
  o["id"] = id;
  o["object"] = object;
  o["created"] = created;
  o["model"] = model;
  return o;
}

}  // namespace

Result<CompletionRequest> parse_chat_request(const json::Value& body, int32_t default_max_tokens,
                                             float default_temperature) {
  CompletionRequest r;
  r.chat = true;
  ENGINE_RETURN_IF_ERROR(read_common(body, default_max_tokens, default_temperature, r));
  const json::Value* msgs = body.find("messages");
  if (!msgs || !msgs->is_array() || msgs->as_array().empty()) return bad("'messages' must be a non-empty array");
  for (const json::Value& m : msgs->as_array()) {
    const json::Value* role = m.find("role");
    if (!role || !role->is_string()) return bad("each message needs a string 'role'");
    ChatMessage cm;
    cm.role = role->as_string() == "developer" ? "system" : role->as_string();
    if (cm.role != "system" && cm.role != "user" && cm.role != "assistant") {
      return bad("unsupported message role '" + role->as_string() + "'");
    }
    const json::Value* content = m.find("content");
    if (!content) return bad("each message needs 'content'");
    ENGINE_RETURN_IF_ERROR(read_content(*content, cm.content));
    r.messages.push_back(std::move(cm));
  }
  return r;
}

Result<CompletionRequest> parse_completion_request(const json::Value& body, int32_t default_max_tokens,
                                                   float default_temperature) {
  CompletionRequest r;
  r.chat = false;
  ENGINE_RETURN_IF_ERROR(read_common(body, default_max_tokens, default_temperature, r));
  const json::Value* p = body.find("prompt");
  if (!p) return bad("'prompt' is required");
  if (p->is_string()) {
    r.prompt = p->as_string();
  } else if (p->is_array() && p->as_array().size() == 1 && p->as_array()[0].is_string()) {
    r.prompt = p->as_array()[0].as_string();
  } else {
    return bad("'prompt' must be a string (batched prompts are not supported)");
  }
  if (r.prompt.empty()) return bad("'prompt' must not be empty");
  return r;
}

std::string_view finish_reason(StreamFinish f) {
  return f == StreamFinish::kLength ? "length" : "stop";
}

std::string chat_completion_json(const std::string& id, const std::string& model, int64_t created,
                                 const std::string& text, StreamFinish finish, const Usage& usage) {
  json::Object msg;
  msg["role"] = "assistant";
  msg["content"] = text;
  json::Object choice;
  choice["index"] = 0;
  choice["message"] = std::move(msg);
  choice["finish_reason"] = finish_value(finish);
  json::Object o = base(id, "chat.completion", created, model);
  o["choices"] = json::Array{std::move(choice)};
  o["usage"] = usage_json(usage);
  return json::dump(o);
}

std::string text_completion_json(const std::string& id, const std::string& model, int64_t created,
                                 const std::string& text, StreamFinish finish, const Usage& usage) {
  json::Object choice;
  choice["index"] = 0;
  choice["text"] = text;
  choice["finish_reason"] = finish_value(finish);
  json::Object o = base(id, "text_completion", created, model);
  o["choices"] = json::Array{std::move(choice)};
  o["usage"] = usage_json(usage);
  return json::dump(o);
}

std::string chat_chunk_json(const std::string& id, const std::string& model, int64_t created, const std::string& delta,
                            bool first, StreamFinish finish) {
  json::Object d;
  if (first) d["role"] = "assistant";
  if (!delta.empty()) d["content"] = delta;
  json::Object choice;
  choice["index"] = 0;
  choice["delta"] = std::move(d);
  choice["finish_reason"] = finish_value(finish);
  json::Object o = base(id, "chat.completion.chunk", created, model);
  o["choices"] = json::Array{std::move(choice)};
  return json::dump(o);
}

std::string text_chunk_json(const std::string& id, const std::string& model, int64_t created,
                            const std::string& delta, StreamFinish finish) {
  json::Object choice;
  choice["index"] = 0;
  choice["text"] = delta;
  choice["finish_reason"] = finish_value(finish);
  json::Object o = base(id, "text_completion", created, model);
  o["choices"] = json::Array{std::move(choice)};
  return json::dump(o);
}

std::string usage_chunk_json(const std::string& id, const std::string& model, int64_t created, bool chat,
                             const Usage& usage) {
  json::Object o = base(id, chat ? "chat.completion.chunk" : "text_completion", created, model);
  o["choices"] = json::Array{};
  o["usage"] = usage_json(usage);
  return json::dump(o);
}

std::string error_json(std::string_view message, std::string_view type, std::string_view code) {
  json::Object e;
  e["message"] = std::string(message);
  e["type"] = std::string(type);
  e["param"] = json::Value();
  e["code"] = code.empty() ? json::Value() : json::Value(std::string(code));
  json::Object o;
  o["error"] = std::move(e);
  return json::dump(o);
}

std::string models_json(const std::string& model_id, int64_t created) {
  json::Object m;
  m["id"] = model_id;
  m["object"] = "model";
  m["created"] = created;
  m["owned_by"] = "dynalm";
  json::Object o;
  o["object"] = "list";
  o["data"] = json::Array{std::move(m)};
  return json::dump(o);
}

}  // namespace engine::api
