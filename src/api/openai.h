#pragma once

// OpenAI-compatible request parsing and response building, independent of the
// HTTP layer (testable without sockets).
//
// Supported: /v1/chat/completions and /v1/completions with messages/prompt,
// max_tokens / max_completion_tokens, stop (string or up to 4 strings),
// stream, stream_options.include_usage, and sampling: temperature, top_p, seed,
// presence_penalty, frequency_penalty, plus the common extensions top_k, min_p,
// repetition_penalty, repeat_last_n and ignore_eos (DD-043). The default
// temperature is the server's (1.0, as in the OpenAI API). Unsupported
// features (tools, n > 1, logprobs,
// non-text content parts) are rejected with a clear error instead of being
// silently ignored.

#include <string>
#include <vector>

#include "api/json.h"
#include "chat_template/chat_template.h"
#include "common/status.h"
#include "runtime/engine.h"

namespace engine::api {

struct CompletionRequest {
  bool chat = true;
  std::vector<ChatMessage> messages;  // chat
  std::string prompt;                 // completions
  GenerateParams params;
  bool stream = false;
  bool stream_usage = false;
  std::string model;
};

Result<CompletionRequest> parse_chat_request(const json::Value& body, int32_t default_max_tokens,
                                             float default_temperature = 1.0f);
Result<CompletionRequest> parse_completion_request(const json::Value& body, int32_t default_max_tokens,
                                                   float default_temperature = 1.0f);

// OpenAI finish_reason for a stream outcome ("stop", "length"); errors and
// cancellations are reported out of band.
std::string_view finish_reason(StreamFinish f);

struct Usage {
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
};

std::string chat_completion_json(const std::string& id, const std::string& model, int64_t created,
                                 const std::string& text, StreamFinish finish, const Usage& usage);
std::string text_completion_json(const std::string& id, const std::string& model, int64_t created,
                                 const std::string& text, StreamFinish finish, const Usage& usage);
// Streaming chunks (one SSE "data:" payload each). `finish` kNone = in progress.
std::string chat_chunk_json(const std::string& id, const std::string& model, int64_t created, const std::string& delta,
                            bool first, StreamFinish finish);
std::string text_chunk_json(const std::string& id, const std::string& model, int64_t created,
                            const std::string& delta, StreamFinish finish);
std::string usage_chunk_json(const std::string& id, const std::string& model, int64_t created, bool chat,
                             const Usage& usage);

// {"error": {"message", "type", "param", "code"}}
std::string error_json(std::string_view message, std::string_view type, std::string_view code = {});

std::string models_json(const std::string& model_id, int64_t created);

}  // namespace engine::api
