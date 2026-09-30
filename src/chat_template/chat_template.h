#pragma once

// Chat templates: turn a message list into the prompt text a model expects.
//
// Model files carry a Jinja template. Instead of embedding a Jinja engine,
// the family is detected from signature strings in the template source and
// rendered natively (the approach llama.cpp used for years). Rendered text
// contains control-token text ("<|im_start|>") and must be tokenized with
// parse_special = true. BOS is never rendered; the tokenizer adds it.
//
// Future: a Jinja subset interpreter for templates no family matches.

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/status.h"

namespace engine {

struct ChatMessage {
  std::string role;  // "system" | "user" | "assistant"
  std::string content;
};

enum class ChatFormat {
  kChatMl,     // <|im_start|>role\n...<|im_end|>   (Qwen, SmolLM, Yi, ...)
  kLlama3,     // <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
  kLlama2,     // [INST] <<SYS>>...<</SYS>> ... [/INST]
  kMistral,    // [INST] ... [/INST] (system prepended to first user turn)
  kGemma,      // <start_of_turn>user\n...<end_of_turn>  (role "model")
  kPhi3,       // <|user|>\n...<|end|>
  kDeepSeek2,  // <｜User｜>...<｜Assistant｜> (DeepSeek-V2/V3/R1)
};

std::string_view chat_format_name(ChatFormat f);
Result<ChatFormat> parse_chat_format(std::string_view name);

class ChatTemplate {
 public:
  // Detects the format from a Jinja template source. Unrecognized templates
  // are kUnsupported; callers may fall back to an explicit format.
  static Result<ChatTemplate> from_jinja(std::string_view jinja);
  explicit ChatTemplate(ChatFormat format, std::string default_system = {})
      : format_(format), default_system_(std::move(default_system)) {}

  ChatFormat format() const { return format_; }
  const std::string& default_system() const { return default_system_; }

  // Renders the conversation. With `add_generation_prompt`, ends with the
  // assistant header so the model continues as the assistant.
  Result<std::string> apply(std::span<const ChatMessage> messages, bool add_generation_prompt) const;

 private:
  ChatFormat format_;
  std::string default_system_;  // used by templates that inject one when absent
};

}  // namespace engine
