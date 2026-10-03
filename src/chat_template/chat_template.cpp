#include "chat_template/chat_template.h"

namespace engine {
namespace {

bool contains(std::string_view s, std::string_view needle) { return s.find(needle) != std::string_view::npos; }

// Extracts the literal default system prompt some ChatML templates inject
// when the conversation has none, e.g.
//   '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud...<|im_end|>\n'
// The "\n" appears as a two-character escape inside the Jinja string.
std::string extract_chatml_default_system(std::string_view jinja) {
  for (std::string_view head : {"<|im_start|>system\\n", "<|im_start|>system\n"}) {
    size_t pos = 0;
    while ((pos = jinja.find(head, pos)) != std::string_view::npos) {
      const size_t start = pos + head.size();
      const size_t end = jinja.find("<|im_end|>", start);
      if (end == std::string_view::npos) break;
      const std::string_view body = jinja.substr(start, end - start);
      // A literal prompt contains no Jinja expressions or string breaks.
      if (!body.empty() && !contains(body, "{{") && !contains(body, "{%") && !contains(body, "'") &&
          !contains(body, "\"") && !contains(body, "+")) {
        std::string out;
        for (size_t i = 0; i < body.size(); ++i) {
          if (body[i] == '\\' && i + 1 < body.size() && body[i + 1] == 'n') {
            out += '\n';
            ++i;
          } else {
            out += body[i];
          }
        }
        return out;
      }
      pos = start;
    }
  }
  return {};
}

// Granite 3.x: the default system prompt is built from string literals,
//   set system_message = "Knowledge Cutoff Date: ... developed by IBM."
//   ... {%- else %}{%- set system_message = system_message + " You are a helpful AI assistant." %}
// (the no-tools, no-documents branch). The turn separator is whatever the
// template writes after <|end_of_text|> (a newline or a space, by release).
std::string literal_after(std::string_view j, std::string_view head) {
  const size_t p = j.find(head);
  if (p == std::string_view::npos) return {};
  const size_t start = p + head.size();
  const size_t end = j.find('"', start);
  return end == std::string_view::npos ? std::string() : std::string(j.substr(start, end - start));
}

ChatTemplate granite_template(std::string_view j) {
  std::string system = literal_after(j, "set system_message = \"");
  const std::string tail = literal_after(j, "{%- else %}{%- set system_message = system_message + \"");
  if (!system.empty()) system += tail;
  std::string sep = "\n";
  constexpr std::string_view kEnd = "<|end_of_text|>";
  if (const size_t p = j.find(kEnd); p != std::string_view::npos) {
    const size_t q = j.find('\'', p + kEnd.size());
    if (q != std::string_view::npos && q - (p + kEnd.size()) <= 2) {
      sep = std::string(j.substr(p + kEnd.size(), q - p - kEnd.size()));
      if (sep == "\\n") sep = "\n";
    }
  }
  return ChatTemplate(ChatFormat::kGranite, std::move(system), std::move(sep));
}

}  // namespace

std::string_view chat_format_name(ChatFormat f) {
  switch (f) {
    case ChatFormat::kChatMl: return "chatml";
    case ChatFormat::kLlama3: return "llama3";
    case ChatFormat::kLlama2: return "llama2";
    case ChatFormat::kMistral: return "mistral";
    case ChatFormat::kGemma: return "gemma";
    case ChatFormat::kPhi3: return "phi3";
    case ChatFormat::kDeepSeek2: return "deepseek2";
    case ChatFormat::kGranite: return "granite";
  }
  return "?";
}

Result<ChatFormat> parse_chat_format(std::string_view name) {
  for (ChatFormat f : {ChatFormat::kChatMl, ChatFormat::kLlama3, ChatFormat::kLlama2, ChatFormat::kMistral,
                       ChatFormat::kGemma, ChatFormat::kPhi3, ChatFormat::kDeepSeek2, ChatFormat::kGranite}) {
    if (chat_format_name(f) == name) return f;
  }
  return InvalidArgument("unknown chat template '" + std::string(name) + "'");
}

Result<ChatTemplate> ChatTemplate::from_jinja(std::string_view j) {
  if (j.empty()) return Unsupported("model has no chat template");
  if (contains(j, "<|im_start|>")) return ChatTemplate(ChatFormat::kChatMl, extract_chatml_default_system(j));
  if (contains(j, "<|start_header_id|>")) return ChatTemplate(ChatFormat::kLlama3);
  if (contains(j, "<|start_of_role|>")) return granite_template(j);
  if (contains(j, "<start_of_turn>")) return ChatTemplate(ChatFormat::kGemma);
  if (contains(j, "<|user|>") && contains(j, "<|end|>")) return ChatTemplate(ChatFormat::kPhi3);
  if (contains(j, "<｜User｜>")) return ChatTemplate(ChatFormat::kDeepSeek2);
  if (contains(j, "<<SYS>>")) return ChatTemplate(ChatFormat::kLlama2);
  if (contains(j, "[INST]")) return ChatTemplate(ChatFormat::kMistral);
  return Unsupported("unrecognized chat template; pass an explicit format");
}

Result<std::string> ChatTemplate::apply(std::span<const ChatMessage> msgs, bool gen) const {
  for (const ChatMessage& m : msgs) {
    if (m.role != "system" && m.role != "user" && m.role != "assistant") {
      return InvalidArgument("unsupported chat role '" + m.role + "'");
    }
  }
  std::string out;
  const bool has_system = !msgs.empty() && msgs.front().role == "system";

  switch (format_) {
    case ChatFormat::kChatMl:
      if (!has_system && !default_system_.empty()) {
        out += "<|im_start|>system\n" + default_system_ + "<|im_end|>\n";
      }
      for (const ChatMessage& m : msgs) out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
      if (gen) out += "<|im_start|>assistant\n";
      break;

    case ChatFormat::kLlama3:
      for (const ChatMessage& m : msgs) {
        out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
      }
      if (gen) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      break;

    case ChatFormat::kPhi3:
      for (const ChatMessage& m : msgs) out += "<|" + m.role + "|>\n" + m.content + "<|end|>\n";
      if (gen) out += "<|assistant|>\n";
      break;

    case ChatFormat::kGemma: {
      // No system role: prepend it to the first user turn.
      std::string pending_system = has_system ? msgs.front().content + "\n\n" : "";
      for (size_t i = has_system ? 1 : 0; i < msgs.size(); ++i) {
        const ChatMessage& m = msgs[i];
        const std::string role = m.role == "assistant" ? "model" : "user";
        out += "<start_of_turn>" + role + "\n";
        if (role == "user") {
          out += pending_system;
          pending_system.clear();
        }
        out += m.content + "<end_of_turn>\n";
      }
      if (gen) out += "<start_of_turn>model\n";
      break;
    }

    case ChatFormat::kLlama2:
    case ChatFormat::kMistral: {
      const bool llama2 = format_ == ChatFormat::kLlama2;
      std::string system = has_system ? msgs.front().content : "";
      for (size_t i = has_system ? 1 : 0; i < msgs.size(); ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user") {
          out += "[INST] ";
          if (!system.empty()) {
            out += llama2 ? "<<SYS>>\n" + system + "\n<</SYS>>\n\n" : system + "\n\n";
            system.clear();
          }
          out += m.content + " [/INST]";
        } else if (m.role == "assistant") {
          out += " " + m.content + "</s>";
        }
      }
      (void)gen;  // the model continues directly after [/INST]
      break;
    }

    case ChatFormat::kGranite: {
      const std::string& system = has_system ? msgs.front().content : default_system_;
      if (!system.empty()) out += "<|start_of_role|>system<|end_of_role|>" + system + "<|end_of_text|>" + separator_;
      for (size_t i = has_system ? 1 : 0; i < msgs.size(); ++i) {
        out += "<|start_of_role|>" + msgs[i].role + "<|end_of_role|>" + msgs[i].content + "<|end_of_text|>" + separator_;
      }
      if (gen) out += "<|start_of_role|>assistant<|end_of_role|>";
      break;
    }

    case ChatFormat::kDeepSeek2:
      if (has_system) out += msgs.front().content;
      for (size_t i = has_system ? 1 : 0; i < msgs.size(); ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user") {
          out += "<｜User｜>" + m.content;
        } else {
          out += "<｜Assistant｜>" + m.content + "<｜end▁of▁sentence｜>";
        }
      }
      if (gen) out += "<｜Assistant｜>";
      break;
  }
  return out;
}

}  // namespace engine
