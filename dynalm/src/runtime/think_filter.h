#pragma once

// Display filter for streamed replies of reasoning models. With reasoning off
// (Qwen3 "/no_think") the model still opens with an empty "<think>\n\n</think>"
// block; this hides it and the blank lines after it. Real thinking is passed
// through as it streams, so the user sees it live.

#include <string>
#include <string_view>

namespace engine {

class ThinkFilter {
 public:
  // Returns the part of `text` to show now (may be empty while undecided).
  std::string feed(std::string_view text) {
    if (state_ == kPass) return std::string(text);
    if (state_ == kAfter) {  // blank lines between the empty block and the answer
      const size_t start = text.find_first_not_of(" \t\r\n");
      if (start == std::string_view::npos) return {};
      state_ = kPass;
      return std::string(text.substr(start));
    }
    held_ += text;
    if (state_ == kStart) {
      constexpr std::string_view kOpen = "<think>";
      if (held_.size() < kOpen.size() && kOpen.starts_with(held_)) return {};  // may still become "<think>"
      if (!std::string_view(held_).starts_with(kOpen)) return release(kPass);
      state_ = kInside;
    }
    const size_t end = held_.find("</think>");
    std::string_view inner = std::string_view(held_).substr(7, end == std::string::npos ? end : end - 7);
    if (end == std::string::npos) {  // a "</think>" may be arriving in pieces: ignore its start
      const size_t lt = inner.rfind('<');
      if (lt != std::string_view::npos && std::string_view("</think>").starts_with(inner.substr(lt))) {
        inner = inner.substr(0, lt);
      }
    }
    if (inner.find_first_not_of(" \t\r\n") != std::string_view::npos) return release(kPass);  // real thinking
    if (end == std::string::npos) return {};
    const size_t answer = held_.find_first_not_of(" \t\r\n", end + 8);
    held_.erase(0, answer == std::string::npos ? held_.size() : answer);
    return release(answer == std::string::npos ? kAfter : kPass);
  }

  // End of the reply: whatever is still held back.
  std::string finish() { return state_ == kStart || state_ == kInside ? release(kPass) : std::string(); }

 private:
  enum State { kStart, kInside, kAfter, kPass };
  std::string release(State next) {
    state_ = next;
    std::string out;
    out.swap(held_);
    return out;
  }
  State state_ = kStart;
  std::string held_;
};

}  // namespace engine
