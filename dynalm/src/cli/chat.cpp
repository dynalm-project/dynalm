// Interactive chat: one model load, many turns. The conversation is re-sent
// each turn; the radix prefix cache recognises the unchanged history, so only
// the new message is prefilled.

#include "cli/chat.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "dynacore/base/timer.h"
#include "runtime/think_filter.h"
#include "common/core.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#endif

namespace dynalm::cli {
namespace {

std::atomic<bool> g_interrupt{false};
void on_sigint(int) {
  g_interrupt.store(true);
  std::signal(SIGINT, on_sigint);  // stay installed (Windows resets handlers)
}

// One line from stdin, UTF-8, without the newline. False at end of input.
bool read_line(std::string& out) {
  out.clear();
#if defined(_WIN32)
  // A console delivers UTF-16; narrow reads lose non-ASCII text.
  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(in, &mode)) {
    std::wstring w;
    wchar_t buf[512];
    for (;;) {
      DWORD got = 0;
      if (!ReadConsoleW(in, buf, 512, &got, nullptr)) return false;
      if (got == 0) {  // Ctrl+C at the prompt (the handler runs on another thread), or EOF
        Sleep(50);
        return g_interrupt.load();
      }
      w.append(buf, got);
      if (w.find(L'\n') != std::wstring::npos) break;
      if (w.size() == 1 && w[0] == 0x1A) return false;  // Ctrl+Z
    }
    while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
    if (w == L"\x1A") return false;
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    out.resize(static_cast<size_t>(n));
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), out.data(), n, nullptr, nullptr);
    return true;
  }
#endif
  char buf[4096];
  bool any = false;
  while (std::fgets(buf, sizeof buf, stdin)) {
    any = true;
    out += buf;
    if (!out.empty() && out.back() == '\n') break;
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return any || g_interrupt.load();
}

std::string trim(std::string s) {
  const size_t b = s.find_first_not_of(" \t");
  const size_t e = s.find_last_not_of(" \t");
  return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

// Reasoning models put their thinking in <think>...</think>; it is shown but
// not kept in the history (the chat templates of these models drop it too).
std::string strip_thinking(const std::string& s) {
  const size_t a = s.find("<think>");
  const size_t b = s.find("</think>");
  if (a == std::string::npos || b == std::string::npos || b < a) return s;
  return trim(s.substr(0, a) + s.substr(b + 8));
}

void print_help() {
  std::printf(
      "Commands:\n"
      "  /help                 this help\n"
      "  /bye                  exit (also /exit, /quit, Ctrl+D or Ctrl+Z+Enter)\n"
      "  /clear                forget the conversation so far\n"
      "  /system [TEXT]        set the system message (no TEXT: show it); clears the conversation\n"
      "  /think on|off         reasoning on or off (Qwen3)\n"
      "  /set temp|top_p|top_k|min_p|repeat_penalty|max_tokens|seed VALUE\n"
      "  /show                 model and current settings\n"
      "  /stats on|off         print speed after each reply\n"
      "  \"\"\"                   start or end a multi-line message\n"
      "Ctrl+C stops the current reply.\n");
}

bool parse_num(const std::string& s, float& out) {
  try {
    size_t used = 0;
    out = std::stof(s, &used);
    return used == s.size();
  } catch (...) {
    return false;
  }
}

}  // namespace

int run_chat_session(Engine& engine, ChatSettings st) {
  const LoadedModel& lm = engine.model();
  if (!lm.chat_template) {
    std::fprintf(stderr, "chat: this model has no chat template; use `dynalm run <model> --raw -p TEXT`\n");
    return 1;
  }
  std::signal(SIGINT, on_sigint);
  std::vector<ChatMessage> history;
  bool think = true;
  // Qwen3 switches its reasoning off with "/no_think" in the user message.
  const bool has_think_switch = lm.config.architecture.starts_with("qwen3");
  const std::string name = lm.config.name.empty() ? lm.config.architecture : lm.config.name;
  std::printf("Chatting with %s. Type /help for commands, /bye to exit.\n", name.c_str());

  for (;;) {
    g_interrupt.store(false);
    std::printf(">>> ");
    std::fflush(stdout);
    std::string line;
    if (!read_line(line)) {
      std::printf("\n");
      break;
    }
    if (g_interrupt.exchange(false) && line.empty()) {
      std::printf("\nUse /bye or Ctrl+D to exit.\n");
      continue;
    }
    if (trim(line) == "\"\"\"") {  // multi-line message
      std::string more, all;
      for (;;) {
        std::printf("... ");
        std::fflush(stdout);
        if (!read_line(more) || trim(more) == "\"\"\"") break;
        all += (all.empty() ? "" : "\n") + more;
      }
      line = all;
    }
    line = trim(line);
    if (line.empty()) continue;

    if (line[0] == '/') {
      const size_t sp = line.find(' ');
      const std::string cmd = line.substr(0, sp);
      const std::string arg = sp == std::string::npos ? "" : trim(line.substr(sp + 1));
      if (cmd == "/bye" || cmd == "/exit" || cmd == "/quit") break;
      if (cmd == "/help" || cmd == "/?") {
        print_help();
      } else if (cmd == "/clear") {
        history.clear();
        std::printf("Conversation cleared.\n");
      } else if (cmd == "/system") {
        if (arg.empty()) {
          std::printf("System message: %s\n", st.system.empty() ? "(none)" : st.system.c_str());
        } else {
          st.system = arg;
          history.clear();
          std::printf("System message set; conversation cleared.\n");
        }
      } else if (cmd == "/think") {
        if (!has_think_switch) {
          std::printf("This model has no thinking mode to switch (supported: Qwen3).\n");
        } else if (arg == "on" || arg == "off") {
          think = arg == "on";
          std::printf("Thinking %s.\n", think ? "on" : "off");
        } else {
          std::printf("usage: /think on|off\n");
        }
      } else if (cmd == "/stats") {
        st.show_stats = arg != "off";
        std::printf("Stats %s.\n", st.show_stats ? "on" : "off");
      } else if (cmd == "/set") {
        const size_t sp2 = arg.find(' ');
        const std::string key = arg.substr(0, sp2);
        float v = 0;
        SamplingParams next = st.params.sampling;
        int32_t max_tokens = st.params.max_tokens;
        bool ok = sp2 != std::string::npos && parse_num(trim(arg.substr(sp2 + 1)), v);
        if (ok) {
          if (key == "temp" || key == "temperature") next.temperature = v;
          else if (key == "top_p") next.top_p = v;
          else if (key == "top_k") next.top_k = static_cast<int32_t>(v);
          else if (key == "min_p") next.min_p = v;
          else if (key == "repeat_penalty") next.repetition_penalty = v;
          else if (key == "seed") next.seed = static_cast<uint64_t>(v), next.has_seed = true;
          else if (key == "max_tokens") max_tokens = static_cast<int32_t>(v);
          else ok = false;
        }
        if (ok && next.validate().ok() && max_tokens > 0) {
          st.params.sampling = next;
          st.params.max_tokens = max_tokens;
          std::printf("Set %s.\n", key.c_str());
        } else {
          std::printf("usage: /set temp|top_p|top_k|min_p|repeat_penalty|max_tokens|seed VALUE\n");
        }
      } else if (cmd == "/show") {
        const SamplingParams& p = st.params.sampling;
        std::printf("Model: %s (%s, %s)\nContext: %d tokens, %zu messages in history\n", name.c_str(),
                    lm.config.architecture.c_str(), lm.quantization.c_str(), st.context, history.size());
        std::printf("temp %.2f  top_k %d  top_p %.2f  min_p %.2f  repeat_penalty %.2f  max_tokens %d  thinking %s\n",
                    p.temperature, p.top_k, p.top_p, p.min_p, p.repetition_penalty, st.params.max_tokens,
                    think ? "on" : "off");
        std::printf("System message: %s\n", st.system.empty() ? "(none)" : st.system.c_str());
      } else {
        std::printf("Unknown command %s. Type /help.\n", cmd.c_str());
      }
      continue;
    }

    // Build the request: system + history + this message. Oldest turns are
    // dropped until the prompt plus a reply fits in the context.
    const std::string user = think || !has_think_switch ? line : line + " /no_think";
    std::vector<ChatMessage> msgs;
    int32_t prompt_tokens = 0;
    for (;;) {
      msgs.clear();
      if (!st.system.empty()) msgs.push_back({"system", st.system});
      msgs.insert(msgs.end(), history.begin(), history.end());
      msgs.push_back({"user", user});
      auto text = lm.chat_template->apply(msgs, true);
      if (!text.ok()) {
        std::fprintf(stderr, "chat: %s\n", text.status().to_string().c_str());
        break;
      }
      prompt_tokens = static_cast<int32_t>(lm.tokenizer->encode(*text, true, true).size());
      if (prompt_tokens + 32 <= st.context || history.size() < 2) break;
      history.erase(history.begin(), history.begin() + 2);  // forget the oldest exchange
    }
    if (prompt_tokens + 32 > st.context) {
      std::printf("That message is too long for the %d-token context (start with -c to raise it).\n", st.context);
      continue;
    }
    GenerateParams params = st.params;
    params.max_tokens = std::min(params.max_tokens, st.context - prompt_tokens);
    auto stream = engine.generate_chat(msgs, params);
    if (!stream.ok()) {
      std::fprintf(stderr, "chat: %s\n", stream.status().to_string().c_str());
      continue;
    }

    const Stopwatch timer;
    double first_ms = -1;
    std::string reply;
    ThinkFilter shown;
    StreamEvent ev;
    bool cancelled = false;
    for (;;) {
      if (g_interrupt.exchange(false) && !cancelled) {
        (*stream)->cancel();
        cancelled = true;
      }
      if (!(*stream)->next(ev, std::chrono::milliseconds(100))) {
        if ((*stream)->finished()) break;
        continue;
      }
      if (!ev.text.empty()) {
        if (first_ms < 0) first_ms = timer.elapsed_ms();
        reply += ev.text;
        const std::string show = shown.feed(ev.text);
        std::fwrite(show.data(), 1, show.size(), stdout);
        std::fflush(stdout);
      }
      if (ev.done) break;
    }
    const std::string rest = shown.finish();
    std::fwrite(rest.data(), 1, rest.size(), stdout);
    std::printf("\n");
    if (ev.finish == StreamFinish::kError) {
      std::fprintf(stderr, "chat: %s\n", ev.error.to_string().c_str());
      continue;
    }
    if (cancelled) std::printf("(stopped)\n");
    if (ev.finish == StreamFinish::kLength && !cancelled) {
      std::printf("(reply cut at %d tokens: /set max_tokens N for longer replies)\n", params.max_tokens);
    }
    history.push_back({"user", user});
    history.push_back({"assistant", strip_thinking(reply)});
    if (st.show_stats) {
      const double total = timer.elapsed_ms();
      const double decode_s = (total - std::max(first_ms, 0.0)) / 1e3;
      std::printf("[%d prompt tokens, first token %.0f ms, %d tokens at %.1f tok/s]\n", ev.prompt_tokens, first_ms,
                  ev.completion_tokens, decode_s > 0 ? (ev.completion_tokens - 1) / decode_s : 0.0);
    }
  }
  std::signal(SIGINT, SIG_DFL);
  return 0;
}

}  // namespace dynalm::cli
