#pragma once

// Interactive multi-turn chat on a loaded Engine (`dynalm chat`, or
// `dynalm run <model>` without -p).

#include <string>

#include "runtime/engine.h"
#include "common/core.h"

namespace dynalm::cli {

struct ChatSettings {
  std::string system;     // system message ("" = none)
  GenerateParams params;  // sampling, stop strings, max_tokens per reply
  int32_t context = 4096; // KV capacity: history is trimmed to fit
  bool show_stats = false;
};

// Runs the read-eval-print loop on stdin/stdout until /bye or end of input.
int run_chat_session(Engine& engine, ChatSettings settings);

}  // namespace dynalm::cli
