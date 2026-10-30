#pragma once

// A child process in the background (dynalm cpu-report starts servers):
// extra environment variables, stdout and stderr appended to a log file,
// stopped politely (SIGTERM / CTRL_BREAK) and then forcibly after a timeout.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dynacore/base/status.h"
#include "common/core.h"

namespace dynalm {

class ChildProcess {
 public:
  using Env = std::vector<std::pair<std::string, std::string>>;
  // argv[0] is a path (not searched in PATH). `env` adds to or replaces the
  // current environment. `log` empty: output discarded.
  static Result<std::unique_ptr<ChildProcess>> start(const std::vector<std::string>& argv, const Env& env,
                                                     const std::string& log);
  ~ChildProcess();
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  bool running();
  // Asks the process to stop, waits up to `grace_ms`, then kills it. Returns its exit code (-1 if killed or unknown).
  int stop(int64_t grace_ms = 10000);

 private:
  ChildProcess() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool done_ = false;
  int exit_code_ = -1;
};

}  // namespace dynalm
