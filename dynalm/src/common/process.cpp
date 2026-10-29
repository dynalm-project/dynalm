#include "common/process.h"

#include <chrono>
#include <cstring>
#include <map>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace dynalm {

#if defined(_WIN32)

struct ChildProcess::Impl {
  PROCESS_INFORMATION pi{};
};

namespace {

// One argument quoted the way CommandLineToArgvW / the MSVC runtime parse it.
std::string quote_arg(const std::string& a) {
  if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) return a;
  std::string q = "\"";
  for (size_t i = 0;; ++i) {
    size_t backslashes = 0;
    while (i < a.size() && a[i] == '\\') ++i, ++backslashes;
    if (i == a.size()) {
      q.append(backslashes * 2, '\\');
      break;
    }
    if (a[i] == '"') {
      q.append(backslashes * 2 + 1, '\\');
      q.push_back('"');
    } else {
      q.append(backslashes, '\\');
      q.push_back(a[i]);
    }
  }
  q.push_back('"');
  return q;
}

struct CaseLess {
  bool operator()(const std::string& a, const std::string& b) const { return _stricmp(a.c_str(), b.c_str()) < 0; }
};

}  // namespace

Result<std::unique_ptr<ChildProcess>> ChildProcess::start(const std::vector<std::string>& argv, const Env& env,
                                                          const std::string& log) {
  if (argv.empty()) return InvalidArgument("no command");
  std::string cmd;
  for (const std::string& a : argv) cmd += (cmd.empty() ? "" : " ") + quote_arg(a);
  // The environment: the current one with `env` applied (names compare case-insensitively).
  std::map<std::string, std::string, CaseLess> vars;
  if (char* block = GetEnvironmentStringsA(); block != nullptr) {
    for (const char* p = block; *p != '\0'; p += std::strlen(p) + 1) {
      const std::string kv(p);
      const size_t eq = kv.find('=', 1);  // entries like "=C:=C:\\" start with '='
      if (eq != std::string::npos) vars[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    FreeEnvironmentStringsA(block);
  }
  for (const auto& [k, v] : env) vars[k] = v;
  std::string envblock;
  for (const auto& [k, v] : vars) envblock += k + "=" + v + '\0';
  envblock.push_back('\0');

  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE out = CreateFileA(log.empty() ? "NUL" : log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (out == INVALID_HANDLE_VALUE) return IoError("cannot open " + (log.empty() ? std::string("NUL") : log));
  HANDLE in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = in;
  si.hStdOutput = out;
  si.hStdError = out;
  auto child = std::unique_ptr<ChildProcess>(new ChildProcess());
  child->impl_ = std::make_unique<Impl>();
  const BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, envblock.data(),
                                 nullptr, &si, &child->impl_->pi);
  CloseHandle(out);
  if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
  if (!ok) return IoError("cannot start " + argv[0] + " (error " + std::to_string(GetLastError()) + ")");
  return child;
}

bool ChildProcess::running() {
  if (done_) return false;
  DWORD code = 0;
  if (GetExitCodeProcess(impl_->pi.hProcess, &code) && code == STILL_ACTIVE) return true;
  done_ = true;
  exit_code_ = static_cast<int>(code);
  return false;
}

int ChildProcess::stop(int64_t /*grace_ms*/) {
  // Servers keep no state worth flushing here; on Windows a process started
  // without a console cannot receive CTRL_BREAK, so it is terminated.
  if (running()) {
    TerminateProcess(impl_->pi.hProcess, 1);
    WaitForSingleObject(impl_->pi.hProcess, 10000);
    done_ = true;
    exit_code_ = -1;
  }
  return exit_code_;
}

ChildProcess::~ChildProcess() {
  if (impl_ == nullptr) return;
  stop();
  CloseHandle(impl_->pi.hThread);
  CloseHandle(impl_->pi.hProcess);
}

#else  // POSIX

struct ChildProcess::Impl {
  pid_t pid = -1;
};

Result<std::unique_ptr<ChildProcess>> ChildProcess::start(const std::vector<std::string>& argv, const Env& env,
                                                          const std::string& log) {
  if (argv.empty()) return InvalidArgument("no command");
  std::map<std::string, std::string> vars;
  for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
    const std::string kv(*e);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos) vars[kv.substr(0, eq)] = kv.substr(eq + 1);
  }
  for (const auto& [k, v] : env) vars[k] = v;
  std::vector<std::string> envs;
  for (const auto& [k, v] : vars) envs.push_back(k + "=" + v);
  std::vector<char*> envp, args;
  for (std::string& s : envs) envp.push_back(s.data());
  envp.push_back(nullptr);
  std::vector<std::string> argv_copy = argv;
  for (std::string& a : argv_copy) args.push_back(a.data());
  args.push_back(nullptr);

  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 1, log.empty() ? "/dev/null" : log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  auto child = std::unique_ptr<ChildProcess>(new ChildProcess());
  child->impl_ = std::make_unique<Impl>();
  const int rc = posix_spawn(&child->impl_->pid, args[0], &fa, nullptr, args.data(), envp.data());
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) {
    child->done_ = true;
    return IoError("cannot start " + argv[0] + ": " + std::strerror(rc));
  }
  return child;
}

bool ChildProcess::running() {
  if (done_) return false;
  int status = 0;
  const pid_t r = waitpid(impl_->pid, &status, WNOHANG);
  if (r == 0) return true;
  done_ = true;
  exit_code_ = r > 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return false;
}

int ChildProcess::stop(int64_t grace_ms) {
  if (!running()) return exit_code_;
  kill(impl_->pid, SIGTERM);
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(grace_ms);
  while (std::chrono::steady_clock::now() < until) {
    if (!running()) return exit_code_;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  kill(impl_->pid, SIGKILL);
  int status = 0;
  waitpid(impl_->pid, &status, 0);
  done_ = true;
  exit_code_ = -1;
  return exit_code_;
}

ChildProcess::~ChildProcess() {
  if (impl_ != nullptr && impl_->pid > 0) stop(2000);
}

#endif

}  // namespace dynalm
