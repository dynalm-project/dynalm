#pragma once

// Error handling for the engine.
//
// Fallible operations return Status or Result<T>. Exceptions are not used for
// control flow: a failing request must never unwind through the scheduler, and
// explicit errors keep failure paths visible at every subsystem boundary.
// The OK path allocates nothing (the message string stays empty).

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace engine {

enum class StatusCode : uint8_t {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kOutOfMemory,
  kResourceExhausted,  // capacity limit reached (e.g. KV blocks), retryable
  kUnsupported,        // valid input the engine does not handle yet
  kIoError,
  kCorrupt,            // malformed file / data
  kCancelled,
  kDeadlineExceeded,
  kInternal,
};

std::string_view status_code_name(StatusCode code);

class [[nodiscard]] Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }
  std::string to_string() const;

  friend bool operator==(const Status& a, const Status& b) { return a.code_ == b.code_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

inline Status InvalidArgument(std::string m) { return {StatusCode::kInvalidArgument, std::move(m)}; }
inline Status NotFound(std::string m) { return {StatusCode::kNotFound, std::move(m)}; }
inline Status OutOfMemory(std::string m) { return {StatusCode::kOutOfMemory, std::move(m)}; }
inline Status ResourceExhausted(std::string m) { return {StatusCode::kResourceExhausted, std::move(m)}; }
inline Status Unsupported(std::string m) { return {StatusCode::kUnsupported, std::move(m)}; }
inline Status IoError(std::string m) { return {StatusCode::kIoError, std::move(m)}; }
inline Status Corrupt(std::string m) { return {StatusCode::kCorrupt, std::move(m)}; }
inline Status Cancelled(std::string m) { return {StatusCode::kCancelled, std::move(m)}; }
inline Status Internal(std::string m) { return {StatusCode::kInternal, std::move(m)}; }

// Result<T>: either a value or a non-OK Status.
template <typename T>
class [[nodiscard]] Result {
 public:
  Result(T value) : v_(std::move(value)) {}  // NOLINT(implicit)
  Result(Status status) : v_(std::move(status)) {  // NOLINT(implicit)
    assert(!std::get<Status>(v_).ok() && "Result constructed from OK status without value");
  }

  bool ok() const { return std::holds_alternative<T>(v_); }

  const Status& status() const {
    static const Status kOk;
    return ok() ? kOk : std::get<Status>(v_);
  }

  T& value() & { assert(ok()); return std::get<T>(v_); }
  const T& value() const& { assert(ok()); return std::get<T>(v_); }
  T&& value() && { assert(ok()); return std::get<T>(std::move(v_)); }

  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }
  T& operator*() & { return value(); }
  const T& operator*() const& { return value(); }

 private:
  std::variant<Status, T> v_;
};

}  // namespace engine

#define ENGINE_CONCAT_INNER(a, b) a##b
#define ENGINE_CONCAT(a, b) ENGINE_CONCAT_INNER(a, b)

// Propagate a non-OK Status.
#define ENGINE_RETURN_IF_ERROR(expr)              \
  do {                                            \
    ::engine::Status _st = (expr);                \
    if (!_st.ok()) return _st;                    \
  } while (0)

// `ENGINE_ASSIGN_OR_RETURN(auto x, make_x());`
#define ENGINE_ASSIGN_OR_RETURN(lhs, expr) \
  ENGINE_ASSIGN_OR_RETURN_IMPL(ENGINE_CONCAT(_res_, __LINE__), lhs, expr)
#define ENGINE_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr) \
  auto tmp = (expr);                                 \
  if (!tmp.ok()) return tmp.status();                \
  lhs = std::move(tmp).value()
