#include "dynacore/base/status.h"

namespace engine {

std::string_view status_code_name(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "OK";
    case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case StatusCode::kNotFound: return "NOT_FOUND";
    case StatusCode::kAlreadyExists: return "ALREADY_EXISTS";
    case StatusCode::kOutOfMemory: return "OUT_OF_MEMORY";
    case StatusCode::kResourceExhausted: return "RESOURCE_EXHAUSTED";
    case StatusCode::kUnsupported: return "UNSUPPORTED";
    case StatusCode::kIoError: return "IO_ERROR";
    case StatusCode::kCorrupt: return "CORRUPT";
    case StatusCode::kCancelled: return "CANCELLED";
    case StatusCode::kDeadlineExceeded: return "DEADLINE_EXCEEDED";
    case StatusCode::kInternal: return "INTERNAL";
  }
  return "UNKNOWN";
}

std::string Status::to_string() const {
  if (ok()) return "OK";
  std::string s(status_code_name(code_));
  if (!message_.empty()) {
    s += ": ";
    s += message_;
  }
  return s;
}

}  // namespace engine

