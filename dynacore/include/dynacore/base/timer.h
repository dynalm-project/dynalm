#pragma once

#include <chrono>
#include <cstdint>

namespace dynacore {

// Monotonic nanoseconds. All latency metrics (TTFT, ITL, queue time) use this.
inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class Stopwatch {
 public:
  Stopwatch() : start_(now_ns()) {}
  void reset() { start_ = now_ns(); }
  int64_t elapsed_ns() const { return now_ns() - start_; }
  double elapsed_ms() const { return static_cast<double>(elapsed_ns()) * 1e-6; }

 private:
  int64_t start_;
};

}  // namespace dynacore
