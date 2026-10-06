#pragma once

// Lock-free runtime metrics with Prometheus text exposition.
//
// Counters and gauges are single atomics; histograms use fixed bucket bounds
// with atomic counts, so recording from any thread never blocks. Rendering
// takes a consistent-enough snapshot for monitoring (no global lock).

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "common/core.h"

namespace dynalm::metrics {

class Counter {
 public:
  void inc(uint64_t n = 1) { v_.fetch_add(n, std::memory_order_relaxed); }
  uint64_t value() const { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> v_{0};
};

class Gauge {
 public:
  void set(double v) { v_.store(v, std::memory_order_relaxed); }
  void add(double d) {
    double cur = v_.load(std::memory_order_relaxed);
    while (!v_.compare_exchange_weak(cur, cur + d, std::memory_order_relaxed)) {
    }
  }
  double value() const { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic<double> v_{0};
};

class Histogram {
 public:
  explicit Histogram(std::vector<double> upper_bounds);
  void observe(double v);
  // Approximate quantile from bucket counts (linear within a bucket).
  double quantile(double q) const;
  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  double sum() const { return sum_.load(std::memory_order_relaxed); }
  void render(std::string_view name, std::string_view help, std::string& out) const;

 private:
  std::vector<double> bounds_;
  std::unique_ptr<std::atomic<uint64_t>[]> buckets_;  // bounds_.size() + 1 (last = +Inf)
  std::atomic<uint64_t> count_{0};
  std::atomic<double> sum_{0};
};

// Latency buckets in milliseconds, 1 ms .. 60 s.
std::vector<double> latency_buckets_ms();

void render_counter(std::string_view name, std::string_view help, double value, std::string& out);
void render_gauge(std::string_view name, std::string_view help, double value, std::string& out);

}  // namespace dynalm::metrics
