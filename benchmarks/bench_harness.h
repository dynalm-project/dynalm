#pragma once

// Minimal benchmark harness. Every result reports P50/P90/P95/P99; never
// optimize from averages alone.
//
// Each sample times `batch` calls and records per-call time, which keeps
// timer overhead out of nanosecond-scale measurements.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common/timer.h"

namespace engine::bench {

struct Stats {
  double mean = 0, p50 = 0, p90 = 0, p95 = 0, p99 = 0, min = 0, max = 0;
  size_t samples = 0;
};

inline double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0;
  // Nearest-rank on the sorted samples.
  const size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
  return sorted[std::clamp<size_t>(rank, 1, sorted.size()) - 1];
}

inline Stats summarize(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.samples = v.size();
  s.mean = sum / v.size();
  s.min = v.front();
  s.max = v.back();
  s.p50 = percentile(v, 50);
  s.p90 = percentile(v, 90);
  s.p95 = percentile(v, 95);
  s.p99 = percentile(v, 99);
  return s;
}

// Prevents the compiler from discarding a computed value.
template <typename T>
inline void do_not_optimize(T const& value) {
  // A volatile read of the value's address is portable across MSVC/GCC/Clang.
  static volatile const void* sink;
  sink = &value;
}

struct Options {
  int warmup_samples = 10;
  int samples = 200;
  int batch = 1000;
};

// Runs fn() `samples * batch` times; returns per-call nanoseconds.
template <typename Fn>
Stats run(Fn&& fn, const Options& opt = {}) {
  for (int i = 0; i < opt.warmup_samples * opt.batch; ++i) fn();
  std::vector<double> ns;
  ns.reserve(opt.samples);
  for (int s = 0; s < opt.samples; ++s) {
    const int64_t t0 = now_ns();
    for (int i = 0; i < opt.batch; ++i) fn();
    ns.push_back(static_cast<double>(now_ns() - t0) / opt.batch);
  }
  return summarize(std::move(ns));
}

inline void print_header() {
  std::printf("%-40s %10s %10s %10s %10s %10s\n", "benchmark (ns/op)", "mean", "p50", "p90",
              "p95", "p99");
}

inline void print_row(const std::string& name, const Stats& s) {
  std::printf("%-40s %10.2f %10.2f %10.2f %10.2f %10.2f\n", name.c_str(), s.mean, s.p50, s.p90,
              s.p95, s.p99);
}

}  // namespace engine::bench
