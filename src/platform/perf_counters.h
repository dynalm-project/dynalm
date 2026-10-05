#pragma once

// Process-wide performance counters for benchmarks (DD-050).
//
// Software counters (context switches, page faults) come from the OS
// accounting of the whole process. Hardware counters (cycles, instructions,
// cache and branch misses) and CPU migrations use Linux perf_event_open, opened
// per thread because inherited counters only fold a thread's counts into the
// parent when it exits, and the kernel worker pool never exits. Where a counter
// cannot be read (Windows, macOS, VMs without a virtual PMU such as WSL2,
// perf_event_paranoid > 2) its value is -1 and `hardware_reason` says why.
// Nothing here runs on the inference hot path.

#include <cstdint>
#include <memory>
#include <string>

namespace engine {

struct PerfSample {
  // -1 = unavailable on this platform / configuration.
  int64_t cycles = -1;
  int64_t instructions = -1;
  int64_t cache_references = -1;  // last-level cache accesses
  int64_t cache_misses = -1;      // last-level cache misses
  int64_t branch_misses = -1;
  int64_t l1d_misses = -1;        // L1 data-cache read misses
  int64_t cpu_migrations = -1;
  int64_t context_switches = -1;  // voluntary + involuntary
  int64_t page_faults = -1;       // minor + major
  int64_t major_faults = -1;      // faults that read from disk

  // b - a per field (stays -1 when either side is unavailable).
  static PerfSample delta(const PerfSample& a, const PerfSample& b);
};

class PerfCounters {
 public:
  // Opens counters for every thread that exists now (call after the engine
  // has started its worker threads). Threads created later are not counted.
  static std::unique_ptr<PerfCounters> open();
  ~PerfCounters();

  PerfSample read() const;
  bool hardware_available() const { return hw_available_; }
  const std::string& hardware_reason() const { return hw_reason_; }

 private:
  PerfCounters() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool hw_available_ = false;
  std::string hw_reason_;
};

// Average current frequency over all logical CPUs in MHz, or -1 if unknown.
double cpu_current_mhz();

}  // namespace engine
