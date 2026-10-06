#pragma once

// Resource usage of the current process (benchmarks, /metrics).

#include <cstdint>

namespace dynacore {

// Resident set size in bytes (0 if unavailable).
int64_t process_rss_bytes();
// Peak resident set size in bytes (0 if unavailable).
int64_t process_peak_rss_bytes();
// User + system CPU time consumed by the process, in seconds.
double process_cpu_seconds();

}  // namespace dynacore
