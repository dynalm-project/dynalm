#pragma once

// Execution-speed hints for compute threads (DD-052).
//
// Hybrid CPUs (Intel P/E cores, Apple Silicon) and power-saving OS policies
// can run a "background-looking" process on efficiency cores or at reduced
// clocks: on Windows 11, EcoQoS power throttling halved decode speed of the
// same engine depending only on which process hosted it. Inference compute
// threads are latency- and throughput-critical, so they ask for full speed:
//   Windows: opt the process and the thread out of power throttling (EcoQoS).
//   macOS:   QOS_CLASS_USER_INITIATED (eligible for performance cores).
//   Linux:   no-op (the scheduler has no equivalent per-thread hint).
// Calls are cheap, idempotent and never fail loudly.

namespace engine {

// Process-wide hint; call once early (engine creation).
void request_full_speed_process();
// Hint for the calling thread; call at the start of every compute thread.
void request_full_speed_thread();

}  // namespace engine
