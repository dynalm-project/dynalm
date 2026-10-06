#include "dynacore/execution/thread_pool.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>

#include "dynacore/base/timer.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/thread_qos.h"

#if ENGINE_ARCH_X86_64
#include <immintrin.h>
#endif

namespace dynacore {
namespace {

// ~100 µs of spinning on current x86 before a worker sleeps.
constexpr int kSpinIterations = 2000;

inline void cpu_relax() {
#if ENGINE_ARCH_X86_64
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

ThreadPool::ThreadPool(int num_threads) {
  const int workers = std::max(0, num_threads - 1);
  workers_.reserve(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i) {
    workers_.emplace_back([this, i] {
      request_full_speed_thread();  // compute threads must not be power-throttled (DD-052)
      if (pinning_enabled()) {
        const auto& cores = cpu_info().core_first_cpu;
        if (static_cast<size_t>(i) + 1 < cores.size()) pin_current_thread(cores[static_cast<size_t>(i) + 1]);
      }
      worker_loop();
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run_chunks() {
  for (;;) {
    const size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
    if (begin >= n_) return;
    (*fn_)(begin, std::min(begin + grain_, n_));
  }
}

void ThreadPool::worker_loop() {
  // Start from the construction-time epoch (0), not a fresh load: a job may
  // already have been published before this thread got scheduled, and it
  // must still be counted as unseen.
  uint64_t seen = 0;
  for (;;) {
    uint64_t e;
    int spins = 0;
    while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
      if (stop_.load(std::memory_order_acquire)) return;
      if (++spins < kSpinIterations) {
        cpu_relax();
        continue;
      }
      sleeps_.fetch_add(1, std::memory_order_relaxed);
      // Announce before re-checking under the lock: with the publisher's
      // seq_cst epoch increment and sleeper check, either the publisher sees
      // this sleeper and notifies, or this thread sees the new epoch (DD-055).
      sleepers_.fetch_add(1, std::memory_order_seq_cst);
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] {
          return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_acquire);
        });
      }
      sleepers_.fetch_sub(1, std::memory_order_seq_cst);
      spins = 0;
    }
    seen = e;
    run_chunks();
    active_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

bool ThreadPool::pinning_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DYNACORE_PIN_THREADS");
    return v && v[0] == '1';
  }();
  return on;
}

void ThreadPool::pin_caller() const {
  if (!pinning_enabled()) return;
  const auto& cores = cpu_info().core_first_cpu;
  if (!cores.empty()) pin_current_thread(cores[0]);
}

void ThreadPool::parallel_for(size_t n, size_t grain, FunctionRef<void(size_t, size_t)> fn) {
  if (n == 0) return;
  grain = std::max<size_t>(grain, 1);
  if (workers_.empty() || n <= grain) {
    if (stats_on_) ++stats_.inline_regions;
    fn(0, n);
    return;
  }
  assert(active_.load() == 0 && "parallel_for is not reentrant");
  const int64_t t0 = stats_on_ ? now_ns() : 0;
  fn_ = &fn;
  n_ = n;
  grain_ = grain;
  next_.store(0, std::memory_order_relaxed);
  active_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
  // Spinning workers see the epoch directly. Only when some worker sleeps
  // (between steps, or after a long op) is the mutex + notify needed; taking
  // the lock before notifying means a worker between its predicate check and
  // its wait cannot miss the wake-up (DD-055).
  epoch_.fetch_add(1, std::memory_order_seq_cst);
  if (sleepers_.load(std::memory_order_seq_cst) > 0) {
    { std::lock_guard<std::mutex> lock(mu_); }
    cv_.notify_all();
  }

  run_chunks();
  const int64_t t_own = stats_on_ ? now_ns() : 0;
  // Wait for workers to finish their last chunk (they never block mid-job).
  while (active_.load(std::memory_order_acquire) != 0) cpu_relax();
  fn_ = nullptr;
  if (stats_on_) {
    const int64_t t1 = now_ns();
    ++stats_.regions;
    stats_.region_ns += t1 - t0;
    stats_.tail_wait_ns += t1 - t_own;
  }
}

}  // namespace dynacore
