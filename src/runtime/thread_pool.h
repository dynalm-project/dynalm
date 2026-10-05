#pragma once

// Persistent worker pool for kernel parallelism.
//
// One pool per model runtime; threads are created once and live for the
// process. parallel_for hands out chunks dynamically from an atomic counter,
// so fast P-cores take more chunks than E-cores on hybrid CPUs instead of
// everyone waiting for the slowest static partition.
//
// Workers spin briefly after each job before sleeping: a decode step launches
// hundreds of small kernels, and a condition-variable wake per launch would
// add milliseconds per token.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "common/function_ref.h"
#include "common/platform.h"

namespace engine {

// Parallel-region accounting (DD-050), collected only while enabled: tells
// dispatch-bound (many tiny regions) from imbalance/synchronization-bound
// (the caller waiting at the end of a region for slower workers).
struct ThreadPoolStats {
  uint64_t regions = 0;         // parallel_for calls that used the workers
  uint64_t inline_regions = 0;  // calls small enough to run on the caller alone
  int64_t region_ns = 0;        // wall time of the parallel regions
  int64_t tail_wait_ns = 0;     // caller idle after its own chunks, waiting for workers
  uint64_t sleeps = 0;          // times a worker gave up spinning and slept (always counted)
};

class ThreadPool {
 public:
  // `num_threads` includes the calling thread; 1 = run inline.
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // One compute thread per physical core, performance cores first (P10,
  // DD-059): with DYNALM_PIN_THREADS=1, worker i is bound to core i + 1 and
  // the thread calling pin_caller() (the scheduler) to core 0.
  static bool pinning_enabled();
  void pin_caller() const;

  // Calls fn(begin, end) over [0, n) in chunks of at most `grain` items,
  // using every thread including the caller. Blocks until all chunks finish.
  // Not reentrant: fn must not call parallel_for on the same pool.
  void parallel_for(size_t n, size_t grain, FunctionRef<void(size_t, size_t)> fn);

  // Statistics are owned by the thread that calls parallel_for (one scheduler
  // thread); toggle and read them from that thread or while it is quiescent.
  void set_stats_enabled(bool on) { stats_on_ = on; }
  ThreadPoolStats stats() const {
    ThreadPoolStats s = stats_;
    s.sleeps = sleeps_.load(std::memory_order_relaxed);
    return s;
  }

 private:
  void worker_loop();
  void run_chunks();

  std::vector<std::thread> workers_;

  // Current job (published by epoch_ increment with release ordering).
  FunctionRef<void(size_t, size_t)>* fn_ = nullptr;
  size_t n_ = 0;
  size_t grain_ = 1;
  alignas(kCacheLineSize) std::atomic<size_t> next_{0};
  alignas(kCacheLineSize) std::atomic<int> active_{0};
  alignas(kCacheLineSize) std::atomic<uint64_t> epoch_{0};

  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};

  bool stats_on_ = false;
  ThreadPoolStats stats_;
  alignas(kCacheLineSize) std::atomic<uint64_t> sleeps_{0};
  alignas(kCacheLineSize) std::atomic<int> sleepers_{0};  // workers blocked on cv_
};

}  // namespace engine
