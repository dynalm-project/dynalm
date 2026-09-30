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

class ThreadPool {
 public:
  // `num_threads` includes the calling thread; 1 = run inline.
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Calls fn(begin, end) over [0, n) in chunks of at most `grain` items,
  // using every thread including the caller. Blocks until all chunks finish.
  // Not reentrant: fn must not call parallel_for on the same pool.
  void parallel_for(size_t n, size_t grain, FunctionRef<void(size_t, size_t)> fn);

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
};

}  // namespace engine
