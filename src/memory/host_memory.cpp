#include "memory/host_memory.h"

#include <cstdlib>

#if ENGINE_OS_WINDOWS
#include <malloc.h>
#endif

namespace engine {
namespace {

// Counters are touched per allocation, not per token; one cache line each
// keeps concurrent allocators from false-sharing.
struct alignas(kCacheLineSize) Counter {
  std::atomic<int64_t> v{0};
};

Counter g_current, g_peak, g_allocs, g_frees;
std::atomic<int64_t> g_limit{0};

void update_peak(int64_t current) {
  int64_t peak = g_peak.v.load(std::memory_order_relaxed);
  while (current > peak &&
         !g_peak.v.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
  }
}

}  // namespace

void* host_alloc(size_t size, size_t alignment) {
  if (size == 0) return nullptr;
  if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) return nullptr;
  const int64_t limit = g_limit.load(std::memory_order_relaxed);
  if (limit > 0 && static_cast<int64_t>(size) > limit) return nullptr;

#if ENGINE_OS_WINDOWS
  void* p = _aligned_malloc(size, alignment);
#else
  void* p = nullptr;
  if (posix_memalign(&p, alignment, size) != 0) p = nullptr;
#endif
  if (!p) return nullptr;

  const int64_t cur = g_current.v.fetch_add(static_cast<int64_t>(size), std::memory_order_relaxed) +
                      static_cast<int64_t>(size);
  update_peak(cur);
  g_allocs.v.fetch_add(1, std::memory_order_relaxed);
  return p;
}

void host_free(void* ptr, size_t size) {
  if (!ptr) return;
#if ENGINE_OS_WINDOWS
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
  g_current.v.fetch_sub(static_cast<int64_t>(size), std::memory_order_relaxed);
  g_frees.v.fetch_add(1, std::memory_order_relaxed);
}

HostMemoryStats host_memory_stats() {
  HostMemoryStats s;
  s.current_bytes = g_current.v.load(std::memory_order_relaxed);
  s.peak_bytes = g_peak.v.load(std::memory_order_relaxed);
  s.alloc_count = g_allocs.v.load(std::memory_order_relaxed);
  s.free_count = g_frees.v.load(std::memory_order_relaxed);
  return s;
}

void set_host_alloc_limit_for_testing(int64_t max_single_alloc_bytes) {
  g_limit.store(max_single_alloc_bytes, std::memory_order_relaxed);
}

}  // namespace engine
