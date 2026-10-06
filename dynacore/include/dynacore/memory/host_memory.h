#pragma once

// Host (CPU) memory: aligned allocation with global statistics.
//
// All engine-owned host buffers go through here so memory usage is observable
// (metrics, `engine info`, OOM diagnostics). This is the bottom allocator;
// pools (KV blocks, scratch arenas) sit on top and allocate large slabs from it
// rather than calling it per object.

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "dynacore/base/platform.h"

namespace dynacore {

inline constexpr size_t kDefaultAlignment = 64;  // cache line; also AVX-512 width

struct HostMemoryStats {
  int64_t current_bytes = 0;
  int64_t peak_bytes = 0;
  int64_t alloc_count = 0;
  int64_t free_count = 0;
};

// Returns nullptr on failure or invalid alignment (must be a power of two
// >= sizeof(void*)). size 0 returns nullptr. Never throws.
void* host_alloc(size_t size, size_t alignment = kDefaultAlignment);
void host_free(void* ptr, size_t size);  // size must match the allocation

HostMemoryStats host_memory_stats();

// Test hook: when > 0, allocations larger than this many bytes fail. Lets tests
// exercise OOM paths deterministically. 0 disables.
void set_host_alloc_limit_for_testing(int64_t max_single_alloc_bytes);

}  // namespace dynacore
