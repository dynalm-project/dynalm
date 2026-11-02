#pragma once

// Expert cache: mixture-of-experts weights read from the model file on
// demand into a fixed RAM budget (docs/perf/moe-ssd-plan.md), so a model
// whose experts do not fit in RAM still runs. Everything else in the model
// stays mapped as usual.
//
// - Each expert (gate, up, down of one layer) has a slot. A slot holds the
//   same bytes as the file, in the same layout, so the matmuls that read it
//   compute exactly what they compute on the mapped weights: outputs are
//   bit-identical with and without the cache.
// - Reads use ordinary file I/O that bypasses the OS file cache (macOS
//   F_NOCACHE; Linux drops the pages after each read; Windows: plain reads),
//   so the bytes live once, in the cache, and never fill the page cache.
// - Replacement: a slot not used by the current step with the lowest use
//   count (halved periodically, so old popularity fades), then the least
//   recently used.
// - Misses are read in parallel by I/O threads, each expert split into
//   chunks. prefetch() starts reads in the background (expert prediction);
//   acquire() waits only for what it needs.
//
// Not thread-safe for callers: acquire() and prefetch() come from the one
// thread that runs forward passes. The I/O threads are internal.

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/tensor/tensor.h"
#include "common/core.h"

namespace dynalm {

class ExpertCache {
 public:
  using Parts = std::array<TensorView, 3>;  // gate, up, down

  struct Options {
    int64_t budget_bytes = 0;
    int io_threads = 8;
    int64_t chunk_bytes = int64_t{4} << 20;  // one read request at most
  };

  struct Stats {
    int64_t hits = 0;           // acquired experts already resident (or being prefetched)
    int64_t misses = 0;         // acquired experts that had to be read on demand
    int64_t prefetched = 0;     // reads started by prefetch()
    int64_t prefetch_used = 0;  // of those, acquired before being evicted
    int64_t evictions = 0;
    int64_t bytes_read = 0;
    double read_ms = 0;  // summed over I/O threads
    double wait_ms = 0;  // acquire() blocked on reads
  };

  // `experts[layer][expert]`: views of each expert's weights inside live
  // MappedFiles (an empty inner vector = a dense layer). The budget must hold
  // at least the experts of the layer with the most of them, so any one
  // step's experts fit at once.
  static Result<std::unique_ptr<ExpertCache>> create(const std::vector<std::vector<Parts>>& experts,
                                                     const Options& options);
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;

  // Makes the distinct experts `ids` of `layer` resident and returns their
  // views in `out` (same order). They stay valid and are never evicted until
  // the next acquire().
  Status acquire(int32_t layer, std::span<const int32_t> ids, std::vector<const Parts*>& out);

  // Starts reading experts of `layer` that are not resident, into slots
  // acquire() does not currently hold. Never blocks on I/O; skips what does
  // not fit. A hint only: acquire() is correct without it.
  void prefetch(int32_t layer, std::span<const int32_t> ids);

  Stats stats() const;
  int64_t slots() const { return static_cast<int64_t>(slots_.size()); }
  int64_t slot_bytes() const { return slot_bytes_; }
  // Bytes of all experts in the file (what the cache stands in for).
  int64_t expert_bytes() const { return expert_bytes_; }

 private:
  struct Source {
    int file = 0;
    uint64_t offset = 0;
    int64_t bytes = 0;
    int64_t slot_offset = 0;  // within the slot
  };
  struct Slot {
    int64_t key = -1;  // layer * experts + expert; -1 = free
    uint32_t uses = 0;
    uint64_t last = 0;
    uint64_t pinned = 0;   // == pin_gen_: held by the current acquire
    int pending = 0;       // chunks not yet read
    bool prefetched = false;
    std::string error;
    std::array<TensorView, 3> views;
  };
  struct Job {
    int32_t slot;
    int file;
    uint64_t offset;
    int64_t bytes;
    std::byte* dst;
  };
  class Reader;

  ExpertCache() = default;
  int64_t key(int32_t layer, int32_t e) const { return static_cast<int64_t>(layer) * max_experts_ + e; }
  // A slot to load `k` into (caller holds mu_), or -1 when every slot is busy.
  int32_t victim();
  void start_load(int32_t slot, int64_t k);  // caller holds mu_
  void io_loop();

  std::vector<std::vector<std::array<Source, 3>>> sources_;  // [layer][expert]
  std::vector<std::vector<Parts>> layouts_;                  // the original views (layouts)
  std::vector<std::unique_ptr<Reader>> files_;
  int32_t max_experts_ = 0;
  int64_t slot_bytes_ = 0, expert_bytes_ = 0, chunk_bytes_ = 0;
  std::unique_ptr<std::byte[]> arena_;
  std::byte* base_ = nullptr;
  std::vector<Slot> slots_;
  std::unordered_map<int64_t, int32_t> where_;  // key -> slot
  uint64_t tick_ = 0, pin_gen_ = 0, decay_at_ = 0;

  mutable std::mutex mu_;
  std::condition_variable io_cv_, done_cv_;
  std::deque<Job> jobs_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
  Stats stats_;
};

}  // namespace dynalm
