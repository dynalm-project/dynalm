// Prompt cache on disk (--prompt-cache DIR): a PrefixCache around the
// in-memory one that also keeps full KV blocks as files, so a prompt prefix
// computed once (a long system prompt, a document, a chat history) is loaded
// instead of recomputed, also after a restart.
//
// - Keys: the hash cache's chained block hash (block i's key covers tokens
//   [0, (i+1) * block_size)). Each file stores its parent key and its
//   tokens, checked on load, so a hash collision can never load wrong KV.
// - The directory is per model and KV format (the caller's choice: the
//   engine names it after the model file and the KV geometry).
// - lookup(): the in-memory match first; then the longest chain of further
//   full blocks found on disk, read into newly allocated blocks and offered
//   to the in-memory cache.
// - insert(): the in-memory cache, then every full block not yet on disk is
//   copied out and written by a background thread (the scheduler does not
//   wait for the disk). At most 512 MiB wait to be written; beyond that,
//   blocks are not saved.
// - Size: at most max_bytes of files; the oldest are deleted first.
//
// Not thread-safe for callers (the scheduler thread); the writer is internal.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "prefix_cache/prefix_cache.h"
#include "common/core.h"

namespace dynalm {
namespace {

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'D', 'L', 'K', 'V', 'B', 'L', 'K', '1'};
constexpr int64_t kMaxPending = int64_t{512} << 20;

inline uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t chain_hash(uint64_t parent, std::span<const TokenId> tokens) {
  uint64_t h = mix(parent ^ 0x5bd1e995ull);
  for (TokenId t : tokens) h = mix(h ^ static_cast<uint32_t>(t));
  return h == 0 ? 1 : h;
}

struct Header {
  char magic[8];
  uint64_t key, parent;
  int64_t block_size, bytes;
};

class DiskPrefixCache final : public PrefixCache {
 public:
  DiskPrefixCache(std::unique_ptr<PrefixCache> inner, KvBlockPool& pool, std::string dir, int64_t max_bytes)
      : inner_(std::move(inner)), pool_(pool), dir_(std::move(dir)), max_bytes_(max_bytes) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    // Existing files, oldest first.
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    for (const auto& e : fs::directory_iterator(dir_, ec)) {
      if (e.is_regular_file(ec) && e.path().extension() == ".kvb") files.emplace_back(e.last_write_time(ec), e.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& [t, p] : files) {
      const uint64_t key = std::strtoull(p.stem().string().c_str(), nullptr, 16);
      if (key == 0) continue;
      on_disk_.insert(key);
      order_.push_back(key);
      bytes_ += static_cast<int64_t>(fs::file_size(p, ec));
    }
    writer_ = std::thread([this] { write_loop(); });
  }

  ~DiskPrefixCache() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    writer_.join();
  }

  Match lookup(std::span<const TokenId> tokens, int32_t max_tokens) override {
    Match m = inner_->lookup(tokens, max_tokens);
    const int32_t bs = pool_.geometry().block_size;
    const auto limit = static_cast<int32_t>(std::min<int64_t>(max_tokens, static_cast<int64_t>(tokens.size())));
    const int32_t avail = limit / bs, have = m.tokens / bs;
    if (avail <= have) return m;
    // Keys of every block up to `avail`; the longest run on disk after `have`.
    keys_.clear();
    uint64_t parent = 0;
    for (int32_t i = 0; i < avail; ++i) {
      parent = chain_hash(parent, tokens.subspan(static_cast<size_t>(i) * bs, static_cast<size_t>(bs)));
      keys_.push_back(parent);
    }
    int32_t end = have;
    {
      std::lock_guard<std::mutex> lock(mu_);
      while (end < avail && on_disk_.count(keys_[static_cast<size_t>(end)]) != 0) ++end;
    }
    if (end == have) return m;
    // Replace a partial last block (radix) with full blocks from disk.
    if (static_cast<int32_t>(m.blocks.size()) > have) {
      for (size_t i = static_cast<size_t>(have); i < m.blocks.size(); ++i) pool_.release(m.blocks[i]);
      m.blocks.resize(static_cast<size_t>(have));
    }
    const int32_t before = static_cast<int32_t>(m.blocks.size());
    buf_.resize(static_cast<size_t>(pool_.block_bytes()));
    for (int32_t i = have; i < end; ++i) {
      auto blk = pool_.allocate();
      if (!blk.ok()) break;
      if (!load(keys_[static_cast<size_t>(i)], i == 0 ? 0 : keys_[static_cast<size_t>(i) - 1],
                tokens.subspan(static_cast<size_t>(i) * bs, static_cast<size_t>(bs)))) {
        pool_.release(*blk);
        break;
      }
      pool_.write_block(*blk, buf_.data());
      m.blocks.push_back(*blk);
    }
    // Full blocks only from here (a partial one was dropped above).
    const auto n = static_cast<int32_t>(m.blocks.size());
    m.tokens = n * bs;
    if (n > before) {
      disk_blocks_ += static_cast<uint64_t>(n - before);
      inner_->insert(tokens, m.blocks, have, n);
    }
    return m;
  }

  void insert(std::span<const TokenId> tokens, std::span<const int32_t> blocks, int32_t first_block,
              int32_t num_full_blocks) override {
    inner_->insert(tokens, blocks, first_block, num_full_blocks);
    const int32_t bs = pool_.geometry().block_size;
    uint64_t parent = 0;
    for (int32_t i = 0; i < num_full_blocks; ++i) {
      const uint64_t key = chain_hash(parent, tokens.subspan(static_cast<size_t>(i) * bs, static_cast<size_t>(bs)));
      if (i >= first_block) {
        std::unique_lock<std::mutex> lock(mu_);
        if (on_disk_.count(key) == 0 && queued_.count(key) == 0) {
          if (pending_bytes_ + pool_.block_bytes() > kMaxPending) return;  // the disk is behind: skip
          queued_.insert(key);
          Job j;
          j.key = key;
          j.parent = parent;
          j.tokens.assign(tokens.begin() + static_cast<ptrdiff_t>(i) * bs, tokens.begin() + static_cast<ptrdiff_t>(i + 1) * bs);
          j.data.resize(static_cast<size_t>(pool_.block_bytes()));
          lock.unlock();
          pool_.read_block(blocks[static_cast<size_t>(i)], j.data.data());
          lock.lock();
          pending_bytes_ += static_cast<int64_t>(j.data.size());
          jobs_.push_back(std::move(j));
          cv_.notify_one();
        }
      }
      parent = key;
    }
  }

  int32_t evict(int32_t n) override { return inner_->evict(n); }
  const PrefixCacheStats& stats() const override {
    merged_ = inner_->stats();
    merged_.disk_loaded_blocks = disk_blocks_;
    merged_.hit_tokens += disk_blocks_ * static_cast<uint64_t>(pool_.geometry().block_size);
    return merged_;
  }

  // Tests: wait until queued writes are on disk.
  void flush() {
    std::unique_lock<std::mutex> lock(mu_);
    idle_cv_.wait(lock, [&] { return jobs_.empty() && !writing_; });
  }

 private:
  struct Job {
    uint64_t key = 0, parent = 0;
    std::vector<TokenId> tokens;
    std::vector<std::byte> data;
  };

  fs::path path_of(uint64_t key) const {
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.kvb", static_cast<unsigned long long>(key));
    return fs::path(dir_) / name;
  }

  // Reads block `key` into buf_ after checking its header and tokens.
  bool load(uint64_t key, uint64_t parent, std::span<const TokenId> tokens) {
    std::ifstream in(path_of(key), std::ios::binary);
    Header h{};
    if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 || h.key != key || h.parent != parent ||
        h.block_size != static_cast<int64_t>(tokens.size()) || h.bytes != pool_.block_bytes()) {
      return false;
    }
    std::vector<TokenId> t(tokens.size());
    if (!in.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(TokenId)))) return false;
    if (!std::equal(t.begin(), t.end(), tokens.begin())) return false;
    return static_cast<bool>(in.read(reinterpret_cast<char*>(buf_.data()), static_cast<std::streamsize>(buf_.size())));
  }

  void write_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
      if (jobs_.empty() && stop_) return;
      Job j = std::move(jobs_.front());
      jobs_.pop_front();
      writing_ = true;
      lock.unlock();
      const fs::path p = path_of(j.key), tmp = fs::path(p.string() + ".tmp");
      bool ok = false;
      {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        Header h{};
        std::memcpy(h.magic, kMagic, sizeof(kMagic));
        h.key = j.key;
        h.parent = j.parent;
        h.block_size = static_cast<int64_t>(j.tokens.size());
        h.bytes = static_cast<int64_t>(j.data.size());
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
        out.write(reinterpret_cast<const char*>(j.tokens.data()), static_cast<std::streamsize>(j.tokens.size() * sizeof(TokenId)));
        out.write(reinterpret_cast<const char*>(j.data.data()), static_cast<std::streamsize>(j.data.size()));
        ok = static_cast<bool>(out);
      }
      std::error_code ec;
      if (ok) fs::rename(tmp, p, ec);
      if (!ok || ec) fs::remove(tmp, ec);
      const int64_t size = static_cast<int64_t>(sizeof(Header) + j.tokens.size() * sizeof(TokenId) + j.data.size());
      lock.lock();
      pending_bytes_ -= static_cast<int64_t>(j.data.size());
      queued_.erase(j.key);
      if (ok && !ec) {
        on_disk_.insert(j.key);
        order_.push_back(j.key);
        bytes_ += size;
        while (max_bytes_ > 0 && bytes_ > max_bytes_ && !order_.empty()) {
          const uint64_t old = order_.front();
          order_.pop_front();
          on_disk_.erase(old);
          lock.unlock();
          const fs::path op = path_of(old);
          const auto n = static_cast<int64_t>(fs::file_size(op, ec));
          fs::remove(op, ec);
          lock.lock();
          if (!ec) bytes_ -= n;
        }
      }
      writing_ = false;
      if (jobs_.empty()) idle_cv_.notify_all();
    }
  }

  std::unique_ptr<PrefixCache> inner_;
  KvBlockPool& pool_;
  std::string dir_;
  int64_t max_bytes_;
  std::vector<uint64_t> keys_;
  std::vector<std::byte> buf_;
  uint64_t disk_blocks_ = 0;  // blocks loaded from disk
  mutable PrefixCacheStats merged_;

  std::mutex mu_;
  std::condition_variable cv_, idle_cv_;
  std::deque<Job> jobs_;
  std::unordered_set<uint64_t> on_disk_, queued_;
  std::list<uint64_t> order_;  // oldest first
  int64_t bytes_ = 0, pending_bytes_ = 0;
  bool stop_ = false, writing_ = false;
  std::thread writer_;
};

}  // namespace

std::unique_ptr<PrefixCache> make_disk_prefix_cache(std::unique_ptr<PrefixCache> inner, KvBlockPool& pool,
                                                    const std::string& dir, int64_t max_bytes) {
  return std::make_unique<DiskPrefixCache>(std::move(inner), pool, dir, max_bytes);
}

void flush_disk_prefix_cache(PrefixCache& cache) {
  if (auto* d = dynamic_cast<DiskPrefixCache*>(&cache)) d->flush();
}

}  // namespace dynalm
