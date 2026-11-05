#include "model/expert_cache.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

#include "dynacore/base/platform.h"
#include "dynacore/memory/mapped_file.h"

#if ENGINE_OS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace dynalm {

namespace {

constexpr int64_t kSlotAlign = 4096;

int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

// Positional reads of one file that leave nothing in the OS file cache.
class ExpertCache::Reader {
 public:
  static Result<std::unique_ptr<Reader>> open(const std::string& path) {
    std::unique_ptr<Reader> r(new Reader());
#if ENGINE_OS_WINDOWS
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), w.data(), n);
    r->h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,
                        nullptr);
    if (r->h_ == INVALID_HANDLE_VALUE) return IoError("cannot open " + path + " (error " + std::to_string(GetLastError()) + ")");
#else
    r->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (r->fd_ < 0) return IoError("cannot open " + path + ": " + std::strerror(errno));
#if defined(__APPLE__)
    fcntl(r->fd_, F_NOCACHE, 1);  // reads bypass the unified buffer cache
#endif
#endif
    r->path_ = path;
    return r;
  }

  ~Reader() {
#if ENGINE_OS_WINDOWS
    if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
#else
    if (fd_ >= 0) ::close(fd_);
#endif
  }

  // Reads exactly `n` bytes at `offset`; an error message, empty on success.
  std::string read(uint64_t offset, int64_t n, std::byte* dst) const {
    int64_t done = 0;
    while (done < n) {
#if ENGINE_OS_WINDOWS
      OVERLAPPED ov{};
      const uint64_t at = offset + static_cast<uint64_t>(done);
      ov.Offset = static_cast<DWORD>(at & 0xffffffffu);
      ov.OffsetHigh = static_cast<DWORD>(at >> 32);
      DWORD got = 0;
      const auto want = static_cast<DWORD>(std::min<int64_t>(n - done, int64_t{1} << 30));
      if (!ReadFile(h_, dst + done, want, &got, &ov) || got == 0) {
        return "read failed in " + path_ + " (error " + std::to_string(GetLastError()) + ")";
      }
#else
      const ssize_t got = ::pread(fd_, dst + done, static_cast<size_t>(n - done), static_cast<off_t>(offset + done));
      if (got < 0) {
        if (errno == EINTR) continue;
        return "read failed in " + path_ + ": " + std::strerror(errno);
      }
      if (got == 0) return "unexpected end of " + path_;
#endif
      done += static_cast<int64_t>(got);
    }
#if defined(__linux__)
    // The cache holds these bytes now; drop the page-cache copy.
    posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(n), POSIX_FADV_DONTNEED);
#endif
    return {};
  }

 private:
  Reader() = default;
  std::string path_;
#if ENGINE_OS_WINDOWS
  HANDLE h_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
};

Result<std::unique_ptr<ExpertCache>> ExpertCache::create(const std::vector<std::vector<Parts>>& experts,
                                                         const Options& o) {
  std::unique_ptr<ExpertCache> c(new ExpertCache());
  c->chunk_bytes_ = std::max<int64_t>(o.chunk_bytes, 1 << 16);
  std::map<std::string, int> file_ids;
  c->sources_.resize(experts.size());
  c->layouts_ = experts;
  for (size_t l = 0; l < experts.size(); ++l) {
    c->max_experts_ = std::max(c->max_experts_, static_cast<int32_t>(experts[l].size()));
    for (const Parts& parts : experts[l]) {
      std::array<Source, 3> src{};
      int64_t at = 0;
      for (size_t i = 0; i < 3; ++i) {
        const TensorView& v = parts[i];
        if (v.data() == nullptr || !v.is_contiguous()) {
          return Unsupported("expert cache: expert weights must be contiguous tensors");
        }
        const int64_t bytes = v.span_bytes();
        const auto loc = dynacore::locate_mapped_bytes(v.data(), static_cast<size_t>(bytes));
        if (!loc) {
          return Unsupported("expert cache: the expert weights are not read from a model file (converted at load?)");
        }
        auto [it, added] = file_ids.emplace(loc->path, static_cast<int>(file_ids.size()));
        if (added) {
          ENGINE_ASSIGN_OR_RETURN(auto r, Reader::open(loc->path));
          c->files_.push_back(std::move(r));
        }
        src[i] = {it->second, loc->offset, bytes, at};
        at = align_up(at + bytes, kSlotAlign);
        c->expert_bytes_ += bytes;
      }
      c->slot_bytes_ = std::max(c->slot_bytes_, at);
      c->sources_[l].push_back(src);
    }
  }
  if (c->max_experts_ == 0) return InvalidArgument("expert cache: the model has no experts");
  const int64_t n = o.budget_bytes / c->slot_bytes_;
  if (n < c->max_experts_) {
    const int64_t need = c->slot_bytes_ * c->max_experts_;
    return InvalidArgument("--expert-cache is too small: at least " + std::to_string((need + (1 << 20) - 1) >> 20) +
                           " MiB (every expert of one layer, " + std::to_string(c->max_experts_) + " x " +
                           std::to_string(c->slot_bytes_ >> 10) + " KiB)");
  }
  // Never more slots than experts: the whole model would then fit.
  int64_t total = 0;
  for (const auto& l : experts) total += static_cast<int64_t>(l.size());
  const int64_t nslots = std::min(n, total);
  c->arena_.reset(new (std::nothrow) std::byte[static_cast<size_t>(nslots * c->slot_bytes_ + kSlotAlign)]);
  if (!c->arena_) return ResourceExhausted("expert cache: cannot allocate " + std::to_string((nslots * c->slot_bytes_) >> 20) + " MiB");
  const auto raw = reinterpret_cast<uintptr_t>(c->arena_.get());
  c->base_ = c->arena_.get() + (align_up(static_cast<int64_t>(raw), kSlotAlign) - static_cast<int64_t>(raw));
  c->slots_.resize(static_cast<size_t>(nslots));
  c->decay_at_ = static_cast<uint64_t>(nslots) * 4;
  const int threads = std::max(1, o.io_threads);
  for (int i = 0; i < threads; ++i) c->threads_.emplace_back([p = c.get()] { p->io_loop(); });
  return c;
}

ExpertCache::~ExpertCache() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  io_cv_.notify_all();
  for (std::thread& t : threads_) t.join();
}

int32_t ExpertCache::victim() {
  int32_t best = -1;
  for (size_t i = 0; i < slots_.size(); ++i) {
    const Slot& s = slots_[i];
    if (s.pinned == pin_gen_ || s.pending > 0) continue;
    if (s.key < 0) return static_cast<int32_t>(i);
    if (best < 0) {
      best = static_cast<int32_t>(i);
      continue;
    }
    const Slot& b = slots_[static_cast<size_t>(best)];
    if (s.uses < b.uses || (s.uses == b.uses && s.last < b.last)) best = static_cast<int32_t>(i);
  }
  if (best >= 0) {
    where_.erase(slots_[static_cast<size_t>(best)].key);
    slots_[static_cast<size_t>(best)].key = -1;
    ++stats_.evictions;
  }
  return best;
}

void ExpertCache::start_load(int32_t slot, int64_t k) {
  Slot& s = slots_[static_cast<size_t>(slot)];
  const auto layer = static_cast<size_t>(k / max_experts_), e = static_cast<size_t>(k % max_experts_);
  s.key = k;
  s.uses = 0;
  s.last = ++tick_;
  s.error.clear();
  s.prefetched = false;
  where_[k] = slot;
  std::byte* base = base_ + static_cast<int64_t>(slot) * slot_bytes_;
  const auto& src = sources_[layer][e];
  const Parts& orig = layouts_[layer][e];
  for (size_t i = 0; i < 3; ++i) {
    s.views[i] = TensorView(base + src[i].slot_offset, orig[i].layout(), orig[i].device());
    for (int64_t at = 0; at < src[i].bytes; at += chunk_bytes_) {
      const int64_t n = std::min(chunk_bytes_, src[i].bytes - at);
      jobs_.push_back({slot, src[i].file, src[i].offset + static_cast<uint64_t>(at), n, base + src[i].slot_offset + at});
      ++s.pending;
    }
  }
  io_cv_.notify_all();
}

void ExpertCache::io_loop() {
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    io_cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) return;
    const Job j = jobs_.front();
    jobs_.pop_front();
    lock.unlock();
    const auto t0 = std::chrono::steady_clock::now();
    const std::string err = files_[static_cast<size_t>(j.file)]->read(j.offset, j.bytes, j.dst);
    const double ms = ms_since(t0);
    lock.lock();
    stats_.read_ms += ms;
    stats_.bytes_read += j.bytes;
    Slot& s = slots_[static_cast<size_t>(j.slot)];
    if (!err.empty() && s.error.empty()) s.error = err;
    if (--s.pending == 0) done_cv_.notify_all();
  }
}

Status ExpertCache::acquire(int32_t layer, std::span<const int32_t> ids, std::vector<const Parts*>& out) {
  std::unique_lock<std::mutex> lock(mu_);
  ++pin_gen_;
  out.clear();
  std::vector<int32_t> held;
  held.reserve(ids.size());
  for (int32_t e : ids) {
    const int64_t k = key(layer, e);
    int32_t s;
    if (auto it = where_.find(k); it != where_.end()) {
      s = it->second;
      ++stats_.hits;
      Slot& slot = slots_[static_cast<size_t>(s)];
      if (slot.prefetched) {
        ++stats_.prefetch_used;
        slot.prefetched = false;
      }
    } else {
      s = victim();
      if (s < 0) return Internal("expert cache: no free slot (more experts in one step than slots)");
      start_load(s, k);
      ++stats_.misses;
    }
    Slot& slot = slots_[static_cast<size_t>(s)];
    slot.pinned = pin_gen_;
    ++slot.uses;
    slot.last = ++tick_;
    held.push_back(s);
  }
  const auto t0 = std::chrono::steady_clock::now();
  done_cv_.wait(lock, [&] {
    return std::all_of(held.begin(), held.end(), [&](int32_t s) { return slots_[static_cast<size_t>(s)].pending == 0; });
  });
  stats_.wait_ms += ms_since(t0);
  for (int32_t s : held) {
    Slot& slot = slots_[static_cast<size_t>(s)];
    if (!slot.error.empty()) {
      const std::string err = slot.error;
      where_.erase(slot.key);
      slot.key = -1;
      slot.error.clear();
      return IoError("expert cache: " + err);
    }
    out.push_back(&slot.views);
  }
  // Fade old popularity: halve every count once per 4 x slots accesses.
  if (tick_ >= decay_at_) {
    for (Slot& slot : slots_) slot.uses >>= 1;
    decay_at_ = tick_ + static_cast<uint64_t>(slots_.size()) * 4;
  }
  return Status::Ok();
}

void ExpertCache::prefetch(int32_t layer, std::span<const int32_t> ids) {
  std::lock_guard<std::mutex> lock(mu_);
  for (int32_t e : ids) {
    const int64_t k = key(layer, e);
    if (where_.count(k) != 0) continue;
    const int32_t s = victim();
    if (s < 0) return;
    start_load(s, k);
    slots_[static_cast<size_t>(s)].prefetched = true;
    ++stats_.prefetched;
  }
}

ExpertCache::Stats ExpertCache::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

}  // namespace dynalm
