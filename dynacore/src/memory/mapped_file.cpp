#include "dynacore/memory/mapped_file.h"

#include <algorithm>
#include <map>
#include <mutex>

#include "dynacore/base/platform.h"

#if ENGINE_OS_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#endif

namespace dynacore {

namespace {
// Live mappings (start -> end), for release_file_pages.
std::mutex g_maps_mu;
std::map<const std::byte*, const std::byte*>& live_maps() {
  static auto* m = new std::map<const std::byte*, const std::byte*>();
  return *m;
}
void add_map(const std::byte* p, size_t n) {
  std::lock_guard<std::mutex> lock(g_maps_mu);
  live_maps()[p] = p + n;
}
void remove_map(const std::byte* p) {
  std::lock_guard<std::mutex> lock(g_maps_mu);
  live_maps().erase(p);
}
bool inside_map(const std::byte* b, const std::byte* e) {
  std::lock_guard<std::mutex> lock(g_maps_mu);
  auto it = live_maps().upper_bound(b);
  if (it == live_maps().begin()) return false;
  --it;
  return b >= it->first && e <= it->second;
}
}  // namespace

#if ENGINE_OS_WINDOWS

namespace {
std::wstring widen(const std::string& s) {
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}
std::string last_error() { return "win32 error " + std::to_string(GetLastError()); }
}  // namespace

Result<std::shared_ptr<MappedFile>> MappedFile::open(const std::string& path) {
  std::shared_ptr<MappedFile> f(new MappedFile());
  f->path_ = path;
  HANDLE file = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    const DWORD err = GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
      return NotFound("no such file: " + path);
    }
    return IoError("cannot open " + path + ": " + last_error());
  }
  f->file_ = file;
  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size)) return IoError("cannot stat " + path + ": " + last_error());
  f->size_ = static_cast<size_t>(size.QuadPart);
  if (f->size_ == 0) return Corrupt("empty file: " + path);

  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) return IoError("cannot map " + path + ": " + last_error());
  f->mapping_ = mapping;
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (!view) return IoError("cannot map view of " + path + ": " + last_error());
  f->data_ = static_cast<const std::byte*>(view);
  add_map(f->data_, f->size_);
  return f;
}

MappedFile::~MappedFile() {
  if (data_) remove_map(data_);
  if (data_) UnmapViewOfFile(data_);
  if (mapping_) CloseHandle(mapping_);
  if (file_) CloseHandle(file_);
}

void MappedFile::prefetch(size_t offset, size_t len) const {
  if (offset >= size_) return;
  WIN32_MEMORY_RANGE_ENTRY range;
  range.VirtualAddress = const_cast<std::byte*>(data_ + offset);
  range.NumberOfBytes = std::min(len, size_ - offset);
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}

#else

Result<std::shared_ptr<MappedFile>> MappedFile::open(const std::string& path) {
  std::shared_ptr<MappedFile> f(new MappedFile());
  f->path_ = path;
  f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (f->fd_ < 0) {
    if (errno == ENOENT) return NotFound("no such file: " + path);
    return IoError("cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat st;
  if (fstat(f->fd_, &st) != 0) return IoError("cannot stat " + path + ": " + std::strerror(errno));
  f->size_ = static_cast<size_t>(st.st_size);
  if (f->size_ == 0) return Corrupt("empty file: " + path);
  void* p = mmap(nullptr, f->size_, PROT_READ, MAP_SHARED, f->fd_, 0);
  if (p == MAP_FAILED) return IoError("cannot mmap " + path + ": " + std::strerror(errno));
  f->data_ = static_cast<const std::byte*>(p);
  add_map(f->data_, f->size_);
  return f;
}

MappedFile::~MappedFile() {
  if (data_) remove_map(data_);
  if (data_) munmap(const_cast<std::byte*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}

void MappedFile::prefetch(size_t offset, size_t len) const {
  if (offset >= size_) return;
  // madvise needs a page-aligned start.
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t start = offset / page * page;
  madvise(const_cast<std::byte*>(data_ + start), std::min(len + (offset - start), size_ - start),
          MADV_WILLNEED);
}

#endif

size_t release_file_pages(const void* p, size_t len) {
#if ENGINE_OS_WINDOWS
  (void)p;
  (void)len;
  return 0;
#else
  const auto* b = static_cast<const std::byte*>(p);
  if (b == nullptr || len == 0 || !inside_map(b, b + len)) return 0;
  const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t start = (reinterpret_cast<uintptr_t>(b) + page - 1) / page * page;
  const uintptr_t end = (reinterpret_cast<uintptr_t>(b) + len) / page * page;
  if (end <= start) return 0;
  void* a = reinterpret_cast<void*>(start);
#if ENGINE_OS_MACOS
  // Darwin ignores MADV_DONTNEED for file pages; MS_INVALIDATE drops them
  // from this mapping (the mapping is read-only, so nothing is written back).
  if (msync(a, end - start, MS_INVALIDATE) != 0) return 0;
#else
  if (madvise(a, end - start, MADV_DONTNEED) != 0) return 0;
#endif
  return end - start;
#endif
}

}  // namespace dynacore
