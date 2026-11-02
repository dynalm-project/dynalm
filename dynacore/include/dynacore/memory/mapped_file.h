#pragma once

// Read-only memory-mapped file.
//
// Model weights are mapped, not read: load time is O(metadata), pages fault in
// on first use and are shared with the OS page cache, and weight tensors are
// zero-copy views into the mapping (Storage::borrow with this as keep-alive).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "dynacore/base/status.h"

namespace dynacore {

class MappedFile {
 public:
  static Result<std::shared_ptr<MappedFile>> open(const std::string& path);

  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::byte* data() const { return data_; }
  size_t size() const { return size_; }
  std::span<const std::byte> bytes() const { return {data_, size_}; }
  const std::string& path() const { return path_; }

  // Asks the OS to read [offset, offset+len) ahead of use. Best effort.
  void prefetch(size_t offset, size_t len) const;

 private:
  MappedFile() = default;

  const std::byte* data_ = nullptr;
  size_t size_ = 0;
  std::string path_;
#if defined(_WIN32)
  void* file_ = nullptr;
  void* mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
};

// Drops the resident pages of [p, p + len) when the whole range lies inside a
// live MappedFile, so they stop counting toward the process's resident memory;
// the bytes stay valid (the next read faults them in again from the file).
// Only whole pages inside the range are released. A range that is not part of
// a mapped file (owned host memory) is never touched. Returns the bytes
// released (0 where unsupported: Windows). For weights a device has copied
// into another layout and no longer reads (CpuDevice packed qmm copies).
size_t release_file_pages(const void* p, size_t len);

// The file and offset of [p, p + len) when the whole range lies inside a live
// MappedFile; nullopt otherwise. For reading weights with ordinary file I/O
// instead of through the mapping (the expert cache reads experts this way so
// their pages never enter the mapping).
struct MappedLocation {
  std::string path;
  uint64_t offset = 0;
};
std::optional<MappedLocation> locate_mapped_bytes(const void* p, size_t len);

}  // namespace dynacore
