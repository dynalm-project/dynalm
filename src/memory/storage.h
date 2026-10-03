#pragma once

// Storage: a contiguous byte range on one device, owned or borrowed.
//
// Owned storage is freed when the last Tensor referencing it goes away.
// Borrowed storage (e.g. a memory-mapped model file) holds a keep-alive handle
// to whatever owns the bytes, so tensors can't outlive the mapping.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/status.h"
#include "memory/host_memory.h"

namespace engine {

// Where a Storage lives. Only kCpu is implemented; GPU types are reserved so
// the interfaces carry them from day one (DD-045). kSimulated is host memory
// that only its backend may touch (tests prove the runtime never dereferences
// device memory directly).
enum class DeviceType : uint8_t { kCpu = 0, kCuda, kHip, kMetal, kVulkan, kSimulated };

struct Device {
  DeviceType type = DeviceType::kCpu;
  int index = 0;
  friend bool operator==(const Device&, const Device&) = default;
};

class Storage {
 public:
  // Allocates `size` bytes of host memory (uninitialized).
  static Result<std::shared_ptr<Storage>> allocate_host(size_t size,
                                                        size_t alignment = kDefaultAlignment);

  // Wraps externally owned bytes. `keep_alive` is retained for the lifetime of
  // the Storage and may be null if the caller guarantees the bytes outlive it.
  static std::shared_ptr<Storage> borrow(void* data, size_t size, Device device,
                                         std::shared_ptr<const void> keep_alive);

  ~Storage();
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  void* data() const { return data_; }
  size_t size() const { return size_; }
  Device device() const { return device_; }
  bool owned() const { return owned_; }

 private:
  Storage() = default;

  void* data_ = nullptr;
  size_t size_ = 0;
  Device device_;
  bool owned_ = false;
  std::shared_ptr<const void> keep_alive_;
};

}  // namespace engine
