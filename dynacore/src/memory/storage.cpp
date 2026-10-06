#include "dynacore/memory/storage.h"

#include <string>

namespace dynacore {

Result<std::shared_ptr<Storage>> Storage::allocate_host(size_t size, size_t alignment) {
  if (size == 0) return InvalidArgument("Storage::allocate_host: size 0");
  void* p = host_alloc(size, alignment);
  if (!p) {
    return OutOfMemory("failed to allocate " + std::to_string(size) + " bytes (alignment " +
                       std::to_string(alignment) + ")");
  }
  std::shared_ptr<Storage> s(new Storage());
  s->data_ = p;
  s->size_ = size;
  s->owned_ = true;
  return s;
}

std::shared_ptr<Storage> Storage::borrow(void* data, size_t size, DeviceLoc device,
                                         std::shared_ptr<const void> keep_alive) {
  std::shared_ptr<Storage> s(new Storage());
  s->data_ = data;
  s->size_ = size;
  s->device_ = device;
  s->keep_alive_ = std::move(keep_alive);
  return s;
}

Storage::~Storage() {
  if (owned_) host_free(data_, size_);
}

}  // namespace dynacore
