#pragma once

// SHA-256 (FIPS 180-4), used by `dynalm pull` to check a downloaded model
// against the checksum the server publishes. Portable C++, no TLS library.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "dynacore/base/status.h"
#include "common/core.h"

namespace dynalm {

class Sha256 {
 public:
  Sha256();
  void update(std::span<const std::byte> data);
  void update(std::string_view s) { update(std::as_bytes(std::span(s.data(), s.size()))); }
  // Lowercase hex digest; the object must not be updated afterwards.
  std::string hex_digest();

 private:
  void block(const uint8_t* p);
  std::array<uint32_t, 8> h_;
  std::array<uint8_t, 64> buf_{};
  size_t buf_len_ = 0;
  uint64_t total_ = 0;  // bytes
};

// Lowercase hex SHA-256 of a whole file, read in chunks.
Result<std::string> sha256_file(const std::string& path);

// The SHA-256 a Hugging Face "resolve" response publishes for a file stored
// with LFS or Xet: the X-Linked-ETag header, a quoted 64-digit hex string.
// `headers` is raw HTTP response headers (several responses allowed, as curl
// writes them when it follows redirects). Empty when absent or not a SHA-256.
std::string linked_etag_sha256(std::string_view headers);

}  // namespace dynalm
