#include "dynacore/tensor/dtype.h"

#include <array>

namespace engine {
namespace {

constexpr int kQK = 32;     // legacy quant block size
constexpr int kQK_K = 256;  // k-quant super-block size

// Block byte sizes follow the GGML block structs (fp16 scales = 2 bytes):
//   q4_0: d + 16 nibble bytes                  = 18
//   q4_1: d, m + 16                            = 20
//   q5_0: d + 4 high-bit bytes + 16            = 22
//   q5_1: d, m + 4 + 16                        = 24
//   q8_0: d + 32                               = 34
//   q8_1: d, s + 32                            = 36
//   q2_K: 16 scales + 64 qs + d, dmin          = 84
//   q3_K: 32 hmask + 64 qs + 12 scales + d     = 110
//   q4_K: d, dmin + 12 scales + 128 qs         = 144
//   q5_K: d, dmin + 12 scales + 32 qh + 128 qs = 176
//   q6_K: 128 ql + 64 qh + 16 scales + d       = 210
//   q8_K: float d + 256 qs + 16 int16 bsums    = 292
constexpr std::array<DTypeInfo, static_cast<size_t>(DType::kCount)> kInfo = {{
    {"f32", 1, 4, false, true},
    {"f16", 1, 2, false, true},
    {"bf16", 1, 2, false, true},
    {"i8", 1, 1, false, false},
    {"i16", 1, 2, false, false},
    {"i32", 1, 4, false, false},
    {"q4_0", kQK, 18, true, false},
    {"q4_1", kQK, 20, true, false},
    {"q5_0", kQK, 22, true, false},
    {"q5_1", kQK, 24, true, false},
    {"q8_0", kQK, 34, true, false},
    {"q8_1", kQK, 36, true, false},
    {"q2_K", kQK_K, 84, true, false},
    {"q3_K", kQK_K, 110, true, false},
    {"q4_K", kQK_K, 144, true, false},
    {"q5_K", kQK_K, 176, true, false},
    {"q6_K", kQK_K, 210, true, false},
    {"q8_K", kQK_K, 292, true, false},
}};

}  // namespace

const DTypeInfo& dtype_info(DType t) { return kInfo[static_cast<size_t>(t)]; }

int64_t dtype_row_bytes(DType t, int64_t n) {
  const DTypeInfo& info = dtype_info(t);
  if (n < 0 || n % info.block_elems != 0) return -1;
  return n / info.block_elems * info.block_bytes;
}

bool parse_dtype(std::string_view s, DType& out) {
  for (size_t i = 0; i < kInfo.size(); ++i) {
    if (kInfo[i].name == s) {
      out = static_cast<DType>(i);
      return true;
    }
  }
  return false;
}

}  // namespace engine
