#pragma once

// EXPERIMENTAL compact variants of the interleaved Q4_K x 8 layout
// (docs/dynakernel.md, "Phase 4: compact packed layouts"). Opt-in, used only
// by the LLVM backend experiments; BlockQ4_Kx8 (DD-078) and every production
// path are unchanged.
//
// Every layout holds exactly the values of BlockQ4_Kx8 (same 4-bit codes in
// the same qs order, same scales, minimums, d and dmin), so kernels on any of
// them compute bit-identical results. They differ only in how the per-block
// parameters are stored:
//
//   layout  d, dmin               6-bit scales and minimums                bytes / 8 x 256
//   x8      fp32 [8] each         unpacked: sc[8][8], mn[8][8] (bytes)      1216 (BlockQ4_Kx8)
//   h       fp16 [8] each         unpacked                                 1184
//   s       fp32 [8] each         GGUF packed: scales[12][8], byte b of     1184
//                                 row r's 12 GGUF scale bytes at [b][r]
//   hs      fp16 [8] each         GGUF packed                              1152 (= 8 GGUF blocks)
//
// fp16 d/dmin are the GGUF block's own fp16 values (BlockQ4_Kx8 widens them
// exactly), so decoding them with a hardware fp16 -> fp32 conversion gives
// the same floats. The packed scales decode with get_scale_min_k4's
// arithmetic on 8 rows at once:
//   j < 4:  sc = s[j] & 63,                          mn = s[j + 4] & 63
//   j >= 4: sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4),
//           mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4)

#include <cstdint>
#include <string_view>

#include "dynacore/quantization/repack.h"

namespace dynacore::dk {

enum class PackedLayout : uint8_t { kX8 = 0, kX8Fp16, kX8Scales6, kX8Fp16Scales6 };
std::string_view packed_layout_name(PackedLayout l);  // "x8" | "h" | "s" | "hs"

struct BlockQ4_Kx8H {
  uint16_t d[8];
  uint16_t dmin[8];
  uint8_t sc[8][8];
  uint8_t mn[8][8];
  uint8_t qs[4][8][32];
};
struct BlockQ4_Kx8S {
  float d[8];
  float dmin[8];
  uint8_t scales[12][8];
  uint8_t qs[4][8][32];
};
struct BlockQ4_Kx8HS {
  uint16_t d[8];
  uint16_t dmin[8];
  uint8_t scales[12][8];
  uint8_t qs[4][8][32];
};
static_assert(sizeof(BlockQ4_Kx8H) == 1184 && sizeof(BlockQ4_Kx8S) == 1184 && sizeof(BlockQ4_Kx8HS) == 1152);

// Byte offsets inside one block (-1: field absent in this layout).
struct PackedLayoutInfo {
  int64_t block_bytes;
  int64_t off_d, off_dmin;  // fp32 or fp16 [8]
  int64_t off_sc, off_mn;   // unpacked [8][8] bytes (x8, h)
  int64_t off_scales;       // GGUF packed [12][8] bytes (s, hs)
  int64_t off_qs;           // [4][8][32]
  bool fp16;
  bool scales6;
};
PackedLayoutInfo packed_layout_info(PackedLayout l);
bool valid_packed_layout(PackedLayout l);

// Packs rows [row0, row0 + 8) of a Q4_K matrix with `k` columns (multiple of
// 256), rows `row_bytes` apart: k / 256 blocks of layout `l` into `out`
// (block_bytes each). kX8 is repack_q4_K_x8.
void repack_q4_K_x8_layout(PackedLayout l, const void* matrix, int64_t row_bytes, int64_t row0, int64_t k, void* out);

// Value of row r, column i of a packed group (reference decode for tests),
// computed exactly as packed_q4_K_x8_value.
float packed_layout_value(PackedLayout l, const void* blocks, int r, int64_t i);

}  // namespace dynacore::dk
