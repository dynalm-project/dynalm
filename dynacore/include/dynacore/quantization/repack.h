#pragma once

// Interleaved weight layouts for multi-row decode (DD-078).
//
// GGUF stores a K-quant matrix row by row. For 2-8 activation rows the int8
// kernels are compute-bound, and with one row per call every 32 values cost
// a maddubs, a madd with a per-row scale, an add and a scale shuffle per
// activation row. Interleaving 8 weight rows four values at a time lets one
// 32-byte maddubs cover 4 values x 8 rows against a broadcast of 4 activation
// bytes; eight such products are summed in int16 before one madd with the 8
// rows' scales, and the 8 output lanes are the 8 rows (no horizontal sums).
//
// The packed copy is built once at model load from the original blocks and
// kept next to them (prefill and single-row decode keep the original layout).

#include <cstddef>
#include <cstdint>

#include "dynacore/quantization/quant_formats.h"

namespace dynacore {

// 8 rows x 256 values of Q4_K. Value v of sub-block j of row r is
//   d[r] * sc[j][r] * q - dmin[r] * mn[j][r]
// with q the low nibble (j = 2 * jp) or high nibble (j = 2 * jp + 1) of
//   qs[jp][c][4 * r + t],  v = 4 * c + t   (c in 0..7, t in 0..3).
struct BlockQ4_Kx8 {
  float d[8];
  float dmin[8];
  uint8_t sc[8][8];  // [sub-block][row]
  uint8_t mn[8][8];
  uint8_t qs[4][8][32];
};
static_assert(sizeof(BlockQ4_Kx8) == 1216);

// Packs rows [row0, row0 + 8) of a Q4_K matrix with `k` columns (a multiple
// of 256) whose rows are `row_bytes` apart: k / 256 blocks into `out`.
void repack_q4_K_x8(const void* matrix, int64_t row_bytes, int64_t row0, int64_t k, BlockQ4_Kx8* out);

// Value of row r, column i of the packed group (reference / tests).
float packed_q4_K_x8_value(const BlockQ4_Kx8* blocks, int r, int64_t i);

}  // namespace dynacore
