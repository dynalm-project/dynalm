#include "dynacore/quantization/repack.h"

#include "dynacore/tensor/fp16.h"

namespace dynacore {

using quant::BlockQ4_K;
using quant::get_scale_min_k4;
using quant::kQK_K;

void repack_q4_K_x8(const void* matrix, int64_t row_bytes, int64_t row0, int64_t k, BlockQ4_Kx8* out) {
  const auto* base = static_cast<const std::byte*>(matrix);
  const int64_t nb = k / kQK_K;
  for (int64_t i = 0; i < nb; ++i) {
    BlockQ4_Kx8& p = out[i];
    for (int r = 0; r < 8; ++r) {
      const auto* b = reinterpret_cast<const BlockQ4_K*>(base + (row0 + r) * row_bytes) + i;
      p.d[r] = fp16_to_fp32(b->d);
      p.dmin[r] = fp16_to_fp32(b->dmin);
      for (int j = 0; j < 8; ++j) get_scale_min_k4(j, b->scales, p.sc[j][r], p.mn[j][r]);
      for (int jp = 0; jp < 4; ++jp) {
        for (int c = 0; c < 8; ++c) {
          for (int t = 0; t < 4; ++t) p.qs[jp][c][4 * r + t] = b->qs[32 * jp + 4 * c + t];
        }
      }
    }
  }
}

float packed_q4_K_x8_value(const BlockQ4_Kx8* blocks, int r, int64_t i) {
  const BlockQ4_Kx8& p = blocks[i / kQK_K];
  const int v = static_cast<int>(i % kQK_K);
  const int j = v / 32, jp = j / 2, c = (v % 32) / 4, t = v % 4;
  const uint8_t byte = p.qs[jp][c][4 * r + t];
  const int q = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
  return p.d[r] * static_cast<float>(p.sc[j][r]) * static_cast<float>(q) - p.dmin[r] * static_cast<float>(p.mn[j][r]);
}

}  // namespace dynacore
