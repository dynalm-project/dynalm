#include "dynacore/quantization/repack.h"

#include <cstring>

#include "dynacore/tensor/fp16.h"

namespace dynacore {

using quant::BlockQ4_K;
using quant::BlockQ6_K;
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

namespace {
// 6-bit code (0..63) of value v of a Q6_K block (layout of dequantize_q6_K).
int q6_code(const BlockQ6_K& b, int v) {
  const int n = v / 128, vv = v % 128, qd = vv / 32, l = vv % 32;
  const uint8_t* ql = b.ql + 64 * n;
  const uint8_t* qh = b.qh + 32 * n;
  const int low = qd == 0 ? (ql[l] & 0xF) : qd == 1 ? (ql[l + 32] & 0xF) : qd == 2 ? (ql[l] >> 4) : (ql[l + 32] >> 4);
  const int high = (qh[l] >> (2 * qd)) & 3;
  return low | (high << 4);
}
}  // namespace

void repack_q6_K_x8(const void* matrix, int64_t row_bytes, int64_t row0, int64_t k, BlockQ6_Kx8* out) {
  const auto* base = static_cast<const std::byte*>(matrix);
  const int64_t nb = k / kQK_K;
  for (int64_t i = 0; i < nb; ++i) {
    BlockQ6_Kx8& p = out[i];
    std::memset(p.qh, 0, sizeof(p.qh));
    for (int r = 0; r < 8; ++r) {
      const auto* b = reinterpret_cast<const BlockQ6_K*>(base + (row0 + r) * row_bytes) + i;
      p.d[r] = fp16_to_fp32(b->d);
      for (int g = 0; g < 16; ++g) p.sc[g][r] = b->scales[g];
      for (int pp = 0; pp < 4; ++pp) {
        for (int c = 0; c < 8; ++c) {
          for (int t = 0; t < 4; ++t) {
            const int qa = q6_code(*b, 64 * pp + 4 * c + t), qb = q6_code(*b, 64 * pp + 32 + 4 * c + t);
            const int kk = 4 * r + t;
            p.ql[pp][c][kk] = static_cast<uint8_t>((qa & 0xF) | ((qb & 0xF) << 4));
            const int h = (qa >> 4) | ((qb >> 4) << 2);  // 4 bits
            uint8_t& hb = p.qh[pp][c][kk % 16];
            hb = static_cast<uint8_t>(hb | (kk < 16 ? h : h << 4));
          }
        }
      }
    }
  }
}

float packed_q6_K_x8_value(const BlockQ6_Kx8* blocks, int r, int64_t i) {
  const BlockQ6_Kx8& p = blocks[i / kQK_K];
  const int v = static_cast<int>(i % kQK_K);
  const int pp = v / 64, within = v % 64, sub = within / 32, c = (within % 32) / 4, t = within % 4;
  const int kk = 4 * r + t;
  const uint8_t byte = p.ql[pp][c][kk];
  const int low = sub == 0 ? (byte & 0xF) : (byte >> 4);
  const uint8_t hb = p.qh[pp][c][kk % 16];
  const int h4 = kk < 16 ? (hb & 0xF) : (hb >> 4);
  const int high = sub == 0 ? (h4 & 3) : ((h4 >> 2) & 3);
  const int q = low | (high << 4);
  return p.d[r] * static_cast<float>(p.sc[v / 16][r]) * static_cast<float>(q - 32);
}

}  // namespace dynacore
