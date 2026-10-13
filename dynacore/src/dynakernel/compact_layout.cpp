#include "dynacore/dynakernel/compact_layout.h"

#include <cstddef>
#include <cstring>

#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"

namespace dynacore::dk {

using quant::BlockQ4_K;
using quant::kQK_K;

std::string_view packed_layout_name(PackedLayout l) {
  switch (l) {
    case PackedLayout::kX8: return "x8";
    case PackedLayout::kX8Fp16: return "h";
    case PackedLayout::kX8Scales6: return "s";
    case PackedLayout::kX8Fp16Scales6: return "hs";
  }
  return "?";
}

bool valid_packed_layout(PackedLayout l) {
  return l == PackedLayout::kX8 || l == PackedLayout::kX8Fp16 || l == PackedLayout::kX8Scales6 ||
         l == PackedLayout::kX8Fp16Scales6;
}

PackedLayoutInfo packed_layout_info(PackedLayout l) {
  switch (l) {
    case PackedLayout::kX8Fp16:
      return {sizeof(BlockQ4_Kx8H), offsetof(BlockQ4_Kx8H, d), offsetof(BlockQ4_Kx8H, dmin), offsetof(BlockQ4_Kx8H, sc),
              offsetof(BlockQ4_Kx8H, mn), -1, offsetof(BlockQ4_Kx8H, qs), true, false};
    case PackedLayout::kX8Scales6:
      return {sizeof(BlockQ4_Kx8S), offsetof(BlockQ4_Kx8S, d), offsetof(BlockQ4_Kx8S, dmin), -1, -1,
              offsetof(BlockQ4_Kx8S, scales), offsetof(BlockQ4_Kx8S, qs), false, true};
    case PackedLayout::kX8Fp16Scales6:
      return {sizeof(BlockQ4_Kx8HS), offsetof(BlockQ4_Kx8HS, d), offsetof(BlockQ4_Kx8HS, dmin), -1, -1,
              offsetof(BlockQ4_Kx8HS, scales), offsetof(BlockQ4_Kx8HS, qs), true, true};
    case PackedLayout::kX8:
    default:
      return {sizeof(BlockQ4_Kx8), offsetof(BlockQ4_Kx8, d), offsetof(BlockQ4_Kx8, dmin), offsetof(BlockQ4_Kx8, sc),
              offsetof(BlockQ4_Kx8, mn), -1, offsetof(BlockQ4_Kx8, qs), false, false};
  }
}

void repack_q4_K_x8_layout(PackedLayout l, const void* matrix, int64_t row_bytes, int64_t row0, int64_t k, void* out) {
  if (l == PackedLayout::kX8) {
    repack_q4_K_x8(matrix, row_bytes, row0, k, static_cast<BlockQ4_Kx8*>(out));
    return;
  }
  const PackedLayoutInfo li = packed_layout_info(l);
  const auto* base = static_cast<const std::byte*>(matrix);
  auto* dst = static_cast<uint8_t*>(out);
  for (int64_t i = 0; i < k / kQK_K; ++i) {
    uint8_t* p = dst + i * li.block_bytes;
    for (int r = 0; r < 8; ++r) {
      const auto* b = reinterpret_cast<const BlockQ4_K*>(base + (row0 + r) * row_bytes) + i;
      if (li.fp16) {
        std::memcpy(p + li.off_d + 2 * r, &b->d, 2);
        std::memcpy(p + li.off_dmin + 2 * r, &b->dmin, 2);
      } else {
        const float d = fp16_to_fp32(b->d), dmin = fp16_to_fp32(b->dmin);
        std::memcpy(p + li.off_d + 4 * r, &d, 4);
        std::memcpy(p + li.off_dmin + 4 * r, &dmin, 4);
      }
      if (li.scales6) {
        for (int s = 0; s < 12; ++s) p[li.off_scales + 8 * s + r] = b->scales[s];
      } else {
        for (int j = 0; j < 8; ++j) quant::get_scale_min_k4(j, b->scales, p[li.off_sc + 8 * j + r], p[li.off_mn + 8 * j + r]);
      }
      for (int jp = 0; jp < 4; ++jp) {
        for (int c = 0; c < 8; ++c) {
          for (int t = 0; t < 4; ++t) p[li.off_qs + 256 * jp + 32 * c + 4 * r + t] = b->qs[32 * jp + 4 * c + t];
        }
      }
    }
  }
}

float packed_layout_value(PackedLayout l, const void* blocks, int r, int64_t i) {
  const PackedLayoutInfo li = packed_layout_info(l);
  const auto* p = static_cast<const uint8_t*>(blocks) + (i / kQK_K) * li.block_bytes;
  const int v = static_cast<int>(i % kQK_K);
  const int j = v / 32, jp = j / 2, c = (v % 32) / 4, t = v % 4;
  const uint8_t byte = p[li.off_qs + 256 * jp + 32 * c + 4 * r + t];
  const int q = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
  float d, dmin;
  if (li.fp16) {
    uint16_t hd, hm;
    std::memcpy(&hd, p + li.off_d + 2 * r, 2);
    std::memcpy(&hm, p + li.off_dmin + 2 * r, 2);
    d = fp16_to_fp32(hd);
    dmin = fp16_to_fp32(hm);
  } else {
    std::memcpy(&d, p + li.off_d + 4 * r, 4);
    std::memcpy(&dmin, p + li.off_dmin + 4 * r, 4);
  }
  uint8_t sc, mn;
  if (li.scales6) {
    uint8_t s[12];
    for (int b = 0; b < 12; ++b) s[b] = p[li.off_scales + 8 * b + r];
    quant::get_scale_min_k4(j, s, sc, mn);
  } else {
    sc = p[li.off_sc + 8 * j + r];
    mn = p[li.off_mn + 8 * j + r];
  }
  return d * static_cast<float>(sc) * static_cast<float>(q) - dmin * static_cast<float>(mn);
}

}  // namespace dynacore::dk
