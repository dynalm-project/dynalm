#pragma once

// qmm weight unpacking (dynacore/cpu/qmm.h): scalar, portable, the reference
// for every tier. A Group is up to 8 consecutive 32-value sub-blocks of one
// weight row as signed int8 codes with
//
//   w[i] = s16[i / 16] * q[i] - mn[i / 32]      (i relative to the group)
//
// s32[j] = s16[2j] = s16[2j + 1] for formats whose scale is constant over 32
// values (kHalf == false). Every format below reproduces its dequantize_*
// reference (quant_formats.cpp) exactly in this form: the codes are the
// format's integer levels (offsets folded in where the offset is a constant),
// s16 and mn are the products the dequantizer forms.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/dtype.h"
#include "dynacore/tensor/fp16.h"

namespace dynacore::qmm {

struct Group {
  alignas(16) int8_t q[256];
  float s16[16];
  float s32[8];
  float mn[8];
};

// Format traits: kSuper = values per storage block (32 or 256), kHalf = scale
// changes every 16 values, kMin = mn may be non-zero.
template <DType T>
struct Fmt;

namespace detail {
inline void set32(Group& g, int j, float s) {
  g.s32[j] = s;
  g.s16[2 * j] = s;
  g.s16[2 * j + 1] = s;
}
}  // namespace detail

template <>
struct Fmt<DType::kQ8_0> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = false;
  using Block = quant::BlockQ8_0;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      std::memcpy(g.q + 32 * j, b[j].qs, 32);
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  }
};

template <>
struct Fmt<DType::kQ4_0> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = false;
  using Block = quant::BlockQ4_0;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      for (int l = 0; l < 16; ++l) {
        g.q[32 * j + l] = static_cast<int8_t>((b[j].qs[l] & 0x0F) - 8);
        g.q[32 * j + l + 16] = static_cast<int8_t>((b[j].qs[l] >> 4) - 8);
      }
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  }
};

template <>
struct Fmt<DType::kQ4_1> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = true;
  using Block = quant::BlockQ4_1;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      for (int l = 0; l < 16; ++l) {
        g.q[32 * j + l] = static_cast<int8_t>(b[j].qs[l] & 0x0F);
        g.q[32 * j + l + 16] = static_cast<int8_t>(b[j].qs[l] >> 4);
      }
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = -fp16_to_fp32(b[j].m);  // w = d q + m
    }
  }
};

template <>
struct Fmt<DType::kQ5_0> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = false;
  using Block = quant::BlockQ5_0;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      uint32_t qh;
      std::memcpy(&qh, b[j].qh, sizeof(qh));
      for (int l = 0; l < 16; ++l) {
        const int h0 = ((qh >> l) << 4) & 0x10, h1 = (qh >> (l + 12)) & 0x10;
        g.q[32 * j + l] = static_cast<int8_t>(((b[j].qs[l] & 0x0F) | h0) - 16);
        g.q[32 * j + l + 16] = static_cast<int8_t>(((b[j].qs[l] >> 4) | h1) - 16);
      }
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  }
};

template <>
struct Fmt<DType::kQ5_1> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = true;
  using Block = quant::BlockQ5_1;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      uint32_t qh;
      std::memcpy(&qh, b[j].qh, sizeof(qh));
      for (int l = 0; l < 16; ++l) {
        const int h0 = ((qh >> l) << 4) & 0x10, h1 = (qh >> (l + 12)) & 0x10;
        g.q[32 * j + l] = static_cast<int8_t>((b[j].qs[l] & 0x0F) | h0);
        g.q[32 * j + l + 16] = static_cast<int8_t>((b[j].qs[l] >> 4) | h1);
      }
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = -fp16_to_fp32(b[j].m);
    }
  }
};

template <>
struct Fmt<DType::kIQ4_NL> {
  static constexpr int kSuper = 32;
  static constexpr bool kHalf = false, kMin = false;
  using Block = quant::BlockIQ4_NL;
  static void unpack(const Block* b, int nsub, Group& g) {
    for (int j = 0; j < nsub; ++j) {
      for (int l = 0; l < 16; ++l) {
        g.q[32 * j + l] = quant::kIQ4Values[b[j].qs[l] & 0x0F];
        g.q[32 * j + l + 16] = quant::kIQ4Values[b[j].qs[l] >> 4];
      }
      detail::set32(g, j, fp16_to_fp32(b[j].d));
      g.mn[j] = 0;
    }
  }
};

// K-quants and IQ4_XS: one storage block = one full group (nsub == 8).
template <>
struct Fmt<DType::kQ4_K> {
  static constexpr int kSuper = 256;
  static constexpr bool kHalf = false, kMin = true;
  using Block = quant::BlockQ4_K;
  static void scales(const Block& b, Group& g) {
    const float d = fp16_to_fp32(b.d), dmin = fp16_to_fp32(b.dmin);
    for (int j = 0; j < 8; ++j) {
      uint8_t sc, mn;
      quant::get_scale_min_k4(j, b.scales, sc, mn);
      detail::set32(g, j, d * sc);
      g.mn[j] = dmin * mn;
    }
  }
  static void unpack(const Block* b, int /*nsub*/, Group& g) {
    scales(*b, g);
    for (int jp = 0; jp < 4; ++jp) {
      for (int l = 0; l < 32; ++l) {
        g.q[64 * jp + l] = static_cast<int8_t>(b->qs[32 * jp + l] & 0x0F);
        g.q[64 * jp + 32 + l] = static_cast<int8_t>(b->qs[32 * jp + l] >> 4);
      }
    }
  }
};

template <>
struct Fmt<DType::kQ5_K> {
  static constexpr int kSuper = 256;
  static constexpr bool kHalf = false, kMin = true;
  using Block = quant::BlockQ5_K;
  static void unpack(const Block* b, int /*nsub*/, Group& g) {
    const float d = fp16_to_fp32(b->d), dmin = fp16_to_fp32(b->dmin);
    for (int j = 0; j < 8; ++j) {
      uint8_t sc, mn;
      quant::get_scale_min_k4(j, b->scales, sc, mn);
      detail::set32(g, j, d * sc);
      g.mn[j] = dmin * mn;
    }
    for (int jp = 0; jp < 4; ++jp) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t ql = b->qs[32 * jp + l], qh = b->qh[l];
        g.q[64 * jp + l] = static_cast<int8_t>((ql & 0x0F) | (((qh >> (2 * jp)) & 1) << 4));
        g.q[64 * jp + 32 + l] = static_cast<int8_t>((ql >> 4) | (((qh >> (2 * jp + 1)) & 1) << 4));
      }
    }
  }
};

template <>
struct Fmt<DType::kQ6_K> {
  static constexpr int kSuper = 256;
  static constexpr bool kHalf = true, kMin = false;
  using Block = quant::BlockQ6_K;
  static void unpack(const Block* b, int /*nsub*/, Group& g) {
    const float d = fp16_to_fp32(b->d);
    for (int h = 0; h < 16; ++h) g.s16[h] = d * static_cast<float>(b->scales[h]);
    for (int j = 0; j < 8; ++j) {
      g.s32[j] = 0;  // per-16 format: s16 only
      g.mn[j] = 0;
    }
    for (int n = 0; n < 2; ++n) {
      const uint8_t* ql = b->ql + 64 * n;
      const uint8_t* qh = b->qh + 32 * n;
      int8_t* y = g.q + 128 * n;
      for (int l = 0; l < 32; ++l) {
        y[l + 0] = static_cast<int8_t>(((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        y[l + 32] = static_cast<int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        y[l + 64] = static_cast<int8_t>(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        y[l + 96] = static_cast<int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
    }
  }
};

template <>
struct Fmt<DType::kQ3_K> {
  static constexpr int kSuper = 256;
  static constexpr bool kHalf = true, kMin = false;
  using Block = quant::BlockQ3_K;
  static void unpack(const Block* b, int /*nsub*/, Group& g) {
    constexpr uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    std::memcpy(aux, b->scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int8_t sc[16];
    std::memcpy(sc, aux, 16);
    const float d = fp16_to_fp32(b->d);
    for (int h = 0; h < 16; ++h) g.s16[h] = d * static_cast<float>(sc[h] - 32);
    for (int j = 0; j < 8; ++j) {
      g.s32[j] = 0;
      g.mn[j] = 0;
    }
    const uint8_t* q = b->qs;
    int8_t* y = g.q;
    uint8_t m = 1;
    for (int n = 0; n < 2; ++n) {
      for (int j = 0; j < 4; ++j) {
        const int shift = 2 * j;
        for (int l = 0; l < 16; ++l) y[l] = static_cast<int8_t>(((q[l] >> shift) & 3) - ((b->hmask[l] & m) ? 0 : 4));
        for (int l = 0; l < 16; ++l) {
          y[16 + l] = static_cast<int8_t>(((q[l + 16] >> shift) & 3) - ((b->hmask[l + 16] & m) ? 0 : 4));
        }
        y += 32;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
};

template <>
struct Fmt<DType::kIQ4_XS> {
  static constexpr int kSuper = 256;
  static constexpr bool kHalf = false, kMin = false;
  using Block = quant::BlockIQ4_XS;
  static void unpack(const Block* b, int /*nsub*/, Group& g) {
    const float d = fp16_to_fp32(b->d);
    for (int ib = 0; ib < 8; ++ib) {
      const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((b->scales_h >> (2 * ib)) & 3) << 4);
      detail::set32(g, ib, d * static_cast<float>(ls - 32));
      g.mn[ib] = 0;
      const uint8_t* qs = b->qs + 16 * ib;
      for (int j = 0; j < 16; ++j) {
        g.q[32 * ib + j] = quant::kIQ4Values[qs[j] & 0xF];
        g.q[32 * ib + j + 16] = quant::kIQ4Values[qs[j] >> 4];
      }
    }
  }
};

// --- Packed 4-row groups (qmm.h, "Packed weights") -----------------------

#pragma pack(push, 1)
struct PackedQ4K {
  uint16_t d[4], dmin[4];
  uint8_t sc[8][4], mn[8][4];   // [sub-block][row]
  uint8_t qs[4][8][4][4];       // [jp][i4][row][byte]
};
#pragma pack(pop)
static_assert(sizeof(PackedQ4K) == 592);

// Packs 4 consecutive Q4_K rows' super-block b.
inline void pack_q4k(const quant::BlockQ4_K* const rows[4], int64_t b, PackedQ4K& out) {
  for (int r = 0; r < 4; ++r) {
    const quant::BlockQ4_K& blk = rows[r][b];
    out.d[r] = blk.d;
    out.dmin[r] = blk.dmin;
    for (int j = 0; j < 8; ++j) quant::get_scale_min_k4(j, blk.scales, out.sc[j][r], out.mn[j][r]);
    for (int jp = 0; jp < 4; ++jp) {
      for (int i4 = 0; i4 < 8; ++i4) std::memcpy(out.qs[jp][i4][r], blk.qs + 32 * jp + 4 * i4, 4);
    }
  }
}

// The Group of row r (0..3) of a packed super-block: exactly what
// Fmt<kQ4_K>::unpack gives for the original block.
inline void unpack_packed_q4k(const PackedQ4K& p, int r, Group& g) {
  const float d = fp16_to_fp32(p.d[r]), dmin = fp16_to_fp32(p.dmin[r]);
  for (int j = 0; j < 8; ++j) {
    detail::set32(g, j, d * p.sc[j][r]);
    g.mn[j] = dmin * p.mn[j][r];
  }
  for (int jp = 0; jp < 4; ++jp) {
    for (int i = 0; i < 32; ++i) {
      const uint8_t byte = p.qs[jp][i / 4][r][i % 4];
      g.q[64 * jp + i] = static_cast<int8_t>(byte & 0x0F);
      g.q[64 * jp + 32 + i] = static_cast<int8_t>(byte >> 4);
    }
  }
}

// Q6_K, 840 bytes per 4-row group and super-block:
//   uint16 d[4]; int8 sc[16][4] (per 16 values, [unit][row]);
//   ql[n 0..1][h 0..1][i4 0..7][row][4] = row bytes ql[64 n + 32 h + 4 i4 ..];
//   qh[n 0..1][i4 0..7][row][4]         = row bytes qh[32 n + 4 i4 ..].
// Value n * 128 + 32 qq + l (l = 4 i4 + t) is, with A = ql[n][0], B = ql[n][1], H = qh[n]:
//   qq 0: (A & 15) | (H & 3) << 4      qq 1: (B & 15) | (H >> 2 & 3) << 4
//   qq 2: (A >> 4) | (H >> 4 & 3) << 4 qq 3: (B >> 4) | (H >> 6) << 4      (minus 32).
#pragma pack(push, 1)
struct PackedQ6K {
  uint16_t d[4];
  int8_t sc[16][4];
  uint8_t ql[2][2][8][4][4];
  uint8_t qh[2][8][4][4];
};
#pragma pack(pop)
static_assert(sizeof(PackedQ6K) == 840);

inline void pack_q6k(const quant::BlockQ6_K* const rows[4], int64_t b, PackedQ6K& out) {
  for (int r = 0; r < 4; ++r) {
    const quant::BlockQ6_K& blk = rows[r][b];
    out.d[r] = blk.d;
    for (int u = 0; u < 16; ++u) out.sc[u][r] = blk.scales[u];
    for (int n = 0; n < 2; ++n) {
      for (int i4 = 0; i4 < 8; ++i4) {
        std::memcpy(out.ql[n][0][i4][r], blk.ql + 64 * n + 4 * i4, 4);
        std::memcpy(out.ql[n][1][i4][r], blk.ql + 64 * n + 32 + 4 * i4, 4);
        std::memcpy(out.qh[n][i4][r], blk.qh + 32 * n + 4 * i4, 4);
      }
    }
  }
}

// Row r's Group of a packed Q6_K super-block (as Fmt<kQ6_K>::unpack).
inline void unpack_packed_q6k(const PackedQ6K& p, int r, Group& g) {
  const float d = fp16_to_fp32(p.d[r]);
  for (int u = 0; u < 16; ++u) g.s16[u] = d * static_cast<float>(p.sc[u][r]);
  for (int j = 0; j < 8; ++j) {
    g.s32[j] = 0;
    g.mn[j] = 0;
  }
  for (int n = 0; n < 2; ++n) {
    for (int l = 0; l < 32; ++l) {
      const uint8_t a = p.ql[n][0][l / 4][r][l % 4], b = p.ql[n][1][l / 4][r][l % 4], h = p.qh[n][l / 4][r][l % 4];
      int8_t* y = g.q + 128 * n;
      y[l + 0] = static_cast<int8_t>(((a & 0xF) | ((h & 3) << 4)) - 32);
      y[l + 32] = static_cast<int8_t>(((b & 0xF) | (((h >> 2) & 3) << 4)) - 32);
      y[l + 64] = static_cast<int8_t>(((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
      y[l + 96] = static_cast<int8_t>(((b >> 4) | (((h >> 6) & 3) << 4)) - 32);
    }
  }
}

// Unpacks sub-blocks [sub0, sub0 + nsub) (nsub <= 8; sub0 a multiple of 8 for
// 256-value formats) of a weight row.
template <DType T>
inline void unpack_group(const void* row, int64_t sub0, int nsub, Group& g) {
  using F = Fmt<T>;
  const auto* blocks = static_cast<const typename F::Block*>(row);
  if constexpr (F::kSuper == 256) {
    F::unpack(blocks + sub0 / 8, 8, g);
  } else {
    F::unpack(blocks + sub0, nsub, g);
  }
}

// The weight types above, for dispatch: calls f(std::integral_constant<DType, T>{}).
template <typename Fn>
inline bool dispatch_format(DType t, Fn&& f) {
  switch (t) {
    case DType::kQ8_0: f(std::integral_constant<DType, DType::kQ8_0>{}); return true;
    case DType::kQ4_0: f(std::integral_constant<DType, DType::kQ4_0>{}); return true;
    case DType::kQ4_1: f(std::integral_constant<DType, DType::kQ4_1>{}); return true;
    case DType::kQ5_0: f(std::integral_constant<DType, DType::kQ5_0>{}); return true;
    case DType::kQ5_1: f(std::integral_constant<DType, DType::kQ5_1>{}); return true;
    case DType::kIQ4_NL: f(std::integral_constant<DType, DType::kIQ4_NL>{}); return true;
    case DType::kQ3_K: f(std::integral_constant<DType, DType::kQ3_K>{}); return true;
    case DType::kQ4_K: f(std::integral_constant<DType, DType::kQ4_K>{}); return true;
    case DType::kQ5_K: f(std::integral_constant<DType, DType::kQ5_K>{}); return true;
    case DType::kQ6_K: f(std::integral_constant<DType, DType::kQ6_K>{}); return true;
    case DType::kIQ4_XS: f(std::integral_constant<DType, DType::kIQ4_XS>{}); return true;
    default: return false;
  }
}

}  // namespace dynacore::qmm
