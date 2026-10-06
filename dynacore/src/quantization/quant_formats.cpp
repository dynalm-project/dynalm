#include "dynacore/quantization/quant_formats.h"

#include <algorithm>
#include <cmath>

#include <cstring>

#include "dynacore/tensor/fp16.h"

namespace dynacore::quant {

void dequantize_q4_0(const BlockQ4_0* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK / 2; ++j) {
      y[j] = static_cast<float>((x[i].qs[j] & 0x0F) - 8) * d;
      y[j + kQK / 2] = static_cast<float>((x[i].qs[j] >> 4) - 8) * d;
    }
  }
}

void dequantize_q4_1(const BlockQ4_1* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d), m = fp16_to_fp32(x[i].m);
    for (int j = 0; j < kQK / 2; ++j) {
      y[j] = static_cast<float>(x[i].qs[j] & 0x0F) * d + m;
      y[j + kQK / 2] = static_cast<float>(x[i].qs[j] >> 4) * d + m;
    }
  }
}

void dequantize_q5_0(const BlockQ5_0* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d);
    uint32_t qh;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    for (int j = 0; j < kQK / 2; ++j) {
      const uint8_t h0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const uint8_t h1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      y[j] = static_cast<float>(((x[i].qs[j] & 0x0F) | h0) - 16) * d;
      y[j + kQK / 2] = static_cast<float>(((x[i].qs[j] >> 4) | h1) - 16) * d;
    }
  }
}

void dequantize_q5_1(const BlockQ5_1* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d), m = fp16_to_fp32(x[i].m);
    uint32_t qh;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    for (int j = 0; j < kQK / 2; ++j) {
      const uint8_t h0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
      const uint8_t h1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
      y[j] = static_cast<float>((x[i].qs[j] & 0x0F) | h0) * d + m;
      y[j + kQK / 2] = static_cast<float>((x[i].qs[j] >> 4) | h1) * d + m;
    }
  }
}

void quantize_q8_0(const float* x, BlockQ8_0* y, int64_t nb) {
  for (int64_t b = 0; b < nb; ++b) {
    const float* xb = x + b * kQK;
    float amax = 0;
    for (int i = 0; i < kQK; ++i) amax = std::max(amax, std::fabs(xb[i]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[b].d = fp32_to_fp16(d);
    for (int i = 0; i < kQK; ++i) y[b].qs[i] = static_cast<int8_t>(std::lround(xb[i] * id));
  }
}

void dequantize_q8_0(const BlockQ8_0* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK; ++j) y[j] = static_cast<float>(x[i].qs[j]) * d;
  }
}

void dequantize_q8_1(const BlockQ8_1* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK; ++j) y[j] = static_cast<float>(x[i].qs[j]) * d;
  }
}

void dequantize_q2_K(const BlockQ2_K* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d), dmin = fp16_to_fp32(x[i].dmin);
    const uint8_t* q = x[i].qs;
    int is = 0;
    for (int n = 0; n < kQK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[is++];
        float dl = d * static_cast<float>(sc & 0xF), ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
        sc = x[i].scales[is++];
        dl = d * static_cast<float>(sc & 0xF);
        ml = dmin * static_cast<float>(sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

void dequantize_q3_K(const BlockQ3_K* x, float* y, int64_t nb) {
  constexpr uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
  for (int64_t i = 0; i < nb; ++i) {
    const float d_all = fp16_to_fp32(x[i].d);
    const uint8_t* q = x[i].qs;
    const uint8_t* hm = x[i].hmask;
    uint8_t m = 1;
    // Unpack sixteen 6-bit scales (12 bytes) into signed bytes.
    uint32_t aux[4];
    std::memcpy(aux, x[i].scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int8_t scales[16];
    std::memcpy(scales, aux, 16);

    int is = 0;
    for (int n = 0; n < kQK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        }
        dl = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * static_cast<float>(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m = static_cast<uint8_t>(m << 1);
      }
      q += 32;
    }
  }
}

void dequantize_q4_K(const BlockQ4_K* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d), dmin = fp16_to_fp32(x[i].dmin);
    const uint8_t* q = x[i].qs;
    int is = 0;
    for (int j = 0; j < kQK_K; j += 64) {
      uint8_t sc, mn;
      get_scale_min_k4(is + 0, x[i].scales, sc, mn);
      const float d1 = d * sc, m1 = dmin * mn;
      get_scale_min_k4(is + 1, x[i].scales, sc, mn);
      const float d2 = d * sc, m2 = dmin * mn;
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void dequantize_q5_K(const BlockQ5_K* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d), dmin = fp16_to_fp32(x[i].dmin);
    const uint8_t* ql = x[i].qs;
    const uint8_t* qh = x[i].qh;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < kQK_K; j += 64) {
      uint8_t sc, mn;
      get_scale_min_k4(is + 0, x[i].scales, sc, mn);
      const float d1 = d * sc, m1 = dmin * mn;
      get_scale_min_k4(is + 1, x[i].scales, sc, mn);
      const float d2 = d * sc, m2 = dmin * mn;
      for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2);
      u2 = static_cast<uint8_t>(u2 << 2);
    }
  }
}

void dequantize_q6_K(const BlockQ6_K* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const uint8_t* ql = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* sc = x[i].scales;
    for (int n = 0; n < kQK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

void dequantize_q8_K(const BlockQ8_K* x, float* y, int64_t nb) {
  for (int64_t i = 0; i < nb; ++i, y += kQK_K) {
    for (int j = 0; j < kQK_K; ++j) y[j] = x[i].d * static_cast<float>(x[i].qs[j]);
  }
}

}  // namespace dynacore::quant
