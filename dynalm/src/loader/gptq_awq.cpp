#include "loader/gptq_awq.h"

#include <cstring>
#include <vector>

#include "dynacore/tensor/fp16.h"
#include "dynacore/quantization/quant_formats.h"

namespace engine::quant {
namespace {

// AWQ packs output column c*8 + kAwqOrder[k] into nibble k; kAwqNibble is
// the inverse (nibble holding column c*8 + j).
constexpr int kAwqNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};

int64_t group_len(const PackedScheme& s, int64_t in) { return s.group_size > 0 ? s.group_size : in; }

}  // namespace

std::string PackedScheme::describe() const {
  std::string d = method == PackedMethod::kGptq ? "GPTQ" : "AWQ";
  d += " int" + std::to_string(bits);
  d += group_size > 0 ? " g" + std::to_string(group_size) : " per-channel";
  d += sym ? " sym" : " asym";
  if (desc_act) d += " act-order";
  return d;
}

std::string_view repack_target_name(RepackTarget t) {
  switch (t) {
    case RepackTarget::kQ4_0: return "Q4_0";
    case RepackTarget::kQ4_1: return "Q4_1";
    case RepackTarget::kQ8_0: return "Q8_0";
    case RepackTarget::kF16: return "F16";
  }
  return "?";
}

Result<PackedShapes> packed_shapes(const PackedScheme& s, int64_t in, int64_t out) {
  if (s.method == PackedMethod::kGptq && s.bits != 4 && s.bits != 8) {
    return Unsupported("GPTQ with " + std::to_string(s.bits) + "-bit weights is not supported (4 and 8 bits are)");
  }
  if (s.method == PackedMethod::kAwq && s.bits != 4) {
    return Unsupported("AWQ with " + std::to_string(s.bits) + "-bit weights is not supported (4 bits is)");
  }
  if (s.group_size == 0 || s.group_size < -1) return InvalidArgument("invalid group_size");
  const int64_t per = 32 / s.bits;
  if (in <= 0 || out <= 0 || out % per != 0 || (s.method == PackedMethod::kGptq && in % per != 0)) {
    return InvalidArgument("packed layer dims " + std::to_string(out) + "x" + std::to_string(in) +
                           " are not multiples of " + std::to_string(per));
  }
  const int64_t g = group_len(s, in);
  PackedShapes p;
  p.groups = (in + g - 1) / g;
  p.qzeros_cols = out / per;
  if (s.method == PackedMethod::kGptq) {
    p.qweight_rows = in / per;
    p.qweight_cols = out;
  } else {
    p.qweight_rows = in;
    p.qweight_cols = out / per;
  }
  return p;
}

Status unpack(const PackedScheme& s, const PackedLinear& w, uint8_t* q, int32_t* zeros) {
  ENGINE_ASSIGN_OR_RETURN(PackedShapes shp, packed_shapes(s, w.in, w.out));
  const int bits = s.bits;
  const int per = 32 / bits;
  const uint32_t mask = (1u << bits) - 1;
  auto word = [](const int32_t* p, int64_t idx) { return static_cast<uint32_t>(p[idx]); };

  // Zero points [groups][out].
  for (int64_t g = 0; g < shp.groups; ++g) {
    for (int64_t o = 0; o < w.out; ++o) {
      const uint32_t wz = word(w.qzeros, g * shp.qzeros_cols + o / per);
      const int shift = s.method == PackedMethod::kAwq ? kAwqNibble[o % 8] * 4 : static_cast<int>(o % per) * bits;
      int32_t z = static_cast<int32_t>((wz >> shift) & mask);
      if (s.method == PackedMethod::kGptq && s.zero_minus_one) ++z;
      zeros[g * w.out + o] = z;
    }
  }
  // Codes [out][in].
  if (s.method == PackedMethod::kGptq) {
    // One packed word holds `per` consecutive inputs of one output: decode it
    // whole so the writes are contiguous.
    for (int64_t r = 0; r < shp.qweight_rows; ++r) {
      const int32_t* row = w.qweight + r * w.out;
      for (int64_t o = 0; o < w.out; ++o) {
        const uint32_t v = word(row, o);
        uint8_t* dst = q + o * w.in + r * per;
        for (int j = 0; j < per; ++j) dst[j] = static_cast<uint8_t>((v >> (j * bits)) & mask);
      }
    }
  } else {
    for (int64_t i = 0; i < w.in; ++i) {
      const int32_t* row = w.qweight + i * shp.qweight_cols;
      for (int64_t o = 0; o < w.out; ++o) {
        q[o * w.in + i] = static_cast<uint8_t>((word(row, o / 8) >> (kAwqNibble[o % 8] * 4)) & mask);
      }
    }
  }
  if (w.g_idx) {
    for (int64_t i = 0; i < w.in; ++i) {
      if (w.g_idx[i] < 0 || w.g_idx[i] >= shp.groups) {
        return Corrupt("g_idx[" + std::to_string(i) + "] = " + std::to_string(w.g_idx[i]) + " is out of range");
      }
    }
  }
  return Status::Ok();
}

namespace {

// Unpacked layer plus per-element group lookup.
struct Unpacked {
  std::vector<uint8_t> q;
  std::vector<int32_t> zeros;
  int64_t groups = 0, g = 0;
  int64_t group_of(const PackedLinear& w, int64_t i) const { return w.g_idx ? w.g_idx[i] : i / g; }
};

Result<Unpacked> unpack_all(const PackedScheme& s, const PackedLinear& w) {
  ENGINE_ASSIGN_OR_RETURN(PackedShapes shp, packed_shapes(s, w.in, w.out));
  Unpacked u;
  u.q.resize(static_cast<size_t>(w.in * w.out));
  u.zeros.resize(static_cast<size_t>(shp.groups * w.out));
  u.groups = shp.groups;
  u.g = group_len(s, w.in);
  ENGINE_RETURN_IF_ERROR(unpack(s, w, u.q.data(), u.zeros.data()));
  return u;
}

}  // namespace

Status dequantize(const PackedScheme& s, const PackedLinear& w, float* out) {
  ENGINE_ASSIGN_OR_RETURN(Unpacked u, unpack_all(s, w));
  for (int64_t o = 0; o < w.out; ++o) {
    for (int64_t i = 0; i < w.in; ++i) {
      const int64_t g = u.group_of(w, i);
      const float scale = fp16_to_fp32(w.scales[g * w.out + o]);
      out[o * w.in + i] = scale * static_cast<float>(static_cast<int32_t>(u.q[o * w.in + i]) - u.zeros[g * w.out + o]);
    }
  }
  return Status::Ok();
}

namespace {

RepackTarget target_for(const PackedScheme& s, const PackedLinear& w, const Unpacked& u) {
  bool contiguous = true;
  for (int64_t i = 0; w.g_idx && i < w.in && contiguous; ++i) contiguous = w.g_idx[i] == i / u.g;
  // Blocks of 32 inputs must share one (scale, zero): groups of 32k, contiguous.
  if (!contiguous || u.g % kQK != 0 || w.in % kQK != 0) return RepackTarget::kF16;
  const int32_t mid = 1 << (s.bits - 1);
  bool symmetric = true;
  for (int32_t z : u.zeros) symmetric = symmetric && z == mid;
  if (s.bits == 4) return symmetric ? RepackTarget::kQ4_0 : RepackTarget::kQ4_1;
  return symmetric ? RepackTarget::kQ8_0 : RepackTarget::kF16;
}

}  // namespace

RepackTarget choose_target(const PackedScheme& s, const PackedLinear& w) {
  auto u = unpack_all(s, w);
  return u.ok() ? target_for(s, w, *u) : RepackTarget::kF16;
}

Result<Tensor> repack(const PackedScheme& s, const PackedLinear& w, RepackTarget* target_out) {
  ENGINE_ASSIGN_OR_RETURN(Unpacked u, unpack_all(s, w));  // once: target choice and conversion share it
  const RepackTarget target = target_for(s, w, u);
  if (target_out) *target_out = target;
  const DType dt = target == RepackTarget::kQ4_0   ? DType::kQ4_0
                   : target == RepackTarget::kQ4_1 ? DType::kQ4_1
                   : target == RepackTarget::kQ8_0 ? DType::kQ8_0
                                                   : DType::kF16;
  ENGINE_ASSIGN_OR_RETURN(Tensor t, Tensor::empty(dt, {w.out, w.in}));
  const int64_t nb = w.in / kQK;
  for (int64_t o = 0; o < w.out; ++o) {
    const uint8_t* qrow = u.q.data() + o * w.in;
    switch (target) {
      case RepackTarget::kQ4_0:
      case RepackTarget::kQ4_1: {
        for (int64_t b = 0; b < nb; ++b) {
          const int64_t g = u.group_of(w, b * kQK);
          const uint16_t d = w.scales[g * w.out + o];
          uint8_t* qs;
          if (target == RepackTarget::kQ4_0) {
            auto* blk = static_cast<BlockQ4_0*>(t.data()) + o * nb + b;
            blk->d = d;  // value = d * (q - 8): exactly GPTQ's scale * (q - zero)
            qs = blk->qs;
          } else {
            auto* blk = static_cast<BlockQ4_1*>(t.data()) + o * nb + b;
            blk->d = d;  // value = d * q + m with m = -scale * zero (rounded to fp16)
            blk->m = fp32_to_fp16(-fp16_to_fp32(d) * static_cast<float>(u.zeros[g * w.out + o]));
            qs = blk->qs;
          }
          for (int j = 0; j < kQK / 2; ++j) {
            qs[j] = static_cast<uint8_t>(qrow[b * kQK + j] | (qrow[b * kQK + j + kQK / 2] << 4));
          }
        }
        break;
      }
      case RepackTarget::kQ8_0: {
        for (int64_t b = 0; b < nb; ++b) {
          const int64_t g = u.group_of(w, b * kQK);
          auto* blk = static_cast<BlockQ8_0*>(t.data()) + o * nb + b;
          blk->d = w.scales[g * w.out + o];
          for (int j = 0; j < kQK; ++j) blk->qs[j] = static_cast<int8_t>(static_cast<int>(qrow[b * kQK + j]) - 128);
        }
        break;
      }
      case RepackTarget::kF16: {
        auto* row = static_cast<uint16_t*>(t.data()) + o * w.in;
        for (int64_t i = 0; i < w.in; ++i) {
          const int64_t g = u.group_of(w, i);
          const float v = fp16_to_fp32(w.scales[g * w.out + o]) *
                          static_cast<float>(static_cast<int32_t>(qrow[i]) - u.zeros[g * w.out + o]);
          row[i] = fp32_to_fp16(v);
        }
        break;
      }
    }
  }
  return t;
}

}  // namespace engine::quant
