#pragma once

// GPTQ and AWQ packed integer weights.
//
// These checkpoints store each linear layer as several tensors:
//   qweight  int32, packed integer codes
//   scales   fp16 [groups, out]
//   qzeros   int32 [groups, out * bits / 32], packed zero points
//   g_idx    int32 [in] (GPTQ only, optional): group of each input column
// and the real weight is  W[o][i] = scale[g(i)][o] * (q[o][i] - zero[g(i)][o]).
//
// Packing (from AutoGPTQ / GPTQModel and AutoAWQ):
//   GPTQ  qweight [in * bits / 32, out]: codes packed along the input dim,
//         element i in bits ((i % per_word) * bits); qzeros packed along the
//         output dim. Checkpoint format "gptq" (v1) stores zero - 1.
//   AWQ   qweight [in, out * bits / 32]: codes packed along the output dim in
//         the interleaved order {0, 2, 4, 6, 1, 3, 5, 7}; qzeros likewise.
//
// The engine does not run these layouts directly. At load they are repacked
// into block formats the CPU kernels already execute, exactly when possible
// (DD-041):
//   4-bit, symmetric (zero = 8), contiguous groups of 32k   -> Q4_0 (bit-exact)
//   8-bit, symmetric (zero = 128), contiguous groups of 32k -> Q8_0 (bit-exact)
//   4-bit, asymmetric, contiguous groups of 32k             -> Q4_1 (min rounded to fp16)
//   anything else (act-order g_idx, other group sizes)      -> F16
// so quantization stays a subsystem concern: model code sees ordinary DTypes.

#include <cstdint>
#include <string>

#include "common/status.h"
#include "tensor/tensor.h"

namespace engine::quant {

enum class PackedMethod : uint8_t { kGptq, kAwq };

struct PackedScheme {
  PackedMethod method = PackedMethod::kGptq;
  int bits = 4;               // GPTQ: 4 or 8; AWQ: 4
  int group_size = 128;       // -1 = one group per output row (whole input dim)
  bool sym = true;            // declared symmetric (verified on the data, not trusted)
  bool desc_act = false;      // GPTQ act-order: g_idx is a permutation of groups
  bool zero_minus_one = true; // GPTQ v1 checkpoints store zero - 1
  std::string describe() const;  // e.g. "GPTQ int4 g128 sym"
};

// One packed linear layer; pointers into mapped file data (little-endian).
struct PackedLinear {
  const int32_t* qweight = nullptr;
  const uint16_t* scales = nullptr;  // fp16
  const int32_t* qzeros = nullptr;
  const int32_t* g_idx = nullptr;    // nullable
  int64_t in = 0, out = 0;
};

// Expected element counts of each component for given dims (validation).
struct PackedShapes {
  int64_t qweight_rows, qweight_cols, groups, qzeros_cols;
};
Result<PackedShapes> packed_shapes(const PackedScheme& s, int64_t in, int64_t out);

// Integer codes q[o][i] and effective zero points z[g][o] (with the v1 offset
// applied). Reference unpacking; validates g_idx range.
Status unpack(const PackedScheme& s, const PackedLinear& w, uint8_t* q /*[out][in]*/, int32_t* zeros /*[groups][out]*/);

// Reference dequantization: W[o][i] = scale * (q - zero), fp32.
Status dequantize(const PackedScheme& s, const PackedLinear& w, float* out /*[out][in]*/);

enum class RepackTarget : uint8_t { kQ4_0, kQ4_1, kQ8_0, kF16 };
std::string_view repack_target_name(RepackTarget t);

// Best executable layout for this layer (inspects g_idx and the zero points).
RepackTarget choose_target(const PackedScheme& s, const PackedLinear& w);

// Converts to an engine tensor [out, in] in `choose_target`'s format.
Result<Tensor> repack(const PackedScheme& s, const PackedLinear& w, RepackTarget* target = nullptr);

}  // namespace engine::quant
