#pragma once

// Parameter types of the device ops (norms, activations, RoPE).
//
// They describe *what* an op computes, independent of any model family or
// file format: the model layer fills them from its own configuration and
// hands them to the backend unchanged.

#include <cstdint>

namespace dynacore {

enum class NormType : uint8_t { kRmsNorm, kLayerNorm };
enum class Activation : uint8_t { kSilu, kGelu, kGeluTanh };

enum class RopeStyle : uint8_t {
  kNone,
  kInterleaved,  // rotate adjacent pairs (x0,x1),(x2,x3)...  (Llama-style checkpoints after the Q/K permute)
  kHalfSplit,    // rotate (x_i, x_{i+d/2})                   (NeoX / HF default)
};

enum class RopeScaling : uint8_t { kNone, kLinear, kYarn, kLlama3 };

struct RopeConfig {
  RopeStyle style = RopeStyle::kHalfSplit;
  int32_t dim = 0;              // rotated dims per head (<= head_dim; partial RoPE when smaller)
  float freq_base = 10000.0f;
  RopeScaling scaling = RopeScaling::kNone;
  float scaling_factor = 1.0f;
  int64_t original_context = 0;  // pre-scaling training context (YaRN / Llama3)
  // YaRN / Llama3 parameters (used only when the matching scaling is set).
  float yarn_beta_fast = 32.0f;
  float yarn_beta_slow = 1.0f;
  float llama3_low_freq_factor = 1.0f;
  float llama3_high_freq_factor = 4.0f;
};

}  // namespace dynacore
