#pragma once

// Token selection. Phase 6: greedy only. Temperature / top-k / top-p / min-p
// and penalties arrive with the sampling phase.

#include <span>

#include "tokenizer/tokenizer.h"

namespace engine {

// Index of the largest logit; ties resolve to the lowest id (deterministic).
TokenId sample_greedy(std::span<const float> logits);

}  // namespace engine
