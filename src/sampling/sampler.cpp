#include "sampling/sampler.h"

namespace engine {

TokenId sample_greedy(std::span<const float> logits) {
  TokenId best = 0;
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[static_cast<size_t>(best)]) best = static_cast<TokenId>(i);
  }
  return best;
}

}  // namespace engine
