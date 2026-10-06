#pragma once

// Token selection (DD-043).
//
// Pipeline per token, on the logits row (modified in place):
//   1. penalties over the last `penalty_last_n` context tokens:
//        repetition (llama/CTRL style: l > 0 ? l / r : l * r),
//        frequency (l -= count * f), presence (l -= p if seen)
//   2. temperature <= 0 (or top_k == 1)  -> greedy argmax, ties to the lowest id
//   3. top-k: keep the k largest logits
//   4. temperature: p_i ∝ exp((l_i - max) / T)
//   5. min-p: drop tokens with p_i < min_p * p_max
//   6. top-p (nucleus): keep the smallest prefix (by probability) with mass >= top_p
//   7. draw from the kept tokens, renormalized
//
// A Sampler belongs to one sequence and owns its RNG (xoshiro256**, the same
// sequence on every platform for a given seed) and scratch buffers, so
// sampling a token allocates nothing once warm.

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "dynacore/base/status.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

struct SamplingParams {
  float temperature = 0.0f;  // <= 0: greedy
  int32_t top_k = 0;         // 0: off
  float top_p = 1.0f;        // 1: off
  float min_p = 0.0f;        // 0: off
  float repetition_penalty = 1.0f;  // 1: off
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int32_t penalty_last_n = 64;  // context window the penalties look at (-1: whole context)
  uint64_t seed = 0;
  bool has_seed = false;        // otherwise seeded from std::random_device

  bool greedy() const { return temperature <= 0.0f || top_k == 1; }
  bool has_penalties() const {
    return repetition_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f;
  }
  Status validate() const;
};

// Index of the largest logit; ties resolve to the lowest id (deterministic).
TokenId sample_greedy(std::span<const float> logits);
float max_logit(std::span<const float> logits);

// Small, fast, portable PRNG (xoshiro256**, seeded through splitmix64).
class Rng {
 public:
  explicit Rng(uint64_t seed);
  uint64_t next();
  double uniform();  // [0, 1), 53 bits
 private:
  uint64_t s_[4];
};

class Sampler {
 public:
  explicit Sampler(const SamplingParams& params);

  const SamplingParams& params() const { return params_; }

  // Samples the next token. `logits` (one vocab row) is modified in place;
  // `context` is the sequence so far (prompt + generated), for penalties.
  TokenId sample(std::span<float> logits, std::span<const TokenId> context);

  // Applies only the penalties (step 1); exposed for tests and speculative
  // decoding, which needs the same adjusted distribution.
  void apply_penalties(std::span<float> logits, std::span<const TokenId> context);

  // Speculative verification of a deterministic draft token (DD-044): accept
  // `draft` with probability p(draft) under this sampler's distribution;
  // otherwise return a token drawn from p with `draft` removed. This is the
  // Leviathan et al. rule for a one-hot draft distribution, so the returned
  // token is distributed exactly as sample()'s. Greedy: accepted iff argmax.
  TokenId sample_speculative(std::span<float> logits, std::span<const TokenId> context, TokenId draft,
                             bool& accepted);

 private:
  // Kept distribution after top-k / temperature / min-p / top-p: n entries in
  // probs_ (token ids via token_at), unnormalized, summing to `total`.
  size_t distribution(std::span<float> logits, double& total);
  TokenId token_at(size_t i) const { return dense_ ? static_cast<TokenId>(i) : cand_[i].second; }
  TokenId draw(size_t n, double total, TokenId exclude);

  SamplingParams params_;
  Rng rng_;
  std::vector<std::pair<float, TokenId>> cand_;  // (logit, id) candidates, reused
  std::vector<float> probs_;
  std::vector<TokenId> history_;
  bool dense_ = false;  // probs_ indexed by token id (temperature-only path)
};

}  // namespace dynalm
