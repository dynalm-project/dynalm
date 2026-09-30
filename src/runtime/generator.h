#pragma once

// Single-sequence generation loop: drives one SequenceState through chunked
// prefill and greedy decode until EOG, the length limit, or cancellation.
//
// The multi-sequence scheduler (continuous batching) replaces this loop in
// later phases; Transformer::forward and the KV cache are shared.

#include <span>
#include <vector>

#include "common/function_ref.h"
#include "common/status.h"
#include "kv_cache/kv_cache.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "tokenizer/tokenizer.h"

namespace engine {

struct GenerationStats {
  int32_t prompt_tokens = 0;
  int32_t generated_tokens = 0;
  double prefill_ms = 0;     // prompt processing
  double ttft_ms = 0;        // start -> first generated token available
  double decode_ms = 0;      // all decode steps after the first token
  std::vector<double> itl_ms;  // inter-token latency per decode step
  FinishReason finish_reason = FinishReason::kNone;

  double prefill_tok_per_s() const { return prefill_ms > 0 ? prompt_tokens / (prefill_ms / 1e3) : 0; }
  double decode_tok_per_s() const {
    return decode_ms > 0 ? static_cast<double>(itl_ms.size()) / (decode_ms / 1e3) : 0;
  }
};

struct GenerateOptions {
  int32_t max_new_tokens = 128;
  bool stop_at_eog = true;
};

class Generator {
 public:
  Generator(Transformer& model, KvCache& cache, const Tokenizer& tokenizer)
      : model_(model), cache_(cache), tokenizer_(tokenizer) {}

  // Calls on_token(id) for each generated token; returning false stops.
  // The sequence's KV blocks are released when generate() returns.
  Status generate(std::span<const TokenId> prompt, const GenerateOptions& opts,
                  FunctionRef<bool(TokenId)> on_token, GenerationStats* stats = nullptr);

 private:
  Transformer& model_;
  KvCache& cache_;
  const Tokenizer& tokenizer_;
  uint64_t next_id_ = 1;
};

}  // namespace engine
