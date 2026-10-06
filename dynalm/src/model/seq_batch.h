#pragma once

// One sequence's slice of a batched forward pass. Shared by the scheduler, the
// execution planner and the model runtime.

#include <cstdint>
#include <span>

#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

// tokens[i] runs at position start_pos + i; K/V are written through
// `block_table`.
struct SeqBatch {
  std::span<const TokenId> tokens;
  int32_t start_pos = 0;
  std::span<const int32_t> block_table;
  bool want_logits = true;  // compute logits for this sequence's last token
  // With want_logits: logits for the last `logits_last` tokens (speculative
  // verification scores every drafted position in one pass).
  int32_t logits_last = 1;
};

}  // namespace dynalm
