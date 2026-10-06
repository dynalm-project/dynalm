#include "runtime/generator.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "dynacore/base/timer.h"
#include "runtime/sequence.h"
#include "sampling/sampler.h"
#include "common/core.h"

namespace dynalm {

Status Generator::generate(std::span<const TokenId> prompt, const GenerateOptions& opts,
                           FunctionRef<bool(TokenId)> on_token, GenerationStats* stats) {
  if (prompt.empty()) return InvalidArgument("empty prompt");
  const ModelConfig& c = model_.config();
  const auto total = static_cast<int64_t>(prompt.size()) + opts.max_new_tokens;
  if (total > c.context_length) {
    return InvalidArgument("prompt + max_new_tokens (" + std::to_string(total) + ") exceeds context length " +
                           std::to_string(c.context_length));
  }

  SequenceState seq(next_id_++, prompt, StopParams{opts.max_new_tokens, opts.stop_at_eog}, cache_);
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  std::vector<int32_t> positions(static_cast<size_t>(model_.max_batch_tokens()));
  GenerationStats st;
  st.prompt_tokens = seq.prompt_len();

  const Stopwatch total_timer;
  Stopwatch step_timer;
  int64_t first_token_ns = 0;

  while (!is_terminal(seq.status())) {
    if (seq.pending() > 0) {
      // Compute K/V (and logits of the last row) for pending tokens: prompt
      // chunks during prefill, the single sampled token during decode.
      const int32_t n = std::min(seq.pending(), model_.max_batch_tokens());
      const int32_t start = seq.num_computed();
      if (Status s = seq.reserve_kv(n); !s.ok()) {
        seq.fail(s);
        break;
      }
      std::iota(positions.begin(), positions.begin() + n, start);
      if (Status s = model_.forward(seq.tokens().subspan(static_cast<size_t>(start), static_cast<size_t>(n)),
                                    {positions.data(), static_cast<size_t>(n)}, cache_, seq.block_table(), logits);
          !s.ok()) {
        seq.fail(s);
        break;
      }
      seq.mark_computed(n);
      if (seq.generated() == 0 && seq.pending() == 0) st.prefill_ms = total_timer.elapsed_ms();
      continue;
    }

    // All tokens computed: sample the next one.
    const TokenId next = sample_greedy(logits);
    if (seq.generated() == 0) {
      st.ttft_ms = total_timer.elapsed_ms();
      first_token_ns = now_ns();
    } else {
      st.itl_ms.push_back(step_timer.elapsed_ms());
    }
    step_timer.reset();
    seq.append_token(next, tokenizer_);
    if (!on_token(next)) seq.cancel();
  }

  st.generated_tokens = seq.generated();
  st.decode_ms = st.itl_ms.empty() ? 0 : static_cast<double>(now_ns() - first_token_ns) * 1e-6;
  st.finish_reason = seq.finish_reason();
  if (stats) *stats = std::move(st);
  return seq.status() == SequenceStatus::kError ? seq.error() : Status::Ok();
}

}  // namespace dynalm
