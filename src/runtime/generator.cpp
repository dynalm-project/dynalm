#include "runtime/generator.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "common/timer.h"
#include "sampling/sampler.h"

namespace engine {

Status Generator::generate(std::span<const TokenId> prompt, const GenerateOptions& opts,
                           FunctionRef<bool(TokenId)> on_token, GenerationStats* stats) {
  if (prompt.empty()) return InvalidArgument("empty prompt");
  const ModelConfig& c = model_.config();
  const auto total = static_cast<int64_t>(prompt.size()) + opts.max_new_tokens;
  if (total > c.context_length) {
    return InvalidArgument("prompt + max_new_tokens (" + std::to_string(total) + ") exceeds context length " +
                           std::to_string(c.context_length));
  }

  KvSequence seq(cache_);
  ENGINE_RETURN_IF_ERROR(seq.reserve(static_cast<int64_t>(prompt.size())));
  std::vector<float> logits(static_cast<size_t>(c.vocab_size));
  std::vector<int32_t> positions(static_cast<size_t>(model_.max_batch_tokens()));
  GenerationStats st;
  st.prompt_tokens = static_cast<int32_t>(prompt.size());

  // Prefill in chunks of at most max_batch_tokens.
  const Stopwatch total_timer;
  const int32_t chunk = model_.max_batch_tokens();
  for (size_t off = 0; off < prompt.size(); off += static_cast<size_t>(chunk)) {
    const size_t n = std::min(prompt.size() - off, static_cast<size_t>(chunk));
    std::iota(positions.begin(), positions.begin() + static_cast<std::ptrdiff_t>(n), static_cast<int32_t>(off));
    ENGINE_RETURN_IF_ERROR(model_.forward(prompt.subspan(off, n), {positions.data(), n}, cache_, seq.block_table(),
                                          logits));
  }
  st.prefill_ms = total_timer.elapsed_ms();

  int32_t pos = static_cast<int32_t>(prompt.size());
  TokenId next = sample_greedy(logits);
  st.ttft_ms = total_timer.elapsed_ms();
  Stopwatch step_timer;
  const Stopwatch decode_timer;

  for (int32_t i = 0; i < opts.max_new_tokens; ++i) {
    if (i > 0) st.itl_ms.push_back(step_timer.elapsed_ms());
    step_timer.reset();
    ++st.generated_tokens;
    const bool eog = tokenizer_.is_eog(next);
    if (!on_token(next) || (eog && opts.stop_at_eog) || i + 1 == opts.max_new_tokens) break;

    ENGINE_RETURN_IF_ERROR(seq.reserve(pos + 1));
    positions[0] = pos;
    ENGINE_RETURN_IF_ERROR(model_.forward({&next, 1}, {positions.data(), 1}, cache_, seq.block_table(), logits));
    ++pos;
    next = sample_greedy(logits);
  }
  st.decode_ms = st.itl_ms.empty() ? 0 : decode_timer.elapsed_ms();
  if (stats) *stats = std::move(st);
  return Status::Ok();
}

}  // namespace engine
