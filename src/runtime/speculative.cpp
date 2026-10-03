#include "runtime/speculative.h"

#include <algorithm>
#include <numeric>
#include <string>

namespace engine {

// --- NgramDrafter ---------------------------------------------------------------

void NgramDrafter::propose(std::span<const TokenId> ctx, int32_t k, std::vector<TokenId>& out) {
  out.clear();
  const auto len = static_cast<int64_t>(ctx.size());
  for (int32_t n = std::min<int32_t>(max_n_, static_cast<int32_t>(len) - 1); n >= min_n_ && k > 0; --n) {
    const std::span<const TokenId> suffix = ctx.subspan(static_cast<size_t>(len - n));
    // Latest earlier occurrence of the suffix whose continuation exists.
    for (int64_t start = len - n - 1; start >= 0; --start) {
      if (!std::equal(suffix.begin(), suffix.end(), ctx.begin() + start)) continue;
      const int64_t from = start + n;
      const int64_t count = std::min<int64_t>(k, len - from);
      out.assign(ctx.begin() + from, ctx.begin() + from + count);
      return;
    }
  }
}

// --- ModelDrafter -----------------------------------------------------------------

ModelDrafter::ModelDrafter(Transformer& draft, KvBlockPool& cache)
    : model_(draft), kv_(cache), logits_(static_cast<size_t>(draft.config().vocab_size)),
      positions_(static_cast<size_t>(draft.max_batch_tokens())) {}

Result<std::unique_ptr<ModelDrafter>> ModelDrafter::create(Transformer& draft, KvBlockPool& cache,
                                                           const Tokenizer& draft_tok, const Tokenizer& target_tok) {
  if (draft_tok.vocab_size() != target_tok.vocab_size()) {
    return InvalidArgument("draft model vocabulary (" + std::to_string(draft_tok.vocab_size()) +
                           ") differs from the target's (" + std::to_string(target_tok.vocab_size()) + ")");
  }
  for (TokenId id = 0; id < draft_tok.vocab_size(); ++id) {
    if (draft_tok.token_text(id) != target_tok.token_text(id)) {
      return InvalidArgument("draft and target vocabularies differ at token " + std::to_string(id));
    }
  }
  return std::unique_ptr<ModelDrafter>(new ModelDrafter(draft, cache));
}

// Computes K/V for `tokens` (appended after computed_) and returns the
// draft's greedy next token.
Status ModelDrafter::feed(std::span<const TokenId> tokens, TokenId& next) {
  size_t done = 0;
  while (done < tokens.size()) {
    const auto n = std::min<size_t>(tokens.size() - done, static_cast<size_t>(model_.max_batch_tokens()));
    const auto start = static_cast<int32_t>(computed_.size());
    ENGINE_RETURN_IF_ERROR(kv_.reserve(start + static_cast<int64_t>(n)));
    std::iota(positions_.begin(), positions_.begin() + static_cast<std::ptrdiff_t>(n), start);
    ENGINE_RETURN_IF_ERROR(model_.forward(tokens.subspan(done, n), {positions_.data(), n}, kv_.pool(),
                                          kv_.block_table(), logits_));
    computed_.insert(computed_.end(), tokens.begin() + static_cast<std::ptrdiff_t>(done),
                     tokens.begin() + static_cast<std::ptrdiff_t>(done + n));
    done += n;
  }
  next = sample_greedy(logits_);
  return Status::Ok();
}

void ModelDrafter::propose(std::span<const TokenId> ctx, int32_t k, std::vector<TokenId>& out) {
  out.clear();
  if (k <= 0 || ctx.empty() || !status_.ok()) return;
  // Roll back to the longest prefix shared with what the draft has computed
  // (rejected drafts and their K/V are dropped). Keep >= 1 token to feed.
  size_t common = 0;
  while (common < computed_.size() && common < ctx.size() && computed_[common] == ctx[common]) ++common;
  common = std::min(common, ctx.size() - 1);
  computed_.resize(common);
  kv_.truncate(static_cast<int64_t>(common));
  if (static_cast<int64_t>(ctx.size()) + k > model_.config().context_length) return;

  TokenId next;
  if (Status s = feed(ctx.subspan(common), next); !s.ok()) {
    status_ = s;  // drafting is an optimization: stop proposing, never fail the request
    return;
  }
  out.push_back(next);
  while (static_cast<int32_t>(out.size()) < k) {
    const TokenId last = out.back();
    if (Status s = feed({&last, 1}, next); !s.ok()) {
      status_ = s;
      return;
    }
    out.push_back(next);
  }
}

// --- SpeculativeGenerator -------------------------------------------------------------

Status SpeculativeGenerator::generate(std::span<const TokenId> prompt, const GenerateOptions& opts,
                                      const SpeculativeOptions& spec, FunctionRef<bool(TokenId)> on_token,
                                      SpeculativeStats* stats_out) {
  if (prompt.empty()) return InvalidArgument("empty prompt");
  if (spec.draft_tokens < 0 || spec.draft_tokens + 1 > model_.max_batch_tokens()) {
    return InvalidArgument("draft_tokens must be in [0, max_batch_tokens - 1]");
  }
  ENGINE_RETURN_IF_ERROR(spec.sampling.validate());
  const ModelConfig& c = model_.config();
  const auto vocab = static_cast<size_t>(c.vocab_size);
  if (static_cast<int64_t>(prompt.size()) + opts.max_new_tokens > c.context_length) {
    return InvalidArgument("prompt + max_new_tokens exceeds context length " + std::to_string(c.context_length));
  }

  std::vector<TokenId> tokens(prompt.begin(), prompt.end());
  KvBlockTable kv(cache_);
  Sampler sampler(spec.sampling);
  std::vector<float> logits(static_cast<size_t>(spec.draft_tokens + 1) * vocab);
  std::vector<TokenId> drafts;
  SpeculativeStats st;

  // Prefill; logits of the last prompt token give the first token.
  int64_t computed = 0;
  while (computed < static_cast<int64_t>(tokens.size())) {
    const int64_t n = std::min<int64_t>(static_cast<int64_t>(tokens.size()) - computed, model_.max_batch_tokens());
    ENGINE_RETURN_IF_ERROR(kv.reserve(computed + n));
    const bool last_chunk = computed + n == static_cast<int64_t>(tokens.size());
    const SeqBatch b{std::span<const TokenId>(tokens).subspan(static_cast<size_t>(computed), static_cast<size_t>(n)),
                     static_cast<int32_t>(computed), kv.block_table(), last_chunk, 1};
    ENGINE_RETURN_IF_ERROR(model_.forward_batch({&b, 1}, cache_, std::span<float>(logits.data(), last_chunk ? vocab : 0)));
    computed += n;
  }

  // Emits one token; false when generation must stop after it.
  bool stop = false;
  auto emit = [&](TokenId t) {
    tokens.push_back(t);
    ++st.generated;
    if (!on_token(t)) stop = true;
    if (opts.stop_at_eog && tokenizer_.is_eog(t)) stop = true;
    if (st.generated >= opts.max_new_tokens) stop = true;
    return !stop;
  };
  emit(sampler.sample(std::span<float>(logits.data(), vocab), tokens));

  while (!stop) {
    // tokens[computed] is the last emitted token, not yet in the KV cache.
    const int32_t room = opts.max_new_tokens - st.generated;  // tokens we may still emit (>= 1)
    drafter_.propose(tokens, std::min(spec.draft_tokens, room - 1), drafts);
    const auto nd = static_cast<int32_t>(drafts.size());
    std::vector<TokenId>& feed = drafts;  // reuse: [last token, drafts...]
    feed.insert(feed.begin(), tokens.back());
    ENGINE_RETURN_IF_ERROR(kv.reserve(computed + nd + 1));
    const SeqBatch b{feed, static_cast<int32_t>(computed), kv.block_table(), true, nd + 1};
    ENGINE_RETURN_IF_ERROR(
        model_.forward_batch({&b, 1}, cache_, std::span<float>(logits.data(), static_cast<size_t>(nd + 1) * vocab)));
    ++st.target_passes;
    st.drafted += nd;

    // Row i scores the token after feed[i]: accept drafts while they match.
    int32_t accepted = 0;
    for (int32_t i = 0; i <= nd && !stop; ++i) {
      const std::span<float> row(logits.data() + static_cast<size_t>(i) * vocab, vocab);
      if (i < nd) {
        bool ok = false;
        const TokenId t = sampler.sample_speculative(row, tokens, feed[static_cast<size_t>(i) + 1], ok);
        emit(t);
        if (!ok) break;
        ++accepted;
      } else {
        emit(sampler.sample(row, tokens));  // all drafts accepted: bonus token
      }
    }
    st.accepted += accepted;
    // Valid K/V: the fed last token and the accepted drafts. Roll back the rest.
    computed += accepted + 1;
    kv.truncate(computed);
  }
  if (stats_out) *stats_out = st;
  return Status::Ok();
}

}  // namespace engine
