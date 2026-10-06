#include "runtime/speculative.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>
#include "common/core.h"

namespace dynalm {

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

// --- SpecController ----------------------------------------------------------------

SpecController::SpecController(int32_t max_k) {
  for (int32_t k : {max_k, std::min(3, max_k), 0}) {
    if (k < 0) continue;
    if (std::none_of(arms_.begin(), arms_.end(), [&](const Arm& a) { return a.k == k; })) arms_.push_back({k});
  }
}

double SpecController::rate(int32_t k) const {
  for (const Arm& a : arms_) {
    if (a.k == k) return a.ms > 0 ? a.tok / a.ms : 0;
  }
  return 0;
}

size_t SpecController::best_index() const {
  size_t best = 0;
  for (size_t i = 1; i < arms_.size(); ++i) {
    if (rate(arms_[i].k) > rate(arms_[best].k)) best = i;
  }
  return best;
}

int32_t SpecController::best_k() const { return arms_[best_index()].k; }

int32_t SpecController::next_k() {
  for (const Arm& a : arms_) {
    if (a.samples < kWarmupRounds) return a.k;
  }
  if (probe_left_ > 0) {
    --probe_left_;
    return arms_[probe_arm_].k;
  }
  const size_t best = best_index();
  if (probing_) {  // a probe finished: back off while it confirms the best arm
    probing_ = false;
    interval_ = arms_[best].k == probe_best_k_ ? std::min(2 * interval_, kReprobeMax) : kReprobeMin;
    next_probe_ = round_ + interval_;
  }
  if (next_probe_ < 0) next_probe_ = round_ + interval_;
  if (arms_.size() > 1 && round_ >= next_probe_) {
    size_t oldest = best == 0 ? 1 : 0;
    for (size_t i = 0; i < arms_.size(); ++i) {
      if (i != best && arms_[i].last_round < arms_[oldest].last_round) oldest = i;
    }
    probe_arm_ = oldest;
    probe_left_ = kProbeRounds - 1;
    probing_ = true;
    probe_best_k_ = arms_[best].k;
    return arms_[oldest].k;
  }
  return arms_[best].k;
}

void SpecController::record(int32_t k, int32_t emitted, double ms) {
  const bool catch_up = k > 0 && last_k_ == 0;
  last_k_ = k;
  for (Arm& a : arms_) {
    if (catch_up) {
      if (a.k == k) a.last_round = round_;
      continue;
    }
    if (a.k != k) continue;
    if (a.samples == 0) {
      a.tok = emitted;
      a.ms = ms;
    } else {
      a.tok += kAlpha * (emitted - a.tok);
      a.ms += kAlpha * (ms - a.ms);
    }
    ++a.samples;
    a.last_round = round_;
  }
  ++round_;
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

  SpecController ctrl(spec.draft_tokens);
  using Clock = std::chrono::steady_clock;
  while (!stop) {
    // tokens[computed] is the last emitted token, not yet in the KV cache.
    const int32_t room = opts.max_new_tokens - st.generated;  // tokens we may still emit (>= 1)
    const int32_t k = spec.adaptive ? ctrl.next_k() : spec.draft_tokens;
    const Clock::time_point t0 = Clock::now();
    const int32_t emitted_before = st.generated;
    drafts.clear();
    if (k > 0) drafter_.propose(tokens, std::min(k, room - 1), drafts);
    if (k == 0) ++st.plain_passes;
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
    if (spec.adaptive) {
      ctrl.record(k, st.generated - emitted_before,
                  std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
  }
  st.final_k = spec.adaptive ? ctrl.best_k() : spec.draft_tokens;
  if (stats_out) *stats_out = st;
  return Status::Ok();
}

}  // namespace dynalm
