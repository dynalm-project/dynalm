#pragma once

// Speculative decoding (DD-044).
//
// A cheap Drafter proposes up to k next tokens; the target model scores all
// of them in ONE forward pass (logits for every drafted position), and the
// Sampler accepts a prefix:
//   greedy   — accept while the target's argmax equals the draft; output is
//              token-for-token identical to plain greedy decoding;
//   sampling — accept draft d with probability p(d), else resample from p
//              without d (Leviathan et al. with a one-hot draft), so outputs
//              follow exactly the target's sampling distribution.
// The first rejected position yields the target's own token, and a fully
// accepted draft yields one bonus token, so every target pass produces
// 1..k+1 tokens. KV of rejected drafts is rolled back (KvBlockTable::truncate).
//
// Drafters:
//   NgramDrafter — prompt lookup: continue the latest earlier occurrence of
//                  the context's suffix (no model; strong on edits, code,
//                  summaries quoting their input).
//   ModelDrafter — a small model sharing the target's vocabulary, greedy.
//
// This is the single-sequence path (Generator's role). Hosting it in the
// continuous-batching scheduler means a decode entry contributing 1 + k rows
// per step with the same verify/rollback, which the batch forward already
// supports (SeqBatch::logits_last).

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "dynacore/base/function_ref.h"
#include "dynacore/base/status.h"
#include "kv_cache/kv_cache.h"
#include "model/transformer.h"
#include "runtime/generator.h"
#include "sampling/sampler.h"
#include "tokenizer/tokenizer.h"
#include "common/core.h"

namespace dynalm {

class Drafter {
 public:
  virtual ~Drafter() = default;
  virtual std::string_view name() const = 0;
  // Proposes up to `k` tokens continuing `context` (fewer or none is fine).
  virtual void propose(std::span<const TokenId> context, int32_t k, std::vector<TokenId>& out) = 0;
};

class NgramDrafter final : public Drafter {
 public:
  explicit NgramDrafter(int32_t max_ngram = 4, int32_t min_ngram = 2) : max_n_(max_ngram), min_n_(min_ngram) {}
  std::string_view name() const override { return "ngram"; }
  void propose(std::span<const TokenId> context, int32_t k, std::vector<TokenId>& out) override;

 private:
  int32_t max_n_, min_n_;
};

class ModelDrafter final : public Drafter {
 public:
  // The draft model must use the target's vocabulary (same token texts).
  static Result<std::unique_ptr<ModelDrafter>> create(Transformer& draft, KvBlockPool& cache,
                                                      const Tokenizer& draft_tokenizer,
                                                      const Tokenizer& target_tokenizer);
  std::string_view name() const override { return "model"; }
  void propose(std::span<const TokenId> context, int32_t k, std::vector<TokenId>& out) override;
  Status status() const { return status_; }

 private:
  ModelDrafter(Transformer& draft, KvBlockPool& cache);
  Status feed(std::span<const TokenId> tokens, TokenId& next);

  Transformer& model_;
  KvBlockTable kv_;
  std::vector<TokenId> computed_;  // tokens whose K/V is in kv_
  std::vector<float> logits_;
  std::vector<int32_t> positions_;
  Status status_;
};

struct SpeculativeOptions {
  int32_t draft_tokens = 4;  // k (the largest k when adaptive)
  SamplingParams sampling;   // greedy by default
  // Choose k per round from measured output tokens/s, including k = 0 (plain
  // decoding) when speculation does not pay off (P13, DD-062). Output is the
  // same either way; only the speed changes.
  bool adaptive = true;
};

// Picks k for each verification round from measured output rates. Arms are
// {0, min(3, max_k), max_k}: plain decoding, a 4-row verification (the int8
// decode path, DD-053) and the configured k. Each arm keeps exponentially
// weighted tokens and milliseconds; their ratio is its output rate. After a
// warm-up of every arm the best one runs. Then the least recently measured
// other arm runs for kProbeRounds rounds, after kReprobeMin rounds at first.
// The gap doubles (up to kReprobeMax) while probes keep confirming the best
// arm and resets when one changes it, so the choice follows changing
// acceptance (e.g. code vs prose) without paying for probes in steady state.
// The first drafting round after plain rounds is not measured: a model
// drafter spends it catching up on the tokens it skipped, a one-off cost of
// switching rather than the arm's rate.
class SpecController {
 public:
  static constexpr int32_t kWarmupRounds = 3;
  static constexpr int32_t kReprobeMin = 16;
  static constexpr int32_t kReprobeMax = 128;
  static constexpr int32_t kProbeRounds = 3;
  static constexpr double kAlpha = 0.25;

  explicit SpecController(int32_t max_k);
  int32_t next_k();
  // One finished round: tokens emitted and wall time (drafting included).
  void record(int32_t k, int32_t emitted, double ms);
  int32_t best_k() const;
  double rate(int32_t k) const;  // tokens per ms; 0 before any sample

 private:
  struct Arm {
    int32_t k = 0;
    double tok = 0, ms = 0;
    int32_t samples = 0;
    int64_t last_round = -1;
  };
  size_t best_index() const;
  std::vector<Arm> arms_;
  int64_t round_ = 0;
  int32_t probe_left_ = 0;
  size_t probe_arm_ = 0;
  bool probing_ = false;     // a probe ran; judge it on the next choice
  int32_t probe_best_k_ = 0;  // best arm when the probe started
  int32_t interval_ = kReprobeMin;
  int64_t next_probe_ = -1;  // round of the next probe; -1 until warm-up ends
  int32_t last_k_ = 0;
};

struct SpeculativeStats {
  int64_t target_passes = 0;  // decode verification passes
  int64_t drafted = 0;
  int64_t accepted = 0;
  int32_t generated = 0;
  int64_t plain_passes = 0;  // adaptive rounds that ran with k = 0
  int32_t final_k = 0;       // the adaptive controller's choice at the end
  double acceptance() const { return drafted ? static_cast<double>(accepted) / static_cast<double>(drafted) : 0; }
  double tokens_per_pass() const {
    return target_passes ? static_cast<double>(generated - 1) / static_cast<double>(target_passes) : 0;
  }
};

class SpeculativeGenerator {
 public:
  SpeculativeGenerator(Transformer& target, KvBlockPool& cache, const Tokenizer& tokenizer, Drafter& drafter)
      : model_(target), cache_(cache), tokenizer_(tokenizer), drafter_(drafter) {}

  // Like Generator::generate; on_token returning false stops.
  Status generate(std::span<const TokenId> prompt, const GenerateOptions& opts, const SpeculativeOptions& spec,
                  FunctionRef<bool(TokenId)> on_token, SpeculativeStats* stats = nullptr);

 private:
  Transformer& model_;
  KvBlockPool& cache_;
  const Tokenizer& tokenizer_;
  Drafter& drafter_;
};

}  // namespace dynalm
