#include "sampling/sampler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <random>
#include <string>

namespace engine {

// exp(x) for x <= 0, ~1e-7 relative error. Our own (not libm's) so that a
// seed gives the same tokens on every platform, and simple enough to
// vectorize: x = n ln2 + r, |r| <= ln2/2, e^r by a degree-6 polynomial,
// 2^n through the exponent bits.
inline float exp_nonpos(float x) {
  x = std::max(x, -87.0f);
  // Round to nearest without a libm call: adding 1.5 * 2^23 pushes the
  // fraction out of the mantissa (exact for |v| < 2^22).
  const float n = (x * 1.44269504088896341f + 12582912.0f) - 12582912.0f;
  const float r = x - n * 0.693145751953125f - n * 1.428606765330187e-06f;
  float p = 1.0f / 720;
  p = p * r + 1.0f / 120;
  p = p * r + 1.0f / 24;
  p = p * r + 1.0f / 6;
  p = p * r + 0.5f;
  p = p * r + 1.0f;
  p = p * r + 1.0f;
  const auto bits = static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23;
  return p * std::bit_cast<float>(bits);
}

// Max over 8 independent lanes: vectorizes without fast-math.
float max_logit(std::span<const float> l) {
  float m[8];
  std::fill(std::begin(m), std::end(m), -INFINITY);
  size_t i = 0;
  for (; i + 8 <= l.size(); i += 8) {
    for (int j = 0; j < 8; ++j) m[j] = std::max(m[j], l[i + static_cast<size_t>(j)]);
  }
  float best = *std::max_element(std::begin(m), std::end(m));
  for (; i < l.size(); ++i) best = std::max(best, l[i]);
  return best;
}

TokenId sample_greedy(std::span<const float> logits) {
  if (logits.empty()) return 0;
  const float best = max_logit(logits);
  for (size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] == best) return static_cast<TokenId>(i);  // lowest id among ties
  }
  return 0;  // NaN-only rows
}

Status SamplingParams::validate() const {
  auto bad = [](const std::string& m) { return InvalidArgument("sampling: " + m); };
  if (!(temperature >= 0.0f) || temperature > 100.0f) return bad("temperature must be in [0, 100]");
  if (top_k < 0) return bad("top_k must be >= 0");
  if (!(top_p > 0.0f) || top_p > 1.0f) return bad("top_p must be in (0, 1]");
  if (!(min_p >= 0.0f) || min_p > 1.0f) return bad("min_p must be in [0, 1]");
  if (!(repetition_penalty > 0.0f)) return bad("repetition_penalty must be > 0");
  if (!(std::abs(frequency_penalty) <= 2.0f)) return bad("frequency_penalty must be in [-2, 2]");
  if (!(std::abs(presence_penalty) <= 2.0f)) return bad("presence_penalty must be in [-2, 2]");
  if (penalty_last_n < -1) return bad("penalty_last_n must be >= -1");
  return Status::Ok();
}

// --- Rng ----------------------------------------------------------------------

namespace {
uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Rng::Rng(uint64_t seed) {
  for (uint64_t& s : s_) s = splitmix64(seed);
}

uint64_t Rng::next() {
  const uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Rng::uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

// --- Sampler --------------------------------------------------------------------

namespace {
uint64_t initial_seed(const SamplingParams& p) {
  if (p.has_seed) return p.seed;
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}
}  // namespace

Sampler::Sampler(const SamplingParams& params) : params_(params), rng_(initial_seed(params)) {}

void Sampler::apply_penalties(std::span<float> logits, std::span<const TokenId> context) {
  if (!params_.has_penalties() || context.empty()) return;
  const size_t n = params_.penalty_last_n < 0 ? context.size()
                                              : std::min(context.size(), static_cast<size_t>(params_.penalty_last_n));
  history_.assign(context.end() - static_cast<std::ptrdiff_t>(n), context.end());
  std::sort(history_.begin(), history_.end());
  for (size_t i = 0; i < history_.size();) {
    size_t j = i;
    while (j < history_.size() && history_[j] == history_[i]) ++j;
    const TokenId id = history_[i];
    const auto count = static_cast<float>(j - i);
    i = j;
    if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
    float& l = logits[static_cast<size_t>(id)];
    if (params_.repetition_penalty != 1.0f) l = l > 0 ? l / params_.repetition_penalty : l * params_.repetition_penalty;
    l -= count * params_.frequency_penalty + params_.presence_penalty;
  }
}

TokenId Sampler::sample(std::span<float> logits, std::span<const TokenId> context) {
  apply_penalties(logits, context);
  if (params_.greedy()) return sample_greedy(logits);

  const size_t vocab = logits.size();
  const float max_l = max_logit(logits);
  const float t = params_.temperature, inv_t = 1.0f / t;
  const bool use_k = params_.top_k > 0 && static_cast<size_t>(params_.top_k) < vocab;

  if (!use_k && params_.top_p >= 1.0f && params_.min_p <= 0.0f) {
    // Temperature only: one pass for the probabilities, one walk to draw.
    probs_.resize(vocab);
    float* pr = probs_.data();
    const float* lg = logits.data();
    for (size_t i = 0; i < vocab; ++i) pr[i] = exp_nonpos((lg[i] - max_l) * inv_t);  // vectorizes
    double lanes[8] = {};
    size_t i = 0;
    for (; i + 8 <= vocab; i += 8) {
      for (int j = 0; j < 8; ++j) lanes[j] += pr[i + static_cast<size_t>(j)];
    }
    double total = 0;
    for (double v : lanes) total += v;
    for (; i < vocab; ++i) total += pr[i];
    double u = rng_.uniform() * total;
    for (size_t k = 0; k < vocab; ++k) {
      u -= probs_[k];
      if (u < 0) return static_cast<TokenId>(k);
    }
    return static_cast<TokenId>(vocab - 1);
  }

  // One pass: a histogram of counts and probability mass over the top 40*T
  // of logit range (exp(-40) is below float resolution of any realistic
  // nucleus), plus the min-p cut. It yields a conservative logit cutoff for
  // top-k and top-p, so the exact steps below only sort the survivors.
  constexpr int kBins = 256;
  const float range = 40.0f * t;
  const float bin_w = range / kBins;
  const float min_p_cut = params_.min_p > 0.0f ? max_l + t * std::log(params_.min_p) : -INFINITY;
  std::array<double, kBins> bin_mass{};
  std::array<int64_t, kBins> bin_count{};
  double total = 0;
  for (size_t i = 0; i < vocab; ++i) {
    const float l = logits[i];
    if (l < min_p_cut) continue;
    const float d = max_l - l;
    if (d >= range) continue;  // negligible mass; never in a nucleus or top-k worth sampling
    const double p = exp_nonpos(-d * inv_t);
    const int b = std::min(kBins - 1, static_cast<int>(d / bin_w));
    bin_mass[static_cast<size_t>(b)] += p;
    ++bin_count[static_cast<size_t>(b)];
    total += p;
  }
  float cutoff = std::max(min_p_cut, max_l - range);
  if (use_k || params_.top_p < 1.0f) {
    int64_t count = 0;
    double mass = 0;
    const double target = params_.top_p * total;
    for (int b = 0; b < kBins; ++b) {
      count += bin_count[static_cast<size_t>(b)];
      mass += bin_mass[static_cast<size_t>(b)];
      const bool k_done = !use_k || count >= params_.top_k;
      const bool p_done = params_.top_p >= 1.0f || use_k || mass >= target;  // with top-k, p applies after k
      if (k_done && p_done) {
        cutoff = std::max(cutoff, max_l - static_cast<float>(b + 1) * bin_w);
        break;
      }
    }
  }

  // Exact steps on the survivors: top-k, then nucleus, then draw.
  cand_.clear();
  for (size_t i = 0; i < vocab; ++i) {
    if (logits[i] >= cutoff) cand_.emplace_back(logits[i], static_cast<TokenId>(i));
  }
  if (cand_.empty()) return sample_greedy(logits);
  size_t n = cand_.size();
  const bool need_order = use_k || params_.top_p < 1.0f;
  if (need_order) {
    std::sort(cand_.begin(), cand_.end(),
              [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    if (use_k) n = std::min(n, static_cast<size_t>(params_.top_k));
  }
  probs_.resize(n);
  double kept_total = 0;
  for (size_t i = 0; i < n; ++i) kept_total += probs_[i] = exp_nonpos((cand_[i].first - max_l) * inv_t);
  if (params_.top_p < 1.0f) {
    // Nucleus over the distribution after top-k / min-p: without top-k that
    // is the full (min-p filtered) mass `total`, of which survivors hold the head.
    const double target = params_.top_p * (use_k ? kept_total : total);
    double mass = 0;
    size_t keep = 0;
    while (keep < n && mass < target) mass += probs_[keep++];
    n = std::max<size_t>(keep, 1);
    kept_total = mass;
  }

  double u = rng_.uniform() * kept_total;
  for (size_t i = 0; i < n; ++i) {
    u -= probs_[i];
    if (u < 0) return cand_[i].second;
  }
  return cand_[n - 1].second;  // rounding: the last kept candidate
}

}  // namespace engine
