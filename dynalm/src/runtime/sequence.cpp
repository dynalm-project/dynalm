#include "runtime/sequence.h"

#include <cassert>

namespace engine {

std::string_view sequence_status_name(SequenceStatus s) {
  switch (s) {
    case SequenceStatus::kWaiting: return "waiting";
    case SequenceStatus::kPrefill: return "prefill";
    case SequenceStatus::kDecode: return "decode";
    case SequenceStatus::kFinished: return "finished";
    case SequenceStatus::kCancelled: return "cancelled";
    case SequenceStatus::kError: return "error";
  }
  return "?";
}

SequenceState::SequenceState(uint64_t id, std::span<const TokenId> prompt, const StopParams& stop, KvBlockPool& cache)
    : id_(id), tokens_(prompt.begin(), prompt.end()), prompt_len_(static_cast<int32_t>(prompt.size())),
      stop_(stop), cache_(&cache), kv_(cache) {}

void SequenceState::mark_computed(int32_t n) {
  assert(!is_terminal(status_) && n > 0 && n <= pending());
  num_computed_ += n;
  // Prefill ends when every prompt token has K/V; afterwards each step
  // computes exactly the one newly sampled token.
  // (After a preemption, generated tokens are recomputed too; the sequence is
  // back in decode once the whole prompt is covered again.)
  status_ = num_computed_ < prompt_len_ ? SequenceStatus::kPrefill : SequenceStatus::kDecode;
}

void SequenceState::append_token(TokenId t, const Tokenizer& tokenizer) {
  assert(status_ == SequenceStatus::kDecode && pending() == 0);
  tokens_.push_back(t);
  if (stop_.stop_at_eog && tokenizer.is_eog(t)) {
    finish(FinishReason::kEog);
  } else if (generated() >= stop_.max_new_tokens) {
    finish(FinishReason::kLength);
  }
}

void SequenceState::finish(FinishReason r) {
  finish_ = r;
  status_ = SequenceStatus::kFinished;
  kv_.release();
}

void SequenceState::cancel() {
  if (is_terminal(status_)) return;
  finish_ = FinishReason::kStopped;
  status_ = SequenceStatus::kCancelled;
  kv_.release();
}

void SequenceState::reset_for_recompute() {
  assert(!is_terminal(status_));
  kv_.release();
  num_computed_ = 0;
  status_ = SequenceStatus::kWaiting;
  ++preemptions_;
}

void SequenceState::adopt_prefix(std::span<const int32_t> blocks, int32_t tokens) {
  assert(num_computed_ == 0 && kv_.num_blocks() == 0 && tokens < static_cast<int32_t>(tokens_.size()));
  for (int32_t b : blocks) kv_.append_shared(b);
  num_computed_ = tokens;
  if (tokens > 0) status_ = tokens < prompt_len_ ? SequenceStatus::kPrefill : SequenceStatus::kDecode;
}

void SequenceState::fail(Status error) {
  error_ = std::move(error);
  status_ = SequenceStatus::kError;
  kv_.release();
}

int64_t SequenceState::kv_bytes() const {
  const KvGeometry& g = cache_->geometry();
  return static_cast<int64_t>(kv_.block_table().size()) * g.bytes_per_layer() / g.num_blocks * g.num_layers;
}

KvGeometry kv_geometry_for(const ModelConfig& c, DType dtype, int32_t block_size, int64_t tokens) {
  KvGeometry g;
  g.num_layers = c.num_layers;
  g.num_kv_heads = c.num_kv_heads;
  g.head_dim = c.head_dim;
  g.head_dim_v = c.head_dim_v;
  g.block_size = block_size;
  g.num_blocks = static_cast<int32_t>((tokens + block_size - 1) / block_size);
  g.dtype = dtype;
  return g;
}

int64_t kv_tokens_for_budget(const ModelConfig& c, DType dtype, int64_t budget_bytes) {
  const int64_t per_token = c.kv_bytes_per_token(dtype);
  return per_token > 0 ? budget_bytes / per_token : 0;
}

}  // namespace engine
