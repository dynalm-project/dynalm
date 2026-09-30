#pragma once

// SequenceState: everything the runtime tracks for one generation stream.
//
// Model-agnostic: holds token ids, positions and a KV block table, never
// model weights or file-format details. The scheduler (later phases) owns many
// of these; today the single-sequence Generator drives one.
//
// Lifecycle:  kWaiting → kPrefill → kDecode → kFinished
//                   ↘         ↘         ↘→ kCancelled / kError
// KV blocks are released when the sequence reaches a terminal state.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "common/status.h"
#include "kv_cache/kv_cache.h"
#include "model_ir/model_config.h"
#include "tokenizer/tokenizer.h"

namespace engine {

enum class SequenceStatus : uint8_t { kWaiting, kPrefill, kDecode, kFinished, kCancelled, kError };

std::string_view sequence_status_name(SequenceStatus s);
inline bool is_terminal(SequenceStatus s) {
  return s == SequenceStatus::kFinished || s == SequenceStatus::kCancelled || s == SequenceStatus::kError;
}

enum class FinishReason : uint8_t { kNone, kEog, kLength, kStopped };

struct StopParams {
  int32_t max_new_tokens = 128;
  bool stop_at_eog = true;
};

class SequenceState {
 public:
  SequenceState(uint64_t id, std::span<const TokenId> prompt, const StopParams& stop, KvBlockPool& cache);

  uint64_t id() const { return id_; }
  SequenceStatus status() const { return status_; }
  FinishReason finish_reason() const { return finish_; }
  const Status& error() const { return error_; }

  // All tokens: prompt followed by generated tokens.
  std::span<const TokenId> tokens() const { return tokens_; }
  int32_t prompt_len() const { return prompt_len_; }
  int32_t generated() const { return static_cast<int32_t>(tokens_.size()) - prompt_len_; }
  // Tokens whose K/V are in the cache; the next token to compute is at this position.
  int32_t num_computed() const { return num_computed_; }
  int32_t pending() const { return static_cast<int32_t>(tokens_.size()) - num_computed_; }

  std::span<const int32_t> block_table() const { return kv_.block_table(); }
  const StopParams& stop() const { return stop_; }

  // --- transitions (return an error on an illegal transition) ---
  // Ensures KV capacity for the next `n` tokens, then marks them computed.
  // Shared (copy-on-write) blocks in that range are copied first.
  Status reserve_kv(int32_t n) {
    ENGINE_RETURN_IF_ERROR(kv_.reserve(num_computed_ + n));
    return kv_.make_writable(num_computed_, num_computed_ + n);
  }
  void mark_computed(int32_t n);
  // Appends a sampled token; finishes on EOG or the length limit.
  void append_token(TokenId t, const Tokenizer& tokenizer);
  void cancel();
  void fail(Status error);

  // Bytes of KV memory currently held.
  int64_t kv_bytes() const;

 private:
  void finish(FinishReason r);

  uint64_t id_;
  std::vector<TokenId> tokens_;
  int32_t prompt_len_;
  int32_t num_computed_ = 0;
  StopParams stop_;
  SequenceStatus status_ = SequenceStatus::kWaiting;
  FinishReason finish_ = FinishReason::kNone;
  Status error_;
  KvBlockPool* cache_;
  KvBlockTable kv_;
};

// KV geometry for a model under a memory budget (bytes) or a token capacity.
KvGeometry kv_geometry_for(const ModelConfig& config, DType dtype, int32_t block_size, int64_t tokens);
int64_t kv_tokens_for_budget(const ModelConfig& config, DType dtype, int64_t budget_bytes);

}  // namespace engine
