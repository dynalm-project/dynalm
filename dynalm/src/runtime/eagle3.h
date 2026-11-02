#pragma once

// EAGLE-3 draft heads (Li et al., 2025) for speculative decoding
// (dynalm serve --spec-eagle DIR): a single decoder layer trained on the
// target model's own features, published per target model (for example
// AngelSlim/Qwen3-4B_eagle3, AngelSlim/Qwen3-a3B_eagle3).
//
// The head reads the residual stream entering three target layers
// (2, L/2 and L-3, L = the target's layer count) at each position t, plus the
// target's embedding of token t+1, and predicts token t+2:
//   h   = fc([f_low | f_mid | f_high])               (3H -> H; first step only)
//   a   = attn([norm_in(embed(token)) | norm_h(h)])  (2H -> heads; o_proj -> H)
//   h'  = h + a;  h'' = h' + mlp(norm_post(h'))
//   d   = argmax(lm_head(norm(h'')))                 (a 32000-token draft vocabulary)
//   token = d + d2t[d]
// Each further draft feeds (h'', the drafted token) back in without fc.
//
// Drafts are only proposals: the target verifies them (Scheduler), so the
// output is exactly the target's whatever the head predicts. The head has
// its own KV cache. It is indexed from the first position it sees, so a
// stream whose prefix came from the prefix cache (no target features for
// those positions) still drafts; RoPE depends only on relative positions.
//
// Not thread-safe: the scheduler thread only.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/device/device.h"
#include "kv_cache/kv_cache.h"
#include "model/transformer.h"
#include "common/core.h"

namespace dynalm {

class Eagle3Drafter {
 public:
  // `path`: a directory with config.json and model.safetensors (or that
  // .safetensors file, with config.json next to it). The head must match
  // the target's hidden size and vocabulary. `kv_tokens`: the head's KV
  // capacity over all streams; `max_rows`: rows per pass.
  static Result<std::unique_ptr<Eagle3Drafter>> create(const std::string& path, const Transformer& target,
                                                       Device& device, int64_t kv_tokens, int32_t max_rows);

  // The target layers whose input the head reads: give them to
  // Transformer::set_capture_layers.
  const std::vector<int32_t>& capture_layers() const { return capture_; }

  // Positions the target computed and kept: rows [start_pos, start_pos + n)
  // with their captured features (n x 3H) and, for each, the token at the
  // next position. After this, each stream has its first draft ready.
  struct Extend {
    uint64_t id = 0;
    int32_t start_pos = 0;
    std::span<const float> features;
    std::span<const TokenId> next_tokens;
  };
  Status extend(std::span<const Extend> rows);

  // Up to k drafts per stream (the first from extend()), batched across
  // streams: k - 1 passes. A stream with nothing ready gets none.
  struct Request {
    uint64_t id = 0;
    int32_t k = 0;
    std::vector<TokenId>* out = nullptr;
  };
  Status draft(std::span<const Request> requests);

  void release(uint64_t id);  // the stream finished
  int64_t passes() const { return passes_; }

 private:
  struct Stream {
    explicit Stream(KvBlockPool& pool) : kv(pool) {}
    KvBlockTable kv;
    int32_t base = -1;  // target position of draft index 0
    int32_t len = 0;    // committed draft rows (positions base .. base + len - 1)
    bool ready = false;
    TokenId next = 0;            // the first draft
    std::vector<float> hidden;   // h'' of the row that produced `next`
  };
  struct Row {
    int32_t stream;  // index into pass_streams_
    int32_t index;   // draft KV index (position)
    TokenId token;
  };

  Eagle3Drafter(Device& d) : device_(d) {}
  Stream& stream(uint64_t id);
  // One pass: rows grouped by stream (pass_streams_), inputs in feat_/hid_.
  // `fc`: rows carry 3H features (else H hidden). Logits for `logit_rows`.
  Status run(std::span<const Row> rows, bool fc, std::span<const int32_t> logit_rows);
  // After a pass of extend(): the first draft and h'' of each logit row's stream.
  Status finish(std::span<const std::pair<Stream*, size_t>> owners, size_t n);
  Result<Tensor> scratch(int64_t rows, int64_t cols);
  Result<TensorView> weight(const Tensor& host);

  Device& device_;
  int64_t H_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, ff_ = 0, vocab_ = 0, draft_vocab_ = 0;
  float eps_ = 1e-6f;
  RopeConfig rope_;
  int32_t max_rows_ = 0;
  std::vector<int32_t> capture_;
  TensorView embed_;  // the target's
  TensorView fc_, wq_, wk_, wv_, wo_, w_gate_, w_up_, w_down_, lm_head_;
  TensorView norm_h_, norm_in_, norm_post_, norm_;
  std::vector<int64_t> d2t_;
  std::vector<Tensor> owned_;
  std::unique_ptr<KvBlockPool> pool_;
  std::unordered_map<uint64_t, std::unique_ptr<Stream>> streams_;

  // Scratch [max_rows, ...]
  Tensor f_, h_, e_, cat_, q_, k_, v_, a_, o_, n2_, g_, u_, act_, d_, hn_, lg_;
  std::vector<int32_t> ids_, pos_, row_seq_;
  std::vector<Stream*> pass_streams_;
  std::vector<KvLayerView> kv_views_;
  std::vector<float> host_;
  int64_t passes_ = 0;
};

}  // namespace dynalm
