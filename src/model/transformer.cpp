#include "model/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "common/timer.h"
#include "quant/dequant.h"

namespace engine {
namespace {

// 2-D fp32 view over raw memory with an explicit row stride (bytes).
TensorView view2d(void* data, int64_t rows, int64_t cols, int64_t row_stride_bytes) {
  TensorLayout l;
  l.dtype = DType::kF32;
  l.shape = TensorShape{rows, cols};
  l.strides = {row_stride_bytes, 4, 0, 0};
  return TensorView(data, l);
}

// First `rows` rows of a [max_batch, cols] scratch tensor.
TensorView rows_of(const Tensor& t, int64_t rows) {
  return view2d(t.data(), rows, t.shape()[1], t.view().stride(0));
}

// Columns [c0, c0 + n) of a 2-D view (strided rows, no copy).
TensorView cols_of(const TensorView& v, int64_t c0, int64_t n) {
  return view2d(static_cast<float*>(v.data()) + c0, v.dim(0), n, v.stride(0));
}

bool present(const TensorView& v) { return v.data() != nullptr; }

}  // namespace

Result<std::unique_ptr<Transformer>> Transformer::create(const ModelConfig& config, const TensorRegistry& weights,
                                                         Backend& backend, int32_t max_batch_tokens) {
  ENGINE_RETURN_IF_ERROR(config.validate());
  if (max_batch_tokens <= 0) return InvalidArgument("max_batch_tokens must be > 0");
  std::unique_ptr<Transformer> t(new Transformer(config, backend));
  ENGINE_RETURN_IF_ERROR(t->init(weights, max_batch_tokens));
  return t;
}

Result<TensorView> Transformer::f32_vector(const Tensor& t) {
  if (t.dtype() == DType::kF32) return t.view();
  ENGINE_ASSIGN_OR_RETURN(Tensor f, Tensor::empty(DType::kF32, {t.numel()}));
  if (!dequantize_row(t.dtype(), t.data(), f.data_as<float>(), t.numel())) {
    return Unsupported("cannot convert " + std::string(dtype_name(t.dtype())) + " vector to f32");
  }
  owned_.push_back(f);
  return f.view();
}

Status Transformer::init(const TensorRegistry& w, int32_t max_batch) {
  const ModelConfig& c = config_;
  max_batch_ = max_batch;

  auto weight = [&](TensorRole role, int layer, TensorView& out) -> Status {
    const Tensor* t = w.find(role, layer);
    if (!t) return Status::Ok();
    if (!backend_.supports_weight_type(t->dtype())) {
      return Unsupported("backend " + std::string(backend_.name()) + " cannot run " +
                         std::string(dtype_name(t->dtype())) + " weight " + std::string(tensor_role_name(role)));
    }
    out = t->view();
    return Status::Ok();
  };
  auto vec = [&](TensorRole role, int layer, TensorView& out) -> Status {
    const Tensor* t = w.find(role, layer);
    if (!t) return Status::Ok();
    ENGINE_ASSIGN_OR_RETURN(out, f32_vector(*t));
    return Status::Ok();
  };

  ENGINE_RETURN_IF_ERROR(weight(TensorRole::kTokenEmbedding, -1, tok_embd_));
  ENGINE_RETURN_IF_ERROR(vec(TensorRole::kOutputNorm, -1, output_norm_));
  ENGINE_RETURN_IF_ERROR(vec(TensorRole::kOutputNormBias, -1, output_norm_b_));
  ENGINE_RETURN_IF_ERROR(weight(TensorRole::kOutput, -1, lm_head_));
  if (!present(lm_head_)) lm_head_ = tok_embd_;  // tied embeddings
  if (const Tensor* f = w.find(TensorRole::kRopeFreqs)) {
    rope_freq_factors_.resize(static_cast<size_t>(f->numel()));
    if (!dequantize_row(f->dtype(), f->data(), rope_freq_factors_.data(), f->numel())) {
      return Unsupported("rope_freqs dtype");
    }
  }

  layers_.resize(static_cast<size_t>(c.num_layers));
  for (int l = 0; l < c.num_layers; ++l) {
    Layer& L = layers_[static_cast<size_t>(l)];
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnNorm, l, L.attn_norm));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnNormBias, l, L.attn_norm_b));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kAttnQ, l, L.wq));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kAttnK, l, L.wk));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kAttnV, l, L.wv));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kAttnQkv, l, L.wqkv));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kAttnOutput, l, L.wo));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnQBias, l, L.bq));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnKBias, l, L.bk));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnVBias, l, L.bv));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnQkvBias, l, L.bqkv));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnOutputBias, l, L.bo));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnQNorm, l, L.q_norm));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kAttnKNorm, l, L.k_norm));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kPostAttnNorm, l, L.post_attn_norm));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kFfnNorm, l, L.ffn_norm));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kFfnNormBias, l, L.ffn_norm_b));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnGate, l, L.w_gate));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnUp, l, L.w_up));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnGateUp, l, L.w_gate_up));
    ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnDown, l, L.w_down));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kFfnUpBias, l, L.b_up));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kFfnDownBias, l, L.b_down));
    ENGINE_RETURN_IF_ERROR(vec(TensorRole::kPostFfnNorm, l, L.post_ffn_norm));
    L.fused_qkv = present(L.wqkv);
    L.fused_gate_up = present(L.w_gate_up);

    if (c.moe.num_experts > 0) {
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnRouter, l, L.router));
      TensorView gate3, up3, down3;
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnGateExperts, l, gate3));
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnUpExperts, l, up3));
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnDownExperts, l, down3));
      if (!present(L.router) || !present(gate3) || !present(up3) || !present(down3)) {
        return InvalidArgument("MoE layer " + std::to_string(l) + " is missing router or expert weights");
      }
      L.experts.resize(static_cast<size_t>(c.moe.num_experts));
      for (int32_t e = 0; e < c.moe.num_experts; ++e) {
        ENGINE_ASSIGN_OR_RETURN(L.experts[static_cast<size_t>(e)][0], gate3.select(0, e));
        ENGINE_ASSIGN_OR_RETURN(L.experts[static_cast<size_t>(e)][1], up3.select(0, e));
        ENGINE_ASSIGN_OR_RETURN(L.experts[static_cast<size_t>(e)][2], down3.select(0, e));
      }
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnGateShared, l, L.sh_gate));
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnUpShared, l, L.sh_up));
      ENGINE_RETURN_IF_ERROR(weight(TensorRole::kFfnDownShared, l, L.sh_down));
      if (w.has(TensorRole::kFfnSharedRouter, l)) {
        TensorView v;
        ENGINE_RETURN_IF_ERROR(vec(TensorRole::kFfnSharedRouter, l, v));
        ENGINE_ASSIGN_OR_RETURN(L.sh_router, v.reshape({1, c.hidden_size}));  // matmul weight [1, hidden]
      }
    }
  }

  // Scratch. qkv holds Q|K|V side by side; ff_a/ff_b hold gate/up (or a
  // fused gate|up of width 2*ff).
  const int64_t q_dim = static_cast<int64_t>(c.num_heads) * c.head_dim;
  const int64_t k_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim;
  const int64_t v_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim_v;
  const int64_t o_dim = static_cast<int64_t>(c.num_heads) * c.head_dim_v;
  auto scratch = [&](int64_t cols, Tensor& t) -> Status {
    ENGINE_ASSIGN_OR_RETURN(t, Tensor::zeros(DType::kF32, {max_batch, cols}));
    return Status::Ok();
  };
  ENGINE_RETURN_IF_ERROR(scratch(c.hidden_size, x_));
  ENGINE_RETURN_IF_ERROR(scratch(c.hidden_size, xn_));
  ENGINE_RETURN_IF_ERROR(scratch(q_dim + k_dim + v_dim, qkv_));
  ENGINE_RETURN_IF_ERROR(scratch(o_dim, attn_));
  ENGINE_RETURN_IF_ERROR(scratch(c.hidden_size, o_));
  ENGINE_RETURN_IF_ERROR(scratch(std::max<int64_t>(1, 2 * c.intermediate_size), ff_a_));
  ENGINE_RETURN_IF_ERROR(scratch(std::max<int64_t>(1, c.intermediate_size), ff_b_));
  if (c.moe.num_experts > 0) {
    const int64_t fw = std::max(c.moe.expert_intermediate_size, c.moe.shared_intermediate_size);
    // Packed routed rows: up to max_batch * experts_per_token (>= max_batch for the shared expert).
    const int64_t packed = static_cast<int64_t>(max_batch) * std::max(1, c.moe.experts_per_token);
    auto packed_scratch = [&](int64_t cols, Tensor& t) -> Status {
      ENGINE_ASSIGN_OR_RETURN(t, Tensor::zeros(DType::kF32, {packed, cols}));
      return Status::Ok();
    };
    ENGINE_RETURN_IF_ERROR(scratch(c.moe.num_experts, router_));
    ENGINE_RETURN_IF_ERROR(packed_scratch(c.hidden_size, moe_x_));
    ENGINE_RETURN_IF_ERROR(packed_scratch(2 * fw, moe_a_));
    ENGINE_RETURN_IF_ERROR(packed_scratch(fw, moe_b_));
    ENGINE_RETURN_IF_ERROR(packed_scratch(c.hidden_size, moe_y_));
    up_jobs_.reserve(static_cast<size_t>(2 * c.moe.num_experts));
    down_jobs_.reserve(static_cast<size_t>(c.moe.num_experts));
    ENGINE_RETURN_IF_ERROR(scratch(1, sh_gate_));
    expert_rows_.resize(static_cast<size_t>(c.moe.num_experts));
    for (auto& v : expert_rows_) v.reserve(static_cast<size_t>(max_batch));
    route_scratch_.resize(static_cast<size_t>(c.moe.num_experts));
  }
  return Status::Ok();
}

// Routed mixture of experts (CPU-side routing; a device backend would add a
// fused route/gather/scatter op behind the same Layer data, DD-042).
void Transformer::moe_mlp(const Layer& L, const TensorView& xn, const TensorView& out, int64_t m) {
  const ModelConfig& c = config_;
  const int32_t ne = c.moe.num_experts, k = c.moe.experts_per_token;
  const int64_t d = c.hidden_size;
  const auto row = [](const TensorView& t, int64_t r) {
    return reinterpret_cast<float*>(static_cast<std::byte*>(t.data()) + r * t.stride(0));
  };

  // 1. Router logits -> softmax -> top-k (ties: lower expert id) -> weights.
  const TensorView logits = rows_of(router_, m);
  backend_.matmul(xn, L.router, nullptr, logits);
  for (auto& v : expert_rows_) v.clear();
  for (int64_t r = 0; r < m; ++r) {
    const float* lg = row(logits, r);
    float mx = lg[0];
    for (int32_t e = 1; e < ne; ++e) mx = std::max(mx, lg[e]);
    double sum = 0;
    for (int32_t e = 0; e < ne; ++e) {
      const float p = std::exp(lg[e] - mx);
      route_scratch_[static_cast<size_t>(e)] = {p, e};
      sum += p;
    }
    std::partial_sort(route_scratch_.begin(), route_scratch_.begin() + k, route_scratch_.end(),
                      [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    double kept = 0;
    for (int32_t j = 0; j < k; ++j) kept += route_scratch_[static_cast<size_t>(j)].first;
    const double denom = c.moe.normalize_topk ? kept : sum;
    for (int32_t j = 0; j < k; ++j) {
      const auto& [p, e] = route_scratch_[static_cast<size_t>(j)];
      expert_rows_[static_cast<size_t>(e)].emplace_back(static_cast<int32_t>(r), static_cast<float>(p / denom));
    }
  }
  mark(ForwardOp::kMoeRoute);

  // 2. Pack the routed rows by expert into one buffer ([sum of counts, d]),
  //    run every active expert's gate+up as one batch of matmuls, one
  //    activation over all packed rows, every down projection as one batch,
  //    then scatter-add with the routing weights. Each active expert reads
  //    its weights once per step however many rows it serves.
  for (int64_t r = 0; r < m; ++r) std::memset(row(out, r), 0, static_cast<size_t>(d) * sizeof(float));
  const int64_t fe = c.moe.expert_intermediate_size;
  int64_t total = 0;
  for (const auto& rows : expert_rows_) total += static_cast<int64_t>(rows.size());
  const TensorView xp = rows_of(moe_x_, total), ap = rows_of(moe_a_, total), bp = rows_of(moe_b_, total);
  const TensorView yp = rows_of(moe_y_, total);
  auto slice = [](const TensorView& v, int64_t r0, int64_t n) {
    return view2d(reinterpret_cast<std::byte*>(v.data()) + r0 * v.stride(0), n, v.dim(1), v.stride(0));
  };
  up_jobs_.clear();
  down_jobs_.clear();
  int64_t off = 0;
  for (int32_t e = 0; e < ne; ++e) {
    const auto& rows = expert_rows_[static_cast<size_t>(e)];
    if (rows.empty()) continue;
    const auto cnt = static_cast<int64_t>(rows.size());
    for (int64_t i = 0; i < cnt; ++i) {
      std::memcpy(row(xp, off + i), row(xn, rows[static_cast<size_t>(i)].first), static_cast<size_t>(d) * sizeof(float));
    }
    const auto& W = L.experts[static_cast<size_t>(e)];
    const TensorView xe = slice(xp, off, cnt), ae = slice(ap, off, cnt), be = slice(bp, off, cnt);
    up_jobs_.push_back({xe, W[0], cols_of(ae, 0, fe)});
    up_jobs_.push_back({xe, W[1], cols_of(ae, fe, fe)});
    down_jobs_.push_back({cols_of(be, 0, fe), W[2], slice(yp, off, cnt)});
    off += cnt;
  }
  mark(ForwardOp::kMoeScatter);
  backend_.matmul_many(up_jobs_);
  backend_.act_mul(c.activation, cols_of(ap, 0, fe), cols_of(ap, fe, fe), cols_of(bp, 0, fe));
  backend_.matmul_many(down_jobs_);
  mark(ForwardOp::kMoeExperts);
  off = 0;
  for (int32_t e = 0; e < ne; ++e) {
    for (const auto& [r, wgt] : expert_rows_[static_cast<size_t>(e)]) {
      float* o = row(out, r);
      const float* y = row(yp, off++);
      for (int64_t j = 0; j < d; ++j) o[j] += wgt * y[j];
    }
  }
  mark(ForwardOp::kMoeScatter);

  // 3. Shared expert (all rows), optionally scaled by sigmoid(x . w_router).
  if (present(L.sh_up)) {
    const int64_t fs = c.moe.shared_intermediate_size;
    const TensorView a = rows_of(moe_a_, m), b = rows_of(moe_b_, m), ys = rows_of(moe_y_, m);
    const TensorView gate = cols_of(a, 0, fs), up = cols_of(a, fs, fs), act = cols_of(b, 0, fs);
    backend_.matmul(xn, L.sh_gate, nullptr, gate);
    backend_.matmul(xn, L.sh_up, nullptr, up);
    backend_.act_mul(c.activation, gate, up, act);
    backend_.matmul(act, L.sh_down, nullptr, ys);
    const TensorView g = rows_of(sh_gate_, m);
    if (present(L.sh_router)) backend_.matmul(xn, L.sh_router, nullptr, g);
    for (int64_t r = 0; r < m; ++r) {
      const float s = present(L.sh_router) ? 1.0f / (1.0f + std::exp(-row(g, r)[0])) : 1.0f;
      float* o = row(out, r);
      const float* y = row(ys, r);
      for (int64_t j = 0; j < d; ++j) o[j] += s * y[j];
    }
  }
}

void Transformer::norm(const TensorView& x, const TensorView& w, const TensorView& b, const TensorView& y) {
  if (config_.norm == NormType::kRmsNorm) {
    backend_.rms_norm(x, w, config_.norm_eps, y);
  } else {
    backend_.layer_norm(x, w, present(b) ? &b : nullptr, config_.norm_eps, y);
  }
}

std::string_view forward_op_name(ForwardOp op) {
  static constexpr std::string_view kNames[] = {"embed",   "norm",     "qkv",     "rope+qknorm", "kv_store", "attention",
                                                "attn_out", "mlp_up",   "act",     "mlp_down",    "moe_route", "moe_experts",
                                                "moe_gather_scatter", "lm_head"};
  static_assert(std::size(kNames) == static_cast<size_t>(ForwardOp::kCount));
  return kNames[static_cast<size_t>(op)];
}

void Transformer::mark(ForwardOp op) {
  if (!profiling_) return;
  const int64_t now = now_ns();
  profile_.ns[static_cast<size_t>(op)] += now - mark_ns_;
  mark_ns_ = now;
}

Status Transformer::forward(std::span<const TokenId> tokens, std::span<const int32_t> positions, KvBlockPool& cache,
                            std::span<const int32_t> block_table, std::span<float> logits) {
  if (tokens.empty() || positions.size() != tokens.size()) return InvalidArgument("forward: empty or mismatched batch");
  for (size_t i = 1; i < positions.size(); ++i) {
    if (positions[i] != positions[0] + static_cast<int32_t>(i)) {
      return InvalidArgument("forward: positions must be consecutive");
    }
  }
  const SeqBatch one{tokens, positions[0], block_table, true};
  return forward_batch({&one, 1}, cache, logits);
}

Status Transformer::forward_batch(std::span<const SeqBatch> seqs, KvBlockPool& cache, std::span<float> logits) {
  const ModelConfig& c = config_;
  // --- validate and flatten the batch ---
  batch_tokens_.clear();
  batch_pos_.clear();
  batch_seq_.clear();
  logit_rows_.clear();
  const int32_t bs = cache.geometry().block_size;
  for (size_t si = 0; si < seqs.size(); ++si) {
    const SeqBatch& sb = seqs[si];
    if (sb.tokens.empty()) return InvalidArgument("forward: sequence with no tokens");
    if (batch_tokens_.size() + sb.tokens.size() > static_cast<size_t>(max_batch_)) {
      return InvalidArgument("forward: batch exceeds max_batch_tokens (" + std::to_string(max_batch_) + ")");
    }
    const int64_t last = static_cast<int64_t>(sb.start_pos) + static_cast<int64_t>(sb.tokens.size()) - 1;
    if (sb.start_pos < 0 || last / bs >= static_cast<int64_t>(sb.block_table.size())) {
      return InvalidArgument("forward: position " + std::to_string(last) + " has no KV block");
    }
    for (size_t i = 0; i < sb.tokens.size(); ++i) {
      const TokenId t = sb.tokens[i];
      if (t < 0 || t >= c.vocab_size) return InvalidArgument("forward: token id " + std::to_string(t) + " out of range");
      batch_tokens_.push_back(t);
      batch_pos_.push_back(sb.start_pos + static_cast<int32_t>(i));
      batch_seq_.push_back(static_cast<int32_t>(si));
    }
    if (sb.want_logits) logit_rows_.push_back(static_cast<int32_t>(batch_tokens_.size()) - 1);
  }
  if (batch_tokens_.empty()) return InvalidArgument("forward: empty batch");
  if (logits.size() != logit_rows_.size() * static_cast<size_t>(c.vocab_size)) {
    return InvalidArgument("forward: logits buffer must hold " + std::to_string(logit_rows_.size()) + " rows");
  }
  const auto m = static_cast<int64_t>(batch_tokens_.size());
  const std::span<const TokenId> tokens = batch_tokens_;
  const std::span<const int32_t> positions = batch_pos_;
  const std::span<const int32_t> row_seq = batch_seq_;

  const int64_t q_dim = static_cast<int64_t>(c.num_heads) * c.head_dim;
  const int64_t k_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim;
  const int64_t v_dim = static_cast<int64_t>(c.num_kv_heads) * c.head_dim_v;
  const int64_t ff = c.intermediate_size;
  const float attn_scale = c.attn_scale > 0 ? c.attn_scale : 1.0f / std::sqrt(static_cast<float>(c.head_dim));
  const float* freq_factors = rope_freq_factors_.empty() ? nullptr : rope_freq_factors_.data();

  const TensorView x = rows_of(x_, m), xn = rows_of(xn_, m), qkv = rows_of(qkv_, m);
  const TensorView attn = rows_of(attn_, m), o = rows_of(o_, m);
  const TensorView ff_a = rows_of(ff_a_, m), ff_b = rows_of(ff_b_, m);
  const TensorView q = cols_of(qkv, 0, q_dim), k = cols_of(qkv, q_dim, k_dim), v = cols_of(qkv, q_dim + k_dim, v_dim);

  if (profiling_) {
    mark_ns_ = now_ns();
    ++profile_.calls;
    profile_.rows += static_cast<uint64_t>(m);
  }
  backend_.embedding(tok_embd_, tokens, x);
  if (c.embedding_scale != 1.0f) backend_.scale(x, c.embedding_scale);
  mark(ForwardOp::kEmbed);

  for (int l = 0; l < c.num_layers; ++l) {
    const Layer& L = layers_[static_cast<size_t>(l)];

    // --- attention ---
    norm(x, L.attn_norm, L.attn_norm_b, xn);
    mark(ForwardOp::kNorm);
    if (L.fused_qkv) {
      backend_.matmul(xn, L.wqkv, present(L.bqkv) ? &L.bqkv : nullptr, qkv);
    } else {
      backend_.matmul(xn, L.wq, present(L.bq) ? &L.bq : nullptr, q);
      backend_.matmul(xn, L.wk, present(L.bk) ? &L.bk : nullptr, k);
      backend_.matmul(xn, L.wv, present(L.bv) ? &L.bv : nullptr, v);
    }
    mark(ForwardOp::kQkv);
    if (c.attn_qk_norm) {
      // Per-head RMSNorm: view each row's heads as [heads, head_dim].
      for (int64_t r = 0; r < m; ++r) {
        auto* qr = reinterpret_cast<float*>(static_cast<std::byte*>(q.data()) + r * q.stride(0));
        auto* kr = reinterpret_cast<float*>(static_cast<std::byte*>(k.data()) + r * k.stride(0));
        const TensorView qh = view2d(qr, c.num_heads, c.head_dim, c.head_dim * 4);
        const TensorView kh = view2d(kr, c.num_kv_heads, c.head_dim, c.head_dim * 4);
        backend_.rms_norm(qh, L.q_norm, c.norm_eps, qh);
        backend_.rms_norm(kh, L.k_norm, c.norm_eps, kh);
      }
    }
    const RopeConfig& rope = c.layer_rope(l);
    backend_.rope(q, c.num_heads, c.head_dim, positions, rope, freq_factors);
    backend_.rope(k, c.num_kv_heads, c.head_dim, positions, rope, freq_factors);
    mark(ForwardOp::kRope);

    kv_views_.clear();
    for (const SeqBatch& sb : seqs) kv_views_.push_back(cache.layer_view(l, sb.block_table));
    backend_.kv_store(k, v, positions, row_seq, kv_views_);
    mark(ForwardOp::kKvStore);

    AttentionParams ap;
    ap.q = q;
    ap.out = attn;
    ap.positions = positions;
    ap.row_seq = row_seq;
    ap.kv = kv_views_;
    ap.num_heads = c.num_heads;
    ap.scale = attn_scale;
    ap.softcap = c.attn_logit_softcap;
    ap.sliding_window = c.layer_uses_sliding_window(l) ? c.sliding_window : 0;
    backend_.attention(ap);
    mark(ForwardOp::kAttention);

    backend_.matmul(attn, L.wo, present(L.bo) ? &L.bo : nullptr, o);
    if (present(L.post_attn_norm)) backend_.rms_norm(o, L.post_attn_norm, c.norm_eps, o);
    if (c.residual_scale != 1.0f) backend_.scale(o, c.residual_scale);
    backend_.add(x, o, x);
    mark(ForwardOp::kAttnOut);

    // --- MLP ---
    norm(x, L.ffn_norm, L.ffn_norm_b, xn);
    mark(ForwardOp::kNorm);
    if (!L.experts.empty()) {
      moe_mlp(L, xn, o, m);
      if (present(L.post_ffn_norm)) backend_.rms_norm(o, L.post_ffn_norm, c.norm_eps, o);
      if (c.residual_scale != 1.0f) backend_.scale(o, c.residual_scale);
      backend_.add(x, o, x);
      mark(ForwardOp::kMlpDown);
      continue;
    }
    if (c.mlp == MlpType::kGated) {
      TensorView gate, up;
      if (L.fused_gate_up) {
        backend_.matmul(xn, L.w_gate_up, nullptr, ff_a);
        gate = cols_of(ff_a, 0, ff);
        up = cols_of(ff_a, ff, ff);
      } else {
        gate = cols_of(ff_a, 0, ff);
        up = ff_b;
        backend_.matmul(xn, L.w_gate, nullptr, gate);
        backend_.matmul(xn, L.w_up, present(L.b_up) ? &L.b_up : nullptr, up);
      }
      mark(ForwardOp::kMlpUp);
      backend_.act_mul(c.activation, gate, up, ff_b);
      mark(ForwardOp::kAct);
    } else {
      backend_.matmul(xn, L.w_up, present(L.b_up) ? &L.b_up : nullptr, ff_b);
      mark(ForwardOp::kMlpUp);
      backend_.activation(c.activation, ff_b, ff_b);
      mark(ForwardOp::kAct);
    }
    backend_.matmul(ff_b, L.w_down, present(L.b_down) ? &L.b_down : nullptr, o);
    if (present(L.post_ffn_norm)) backend_.rms_norm(o, L.post_ffn_norm, c.norm_eps, o);
    if (c.residual_scale != 1.0f) backend_.scale(o, c.residual_scale);
    backend_.add(x, o, x);
    mark(ForwardOp::kMlpDown);
  }

  // Logits only for requested rows: gather them into contiguous scratch,
  // then final norm + LM head on just those rows.
  if (!logit_rows_.empty()) {
    const auto n_out = static_cast<int64_t>(logit_rows_.size());
    const TensorView gathered = rows_of(o_, n_out);  // o_ is free after the last layer
    for (int64_t i = 0; i < n_out; ++i) {
      std::memcpy(static_cast<std::byte*>(gathered.data()) + i * gathered.stride(0),
                  static_cast<const std::byte*>(x.data()) + logit_rows_[static_cast<size_t>(i)] * x.stride(0),
                  static_cast<size_t>(c.hidden_size) * sizeof(float));
    }
    const TensorView normed = rows_of(xn_, n_out);
    norm(gathered, output_norm_, output_norm_b_, normed);
    const TensorView out = view2d(logits.data(), n_out, c.vocab_size, c.vocab_size * 4);
    backend_.matmul(normed, lm_head_, nullptr, out);
    if (c.logit_scale != 1.0f) backend_.scale(out, 1.0f / c.logit_scale);
    if (c.final_logit_softcap > 0) backend_.softcap(out, c.final_logit_softcap);
    mark(ForwardOp::kLmHead);
  }
  return Status::Ok();
}

}  // namespace engine
