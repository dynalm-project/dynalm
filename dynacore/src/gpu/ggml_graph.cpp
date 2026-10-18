// Lowering a recorded segment to one ggml graph (GraphBuilder) and preparing it
// (GgmlDevice::build). Split from ggml_device.cpp; see ggml_device_internal.h.

#include "ggml_device_internal.h"

#if DYNACORE_HAS_GGML

namespace dynacore::ggml_detail {

// Builds one ggml graph for a recorded segment. Values flow through ggml
// nodes; each DynaLM view written in the segment is copied back to its buffer
// at the end (so memory reads in later segments see it), and any read of
// memory written through a different, non-derivable view is refused.
class GraphBuilder {
 public:
  GraphBuilder(GgmlDevice& d, Prepared& g, std::span<const DeviceCall* const> calls,
               const std::vector<std::vector<SeqRows>>& seqs)
      : d_(d), g_(g), calls_(calls), seqs_(seqs) {}

  Status build() {
    const size_t n_calls = calls_.size();
    const size_t max_nodes = std::max<size_t>(1024, n_calls * 48);
    g_.ctx = meta_ctx(max_nodes * 2, max_nodes);
    g_.ctx_in = meta_ctx(n_calls * 4 + 16);
    ctx_ = g_.ctx.get();
    g_.graph = ggml_new_graph_custom(ctx_, max_nodes, false);
    for (size_t i = 0; i < n_calls; ++i) {
      ci_ = static_cast<int32_t>(i);
      ENGINE_RETURN_IF_ERROR(lower(*calls_[i]));
    }
    // Write back every view written in this segment, in write order.
    for (const auto& [k, t] : live_) ENGINE_RETURN_IF_ERROR(materialize(k, t));
    for (ggml_tensor* t : pools_written_) ggml_build_forward_expand(g_.graph, t);
    // Every node must be supported by the backend: no silent CPU fallback.
    for (int i = 0; i < ggml_graph_n_nodes(g_.graph); ++i) {
      ggml_tensor* n = ggml_graph_node(g_.graph, i);
      if (!ggml_backend_supports_op(d_.backend_, n)) {
        return Unsupported(d_.name_ + " does not support op " + std::string(ggml_op_desc(n)) + " (" +
                           ggml_type_name(n->type) + ") at " + std::to_string(n->ne[0]) + "x" +
                           std::to_string(n->ne[1]) + "x" + std::to_string(n->ne[2]));
      }
    }
    return Status::Ok();
  }

 private:
  using M = DeviceCall::M;

  // --- values -------------------------------------------------------------------
  // A leaf bound to runtime memory at `p` with the given type and shape.
  Result<ggml_tensor*> bind(const std::byte* p, ggml_type type, int64_t ne0, int64_t ne1, size_t nb1) {
    const auto key = std::make_tuple(p, static_cast<int>(type), ne0, ne1, nb1);
    if (auto it = leaves_.find(key); it != leaves_.end()) return it->second;
    ggml_tensor* t = ggml_new_tensor_2d(ctx_, type, ne0, ne1);
    t->nb[1] = nb1;
    t->nb[2] = t->nb[3] = nb1 * static_cast<size_t>(ne1);
    const size_t bytes = ggml_nbytes(t);
    const BufferInfo* b = d_.find(p, bytes);
    if (b == nullptr) return InvalidArgument(d_.name_ + ": tensor memory is not owned by this device");
    if (ggml_backend_tensor_alloc(b->buf, t, b->real_at(p)) != GGML_STATUS_SUCCESS) {
      return Internal(d_.name_ + ": cannot bind tensor memory");
    }
    leaves_.emplace(key, t);
    return t;
  }

  // Weight or constant vector: any supported dtype, rank 1 or 2, dense rows.
  // ggml-cpu: a matmul weight's repacked copy (GgmlDevice::prepack_weight),
  // else the weight itself. Only for the src0 of a matmul: other ops (an
  // embedding lookup of a tied table) must read the original layout.
  Result<ggml_tensor*> matmul_weight(const TensorView& w) {
    if (w.rank() == 2) {
      if (ggml_tensor* r = d_.repacked(static_cast<const std::byte*>(w.data()), w.dim(1), w.dim(0))) return r;
    }
    return weight(w);
  }

  Result<ggml_tensor*> weight(const TensorView& w) {
    ggml_type type;
    if (!to_ggml_type(w.dtype(), type)) return Unsupported("weight type " + std::string(dtype_name(w.dtype())));
    if (w.rank() > 2) return Unsupported("weights of rank > 2 (MoE expert tensors)");
    const int64_t ne0 = cols_of(w), ne1 = rows_of(w);
    const size_t row = ggml_row_size(type, ne0);
    const size_t nb1 = w.rank() > 1 ? static_cast<size_t>(w.stride(0)) : row;
    if (nb1 < row) return Unsupported("weight rows overlap");
    return bind(static_cast<const std::byte*>(w.data()), type, ne0, ne1, nb1);
  }

  // The current value of an fp32 activation view.
  Result<ggml_tensor*> read(const TensorView& v) {
    if (v.dtype() != DType::kF32) return Unsupported("activation dtype " + std::string(dtype_name(v.dtype())));
    const ViewKey k = key_of(v);
    for (const auto& [lk, t] : live_) {
      if (lk == k) return t;
    }
    // The same bytes reshaped (per-head norms read Q as [rows x heads, head_dim]).
    for (const auto& [lk, t] : live_) {
      if (lk.same_range(k)) return ggml_reshape_2d(ctx_, ggml_is_contiguous(t) ? t : ggml_cont(ctx_, t), k.cols, k.rows);
    }
    // A sub-view (rows and/or columns) of a value written in this segment.
    for (const auto& [lk, t] : live_) {
      if (lk.stride != k.stride || k.p < lk.p || !lk.overlaps(k)) continue;
      const int64_t off = k.p - lk.p;
      const int64_t r0 = off / lk.stride, c0 = (off % lk.stride) / 4;
      if ((off % lk.stride) % 4 == 0 && r0 + k.rows <= lk.rows && c0 + k.cols <= lk.cols) {
        return ggml_view_2d(ctx_, t, k.cols, k.rows, t->nb[1], static_cast<size_t>(r0) * t->nb[1] + c0 * 4);
      }
    }
    for (const ViewKey& w : written_) {
      if (w.overlaps(k)) {
        return Unsupported("a step reads memory written through a different view (aliasing): call " +
                           std::to_string(ci_) + " (" + call_name(calls_[static_cast<size_t>(ci_)]->m) + ") reads " +
                           describe(k) + ", written as " + describe(w));
      }
    }
    return bind(k.p, GGML_TYPE_F32, k.cols, k.rows, static_cast<size_t>(k.stride));
  }

  // `t` ([cols, rows] or reshapeable to it) becomes the value of view `v`.
  Status write(const TensorView& v, ggml_tensor* t) {
    const ViewKey k = key_of(v);
    if (ggml_nelements(t) != k.rows * k.cols) return Internal("value shape does not match its view");
    if (t->ne[0] != k.cols || t->ne[1] != k.rows || t->ne[2] != 1 || t->ne[3] != 1) {
      if (!ggml_is_contiguous(t)) t = ggml_cont(ctx_, t);
      t = ggml_reshape_2d(ctx_, t, k.cols, k.rows);
    }
    // Overlapping values written earlier: write them back first, then forget
    // them (a later read of what they covered is refused or re-derived). A
    // value that contains the new one as a block (rope writing Q inside a
    // fused Q|K|V) stays live as the views around that block instead, so K
    // and V remain readable and nothing is copied.
    std::vector<std::pair<ViewKey, ggml_tensor*>> rest;
    for (auto it = live_.begin(); it != live_.end();) {
      const ViewKey& lk = it->first;
      const int64_t off = k.p - lk.p;
      if (!(lk == k) && lk.same_range(k)) {
        it = live_.erase(it);  // fully overwritten: nothing to write back
      } else if (!(lk == k) && lk.overlaps(k) && lk.stride == k.stride && off >= 0 && (off % lk.stride) % 4 == 0 &&
                 off / lk.stride + k.rows <= lk.rows && (off % lk.stride) / 4 + k.cols <= lk.cols) {
        ggml_tensor* pt = it->second;
        const int64_t r0 = off / lk.stride, c0 = (off % lk.stride) / 4;
        auto keep = [&](int64_t r, int64_t c, int64_t rows, int64_t cols) {
          if (rows <= 0 || cols <= 0) return;
          ViewKey part{lk.p + r * lk.stride + c * 4, rows, cols, lk.stride};
          rest.emplace_back(part, ggml_view_2d(ctx_, pt, cols, rows, pt->nb[1], static_cast<size_t>(r) * pt->nb[1] + c * 4));
        };
        keep(0, 0, lk.rows, c0);                                         // columns left of the block
        keep(0, c0 + k.cols, lk.rows, lk.cols - c0 - k.cols);            // columns right of it
        keep(0, c0, r0, k.cols);                                         // rows above it
        keep(r0 + k.rows, c0, lk.rows - r0 - k.rows, k.cols);            // rows below it
        it = live_.erase(it);
      } else if (!(lk == k) && lk.overlaps(k)) {
        ENGINE_RETURN_IF_ERROR(materialize(it->first, it->second));
        it = live_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto& r : rest) live_.push_back(std::move(r));
    bool found = false;
    for (auto& [lk, lt] : live_) {
      if (lk == k) {
        lt = t;
        found = true;
      }
    }
    if (!found) live_.emplace_back(k, t);
    written_.push_back(k);
    return Status::Ok();
  }

  Status materialize(const ViewKey& k, ggml_tensor* t) {
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * dst, bind(k.p, GGML_TYPE_F32, k.cols, k.rows, static_cast<size_t>(k.stride)));
    ggml_build_forward_expand(g_.graph, ggml_cpy(ctx_, t, dst));
    return Status::Ok();
  }

  // A dynamic input tensor, shared by calls whose source data are identical.
  ggml_tensor* input(InKind kind, ggml_type type, int64_t ne0, int64_t ne1, int32_t seq = 0, int32_t r0 = 0,
                     int32_t n = 0, int32_t span = 0) {
    for (Input& in : g_.inputs) {
      if (in.kind == kind && in.seq == seq && in.r0 == r0 && in.n == n && in.span == span &&
          in.t->ne[0] == ne0 && in.t->ne[1] == ne1 && same_source(kind, *calls_[in.call], *calls_[ci_], seq)) {
        in.members.push_back(ci_);
        return in.t;
      }
    }
    Input in;
    in.kind = kind;
    in.t = ggml_new_tensor_2d(g_.ctx_in.get(), type, ne0, ne1);
    ggml_set_input(in.t);
    in.call = ci_;
    in.seq = seq;
    in.r0 = r0;
    in.n = n;
    in.span = span;
    g_.inputs.push_back(std::move(in));
    return g_.inputs.back().t;
  }

  // --- lowering -----------------------------------------------------------------
  Status lower(const DeviceCall& c) {
    switch (c.m) {
      case M::kEmbedding: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * table, weight(c.a));
        ggml_tensor* ids = input(InKind::kIds, GGML_TYPE_I32, static_cast<int64_t>(c.ints0.size()), 1);
        return write(c.b, ggml_get_rows(ctx_, table, ids));
      }
      case M::kMatmul: return matmul(c.a, c.b, c.has_c ? &c.c : nullptr, c.d);
      case M::kMatmulMany:
        for (const Device::MatmulJob& j : c.jobs) ENGINE_RETURN_IF_ERROR(matmul(j.x, j.w, j.bias, j.y));
        return Status::Ok();
      case M::kRmsNorm:
      case M::kLayerNorm: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * w, weight(c.b));
        if (!ggml_is_contiguous_rows(x)) x = ggml_cont(ctx_, x);
        ggml_tensor* y = c.m == M::kRmsNorm ? ggml_rms_norm(ctx_, x, c.f) : ggml_norm(ctx_, x, c.f);
        y = ggml_mul(ctx_, y, w);
        if (c.has_c) {
          ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, weight(c.c));
          y = ggml_add(ctx_, y, b);
        }
        return write(c.d, y);
      }
      case M::kRope: return rope(c);
      case M::kKvStore: return kv_store(c);
      case M::kAttention: return attention(c);
      case M::kActMul: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * g, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * u, read(c.b));
        if (!ggml_is_contiguous(g)) g = ggml_cont(ctx_, g);
        if (!ggml_is_contiguous(u)) u = ggml_cont(ctx_, u);
        ggml_tensor* y = c.act == Activation::kSilu   ? ggml_swiglu_split(ctx_, g, u)
                         : c.act == Activation::kGelu ? ggml_geglu_erf_split(ctx_, g, u)
                                                      : ggml_geglu_split(ctx_, g, u);
        return write(c.d, y);
      }
      case M::kActivation: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
        ggml_tensor* y = c.act == Activation::kSilu   ? ggml_silu(ctx_, x)
                         : c.act == Activation::kGelu ? ggml_gelu_erf(ctx_, x)
                                                      : ggml_gelu(ctx_, x);
        return write(c.d, y);
      }
      case M::kAdd: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * a, read(c.a));
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, read(c.b));
        return write(c.d, ggml_add(ctx_, a, b));
      }
      case M::kScale: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        return write(c.a, ggml_scale(ctx_, x, c.f));
      }
      case M::kSoftcap: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        return write(c.a, ggml_scale(ctx_, ggml_tanh(ctx_, ggml_scale(ctx_, x, 1.0f / c.f)), c.f));
      }
      case M::kFill: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
        return write(c.a, ggml_fill(ctx_, x, c.f));
      }
      case M::kGather: {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * src, read(c.a));
        ggml_tensor* rows = input(InKind::kRows, GGML_TYPE_I32, static_cast<int64_t>(c.ints0.size()), 1);
        return write(c.d, ggml_get_rows(ctx_, src, rows));
      }
      case M::kScatterAdd: return Unsupported("scatter_add_rows (host-routed mixture of experts) on the GPU");
      case M::kMoeFfn: return moe_ffn(c.moe);
    }
    return Internal("unknown recorded call");
  }

  Status matmul(const TensorView& xv, const TensorView& wv, const TensorView* bv, const TensorView& yv) {
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(xv));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * w, matmul_weight(wv));
    if (w->ne[0] != x->ne[0]) return Internal("matmul inner dimensions differ");
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
    ggml_tensor* y = ggml_mul_mat(ctx_, w, x);
    // fp32 accumulation (DD-053): Vulkan and CUDA otherwise may accumulate
    // matrix-matrix products in fp16; Metal always uses fp32.
    (void)ggml_prec_set_acc(y, GGML_PREC_F32);
    if (bv != nullptr) {
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * b, weight(*bv));
      y = ggml_add(ctx_, y, b);
    }
    return write(yv, y);
  }

  // A 3-D expert weight [experts, rows, cols] (contiguous experts) as
  // ggml [cols, rows, experts].
  Result<ggml_tensor*> expert_weight(const TensorView& w) {
    ggml_type type;
    if (!to_ggml_type(w.dtype(), type)) return Unsupported("weight type " + std::string(dtype_name(w.dtype())));
    if (w.rank() != 3) return Internal("expert weights must be 3-D");
    const int64_t ne = w.dim(0), rows = w.dim(1), cols = w.dim(2);
    const auto nb1 = static_cast<size_t>(w.stride(1)), nb2 = static_cast<size_t>(w.stride(0));
    if (nb1 < ggml_row_size(type, cols) || nb2 != nb1 * static_cast<size_t>(rows)) {
      return Unsupported("expert weights with padded rows or experts");
    }
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * flat, bind(static_cast<const std::byte*>(w.data()), type, cols, ne * rows, nb1));
    return ggml_view_3d(ctx_, flat, cols, rows, ne, nb1, nb2, 0);
  }

  // Device-side routed MoE (MoeFfnParams), as llama.cpp builds it: router ->
  // softmax -> top-k -> expert-indexed matmuls (mul_mat_id) -> weighted sum.
  Status moe_ffn(const MoeFfnParams& p) {
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(p.x));
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * router, weight(p.router));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * wg, expert_weight(p.gate));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * wu, expert_weight(p.up));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * wd, expert_weight(p.down));
    const int64_t d = x->ne[0], m = x->ne[1], ne = p.experts, k = p.top_k;
    if (router->ne[1] != ne || wg->ne[2] != ne || k <= 0 || k > ne) return Internal("moe_ffn: inconsistent shapes");
    ggml_tensor* logits = ggml_mul_mat(ctx_, router, x);  // [experts, m]
    (void)ggml_prec_set_acc(logits, GGML_PREC_F32);
    ggml_tensor* probs = ggml_soft_max(ctx_, logits);
    ggml_tensor* sel = ggml_argsort_top_k(ctx_, probs, static_cast<int>(k));  // [k, m] expert ids
    ggml_tensor* w = ggml_get_rows(ctx_, ggml_reshape_3d(ctx_, probs, 1, ne, m), sel);  // [1, k, m]
    if (p.normalize_topk) {
      ggml_tensor* w2 = ggml_reshape_2d(ctx_, w, k, m);
      w = ggml_reshape_3d(ctx_, ggml_div(ctx_, w2, ggml_sum_rows(ctx_, w2)), 1, k, m);
    }
    ggml_tensor* x3 = ggml_reshape_3d(ctx_, x, d, 1, m);
    // fp32 accumulation, as for every matmul (DD-053; Vulkan / CUDA may use fp16).
    ggml_tensor* g = ggml_mul_mat_id(ctx_, wg, x3, sel);  // [ff, k, m]
    ggml_tensor* u = ggml_mul_mat_id(ctx_, wu, x3, sel);
    (void)ggml_prec_set_acc(g, GGML_PREC_F32);
    (void)ggml_prec_set_acc(u, GGML_PREC_F32);
    ggml_tensor* h = p.act == Activation::kSilu   ? ggml_swiglu_split(ctx_, g, u)
                     : p.act == Activation::kGelu ? ggml_geglu_erf_split(ctx_, g, u)
                                                  : ggml_geglu_split(ctx_, g, u);
    ggml_tensor* yd = ggml_mul_mat_id(ctx_, wd, h, sel);
    (void)ggml_prec_set_acc(yd, GGML_PREC_F32);
    ggml_tensor* y = ggml_mul(ctx_, yd, w);  // [d, k, m], weighted
    ggml_tensor* sum = nullptr;
    for (int64_t j = 0; j < k; ++j) {
      ggml_tensor* yj = ggml_view_2d(ctx_, y, d, m, y->nb[2], static_cast<size_t>(j) * y->nb[1]);
      sum = sum == nullptr ? yj : ggml_add(ctx_, sum, yj);
    }
    if (k == 1) sum = ggml_cont(ctx_, sum);
    if (p.sh_up.data() != nullptr) {  // shared expert, optionally sigmoid-gated per row
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * sg, weight(p.sh_gate));
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * su, weight(p.sh_up));
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * sd, weight(p.sh_down));
      ggml_tensor* a = ggml_mul_mat(ctx_, sg, x);
      ggml_tensor* b = ggml_mul_mat(ctx_, su, x);
      (void)ggml_prec_set_acc(a, GGML_PREC_F32);
      (void)ggml_prec_set_acc(b, GGML_PREC_F32);
      ggml_tensor* hs = p.act == Activation::kSilu   ? ggml_swiglu_split(ctx_, a, b)
                        : p.act == Activation::kGelu ? ggml_geglu_erf_split(ctx_, a, b)
                                                     : ggml_geglu_split(ctx_, a, b);
      ggml_tensor* ys = ggml_mul_mat(ctx_, sd, hs);  // [d, m]
      (void)ggml_prec_set_acc(ys, GGML_PREC_F32);
      if (p.sh_router.data() != nullptr) {
        ENGINE_ASSIGN_OR_RETURN(ggml_tensor * sr, weight(p.sh_router));
        ggml_tensor* gate = ggml_mul_mat(ctx_, sr, x);  // [1, m]
        (void)ggml_prec_set_acc(gate, GGML_PREC_F32);
        ys = ggml_mul(ctx_, ys, ggml_sigmoid(ctx_, gate));
      }
      sum = ggml_add(ctx_, sum, ys);
    }
    return write(p.out, sum);
  }

  Status rope(const DeviceCall& c) {
    if (c.rope.style == RopeStyle::kNone) return Status::Ok();
    float freq_scale = 1.0f;
    if (c.rope.scaling == RopeScaling::kLinear) {
      freq_scale = 1.0f / c.rope.scaling_factor;
    } else if (c.rope.scaling != RopeScaling::kNone) {
      return Unsupported("RoPE scaling other than none/linear (YaRN, Llama 3) on the GPU");
    }
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * x, read(c.a));
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx_, x);
    const int64_t m = static_cast<int64_t>(c.ints0.size());
    ggml_tensor* x3 = ggml_reshape_3d(ctx_, x, c.i1, c.i0, m);
    ggml_tensor* pos = input(InKind::kPositions, GGML_TYPE_I32, m, 1);
    ggml_tensor* factors = nullptr;
    if (c.freq != nullptr) {
      factors = input(InKind::kConst, GGML_TYPE_F32, c.rope.dim / 2, 1);
      Input& in = g_.inputs.back();
      if (in.t == factors && in.constant.empty()) in.constant.assign(c.freq, c.freq + c.rope.dim / 2);
    }
    const int mode = c.rope.style == RopeStyle::kHalfSplit ? GGML_ROPE_TYPE_NEOX : 0;
    ggml_tensor* y = ggml_rope_ext(ctx_, x3, pos, factors, c.rope.dim, mode, static_cast<int>(c.rope.original_context),
                                   c.rope.freq_base, freq_scale, 0.0f, c.rope.attn_factor, 0.0f, 0.0f);
    return write(c.a, ggml_reshape_2d(ctx_, y, static_cast<int64_t>(c.i0) * c.i1, m));
  }

  // The current K or V pool tensor of a layer. Head-major: rows of head_dim
  // elements (one per block, head, slot). Slot-major: one row per cell, all
  // KV heads of one token (n_kv_heads * head_dim elements).
  Result<ggml_tensor*> pool(const KvLayerView& kv, bool is_v) {
    const KvGeometry& g = *kv.geom;
    ggml_type type;
    if (g.dtype != DType::kF16 && g.dtype != DType::kF32 && g.dtype != DType::kQ8_0) {
      return Unsupported("KV cache dtype " + std::string(dtype_name(g.dtype)) + " on the GPU (f16, f32, q8_0)");
    }
    to_ggml_type(g.dtype, type);
    const auto* base = static_cast<const std::byte*>(is_v ? kv.v : kv.k);
    if (auto it = pool_now_.find(base); it != pool_now_.end()) return it->second;
    const int64_t dim = is_v ? g.head_dim_v : g.head_dim;
    const bool slot = g.layout == KvLayout::kSlotMajor;
    const int64_t ne0 = slot ? dim * g.num_kv_heads : dim;
    const int64_t rows = static_cast<int64_t>(g.num_blocks) * g.block_size * (slot ? 1 : g.num_kv_heads);
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * t, bind(base, type, ne0, rows, ggml_row_size(type, ne0)));
    pool_now_[base] = t;
    return t;
  }

  Status kv_store(const DeviceCall& c) {
    if (c.kv.empty()) return Status::Ok();
    for (const KvLayerView& v : c.kv) {
      if (v.k != c.kv[0].k || v.v != c.kv[0].v) return Unsupported("kv_store across several KV pools");
    }
    const KvGeometry& g = *c.kv[0].geom;
    const bool slot = g.layout == KvLayout::kSlotMajor;
    const int64_t m = static_cast<int64_t>(c.ints0.size());
    // Slot-major: one row (cell) per token; head-major: one row per token and head.
    ggml_tensor* idx = input(InKind::kKvIndex, GGML_TYPE_I64, slot ? m : g.num_kv_heads * m, 1);
    for (int which = 0; which < 2; ++which) {
      const bool is_v = which == 1;
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * src, read(is_v ? c.b : c.a));
      if (!ggml_is_contiguous(src)) src = ggml_cont(ctx_, src);
      const int64_t dim = is_v ? g.head_dim_v : g.head_dim;
      ggml_tensor* rows = slot ? ggml_reshape_2d(ctx_, src, dim * g.num_kv_heads, m)
                               : ggml_reshape_2d(ctx_, src, dim, g.num_kv_heads * m);
      ENGINE_ASSIGN_OR_RETURN(ggml_tensor * dst, pool(c.kv[0], is_v));
      ggml_tensor* out = ggml_set_rows(ctx_, dst, rows, idx);
      pool_now_[static_cast<const std::byte*>(is_v ? c.kv[0].v : c.kv[0].k)] = out;
      pools_written_.push_back(out);
    }
    return Status::Ok();
  }

  Status attention(const DeviceCall& c) {
    const AttentionParams& p = c.ap;
    if (p.kv.empty()) return Internal("attention without KV");
    const KvGeometry& g = *p.kv[0].geom;
    for (const KvLayerView& v : p.kv) {
      if (v.k != p.kv[0].k || v.v != p.kv[0].v) return Unsupported("attention across several KV pools");
    }
    const int32_t hd = g.head_dim, hdv = g.head_dim_v, nkv = g.num_kv_heads, nh = p.num_heads;
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * q, read(p.q));
    if (!ggml_is_contiguous(q)) q = ggml_cont(ctx_, q);
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * kpool, pool(p.kv[0], false));
    ENGINE_ASSIGN_OR_RETURN(ggml_tensor * vpool, pool(p.kv[0], true));
    const std::vector<SeqRows>& seqs = seqs_[static_cast<size_t>(ci_)];
    if (g.layout == KvLayout::kSlotMajor) {
      // In-place paged attention: K/V are strided views of the pool's first
      // `span` cells ([dim, cells, kv_heads]); the mask admits, per row, exactly
      // the cells of its own sequence at positions [lo, pos] (block tables,
      // causality, window). Keys are stored after RoPE, so attention over
      // physical cell order is exact up to summation order. Fully masked
      // chunks are skipped by the kernel; nothing is gathered or copied.
      const int64_t m = static_cast<int64_t>(p.positions.size()), L = seqs.front().span;
      ggml_tensor* qs = ggml_permute(ctx_, ggml_reshape_3d(ctx_, q, hd, nh, m), 0, 2, 1, 3);  // [hd, m, nh]
      ggml_tensor* ks = ggml_view_3d(ctx_, kpool, hd, L, nkv, kpool->nb[1], ggml_row_size(kpool->type, hd), 0);
      ggml_tensor* vs = ggml_view_3d(ctx_, vpool, hdv, L, nkv, vpool->nb[1], ggml_row_size(vpool->type, hdv), 0);
      ggml_tensor* mask = input(InKind::kPagedMask, GGML_TYPE_F16, L, m, -1, 0, static_cast<int32_t>(m), static_cast<int32_t>(L));
      ggml_tensor* o = ggml_flash_attn_ext(ctx_, qs, ks, vs, mask, p.scale, 0.0f, p.softcap);  // [hdv, nh, m]
      (void)ggml_prec_set_acc(o, GGML_PREC_F32);  // fp32 softmax/accumulation, as on Metal
      // Sparse K/V: each row visits only its finite mask entries (its own
      // sequence) instead of the whole span; the backend decides when that
      // pays (Metal: <= 4096, Vulkan: large spans; the CPU ignores it).
      if (seqs.front().kv_max > 0) ggml_flash_attn_ext_set_n_kv_max(o, seqs.front().kv_max);
      return write(p.out, ggml_reshape_2d(ctx_, o, static_cast<int64_t>(hdv) * nh, m));
    }
    ggml_tensor* out = nullptr;
    for (const SeqRows& s : seqs) {
      const int64_t n = s.n, L = s.span;
      ggml_tensor* qs = ggml_view_2d(ctx_, q, q->ne[0], n, q->nb[1], static_cast<size_t>(s.r0) * q->nb[1]);
      qs = ggml_permute(ctx_, ggml_reshape_3d(ctx_, ggml_cont(ctx_, qs), hd, nh, n), 0, 2, 1, 3);  // [hd, n, nh]
      ggml_tensor* gidx = input(InKind::kGatherIdx, GGML_TYPE_I32, nkv * L, 1, s.seq, s.r0, s.n, s.span);
      ggml_tensor* ks = ggml_reshape_3d(ctx_, ggml_get_rows(ctx_, kpool, gidx), hd, L, nkv);
      ggml_tensor* vs = ggml_reshape_3d(ctx_, ggml_get_rows(ctx_, vpool, gidx), hdv, L, nkv);
      ggml_tensor* mask = input(InKind::kMask, GGML_TYPE_F16, L, n, s.seq, s.r0, s.n, s.span);
      ggml_tensor* o = ggml_flash_attn_ext(ctx_, qs, ks, vs, mask, p.scale, 0.0f, p.softcap);  // [hdv, nh, n]
      (void)ggml_prec_set_acc(o, GGML_PREC_F32);
      o = ggml_reshape_2d(ctx_, o, static_cast<int64_t>(hdv) * nh, n);
      out = out == nullptr ? o : ggml_concat(ctx_, out, o, 1);
    }
    return write(p.out, out);
  }

  static std::string describe(const ViewKey& k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "[%lldx%lld stride %lld @%p]", static_cast<long long>(k.rows),
                  static_cast<long long>(k.cols), static_cast<long long>(k.stride), static_cast<const void*>(k.p));
    return buf;
  }
  static const char* call_name(M m) {
    switch (m) {
      case M::kEmbedding: return "embedding";
      case M::kMatmul: return "matmul";
      case M::kMatmulMany: return "matmul_many";
      case M::kRmsNorm: return "rms_norm";
      case M::kLayerNorm: return "layer_norm";
      case M::kRope: return "rope";
      case M::kKvStore: return "kv_store";
      case M::kAttention: return "attention";
      case M::kActMul: return "act_mul";
      case M::kActivation: return "activation";
      case M::kAdd: return "add";
      case M::kScale: return "scale";
      case M::kSoftcap: return "softcap";
      case M::kFill: return "fill";
      case M::kGather: return "gather_rows";
      case M::kScatterAdd: return "scatter_add_rows";
      case M::kMoeFfn: return "moe_ffn";
    }
    return "?";
  }

  GgmlDevice& d_;
  Prepared& g_;
  std::span<const DeviceCall* const> calls_;
  const std::vector<std::vector<SeqRows>>& seqs_;
  ggml_context* ctx_ = nullptr;
  int32_t ci_ = 0;
  std::vector<std::pair<ViewKey, ggml_tensor*>> live_;  // values written in this segment, in write order
  std::vector<ViewKey> written_;
  std::map<std::tuple<const std::byte*, int, int64_t, int64_t, size_t>, ggml_tensor*> leaves_;
  std::map<const std::byte*, ggml_tensor*> pool_now_;  // latest K/V pool value per layer
  std::vector<ggml_tensor*> pools_written_;
};

Result<std::unique_ptr<Prepared>> GgmlDevice::build(std::span<const DeviceCall* const> calls, std::vector<uint64_t> key,
                                                    const std::vector<std::vector<SeqRows>>& seqs) {
  auto g = std::make_unique<Prepared>();
  g->key = std::move(key);
  GraphBuilder b(*this, *g, calls, seqs);
  ENGINE_RETURN_IF_ERROR(b.build());
  g->in_buf = ggml_backend_alloc_ctx_tensors(g->ctx_in.get(), backend_);
  if (g->in_buf == nullptr && !g->inputs.empty()) return OutOfMemory(name_ + ": cannot allocate graph inputs");
  g->galloc = ggml_gallocr_new(buft_);
  if (g->galloc == nullptr || !ggml_gallocr_alloc_graph(g->galloc, g->graph)) {
    return OutOfMemory(name_ + ": cannot allocate the graph workspace");
  }
  g->workspace_bytes = static_cast<int64_t>(ggml_gallocr_get_buffer_size(g->galloc, 0)) +
                       (g->in_buf ? static_cast<int64_t>(ggml_backend_buffer_get_size(g->in_buf)) : 0);
  // Constant inputs are written once.
  for (Input& in : g->inputs) {
    if (in.kind == InKind::kConst) ggml_backend_tensor_set(in.t, in.constant.data(), 0, ggml_nbytes(in.t));
  }
  return g;
}

}  // namespace dynacore::ggml_detail

#endif  // DYNACORE_HAS_GGML
