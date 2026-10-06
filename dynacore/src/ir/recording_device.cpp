#include "dynacore/ir/recording_device.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>

#include "dynacore/ir/text.h"

namespace dynacore::ir {
namespace {

using Clock = std::chrono::steady_clock;

int64_t ns_since(Clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
}

std::string_view act_name(Activation a) {
  switch (a) {
    case Activation::kSilu: return "silu";
    case Activation::kGelu: return "gelu";
    case Activation::kGeluTanh: return "gelu_tanh";
  }
  return "silu";
}

std::string_view rope_style_word(RopeStyle s) {
  switch (s) {
    case RopeStyle::kNone: return "none";
    case RopeStyle::kInterleaved: return "interleaved";
    case RopeStyle::kHalfSplit: return "half_split";
  }
  return "none";
}

// IR type of a view as the device sees it.
Type type_of(const TensorView& t, ValueKind kind) {
  Type ty;
  ty.kind = kind;
  ty.dtype = t.dtype();
  for (int i = 0; i < t.rank(); ++i) ty.shape.push_back(Dim::of(t.dim(i)));
  if (dtype_is_quantized(t.dtype())) {
    ty.layout = Layout{LayoutKind::kBlocked, dtype_block_elems(t.dtype())};
  } else if (t.rank() == 2) {
    const int64_t elem = dtype_block_bytes(t.dtype());
    if (elem > 0 && t.stride(0) != t.dim(1) * elem) ty.layout = Layout{LayoutKind::kStrided, t.stride(0) / elem};
  }
  return ty;
}

uint64_t ptr_word(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }
uint64_t float_word(float f) { return std::bit_cast<uint32_t>(f); }

uint64_t plan_word(const KernelPlan& p) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  mix(static_cast<uint64_t>(p.expand_min_rows));
  mix(static_cast<uint64_t>(p.gemm_k_block));
  mix(static_cast<uint64_t>(p.matmul_chunks_per_thread));
  mix(static_cast<uint64_t>(p.int8_decode_max_rows));
  mix(p.int8_ffn_down ? 1 : 0);
  mix(static_cast<uint64_t>(p.attention_full));
  mix(static_cast<uint64_t>(p.attention_window));
  mix(static_cast<uint64_t>(p.attention_chunk));
  mix(p.grouped_attention ? 1 : 0);
  return h;
}

}  // namespace

// One recorded Device call with private copies of everything that is not
// device memory (spans, tables, parameter structs). Objects are reused across
// segments, so vectors keep their capacity.
struct RecordingDevice::Call {
  enum class M : uint8_t {
    kEmbedding, kMatmul, kMatmulMany, kRmsNorm, kLayerNorm, kRope, kKvStore, kAttention,
    kActMul, kActivation, kAdd, kScale, kSoftcap, kFill, kGather, kScatterAdd,
  };
  M m{};
  TensorView a, b, c, d;
  bool has_c = false;
  float f = 0;
  int32_t i0 = 0, i1 = 0;
  Activation act{};
  RopeConfig rope{};
  const float* freq = nullptr;
  std::vector<int32_t> ints0, ints1;
  std::vector<float> floats;
  std::vector<std::vector<int32_t>> tables;
  std::vector<KvLayerView> kv;
  std::vector<MatmulJob> jobs;
  std::vector<TensorView> biases;  // matmul_many biases (jobs point here)
  AttentionParams ap{};
  std::shared_ptr<const KernelPlan> plan;  // plan in effect when recorded

  void reset(M kind) {
    m = kind;
    a = b = c = d = TensorView();
    has_c = false;
    f = 0;
    i0 = i1 = 0;
    freq = nullptr;
    ints0.clear();
    ints1.clear();
    floats.clear();
    kv.clear();
    jobs.clear();
    biases.clear();
    ap = AttentionParams{};
  }

  void copy_kv(std::span<const KvLayerView> src) {
    if (tables.size() < src.size()) tables.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) tables[i].assign(src[i].block_table.begin(), src[i].block_table.end());
    kv.assign(src.begin(), src.end());
    for (size_t i = 0; i < kv.size(); ++i) kv[i].block_table = tables[i];
  }
};

RecordingDevice::RecordingDevice(Device& inner, Mode mode)
    : inner_(inner),
      mode_(mode),
      name_(std::string(mode == Mode::kTrace ? "trace/" : "compiled/") + std::string(inner.name())),
      plan_(std::make_shared<KernelPlan>(KernelPlan::defaults())) {
  begin_graph("step");
}

RecordingDevice::~RecordingDevice() {
  // Never drop recorded work silently.
  if (ncalls_ > 0) (void)flush();
}

void RecordingDevice::set_name(const void* data, std::string name) { names_[data] = std::move(name); }

void RecordingDevice::set_planner(ExecutionPlanner planner) {
  planner_ = std::move(planner);
  cache_.clear();
}

void RecordingDevice::begin_graph(std::string name) {
  graph_ = Graph(std::move(name));
  op_ns_.clear();
  live_.clear();
  buffers_.clear();
  kv_values_.clear();
}

// --- recording ------------------------------------------------------------------

RecordingDevice::Call& RecordingDevice::next_call(int kind) {
  if (ncalls_ == calls_.size()) calls_.push_back(std::make_unique<Call>());
  Call& c = *calls_[ncalls_++];
  c.reset(static_cast<Call::M>(kind));
  c.plan = plan_;
  return c;
}

void RecordingDevice::submit(Call& c) {
  ++stats_.ops;
  if (mode_ == Mode::kTrace) {
    const auto t0 = Clock::now();
    const auto first = static_cast<size_t>(build_ir(c));
    const size_t last = graph_.ops().size();
    stats_.record_ns += ns_since(t0);
    const auto t1 = Clock::now();
    run_call(c);
    // The call's time goes to its last op (the compute op; leaves come first).
    if (timing_ && last > first) op_ns_[last - 1] = ns_since(t1);
    --ncalls_;  // trace mode keeps nothing pending
    return;
  }
  const auto t0 = Clock::now();
  sign(c);
  stats_.record_ns += ns_since(t0);
}

void RecordingDevice::sign(const Call& c) {
  std::vector<uint64_t>& s = signature_;
  auto view = [&](const TensorView& t) {
    s.push_back(ptr_word(t.data()));
    s.push_back(static_cast<uint64_t>(t.rank() > 0 ? t.dim(0) : 0) << 32 |
                static_cast<uint64_t>(t.rank() > 1 ? t.dim(1) : 0));
    s.push_back(static_cast<uint64_t>(t.rank() > 1 ? t.stride(0) : 0) << 8 | static_cast<uint64_t>(t.dtype()));
  };
  s.push_back(static_cast<uint64_t>(c.m) | static_cast<uint64_t>(c.has_c) << 8 |
              static_cast<uint64_t>(c.act) << 16 | static_cast<uint64_t>(c.i0) << 24 |
              static_cast<uint64_t>(static_cast<uint32_t>(c.i1)) << 40);
  view(c.a);
  view(c.b);
  view(c.c);
  view(c.d);
  s.push_back(float_word(c.f) | static_cast<uint64_t>(c.ints0.size()) << 32);
  s.push_back(static_cast<uint64_t>(c.ints1.size()) | static_cast<uint64_t>(c.kv.size()) << 32);
  if (!c.kv.empty()) s.push_back(ptr_word(c.kv[0].k) ^ ptr_word(c.kv[0].geom) << 1);
  for (const MatmulJob& j : c.jobs) {
    view(j.x);
    view(j.w);
    view(j.y);
    s.push_back(j.bias ? ptr_word(j.bias->data()) : 0);
  }
  if (c.m == Call::M::kAttention) {
    view(c.ap.q);
    view(c.ap.out);
    s.push_back(static_cast<uint64_t>(c.ap.num_heads) | float_word(c.ap.scale) << 32);
    s.push_back(float_word(c.ap.softcap) | static_cast<uint64_t>(static_cast<uint32_t>(c.ap.sliding_window)) << 32);
  }
  if (c.m == Call::M::kRope) {
    s.push_back(static_cast<uint64_t>(c.rope.style) | static_cast<uint64_t>(c.rope.dim) << 8 |
                float_word(c.rope.freq_base) << 32);
    s.push_back(ptr_word(c.freq));
  }
  s.push_back(plan_word(*c.plan));
}

// --- IR construction ------------------------------------------------------------

ValueId RecordingDevice::use(const TensorView& t, Role role) {
  const auto key = std::make_tuple(static_cast<const void*>(t.data()), t.rank() > 0 ? t.dim(0) : 1,
                                   t.rank() > 1 ? t.dim(1) : 1, t.rank() > 1 ? t.stride(0) : 0);
  if (auto it = live_.find(key); it != live_.end()) return it->second;
  // First sight of this data: a leaf. Matmul weights, norm weights and
  // embedding tables are parameters; everything else is a step input.
  std::string name;
  if (auto n = names_.find(t.data()); n != names_.end()) name = n->second;
  const ValueKind kind = role == Role::kWeight ? ValueKind::kWeight : ValueKind::kTensor;
  const ValueId id =
      graph_.add_leaf(role == Role::kWeight ? OpKind::kWeight : OpKind::kInput, type_of(t, kind), std::move(name));
  if (auto b = buffers_.find(t.data()); b != buffers_.end()) {
    graph_.mutable_value(id).buffer = b->second;
  } else {
    buffers_[t.data()] = graph_.value(id).buffer;
  }
  op_ns_.push_back(0);
  live_[key] = id;
  return id;
}

ValueId RecordingDevice::index_const(std::span<const int32_t> data, std::string_view name) {
  ConstData c;
  c.i32.assign(data.begin(), data.end());
  const ValueId id = graph_.add_const(index_type({Dim::of(static_cast<int64_t>(data.size()))}), std::move(c),
                                      std::string(name));
  op_ns_.push_back(0);
  return id;
}

ValueId RecordingDevice::kv_value(const KvLayerView& kv) {
  if (auto it = kv_values_.find(kv.k); it != kv_values_.end()) return it->second;
  const KvGeometry& g = *kv.geom;
  std::string name;
  if (auto n = names_.find(kv.k); n != names_.end()) name = n->second;
  const ValueId id = graph_.add_leaf(
      OpKind::kKvCache, kv_type(g.dtype, g.num_blocks, g.num_kv_heads, g.block_size, g.head_dim), std::move(name));
  op_ns_.push_back(0);
  kv_values_[kv.k] = id;
  return id;
}

ValueId RecordingDevice::record_op(const Call* c, OpKind kind, std::vector<ValueId> inputs, Attrs attrs,
                                   const TensorView& out, bool in_place) {
  if (c != nullptr && (kind == OpKind::kMatMul || kind == OpKind::kQuantizedMatMul)) {
    attrs["int8_rows"] = static_cast<int64_t>(c->plan->int8_decode_max_rows);  // the plan's limit for this op
  }
  std::vector<Type> types;
  for (ValueId v : inputs) types.push_back(graph_.value(v).type);
  Result<Type> inferred = infer_type(kind, types, attrs);
  // The recorded type is what the op computes; a strided destination (a
  // column slice of a wider buffer) is a physical placement of that value.
  Type result = inferred.ok() ? *inferred : type_of(out, ValueKind::kTensor);
  if (kind != OpKind::kKvWrite) {
    const Type physical = type_of(out, ValueKind::kTensor);
    if (physical.layout.kind == LayoutKind::kStrided) result.layout = physical.layout;
  }
  int32_t buffer = -1;
  if (in_place && !inputs.empty()) {
    buffer = graph_.value(inputs[0]).buffer;
  } else if (kind != OpKind::kKvWrite) {
    if (auto b = buffers_.find(out.data()); b != buffers_.end()) buffer = b->second;
  }
  const ValueId r = graph_.add_op(kind, std::move(inputs), std::move(result), std::move(attrs), buffer);
  if (kind != OpKind::kKvWrite) {
    buffers_[out.data()] = graph_.value(r).buffer;
    live_[std::make_tuple(static_cast<const void*>(out.data()), out.rank() > 0 ? out.dim(0) : 1,
                          out.rank() > 1 ? out.dim(1) : 1, out.rank() > 1 ? out.stride(0) : 0)] = r;
  }
  op_ns_.push_back(0);
  return r;
}

int32_t RecordingDevice::build_ir(const Call& c) {
  const auto first = static_cast<int32_t>(graph_.ops().size());
  using M = Call::M;
  switch (c.m) {
    case M::kEmbedding: {
      const ValueId t = use(c.a, Role::kWeight);
      const ValueId i = index_const(c.ints0, "tokens");
      record_op(&c, OpKind::kEmbedding, {t, i}, {}, c.b, false);
      break;
    }
    case M::kMatmul: {
      std::vector<ValueId> in = {use(c.a, Role::kActivation), use(c.b, Role::kWeight)};
      if (c.has_c) in.push_back(use(c.c, Role::kWeight));
      record_op(&c, dtype_is_quantized(c.b.dtype()) ? OpKind::kQuantizedMatMul : OpKind::kMatMul, std::move(in), {},
                c.d, false);
      break;
    }
    case M::kMatmulMany: {
      // One call for the whole group (it already is one region); one IR op
      // per job, tagged with a shared group id.
      const int64_t group = static_cast<int64_t>(graph_.ops().size());
      for (const MatmulJob& j : c.jobs) {
        std::vector<ValueId> in = {use(j.x, Role::kActivation), use(j.w, Role::kWeight)};
        if (j.bias) in.push_back(use(*j.bias, Role::kWeight));
        record_op(&c, dtype_is_quantized(j.w.dtype()) ? OpKind::kQuantizedMatMul : OpKind::kMatMul, std::move(in),
                  {{"group", group}}, j.y, false);
      }
      break;
    }
    case M::kRmsNorm: {
      const ValueId xv = use(c.a, Role::kActivation);
      const ValueId wv = use(c.b, Role::kWeight);
      record_op(&c, OpKind::kRmsNorm, {xv, wv}, {{"eps", static_cast<double>(c.f)}}, c.d, c.a.data() == c.d.data());
      break;
    }
    case M::kLayerNorm: {
      std::vector<ValueId> in = {use(c.a, Role::kActivation), use(c.b, Role::kWeight)};
      if (c.has_c) in.push_back(use(c.c, Role::kWeight));
      record_op(&c, OpKind::kLayerNorm, std::move(in), {{"eps", static_cast<double>(c.f)}}, c.d,
                c.a.data() == c.d.data());
      break;
    }
    case M::kRope: {
      const ValueId xv = use(c.a, Role::kActivation);
      const ValueId pv = index_const(c.ints0, "positions");
      Attrs a{{"heads", static_cast<int64_t>(c.i0)},
              {"head_dim", static_cast<int64_t>(c.i1)},
              {"dim", static_cast<int64_t>(c.rope.dim > 0 ? c.rope.dim : c.i1)},
              {"style", std::string(rope_style_word(c.rope.style))},
              {"base", static_cast<double>(c.rope.freq_base)}};
      if (c.rope.scaling != RopeScaling::kNone) a["scaling_factor"] = static_cast<double>(c.rope.scaling_factor);
      if (c.freq) a["freq_factors"] = int64_t{1};
      record_op(&c, OpKind::kRope, {xv, pv}, std::move(a), c.a, true);
      break;
    }
    case M::kKvStore: {
      const ValueId cache = kv_value(c.kv[0]);
      const ValueId kvv = use(c.a, Role::kActivation), vv = use(c.b, Role::kActivation);
      const ValueId pv = index_const(c.ints0, "positions"), sv = index_const(c.ints1, "row_seq");
      Attrs a{{"seqs", static_cast<int64_t>(c.kv.size())}};
      if (c.kv[0].geom->head_dim_v != c.kv[0].geom->head_dim) {
        a["head_dim_v"] = static_cast<int64_t>(c.kv[0].geom->head_dim_v);
      }
      // Later reads of this cache see the written version.
      kv_values_[c.kv[0].k] = record_op(&c, OpKind::kKvWrite, {cache, kvv, vv, pv, sv}, std::move(a), c.a, true);
      break;
    }
    case M::kAttention: {
      const KvGeometry& g = *c.kv[0].geom;
      const ValueId qv = use(c.ap.q, Role::kActivation);
      const ValueId cache = kv_value(c.kv[0]);
      const ValueId pv = index_const(c.ints0, "positions"), sv = index_const(c.ints1, "row_seq");
      Attrs a{{"heads", static_cast<int64_t>(c.ap.num_heads)},
              {"kv_heads", static_cast<int64_t>(g.num_kv_heads)},
              {"head_dim", static_cast<int64_t>(g.head_dim)},
              {"scale", static_cast<double>(c.ap.scale)},
              {"block_size", static_cast<int64_t>(g.block_size)},
              {"seqs", static_cast<int64_t>(c.kv.size())}};
      if (g.head_dim_v != g.head_dim) a["head_dim_v"] = static_cast<int64_t>(g.head_dim_v);
      if (c.ap.softcap != 0) a["softcap"] = static_cast<double>(c.ap.softcap);
      if (c.ap.sliding_window != 0) a["window"] = static_cast<int64_t>(c.ap.sliding_window);
      record_op(&c, c.ap.num_heads != g.num_kv_heads ? OpKind::kGqa : OpKind::kAttention, {qv, cache, pv, sv},
                std::move(a), c.ap.out, false);
      break;
    }
    case M::kActMul: {
      const ValueId g = use(c.a, Role::kActivation), u = use(c.b, Role::kActivation);
      record_op(&c, OpKind::kActMul, {g, u}, {{"act", std::string(act_name(c.act))}}, c.d, false);
      break;
    }
    case M::kActivation: {
      const ValueId xv = use(c.a, Role::kActivation);
      record_op(&c, OpKind::kActivation, {xv}, {{"act", std::string(act_name(c.act))}}, c.d, c.a.data() == c.d.data());
      break;
    }
    case M::kAdd: {
      const ValueId av = use(c.a, Role::kActivation), bv = use(c.b, Role::kActivation);
      record_op(&c, OpKind::kAdd, {av, bv}, {}, c.d, false);
      break;
    }
    case M::kScale:
      record_op(&c, OpKind::kScale, {use(c.a, Role::kActivation)}, {{"s", static_cast<double>(c.f)}}, c.a, true);
      break;
    case M::kSoftcap:
      record_op(&c, OpKind::kSoftcap, {use(c.a, Role::kActivation)}, {{"cap", static_cast<double>(c.f)}}, c.a, true);
      break;
    case M::kFill:
      record_op(&c, OpKind::kFill, {use(c.a, Role::kActivation)}, {{"value", static_cast<double>(c.f)}}, c.a, true);
      break;
    case M::kGather: {
      const ValueId s = use(c.a, Role::kActivation);
      record_op(&c, OpKind::kGather, {s, index_const(c.ints0, "rows")}, {}, c.d, false);
      break;
    }
    case M::kScatterAdd: {
      const ValueId d = use(c.d, Role::kActivation), s = use(c.a, Role::kActivation);
      ConstData w;
      w.f32 = c.floats;
      const ValueId wv = graph_.add_const(tensor_type(DType::kF32, {Dim::of(static_cast<int64_t>(c.floats.size()))}),
                                          std::move(w), "weights");
      op_ns_.push_back(0);
      record_op(&c, OpKind::kScatterAdd, {d, s, index_const(c.ints0, "rows"), wv}, {}, c.d, true);
      break;
    }
  }
  return first;
}

// --- execution ----------------------------------------------------------------------

void RecordingDevice::apply_plan(const Call& c) {
  if (!c.plan || c.plan == applied_ || (applied_ && *c.plan == *applied_)) return;
  inner_.set_kernel_plan(*c.plan);
  applied_ = c.plan;
}

void RecordingDevice::run_call(const Call& c) {
  apply_plan(c);
  Device& d = inner_;
  using M = Call::M;
  switch (c.m) {
    case M::kEmbedding: d.embedding(c.a, c.ints0, c.b); break;
    case M::kMatmul: d.matmul(c.a, c.b, c.has_c ? &c.c : nullptr, c.d); break;
    case M::kMatmulMany: d.matmul_many(c.jobs); break;
    case M::kRmsNorm: d.rms_norm(c.a, c.b, c.f, c.d); break;
    case M::kLayerNorm: d.layer_norm(c.a, c.b, c.has_c ? &c.c : nullptr, c.f, c.d); break;
    case M::kRope: d.rope(c.a, c.i0, c.i1, c.ints0, c.rope, c.freq); break;
    case M::kKvStore: d.kv_store(c.a, c.b, c.ints0, c.ints1, c.kv); break;
    case M::kAttention: d.attention(c.ap); break;
    case M::kActMul: d.act_mul(c.act, c.a, c.b, c.d); break;
    case M::kActivation: d.activation(c.act, c.a, c.d); break;
    case M::kAdd: d.add(c.a, c.b, c.d); break;
    case M::kScale: d.scale(c.a, c.f); break;
    case M::kSoftcap: d.softcap(c.a, c.f); break;
    case M::kFill: d.fill(c.a, c.f); break;
    case M::kGather: d.gather_rows(c.a, c.ints0, c.d); break;
    case M::kScatterAdd: d.scatter_add_rows(c.a, c.ints0, c.floats, c.d); break;
  }
}

void RecordingDevice::execute(const std::vector<CallStep>& steps) {
  std::vector<MatmulJob> jobs;
  for (const CallStep& step : steps) {
    switch (step.kind) {
      case ExecStep::Kind::kOp:
        for (int32_t ci : step.calls) run_call(*calls_[static_cast<size_t>(ci)]);
        break;
      case ExecStep::Kind::kMatmulGroup: {
        apply_plan(*calls_[static_cast<size_t>(step.calls[0])]);
        jobs.clear();
        for (int32_t ci : step.calls) {
          const Call& c = *calls_[static_cast<size_t>(ci)];
          jobs.push_back(MatmulJob{c.a, c.b, c.d, c.has_c ? &c.c : nullptr});
        }
        inner_.matmul_many(jobs);
        break;
      }
      case ExecStep::Kind::kGatedMatmul: {
        const Call& g = *calls_[static_cast<size_t>(step.calls[0])];
        const Call& u = *calls_[static_cast<size_t>(step.calls[1])];
        const Call& a = *calls_[static_cast<size_t>(step.calls[2])];
        apply_plan(g);
        inner_.matmul_gated(a.act, g.a, g.b, u.b, a.d, g.d, u.d);
        break;
      }
    }
  }
}

// Builds the IR of the pending calls, plans it, and turns the plan into call
// indices. Any problem gives the in-order plan.
std::vector<RecordingDevice::CallStep> RecordingDevice::compile_pending() {
  std::vector<CallStep> in_order;
  for (size_t i = 0; i < ncalls_; ++i) in_order.push_back({ExecStep::Kind::kOp, {static_cast<int32_t>(i)}});
  if (!planner_) return in_order;

  begin_graph(graph_.name());
  std::vector<int32_t> call_of_op;  // IR op index -> call index (-1 for leaves)
  for (size_t i = 0; i < ncalls_; ++i) {
    build_ir(*calls_[i]);
    call_of_op.resize(graph_.ops().size(), -1);
    // Every non-leaf op added by this call belongs to it.
    for (size_t op = call_of_op.size(); op-- > 0;) {
      if (call_of_op[op] != -1) break;
      const OpKind k = graph_.op(static_cast<int32_t>(op)).kind;
      const bool leaf = k == OpKind::kInput || k == OpKind::kWeight || k == OpKind::kKvCache || k == OpKind::kConstant;
      call_of_op[op] = leaf ? -2 : static_cast<int32_t>(i);
    }
  }
  for (int32_t& c : call_of_op) c = c == -2 ? -1 : c;

  Result<ExecPlan> plan = planner_(graph_);
  if (!plan.ok()) {
    ++stats_.fallbacks;
    return in_order;
  }
  if (observer_) observer_(graph_, *plan);
  std::vector<CallStep> steps;
  std::vector<int32_t> covered(ncalls_, 0);
  bool valid = true;
  int32_t last_call = -1;
  for (const ExecStep& s : plan->steps) {
    CallStep cs;
    cs.kind = s.kind;
    for (int32_t op : s.ops) {
      if (op < 0 || op >= static_cast<int32_t>(call_of_op.size())) {
        valid = false;
        continue;
      }
      const int32_t ci = call_of_op[static_cast<size_t>(op)];
      if (ci < 0) continue;
      // A recorded matmul_many is one call but several IR ops: run it once.
      if (s.kind == ExecStep::Kind::kOp && (ci == last_call || (!cs.calls.empty() && cs.calls.back() == ci))) continue;
      cs.calls.push_back(ci);
    }
    if (cs.calls.empty()) continue;
    // Fused steps: right call kinds, one kernel plan, distinct calls.
    if (cs.kind != ExecStep::Kind::kOp) {
      const Call& first = *calls_[static_cast<size_t>(cs.calls[0])];
      for (int32_t ci : cs.calls) {
        const Call& c = *calls_[static_cast<size_t>(ci)];
        if (!(*c.plan == *first.plan)) valid = false;
      }
      if (cs.kind == ExecStep::Kind::kMatmulGroup) {
        for (int32_t ci : cs.calls) valid = valid && calls_[static_cast<size_t>(ci)]->m == Call::M::kMatmul;
      } else {
        valid = valid && cs.calls.size() == 3;
        if (valid) {
          const Call& g = *calls_[static_cast<size_t>(cs.calls[0])];
          const Call& u = *calls_[static_cast<size_t>(cs.calls[1])];
          const Call& a = *calls_[static_cast<size_t>(cs.calls[2])];
          valid = g.m == Call::M::kMatmul && u.m == Call::M::kMatmul && !g.has_c && !u.has_c &&
                  a.m == Call::M::kActMul && g.a.data() == u.a.data();
        }
      }
    }
    for (int32_t ci : cs.calls) ++covered[static_cast<size_t>(ci)];
    last_call = cs.calls.back();
    steps.push_back(std::move(cs));
  }
  for (int32_t n : covered) valid = valid && n == 1;
  if (!valid) {
    ++stats_.fallbacks;
    return in_order;
  }
  return steps;
}

Status RecordingDevice::flush() {
  if (ncalls_ == 0) return Status::Ok();
  ++stats_.segments;
  uint64_t h = 1469598103934665603ull;
  for (uint64_t w : signature_) h = (h ^ w) * 1099511628211ull;
  const CacheEntry* entry = nullptr;
  if (cache_on_) {
    auto it = cache_.find(h);
    if (it != cache_.end() && it->second.signature == signature_) entry = &it->second;
  }
  std::vector<CallStep> compiled;
  if (entry != nullptr) {
    ++stats_.cache_hits;
  } else {
    ++stats_.cache_misses;
    const auto t0 = Clock::now();
    compiled = compile_pending();
    stats_.plan_ns += ns_since(t0);
    if (cache_on_) {
      if (cache_.size() >= 256) cache_.clear();  // bounded; decode/prefill shapes are few
      CacheEntry& e = cache_[h];
      e.signature = signature_;
      e.steps = compiled;
      entry = &e;
    }
  }
  const std::vector<CallStep>& steps = entry != nullptr ? entry->steps : compiled;
  stats_.planned_steps += static_cast<int64_t>(steps.size());
  execute(steps);
  ncalls_ = 0;
  signature_.clear();
  return Status::Ok();
}

// --- sync points -------------------------------------------------------------------

Result<std::shared_ptr<Storage>> RecordingDevice::allocate(size_t bytes) {
  (void)flush();
  return inner_.allocate(bytes);
}

void RecordingDevice::copy(void* dst, const void* src, size_t bytes) {
  (void)flush();
  inner_.copy(dst, src, bytes);
}

void RecordingDevice::synchronize() {
  (void)flush();
  inner_.synchronize();
}

Result<Tensor> RecordingDevice::upload(const Tensor& host) {
  (void)flush();
  return inner_.upload(host);
}

void RecordingDevice::download(const TensorView& src, std::span<float> dst) {
  (void)flush();
  if (inner_.host_accessible()) {
    // The inner device's memory is host memory: copy the rows directly.
    const int64_t rows = src.rank() > 1 ? src.dim(0) : 1, cols = src.dim(src.rank() - 1);
    for (int64_t r = 0; r < rows; ++r) {
      const auto* row = reinterpret_cast<const float*>(static_cast<const std::byte*>(src.data()) +
                                                       (src.rank() > 1 ? r * src.stride(0) : 0));
      std::copy(row, row + cols, dst.data() + r * cols);
    }
    return;
  }
  inner_.download(src, dst);
}

void RecordingDevice::set_kernel_plan(const KernelPlan& plan) {
  if (mode_ == Mode::kTrace) {
    inner_.set_kernel_plan(plan);
    applied_ = nullptr;
  }
  if (!(*plan_ == plan)) plan_ = std::make_shared<KernelPlan>(plan);
}

// --- ops ---------------------------------------------------------------------------

void RecordingDevice::embedding(const TensorView& table, std::span<const int32_t> ids, const TensorView& out) {
  Call& c = next_call(static_cast<int>(Call::M::kEmbedding));
  c.a = table;
  c.b = out;
  c.ints0.assign(ids.begin(), ids.end());
  submit(c);
}

void RecordingDevice::matmul(const TensorView& x, const TensorView& w, const TensorView* bias, const TensorView& y) {
  Call& c = next_call(static_cast<int>(Call::M::kMatmul));
  c.a = x;
  c.b = w;
  if (bias) {
    c.c = *bias;
    c.has_c = true;
  }
  c.d = y;
  submit(c);
}

void RecordingDevice::matmul_many(std::span<const MatmulJob> jobs) {
  Call& c = next_call(static_cast<int>(Call::M::kMatmulMany));
  c.biases.reserve(jobs.size());
  for (const MatmulJob& j : jobs) c.biases.push_back(j.bias ? *j.bias : TensorView());
  c.jobs.assign(jobs.begin(), jobs.end());
  for (size_t i = 0; i < jobs.size(); ++i) c.jobs[i].bias = jobs[i].bias ? &c.biases[i] : nullptr;
  submit(c);
}

void RecordingDevice::rms_norm(const TensorView& x, const TensorView& weight, float eps, const TensorView& y) {
  Call& c = next_call(static_cast<int>(Call::M::kRmsNorm));
  c.a = x;
  c.b = weight;
  c.f = eps;
  c.d = y;
  submit(c);
}

void RecordingDevice::layer_norm(const TensorView& x, const TensorView& weight, const TensorView* bias, float eps,
                                 const TensorView& y) {
  Call& c = next_call(static_cast<int>(Call::M::kLayerNorm));
  c.a = x;
  c.b = weight;
  if (bias) {
    c.c = *bias;
    c.has_c = true;
  }
  c.f = eps;
  c.d = y;
  submit(c);
}

void RecordingDevice::rope(const TensorView& x, int32_t num_heads, int32_t head_dim, std::span<const int32_t> positions,
                           const RopeConfig& rope, const float* freq_factors) {
  Call& c = next_call(static_cast<int>(Call::M::kRope));
  c.a = x;
  c.i0 = num_heads;
  c.i1 = head_dim;
  c.ints0.assign(positions.begin(), positions.end());
  c.rope = rope;
  c.freq = freq_factors;
  submit(c);
}

void RecordingDevice::kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> positions,
                               std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) {
  Call& c = next_call(static_cast<int>(Call::M::kKvStore));
  c.a = k;
  c.b = v;
  c.ints0.assign(positions.begin(), positions.end());
  c.ints1.assign(row_seq.begin(), row_seq.end());
  c.copy_kv(kv);
  submit(c);
}

void RecordingDevice::attention(const AttentionParams& p) {
  Call& c = next_call(static_cast<int>(Call::M::kAttention));
  c.ints0.assign(p.positions.begin(), p.positions.end());
  c.ints1.assign(p.row_seq.begin(), p.row_seq.end());
  c.copy_kv(p.kv);
  c.ap = p;
  c.ap.positions = c.ints0;
  c.ap.row_seq = c.ints1;
  c.ap.kv = c.kv;
  submit(c);
}

void RecordingDevice::act_mul(Activation act, const TensorView& gate, const TensorView& up, const TensorView& out) {
  Call& c = next_call(static_cast<int>(Call::M::kActMul));
  c.act = act;
  c.a = gate;
  c.b = up;
  c.d = out;
  submit(c);
}

void RecordingDevice::activation(Activation act, const TensorView& x, const TensorView& out) {
  Call& c = next_call(static_cast<int>(Call::M::kActivation));
  c.act = act;
  c.a = x;
  c.d = out;
  submit(c);
}

void RecordingDevice::add(const TensorView& a, const TensorView& b, const TensorView& y) {
  Call& c = next_call(static_cast<int>(Call::M::kAdd));
  c.a = a;
  c.b = b;
  c.d = y;
  submit(c);
}

void RecordingDevice::scale(const TensorView& x, float s) {
  Call& c = next_call(static_cast<int>(Call::M::kScale));
  c.a = x;
  c.f = s;
  submit(c);
}

void RecordingDevice::softcap(const TensorView& x, float cap) {
  Call& c = next_call(static_cast<int>(Call::M::kSoftcap));
  c.a = x;
  c.f = cap;
  submit(c);
}

void RecordingDevice::fill(const TensorView& x, float value) {
  Call& c = next_call(static_cast<int>(Call::M::kFill));
  c.a = x;
  c.f = value;
  submit(c);
}

void RecordingDevice::gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) {
  Call& c = next_call(static_cast<int>(Call::M::kGather));
  c.a = src;
  c.d = dst;
  c.ints0.assign(rows.begin(), rows.end());
  submit(c);
}

void RecordingDevice::scatter_add_rows(const TensorView& src, std::span<const int32_t> rows,
                                       std::span<const float> weights, const TensorView& dst) {
  Call& c = next_call(static_cast<int>(Call::M::kScatterAdd));
  c.a = src;
  c.d = dst;
  c.ints0.assign(rows.begin(), rows.end());
  c.floats.assign(weights.begin(), weights.end());
  submit(c);
}

}  // namespace dynacore::ir
