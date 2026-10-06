#include "dynacore/ir/executor.h"

#include <cstring>
#include <random>

#include "dynacore/attention/paged_kv.h"

namespace dynacore::ir {

struct GraphExecutor::KvState {
  KvGeometry geom;
  std::shared_ptr<Storage> storage;
  std::vector<int32_t> table;
  KvLayerView view;
};

namespace {

bool executable(OpKind k) {
  switch (k) {
    case OpKind::kInput: case OpKind::kWeight: case OpKind::kKvCache: case OpKind::kConstant:
    case OpKind::kMatMul: case OpKind::kQuantizedMatMul: case OpKind::kEmbedding: case OpKind::kRmsNorm:
    case OpKind::kLayerNorm: case OpKind::kRope: case OpKind::kKvWrite: case OpKind::kAttention: case OpKind::kGqa:
    case OpKind::kActivation: case OpKind::kActMul: case OpKind::kAdd: case OpKind::kScale: case OpKind::kSoftcap:
    case OpKind::kFill: case OpKind::kGather: case OpKind::kScatterAdd:
      return true;
    default:
      return false;
  }
}

Activation parse_act(const std::string& s) {
  if (s == "gelu") return Activation::kGelu;
  if (s == "gelu_tanh") return Activation::kGeluTanh;
  return Activation::kSilu;
}

}  // namespace

GraphExecutor::GraphExecutor(const Graph& g, Device& d) : g_(g), dev_(d) {}
GraphExecutor::~GraphExecutor() = default;

Result<std::unique_ptr<GraphExecutor>> GraphExecutor::create(const Graph& g, Device& device,
                                                             const ExecutorOptions& opts) {
  for (const Op& op : g.ops()) {
    if (!executable(op.kind)) {
      return Unsupported("executor: op '" + std::string(op_name(op.kind)) +
                         "' has no device lowering yet (supported: matmul, qmatmul, embedding, norms, rope, kv_write, "
                         "attention, gqa, activations, add, scale, softcap, fill, gather, scatter_add)");
    }
  }
  std::unique_ptr<GraphExecutor> e(new GraphExecutor(g, device));
  ENGINE_RETURN_IF_ERROR(e->allocate(opts));
  return e;
}

Status GraphExecutor::allocate(const ExecutorOptions& opts) {
  symbols_ = opts.symbols;
  auto size = [&](const Dim& d) -> int64_t {
    if (d.known()) return d.size;
    auto it = symbols_.find(d.sym);
    return it == symbols_.end() ? 1 : it->second;
  };
  // Roles of index values, from the ops that read them.
  std::map<ValueId, char> role;  // 'p' positions, 's' row->sequence, 'r' rows / ids
  for (const Op& op : g_.ops()) {
    switch (op.kind) {
      case OpKind::kRope: role[op.inputs[1]] = 'p'; break;
      case OpKind::kKvWrite: role[op.inputs[3]] = 'p'; role[op.inputs[4]] = 's'; break;
      case OpKind::kAttention: case OpKind::kGqa: role[op.inputs[2]] = 'p'; role[op.inputs[3]] = 's'; break;
      default: break;
    }
  }
  std::mt19937 rng(opts.seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::map<int32_t, TensorView> buffer_view;  // first view per buffer (in-place results reuse it)

  for (const Value& v : g_.values()) {
    const Op& producer = g_.op(v.producer);
    if (v.type.kind == ValueKind::kIndex || (producer.kind == OpKind::kConstant && v.type.kind == ValueKind::kTensor)) {
      if (const ConstData* c = g_.const_data(v.id)) {
        if (!c->i32.empty()) index_[v.id] = c->i32;
        if (!c->f32.empty()) floats_[v.id] = c->f32;
        continue;
      }
      std::vector<int32_t> idx(static_cast<size_t>(size(v.type.shape.at(0))));
      const char r = role.count(v.id) ? role[v.id] : 'r';
      for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = r == 'p' ? opts.context + static_cast<int32_t>(i) : r == 's' ? 0 : static_cast<int32_t>(i);
      }
      index_[v.id] = std::move(idx);
      continue;
    }
    if (v.type.kind == ValueKind::kKv) {
      if (kv_.count(v.buffer)) continue;  // kv_write results share the cache
      auto st = std::make_unique<KvState>();
      const Shape& s = v.type.shape;
      st->geom = KvGeometry{1, static_cast<int32_t>(size(s[1])), static_cast<int32_t>(size(s[3])),
                            static_cast<int32_t>(size(s[3])), static_cast<int32_t>(size(s[2])),
                            static_cast<int32_t>(size(s[0])), v.type.dtype};
      const int64_t bytes = st->geom.total_bytes();
      ENGINE_ASSIGN_OR_RETURN(st->storage, dev_.allocate(static_cast<size_t>(bytes)));
      std::memset(st->storage->data(), 0, static_cast<size_t>(bytes));
      st->table.resize(static_cast<size_t>(st->geom.num_blocks));
      for (size_t i = 0; i < st->table.size(); ++i) st->table[i] = static_cast<int32_t>(i);
      auto* base = static_cast<std::byte*>(st->storage->data());
      st->view.k = base;
      st->view.v = base + (st->geom.k_block_elems() * st->geom.num_blocks * dtype_block_bytes(st->geom.dtype));
      st->view.geom = &st->geom;
      st->view.block_table = st->table;
      if (opts.context + 64 > st->geom.num_blocks * st->geom.block_size) {
        return InvalidArgument("executor: KV cache of " + std::to_string(st->geom.num_blocks * st->geom.block_size) +
                               " tokens is too small for context " + std::to_string(opts.context));
      }
      kv_[v.buffer] = std::move(st);
      continue;
    }
    // Tensors and weights: one allocation per buffer; in-place results alias it.
    if (auto it = buffer_view.find(v.buffer); it != buffer_view.end()) {
      views_[v.id] = it->second;
      continue;
    }
    std::vector<int64_t> dims;
    for (const Dim& d : v.type.shape) dims.push_back(size(d));
    ENGINE_ASSIGN_OR_RETURN(TensorShape shape, TensorShape::from(dims));
    ENGINE_ASSIGN_OR_RETURN(TensorLayout layout, TensorLayout::contiguous(v.type.dtype, shape));
    ENGINE_ASSIGN_OR_RETURN(auto st, dev_.allocate(static_cast<size_t>(std::max<int64_t>(64, layout.span_bytes()))));
    auto* p = static_cast<std::byte*>(st->data());
    const auto bytes = static_cast<size_t>(layout.span_bytes());
    if (v.type.kind == ValueKind::kWeight) {
      if (v.type.dtype == DType::kF32) {
        for (size_t i = 0; i < bytes / 4; ++i) reinterpret_cast<float*>(p)[i] = 0.01f;
      } else {
        std::memset(p, 0x11, bytes);  // fp16 0x1111 ~ 6e-5 scales: finite, no NaN
      }
    } else if (v.type.dtype == DType::kF32) {
      for (size_t i = 0; i < bytes / 4; ++i) reinterpret_cast<float*>(p)[i] = 0.1f * nd(rng);
    } else {
      std::memset(p, 0, bytes);
    }
    const TensorView tv(p, layout, dev_.device());
    storage_.push_back(std::move(st));
    buffer_view[v.buffer] = tv;
    views_[v.id] = tv;
  }
  return Status::Ok();
}

Status GraphExecutor::bind(ValueId v, const TensorView& view) {
  const Value& val = g_.value(v);
  if (view.dtype() != val.type.dtype || view.rank() != val.type.rank()) {
    return InvalidArgument("executor: binding for %" + std::to_string(v) + " has a different dtype or rank");
  }
  for (const Value& other : g_.values()) {
    if (other.buffer == val.buffer && views_.count(other.id)) views_[other.id] = view;
  }
  return Status::Ok();
}

TensorView GraphExecutor::view(ValueId v) const {
  auto it = views_.find(v);
  return it == views_.end() ? TensorView() : it->second;
}

Status GraphExecutor::run_op(int32_t i) {
  const Op& op = g_.op(i);
  auto in = [&](size_t k) { return views_.at(op.inputs[k]); };
  auto idx = [&](size_t k) -> std::span<const int32_t> { return index_.at(op.inputs[k]); };
  const TensorView out = views_.count(op.result) ? views_.at(op.result) : TensorView();
  switch (op.kind) {
    case OpKind::kInput: case OpKind::kWeight: case OpKind::kKvCache: case OpKind::kConstant:
      return Status::Ok();
    case OpKind::kMatMul:
    case OpKind::kQuantizedMatMul: {
      const TensorView bias = op.inputs.size() > 2 ? in(2) : TensorView();
      dev_.matmul(in(0), in(1), op.inputs.size() > 2 ? &bias : nullptr, out);
      return Status::Ok();
    }
    case OpKind::kEmbedding: dev_.embedding(in(0), idx(1), out); return Status::Ok();
    case OpKind::kRmsNorm: dev_.rms_norm(in(0), in(1), static_cast<float>(op.float_attr("eps")), out); return Status::Ok();
    case OpKind::kLayerNorm: {
      const TensorView b = op.inputs.size() > 2 ? in(2) : TensorView();
      dev_.layer_norm(in(0), in(1), op.inputs.size() > 2 ? &b : nullptr, static_cast<float>(op.float_attr("eps")), out);
      return Status::Ok();
    }
    case OpKind::kRope: {
      RopeConfig rc;
      const std::string style = op.str_attr("style", "half_split");
      rc.style = style == "interleaved" ? RopeStyle::kInterleaved : style == "none" ? RopeStyle::kNone : RopeStyle::kHalfSplit;
      rc.dim = static_cast<int32_t>(op.int_attr("dim", op.int_attr("head_dim")));
      rc.freq_base = static_cast<float>(op.float_attr("base", 10000.0));
      dev_.rope(in(0), static_cast<int32_t>(op.int_attr("heads")), static_cast<int32_t>(op.int_attr("head_dim")), idx(1),
                rc, nullptr);
      return Status::Ok();
    }
    case OpKind::kKvWrite: {
      const KvState& kv = *kv_.at(g_.value(op.inputs[0]).buffer);
      dev_.kv_store(in(1), in(2), idx(3), idx(4), std::span<const KvLayerView>(&kv.view, 1));
      return Status::Ok();
    }
    case OpKind::kAttention:
    case OpKind::kGqa: {
      const KvState& kv = *kv_.at(g_.value(op.inputs[1]).buffer);
      AttentionParams p;
      p.q = in(0);
      p.out = out;
      p.positions = idx(2);
      p.row_seq = idx(3);
      p.kv = std::span<const KvLayerView>(&kv.view, 1);
      p.num_heads = static_cast<int32_t>(op.int_attr("heads"));
      p.scale = static_cast<float>(op.float_attr("scale", 1.0));
      p.softcap = static_cast<float>(op.float_attr("softcap", 0.0));
      p.sliding_window = static_cast<int32_t>(op.int_attr("window", 0));
      dev_.attention(p);
      return Status::Ok();
    }
    case OpKind::kActivation: dev_.activation(parse_act(op.str_attr("act")), in(0), out); return Status::Ok();
    case OpKind::kActMul: dev_.act_mul(parse_act(op.str_attr("act")), in(0), in(1), out); return Status::Ok();
    case OpKind::kAdd: dev_.add(in(0), in(1), out); return Status::Ok();
    case OpKind::kScale: dev_.scale(in(0), static_cast<float>(op.float_attr("s", 1.0))); return Status::Ok();
    case OpKind::kSoftcap: dev_.softcap(in(0), static_cast<float>(op.float_attr("cap", 1.0))); return Status::Ok();
    case OpKind::kFill: dev_.fill(in(0), static_cast<float>(op.float_attr("value", 0.0))); return Status::Ok();
    case OpKind::kGather: dev_.gather_rows(in(0), idx(1), out); return Status::Ok();
    case OpKind::kScatterAdd:
      dev_.scatter_add_rows(in(1), idx(2), floats_.at(op.inputs[3]), in(0));
      return Status::Ok();
    default:
      return Unsupported("executor: " + std::string(op_name(op.kind)));
  }
}

Status GraphExecutor::run(const ExecPlan* plan) {
  if (plan == nullptr) {
    for (int32_t i = 0; i < static_cast<int32_t>(g_.ops().size()); ++i) ENGINE_RETURN_IF_ERROR(run_op(i));
    return Status::Ok();
  }
  std::vector<Device::MatmulJob> jobs;
  std::vector<TensorView> biases;
  for (const ExecStep& s : plan->steps) {
    switch (s.kind) {
      case ExecStep::Kind::kOp:
        for (int32_t op : s.ops) ENGINE_RETURN_IF_ERROR(run_op(op));
        break;
      case ExecStep::Kind::kMatmulGroup: {
        jobs.clear();
        biases.assign(s.ops.size(), TensorView());
        for (size_t j = 0; j < s.ops.size(); ++j) {
          const Op& op = g_.op(s.ops[j]);
          if (op.inputs.size() > 2) biases[j] = views_.at(op.inputs[2]);
          jobs.push_back({views_.at(op.inputs[0]), views_.at(op.inputs[1]), views_.at(op.result),
                          op.inputs.size() > 2 ? &biases[j] : nullptr});
        }
        dev_.matmul_many(jobs);
        break;
      }
      case ExecStep::Kind::kGatedMatmul: {
        const Op& gate = g_.op(s.ops[0]);
        const Op& up = g_.op(s.ops[1]);
        const Op& act = g_.op(s.ops[2]);
        dev_.matmul_gated(parse_act(act.str_attr("act")), views_.at(gate.inputs[0]), views_.at(gate.inputs[1]),
                          views_.at(up.inputs[1]), views_.at(act.result), views_.at(gate.result), views_.at(up.result));
        break;
      }
    }
  }
  return Status::Ok();
}

}  // namespace dynacore::ir
