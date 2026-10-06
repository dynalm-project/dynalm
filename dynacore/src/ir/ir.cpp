#include "dynacore/ir/ir.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace dynacore::ir {

// --- types -------------------------------------------------------------------

int64_t Type::numel() const {
  int64_t n = 1;
  for (const Dim& d : shape) {
    if (!d.known()) return -1;
    n *= d.size;
  }
  return n;
}

int64_t Type::bytes() const {
  const int64_t n = numel();
  if (n < 0) return -1;
  if (kind == ValueKind::kIndex) return n * 4;
  const int64_t be = dtype_block_elems(dtype);
  if (n % be != 0) return -1;
  return n / be * dtype_block_bytes(dtype);
}

Type tensor_type(DType dtype, Shape shape) { return Type{ValueKind::kTensor, dtype, std::move(shape), {}, {}}; }

Type weight_type(DType dtype, Shape shape) {
  Type t{ValueKind::kWeight, dtype, std::move(shape), {}, {}};
  if (dtype_is_quantized(dtype)) t.layout = Layout{LayoutKind::kBlocked, dtype_block_elems(dtype)};
  return t;
}

Type index_type(Shape shape) { return Type{ValueKind::kIndex, DType::kI32, std::move(shape), {}, {}}; }

Type kv_type(DType dtype, int64_t blocks, int64_t kv_heads, int64_t block_size, int64_t head_dim) {
  return Type{ValueKind::kKv,
              dtype,
              {Dim::of(blocks), Dim::of(kv_heads), Dim::of(block_size), Dim::of(head_dim)},
              Layout{LayoutKind::kPaged, block_size},
              {}};
}

std::string_view value_kind_name(ValueKind k) {
  switch (k) {
    case ValueKind::kTensor: return "tensor";
    case ValueKind::kWeight: return "weight";
    case ValueKind::kKv: return "kv";
    case ValueKind::kIndex: return "index";
  }
  return "tensor";
}

std::string_view layout_kind_name(LayoutKind k) {
  switch (k) {
    case LayoutKind::kRowMajor: return "row_major";
    case LayoutKind::kColMajor: return "col_major";
    case LayoutKind::kStrided: return "strided";
    case LayoutKind::kBlocked: return "blocked";
    case LayoutKind::kPacked: return "packed";
    case LayoutKind::kPaged: return "paged";
  }
  return "row_major";
}

std::string type_to_string(const Type& t) {
  std::string s(value_kind_name(t.kind));
  s += "<";
  s += dtype_name(t.dtype);
  s += ">[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i) s += ",";
    s += t.shape[i].known() ? std::to_string(t.shape[i].size) : t.shape[i].sym;
  }
  s += "]";
  // Row-major is the default and is not printed; blocked weights print their
  // block so the quantization granularity is visible.
  if (t.layout.kind != LayoutKind::kRowMajor) {
    s += " ";
    s += layout_kind_name(t.layout.kind);
    if (t.layout.param != 0) s += "(" + std::to_string(t.layout.param) + ")";
  }
  if (t.device != DeviceType::kCpu) s += " @gpu";
  return s;
}

// --- ops -----------------------------------------------------------------------

namespace {

constexpr std::string_view kOpNames[] = {
    "input",   "weight",   "kv_cache", "constant",   "matmul",  "qmatmul", "embedding", "kv_write",
    "kv_read", "attention", "gqa",     "rope",       "rmsnorm", "layernorm", "softmax", "activation",
    "act_mul", "add",       "mul",     "scale",      "softcap", "fill",    "gather",    "scatter_add",
    "transpose", "reshape", "cast",    "reduce",     "broadcast",
};
static_assert(std::size(kOpNames) == static_cast<size_t>(OpKind::kCount), "op name table");

bool dims_compatible(const Dim& a, const Dim& b) {
  if (a.known() && b.known()) return a.size == b.size;
  if (!a.known() && !b.known()) return a.sym == b.sym;
  return true;  // a symbol may take any size
}

std::string dim_str(const Dim& d) { return d.known() ? std::to_string(d.size) : d.sym; }

Status shape_error(std::string_view op, const std::string& what) {
  return InvalidArgument(std::string(op) + ": " + what);
}

int64_t attr_int(const Attrs& a, std::string_view k, int64_t def) {
  auto it = a.find(k);
  if (it == a.end()) return def;
  if (const auto* i = std::get_if<int64_t>(&it->second)) return *i;
  if (const auto* d = std::get_if<double>(&it->second)) return static_cast<int64_t>(*d);
  return def;
}

bool is_float(DType t) { return t == DType::kF32 || t == DType::kF16 || t == DType::kBF16; }

Status expect_rank(std::string_view op, const Type& t, int64_t rank, std::string_view role) {
  if (t.rank() != rank) {
    return shape_error(op, std::string(role) + " must have rank " + std::to_string(rank) + ", got " +
                               type_to_string(t));
  }
  return Status::Ok();
}

Status expect_kind(std::string_view op, const Type& t, ValueKind k, std::string_view role) {
  if (t.kind != k) {
    return shape_error(op, std::string(role) + " must be a " + std::string(value_kind_name(k)) + ", got " +
                               type_to_string(t));
  }
  return Status::Ok();
}

Status expect_same_shape(std::string_view op, const Type& a, const Type& b) {
  bool ok = a.rank() == b.rank();
  for (size_t i = 0; ok && i < a.shape.size(); ++i) ok = dims_compatible(a.shape[i], b.shape[i]);
  if (!ok) return shape_error(op, "operand shapes differ: " + type_to_string(a) + " vs " + type_to_string(b));
  return Status::Ok();
}

Type dense_f32(Shape s) { return tensor_type(DType::kF32, std::move(s)); }

// Rows of a [M, cols] operand must equal the rows of index arrays [M].
Status expect_rows(std::string_view op, const Type& t, const Type& idx, std::string_view role) {
  if (idx.rank() != 1 || !dims_compatible(t.shape[0], idx.shape[0])) {
    return shape_error(op, std::string(role) + " must have one entry per row (" + dim_str(t.shape[0]) + "), got " +
                               type_to_string(idx));
  }
  return Status::Ok();
}

}  // namespace

std::string_view op_name(OpKind k) { return kOpNames[static_cast<size_t>(k)]; }

bool parse_op_name(std::string_view s, OpKind& out) {
  for (size_t i = 0; i < std::size(kOpNames); ++i) {
    if (kOpNames[i] == s) {
      out = static_cast<OpKind>(i);
      return true;
    }
  }
  return false;
}

int64_t Op::int_attr(std::string_view key, int64_t def) const { return attr_int(attrs, key, def); }

double Op::float_attr(std::string_view key, double def) const {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  if (const auto* d = std::get_if<double>(&it->second)) return *d;
  if (const auto* i = std::get_if<int64_t>(&it->second)) return static_cast<double>(*i);
  return def;
}

std::string Op::str_attr(std::string_view key, std::string_view def) const {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::string(def);
  if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
  return std::string(def);
}

Result<Type> infer_type(OpKind kind, std::span<const Type> in, const Attrs& attrs) {
  const std::string_view name = op_name(kind);
  auto need = [&](size_t lo, size_t hi) -> Status {
    if (in.size() < lo || in.size() > hi) {
      return shape_error(name, "expects " + std::to_string(lo) + (lo == hi ? "" : "-" + std::to_string(hi)) +
                                   " operands, got " + std::to_string(in.size()));
    }
    return Status::Ok();
  };
  switch (kind) {
    case OpKind::kInput:
    case OpKind::kWeight:
    case OpKind::kKvCache:
    case OpKind::kConstant:
      return shape_error(name, "is a leaf; its type is given, not inferred");

    case OpKind::kMatMul:
    case OpKind::kQuantizedMatMul: {
      ENGINE_RETURN_IF_ERROR(need(2, 3));
      const Type &x = in[0], &w = in[1];
      ENGINE_RETURN_IF_ERROR(expect_rank(name, x, 2, "x"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, w, 2, "weight"));
      if (!is_float(x.dtype)) return shape_error(name, "x must be f32/f16/bf16, got " + type_to_string(x));
      if (!dims_compatible(x.shape[1], w.shape[1])) {
        return shape_error(name, "inner dimensions differ: x " + type_to_string(x) + ", weight " + type_to_string(w));
      }
      if (w.quantized() && w.shape[1].known() && w.shape[1].size % dtype_block_elems(w.dtype) != 0) {
        return shape_error(name, "weight rows must hold whole " + std::string(dtype_name(w.dtype)) + " blocks (" +
                                     std::to_string(dtype_block_elems(w.dtype)) + " elements), K = " +
                                     dim_str(w.shape[1]));
      }
      if (kind == OpKind::kQuantizedMatMul && !w.quantized()) {
        return shape_error(name, "weight must be block-quantized, got " + type_to_string(w));
      }
      if (in.size() == 3 && (in[2].rank() != 1 || !dims_compatible(in[2].shape[0], w.shape[0]))) {
        return shape_error(name, "bias must be [N] = [" + dim_str(w.shape[0]) + "], got " + type_to_string(in[2]));
      }
      return dense_f32({x.shape[0], w.shape[0]});
    }

    case OpKind::kEmbedding: {
      ENGINE_RETURN_IF_ERROR(need(2, 2));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "table"));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, in[1], ValueKind::kIndex, "ids"));
      return dense_f32({in[1].shape.at(0), in[0].shape[1]});
    }

    case OpKind::kRmsNorm:
    case OpKind::kLayerNorm: {
      ENGINE_RETURN_IF_ERROR(need(2, kind == OpKind::kLayerNorm ? 3 : 2));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "x"));
      if (in[1].rank() != 1 || !dims_compatible(in[1].shape[0], in[0].shape[1])) {
        return shape_error(name, "weight must be [" + dim_str(in[0].shape[1]) + "], got " + type_to_string(in[1]));
      }
      if (!attrs.count("eps")) return shape_error(name, "needs eps");
      Type r = in[0];
      r.kind = ValueKind::kTensor;
      return r;
    }

    case OpKind::kRope: {
      ENGINE_RETURN_IF_ERROR(need(2, 2));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "x"));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, in[1], ValueKind::kIndex, "positions"));
      ENGINE_RETURN_IF_ERROR(expect_rows(name, in[0], in[1], "positions"));
      const int64_t heads = attr_int(attrs, "heads", 0), hd = attr_int(attrs, "head_dim", 0);
      if (heads <= 0 || hd <= 0) return shape_error(name, "needs heads and head_dim");
      if (in[0].shape[1].known() && in[0].shape[1].size != heads * hd) {
        return shape_error(name, "x has " + dim_str(in[0].shape[1]) + " columns, heads * head_dim = " +
                                     std::to_string(heads * hd));
      }
      if (attr_int(attrs, "dim", hd) > hd || attr_int(attrs, "dim", hd) % 2 != 0) {
        return shape_error(name, "rotated dim must be even and <= head_dim");
      }
      return in[0];
    }

    case OpKind::kKvWrite: {
      ENGINE_RETURN_IF_ERROR(need(5, 5));
      const Type &kv = in[0], &k = in[1], &v = in[2];
      ENGINE_RETURN_IF_ERROR(expect_kind(name, kv, ValueKind::kKv, "cache"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, kv, 4, "cache"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, k, 2, "k"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, v, 2, "v"));
      const Dim kvh = kv.shape[1], hd = kv.shape[3];
      const int64_t hdv = attr_int(attrs, "head_dim_v", hd.known() ? hd.size : 0);
      if (kvh.known() && hd.known() && k.shape[1].known() && k.shape[1].size != kvh.size * hd.size) {
        return shape_error(name, "k has " + dim_str(k.shape[1]) + " columns, kv_heads * head_dim = " +
                                     std::to_string(kvh.size * hd.size));
      }
      if (kvh.known() && v.shape[1].known() && v.shape[1].size != kvh.size * hdv) {
        return shape_error(name, "v has " + dim_str(v.shape[1]) + " columns, kv_heads * head_dim_v = " +
                                     std::to_string(kvh.size * hdv));
      }
      ENGINE_RETURN_IF_ERROR(expect_rows(name, k, in[3], "positions"));
      ENGINE_RETURN_IF_ERROR(expect_rows(name, k, in[4], "row_seq"));
      return kv;
    }

    case OpKind::kKvRead: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, in[0], ValueKind::kKv, "cache"));
      const int64_t tokens = attr_int(attrs, "tokens", 0);
      if (tokens <= 0) return shape_error(name, "needs tokens");
      return tensor_type(in[0].dtype, {Dim::of(tokens), Dim::of(in[0].shape[1].size * in[0].shape[3].size)});
    }

    case OpKind::kAttention:
    case OpKind::kGqa: {
      ENGINE_RETURN_IF_ERROR(need(4, 4));
      const Type &q = in[0], &kv = in[1];
      ENGINE_RETURN_IF_ERROR(expect_rank(name, q, 2, "q"));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, kv, ValueKind::kKv, "cache"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, kv, 4, "cache"));
      const int64_t heads = attr_int(attrs, "heads", 0), kv_heads = attr_int(attrs, "kv_heads", 0);
      const int64_t hd = attr_int(attrs, "head_dim", 0), hdv = attr_int(attrs, "head_dim_v", hd);
      if (heads <= 0 || kv_heads <= 0 || hd <= 0) return shape_error(name, "needs heads, kv_heads and head_dim");
      if (heads % kv_heads != 0) {
        return shape_error(name, "GQA requires query_heads % kv_heads == 0 (query_heads = " + std::to_string(heads) +
                                     ", kv_heads = " + std::to_string(kv_heads) + ")");
      }
      if (kind == OpKind::kAttention && heads != kv_heads) {
        return shape_error(name, "heads != kv_heads: this is grouped-query attention (gqa)");
      }
      if (kind == OpKind::kGqa && heads == kv_heads) {
        return shape_error(name, "heads == kv_heads: plain attention, not gqa");
      }
      if (kv.shape[1].known() && kv.shape[1].size != kv_heads) {
        return shape_error(name, "cache has " + dim_str(kv.shape[1]) + " KV heads, attribute says " +
                                     std::to_string(kv_heads));
      }
      if (kv.shape[3].known() && kv.shape[3].size != hd) {
        return shape_error(name, "cache head_dim " + dim_str(kv.shape[3]) + " != " + std::to_string(hd));
      }
      if (q.shape[1].known() && q.shape[1].size != heads * hd) {
        return shape_error(name, "q has " + dim_str(q.shape[1]) + " columns, heads * head_dim = " +
                                     std::to_string(heads * hd));
      }
      if (kv.layout.kind == LayoutKind::kPaged && attrs.count("block_size") &&
          attr_int(attrs, "block_size", 0) != kv.layout.param) {
        return shape_error(name, "block_size attribute differs from the cache's paged layout");
      }
      ENGINE_RETURN_IF_ERROR(expect_rows(name, q, in[2], "positions"));
      ENGINE_RETURN_IF_ERROR(expect_rows(name, q, in[3], "row_seq"));
      return dense_f32({q.shape[0], Dim::of(heads * hdv)});
    }

    case OpKind::kSoftmax:
    case OpKind::kActivation:
    case OpKind::kScale:
    case OpKind::kSoftcap:
    case OpKind::kFill: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      if (kind == OpKind::kActivation && !attrs.count("act")) return shape_error(name, "needs act");
      if (kind == OpKind::kScale && !attrs.count("s")) return shape_error(name, "needs s");
      if (kind == OpKind::kSoftcap && !attrs.count("cap")) return shape_error(name, "needs cap");
      Type r = in[0];
      r.kind = ValueKind::kTensor;
      return r;
    }

    case OpKind::kActMul:
    case OpKind::kAdd:
    case OpKind::kMul: {
      ENGINE_RETURN_IF_ERROR(need(2, 2));
      ENGINE_RETURN_IF_ERROR(expect_same_shape(name, in[0], in[1]));
      if (kind == OpKind::kActMul && !attrs.count("act")) return shape_error(name, "needs act");
      Type r = in[0];
      r.kind = ValueKind::kTensor;
      r.layout = Layout{};
      return r;
    }

    case OpKind::kGather: {
      ENGINE_RETURN_IF_ERROR(need(2, 2));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "src"));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, in[1], ValueKind::kIndex, "rows"));
      return dense_f32({in[1].shape.at(0), in[0].shape[1]});
    }

    case OpKind::kScatterAdd: {
      ENGINE_RETURN_IF_ERROR(need(4, 4));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "dst"));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[1], 2, "src"));
      ENGINE_RETURN_IF_ERROR(expect_kind(name, in[2], ValueKind::kIndex, "rows"));
      ENGINE_RETURN_IF_ERROR(expect_rows(name, in[1], in[2], "rows"));
      if (!dims_compatible(in[0].shape[1], in[1].shape[1])) return shape_error(name, "dst and src widths differ");
      return in[0];
    }

    case OpKind::kTranspose: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "x"));
      Type r = in[0];
      std::swap(r.shape[0], r.shape[1]);
      r.layout = Layout{};
      return r;
    }

    case OpKind::kReshape: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      auto it = attrs.find("shape");
      const std::string* spec = it == attrs.end() ? nullptr : std::get_if<std::string>(&it->second);
      if (spec == nullptr) return shape_error(name, "needs shape=\"d0,d1,...\"");
      Shape s;
      size_t pos = 0;
      while (pos <= spec->size()) {
        const size_t comma = std::min(spec->find(',', pos), spec->size());
        const std::string part = spec->substr(pos, comma - pos);
        if (part.empty()) return shape_error(name, "empty dimension in shape");
        if (std::isdigit(static_cast<unsigned char>(part[0]))) s.push_back(Dim::of(std::stoll(part)));
        else s.push_back(Dim::symbol(part));
        pos = comma + 1;
      }
      Type r = in[0];
      r.shape = s;
      r.layout = Layout{};
      if (in[0].numel() >= 0 && r.numel() >= 0 && in[0].numel() != r.numel()) {
        return shape_error(name, "element count changes: " + type_to_string(in[0]) + " -> " + type_to_string(r));
      }
      return r;
    }

    case OpKind::kCast: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      auto it = attrs.find("dtype");
      const std::string* d = it == attrs.end() ? nullptr : std::get_if<std::string>(&it->second);
      DType to;
      if (d == nullptr || !parse_dtype(*d, to)) return shape_error(name, "needs dtype=<f32|f16|bf16|i8|...>");
      if (dtype_is_quantized(to)) return shape_error(name, "casting to a block format is quantization, not a cast");
      Type r = in[0];
      r.dtype = to;
      r.layout = Layout{};
      return r;
    }

    case OpKind::kReduce: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 2, "x"));
      return dense_f32({in[0].shape[0]});
    }

    case OpKind::kBroadcast: {
      ENGINE_RETURN_IF_ERROR(need(1, 1));
      ENGINE_RETURN_IF_ERROR(expect_rank(name, in[0], 1, "x"));
      const int64_t rows = attr_int(attrs, "rows", 0);
      if (rows <= 0) return shape_error(name, "needs rows");
      return dense_f32({Dim::of(rows), in[0].shape[0]});
    }

    case OpKind::kCount: break;
  }
  return Internal("infer_type: unknown op");
}

bool result_type_matches(const Type& inferred, const Type& actual) {
  if (inferred == actual) return true;
  Type relaxed = actual;
  if (actual.layout.kind == LayoutKind::kStrided && inferred.layout.kind == LayoutKind::kRowMajor) {
    relaxed.layout = inferred.layout;
  }
  return relaxed == inferred;
}

// --- graph ---------------------------------------------------------------------

const ConstData* Graph::const_data(ValueId id) const {
  auto it = consts_.find(id);
  return it == consts_.end() ? nullptr : &it->second;
}

ValueId Graph::add_leaf(OpKind kind, Type type, std::string name, Attrs attrs) {
  const auto id = static_cast<ValueId>(values_.size());
  values_.push_back(Value{id, std::move(type), std::move(name), new_buffer(), static_cast<int32_t>(ops_.size())});
  Op op;
  op.kind = kind;
  op.result = id;
  op.attrs = std::move(attrs);
  ops_.push_back(std::move(op));
  return id;
}

ValueId Graph::add_const(Type type, ConstData data, std::string name) {
  const ValueId id = add_leaf(OpKind::kConstant, std::move(type), std::move(name));
  consts_[id] = std::move(data);
  return id;
}

ValueId Graph::add_op(OpKind kind, std::vector<ValueId> inputs, Type result, Attrs attrs, int32_t buffer) {
  const auto id = static_cast<ValueId>(values_.size());
  values_.push_back(Value{id, std::move(result), {}, buffer >= 0 ? buffer : new_buffer(),
                          static_cast<int32_t>(ops_.size())});
  Op op;
  op.kind = kind;
  op.inputs = std::move(inputs);
  op.result = id;
  op.attrs = std::move(attrs);
  ops_.push_back(std::move(op));
  return id;
}

void Graph::set_ops(std::vector<Op> ops) {
  ops_ = std::move(ops);
  for (Value& v : values_) v.producer = -1;
  for (size_t i = 0; i < ops_.size(); ++i) {
    if (ops_[i].result != kNoValue) values_[static_cast<size_t>(ops_[i].result)].producer = static_cast<int32_t>(i);
  }
}

// --- builder ---------------------------------------------------------------------

ValueId Builder::constant(std::vector<int32_t> data, std::string name) {
  const auto n = static_cast<int64_t>(data.size());
  ConstData c;
  c.i32 = std::move(data);
  return g_.add_const(index_type({Dim::of(n)}), std::move(c), std::move(name));
}

ValueId Builder::emit(OpKind kind, std::vector<ValueId> inputs, Attrs attrs, int32_t buffer) {
  std::vector<Type> types;
  for (ValueId v : inputs) {
    if (v < 0 || v >= static_cast<ValueId>(g_.values().size())) {
      if (status_.ok()) status_ = InvalidArgument(std::string(op_name(kind)) + ": operand is not a value");
      return g_.add_op(kind, {}, tensor_type(DType::kF32, {}), std::move(attrs));
    }
    types.push_back(g_.value(v).type);
  }
  Result<Type> t = infer_type(kind, types, attrs);
  if (!t.ok() && status_.ok()) status_ = t.status();
  Type result = t.ok() ? *t : (types.empty() ? tensor_type(DType::kF32, {}) : types[0]);
  return g_.add_op(kind, std::move(inputs), std::move(result), std::move(attrs), buffer);
}

ValueId Builder::matmul(ValueId x, ValueId w, ValueId bias) {
  std::vector<ValueId> in = {x, w};
  if (bias != kNoValue) in.push_back(bias);
  return emit(OpKind::kMatMul, std::move(in), {});
}

ValueId Builder::embedding(ValueId table, ValueId ids) { return emit(OpKind::kEmbedding, {table, ids}, {}); }

ValueId Builder::rms_norm(ValueId x, ValueId w, double eps) { return emit(OpKind::kRmsNorm, {x, w}, {{"eps", eps}}); }

ValueId Builder::layer_norm(ValueId x, ValueId w, ValueId b, double eps) {
  std::vector<ValueId> in = {x, w};
  if (b != kNoValue) in.push_back(b);
  return emit(OpKind::kLayerNorm, std::move(in), {{"eps", eps}});
}

ValueId Builder::rope(ValueId x, ValueId positions, int64_t heads, int64_t head_dim, Attrs rope_attrs) {
  rope_attrs["heads"] = heads;
  rope_attrs["head_dim"] = head_dim;
  return emit(OpKind::kRope, {x, positions}, std::move(rope_attrs), g_.value(x).buffer);
}

ValueId Builder::kv_write(ValueId kv, ValueId k, ValueId v, ValueId positions, ValueId row_seq) {
  return emit(OpKind::kKvWrite, {kv, k, v, positions, row_seq}, {}, g_.value(kv).buffer);
}

ValueId Builder::attention(ValueId q, ValueId kv, ValueId positions, ValueId row_seq, int64_t heads,
                           int64_t kv_heads, int64_t head_dim, double scale, Attrs extra) {
  extra["heads"] = heads;
  extra["kv_heads"] = kv_heads;
  extra["head_dim"] = head_dim;
  extra["scale"] = scale;
  return emit(heads != kv_heads ? OpKind::kGqa : OpKind::kAttention, {q, kv, positions, row_seq}, std::move(extra));
}

ValueId Builder::activation(ValueId x, std::string act) {
  return emit(OpKind::kActivation, {x}, {{"act", std::move(act)}});
}

ValueId Builder::act_mul(ValueId gate, ValueId up, std::string act) {
  return emit(OpKind::kActMul, {gate, up}, {{"act", std::move(act)}});
}

ValueId Builder::add(ValueId a, ValueId b) { return emit(OpKind::kAdd, {a, b}, {}); }
ValueId Builder::mul(ValueId a, ValueId b) { return emit(OpKind::kMul, {a, b}, {}); }
ValueId Builder::scale(ValueId x, double s) { return emit(OpKind::kScale, {x}, {{"s", s}}, g_.value(x).buffer); }
ValueId Builder::softmax(ValueId x) { return emit(OpKind::kSoftmax, {x}, {}); }

ValueId Builder::generic(OpKind kind, std::vector<ValueId> inputs, Attrs attrs) {
  return emit(kind, std::move(inputs), std::move(attrs));
}

}  // namespace dynacore::ir
