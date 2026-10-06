#pragma once

// DynaCore IR: an inference-specific intermediate representation (DD-071).
//
// A Graph is a list of operations in execution order over SSA values. It
// describes *what* a forward step computes, in inference terms: quantized
// weights, paged KV caches, grouped-query attention, RoPE, norms. It is not
// a general compute graph. Backends decide *how* (kernels, tiles, threads).
//
// Values carry a full type: kind (activation tensor, weight, KV cache, index
// array), element type (any DType, including block-quantized formats), shape
// (static sizes or named symbols such as M), layout (row-major, strided,
// blocked, paged, ...) and device. Every op's result type follows from its
// operands and attributes (infer_types); the verifier rejects anything else.
//
// In-place device ops (rope, scale, add into an operand) become SSA: the op
// produces a new value that shares the operand's buffer (Value::buffer), so
// passes see true data flow and the executor still writes in place.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/memory/storage.h"
#include "dynacore/tensor/dtype.h"

namespace dynacore::ir {

// --- types -------------------------------------------------------------------

enum class ValueKind : uint8_t {
  kTensor,  // activations: computed per step
  kWeight,  // read-only model parameters (may be block-quantized)
  kKv,      // KV cache storage of one layer: K and V, possibly paged
  kIndex,   // small int32 host arrays: token ids, positions, row->sequence, rows
};

enum class LayoutKind : uint8_t {
  kRowMajor,  // dense, innermost dimension contiguous
  kColMajor,  // dense, outermost dimension contiguous
  kStrided,   // rows `param` elements apart (a column slice of a wider buffer)
  kBlocked,   // quantized blocks of `param` elements along the innermost dimension
  kPacked,    // reordered for a specific kernel (`param` = panel rows)
  kPaged,     // KV: fixed blocks of `param` tokens addressed through a block table
};

struct Layout {
  LayoutKind kind = LayoutKind::kRowMajor;
  int64_t param = 0;
  friend bool operator==(const Layout&, const Layout&) = default;
};

// A dimension: a static size, or a symbol bound per step (M = rows).
struct Dim {
  int64_t size = -1;  // -1 = symbolic
  std::string sym;
  static Dim of(int64_t n) { return Dim{n, {}}; }
  static Dim symbol(std::string s) { return Dim{-1, std::move(s)}; }
  bool known() const { return size >= 0; }
  friend bool operator==(const Dim&, const Dim&) = default;
};
using Shape = std::vector<Dim>;

struct Type {
  ValueKind kind = ValueKind::kTensor;
  DType dtype = DType::kF32;
  Shape shape;
  Layout layout;
  DeviceType device = DeviceType::kCpu;
  friend bool operator==(const Type&, const Type&) = default;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  bool quantized() const { return dtype_is_quantized(dtype); }
  // Elements, or -1 if any dimension is symbolic.
  int64_t numel() const;
  // Storage bytes for a dense or blocked layout, or -1 if unknown.
  int64_t bytes() const;
};

Type tensor_type(DType dtype, Shape shape);
Type weight_type(DType dtype, Shape shape);
Type index_type(Shape shape);
// One layer's KV cache: [blocks, kv_heads, block_size, head_dim], paged.
Type kv_type(DType dtype, int64_t blocks, int64_t kv_heads, int64_t block_size, int64_t head_dim);

std::string type_to_string(const Type& t);
std::string_view value_kind_name(ValueKind k);
std::string_view layout_kind_name(LayoutKind k);

// --- operations ----------------------------------------------------------------

enum class OpKind : uint8_t {
  // Leaves.
  kInput,     // activation or index array supplied by the caller
  kWeight,    // model parameter
  kKvCache,   // one layer's KV cache
  kConstant,  // small data owned by the graph (positions, block tables, ...)
  // Linear algebra.
  kMatMul,           // y[M,N] = x[M,K] . w[N,K]^T (+ bias[N]); f16/f32 weights
  kQuantizedMatMul,  // the same with a block-quantized weight (canonical form)
  kEmbedding,        // y[M,D] = table[ids[M], :]
  // Attention.
  kKvWrite,    // writes k/v rows into the cache; result = the updated cache
  kKvRead,     // a sequence's K or V as dense rows (debugging, references)
  kAttention,  // softmax(q k^T * scale) v over a KV cache, heads == kv_heads
  kGqa,        // grouped-query attention: heads % kv_heads == 0, heads > kv_heads
  kRope,       // rotary position embedding, in place
  // Normalization and elementwise.
  kRmsNorm,
  kLayerNorm,
  kSoftmax,     // row-wise
  kActivation,  // act(x)
  kActMul,      // act(gate) * up (SwiGLU / GeGLU)
  kAdd,
  kMul,         // elementwise product
  kScale,       // x * s
  kSoftcap,     // cap * tanh(x / cap)
  kFill,
  // Data movement.
  kGather,      // dst[i] = src[rows[i]]
  kScatterAdd,  // dst[rows[i]] += w[i] * src[i]
  kTranspose,
  kReshape,
  kCast,
  kReduce,     // sum over the last dimension
  kBroadcast,  // row vector to [M, N]
  kCount,
};

std::string_view op_name(OpKind k);
bool parse_op_name(std::string_view s, OpKind& out);

// Attribute values: integers, floats, strings (enums print as words).
using Attr = std::variant<int64_t, double, std::string>;
using Attrs = std::map<std::string, Attr, std::less<>>;

using ValueId = int32_t;
constexpr ValueId kNoValue = -1;

struct Value {
  ValueId id = kNoValue;
  Type type;
  std::string name;   // optional debug name ("blk.3.attn_q", "x")
  int32_t buffer = -1;  // storage identity: in-place results share their operand's buffer
  int32_t producer = -1;  // op index, -1 for none
};

struct Op {
  OpKind kind = OpKind::kInput;
  std::vector<ValueId> inputs;
  ValueId result = kNoValue;
  Attrs attrs;
  // Filled by passes.
  std::string kernel;  // selected implementation, e.g. "cpu.int8_gemv.q4_K"
  int32_t fusion_group = -1;

  int64_t int_attr(std::string_view key, int64_t def = 0) const;
  double float_attr(std::string_view key, double def = 0) const;
  std::string str_attr(std::string_view key, std::string_view def = {}) const;
  bool has(std::string_view key) const { return attrs.find(key) != attrs.end(); }
};

// Small graph-owned data (positions, block tables, ...): constants keep the
// recorded step self-contained after the caller's buffers are reused.
struct ConstData {
  std::vector<int32_t> i32;
  std::vector<float> f32;
};

class Graph {
 public:
  explicit Graph(std::string name = "graph") : name_(std::move(name)) {}

  const std::string& name() const { return name_; }
  std::span<const Value> values() const { return values_; }
  std::span<const Op> ops() const { return ops_; }
  std::span<Op> mutable_ops() { return ops_; }
  const Value& value(ValueId id) const { return values_[static_cast<size_t>(id)]; }
  Value& mutable_value(ValueId id) { return values_[static_cast<size_t>(id)]; }
  const Op& op(int32_t i) const { return ops_[static_cast<size_t>(i)]; }
  const ConstData* const_data(ValueId id) const;

  // Low-level construction (the builder wraps these). The result type is
  // given explicitly; infer_types() checks it.
  ValueId add_leaf(OpKind kind, Type type, std::string name, Attrs attrs = {});
  ValueId add_const(Type type, ConstData data, std::string name = {});
  ValueId add_op(OpKind kind, std::vector<ValueId> inputs, Type result, Attrs attrs = {}, int32_t buffer = -1);
  // Replaces the op list (passes that reorder or remove ops). Value producers
  // are recomputed.
  void set_ops(std::vector<Op> ops);

  int32_t num_buffers() const { return next_buffer_; }
  int32_t new_buffer() { return next_buffer_++; }

 private:
  std::string name_;
  std::vector<Value> values_;
  std::vector<Op> ops_;
  std::map<ValueId, ConstData> consts_;
  int32_t next_buffer_ = 0;
};

// Result type of an op from its operand types and attributes, or why the op
// is malformed. Used by the builder, the verifier and the parser.
Result<Type> infer_type(OpKind kind, std::span<const Type> inputs, const Attrs& attrs);

// Whether `actual` is a valid placement of an op result whose inferred type
// is `inferred`: identical, or the same value written into rows of a wider
// buffer (strided instead of row-major).
bool result_type_matches(const Type& inferred, const Type& actual);

// --- builder ---------------------------------------------------------------------

// Typed construction with inferred result types. Errors (shape or dtype
// mismatch, bad GQA grouping) are kept and reported by status(): building
// continues so one call site can check once.
class Builder {
 public:
  explicit Builder(Graph& g) : g_(g) {}

  ValueId input(std::string name, Type t) { return g_.add_leaf(OpKind::kInput, std::move(t), std::move(name)); }
  ValueId weight(std::string name, Type t) { return g_.add_leaf(OpKind::kWeight, std::move(t), std::move(name)); }
  ValueId kv_cache(std::string name, Type t) { return g_.add_leaf(OpKind::kKvCache, std::move(t), std::move(name)); }
  ValueId constant(std::vector<int32_t> data, std::string name = {});

  ValueId matmul(ValueId x, ValueId w, ValueId bias = kNoValue);
  ValueId embedding(ValueId table, ValueId ids);
  ValueId rms_norm(ValueId x, ValueId w, double eps);
  ValueId layer_norm(ValueId x, ValueId w, ValueId b, double eps);
  ValueId rope(ValueId x, ValueId positions, int64_t heads, int64_t head_dim, Attrs rope_attrs = {});
  ValueId kv_write(ValueId kv, ValueId k, ValueId v, ValueId positions, ValueId row_seq);
  // Attention over `kv` for query rows q[M, heads*head_dim]. kv_heads <
  // heads builds a gqa op.
  ValueId attention(ValueId q, ValueId kv, ValueId positions, ValueId row_seq, int64_t heads, int64_t kv_heads,
                    int64_t head_dim, double scale, Attrs extra = {});
  ValueId activation(ValueId x, std::string act);
  ValueId act_mul(ValueId gate, ValueId up, std::string act);
  ValueId add(ValueId a, ValueId b);
  ValueId mul(ValueId a, ValueId b);
  ValueId scale(ValueId x, double s);
  ValueId softmax(ValueId x);
  ValueId generic(OpKind kind, std::vector<ValueId> inputs, Attrs attrs = {});

  const Status& status() const { return status_; }

 private:
  ValueId emit(OpKind kind, std::vector<ValueId> inputs, Attrs attrs, int32_t buffer = -1);
  Graph& g_;
  Status status_;
};

}  // namespace dynacore::ir
