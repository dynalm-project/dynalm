#pragma once

// DynaKernel IR v0: a kernel-level program (docs/dynakernel.md).
//
// DynaCore IR (dynacore/ir) describes *what* a forward step computes: SSA
// values with symbolic shapes (M), inference ops (attention, rope, kv_write),
// and a kernel name string chosen by passes. DynaKernel sits one level below:
// one program is a short list of kernel invocations over concrete buffers.
// Every value has a fixed shape, byte strides (dynacore::TensorLayout, the
// same layout the kernels receive), a required alignment and a role; every op
// writes into an explicitly declared destination (no allocation, no aliasing),
// and carries the numerical contract a backend must honor: accumulation type
// and the activation format it may use.
//
// That is the information a backend needs to pick, verify, or generate an
// implementation for one shape. Nothing here depends on a CPU, ISA or kernel:
// target capabilities and kernel choices live in the backend (backend.h).
//
// v0 defines one executable op, matmul. Other kinds are declared so programs
// and diagnostics can name them, and the verifier rejects them.

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/tensor/dtype.h"
#include "dynacore/tensor/tensor.h"

namespace dynacore::dk {

// --- values ----------------------------------------------------------------------

enum class ValueRole : uint8_t {
  kActivation,  // per-call input data
  kWeight,      // read-only parameters; block-quantized dtypes keep their GGUF block layout
  kBias,        // read-only fp32 vector added to a result
  kResult,      // destination of exactly one op
};
std::string_view value_role_name(ValueRole r);

using ValueId = int32_t;
inline constexpr ValueId kNoValue = -1;

struct Value {
  ValueId id = kNoValue;
  std::string name;
  ValueRole role = ValueRole::kActivation;
  // dtype, shape and byte strides, with TensorView's rules: row-major order,
  // and the innermost dimension of a quantized dtype is packed (stride 0).
  TensorLayout layout;
  int64_t alignment = 0;  // required byte alignment of the data pointer (0 = none; else a power of two)
};

// --- operations -------------------------------------------------------------------

enum class OpKind : uint8_t {
  kMatMul,  // y[M,N] = x[M,K] . w[N,K]^T (+ bias[N]); the only executable kind in v0
  // Declared extension points (rejected by the v0 verifier):
  kQuantize,
  kDequantize,
  kBiasAdd,
  kActivation,
  kRmsNorm,
  kAttention,
  kFusedGroup,
  kCount,
};
std::string_view op_kind_name(OpKind k);

enum class AccumType : uint8_t { kF32 };
std::string_view accum_type_name(AccumType a);

// Activation format a matmul may use. kAuto lets the backend use its verified
// default dispatch for the shape (on CPU: KernelPlan, which may quantize
// activations of small row counts to int8 within the DD-053 accuracy
// contract). kF32 requires fp32 activations end to end, so results do not
// depend on which other rows share the call (batch invariance, DD-031).
enum class ActFormat : uint8_t { kAuto, kF32, kInt8, kInt16 };
std::string_view act_format_name(ActFormat a);

struct MatMulAttrs {
  int64_t m = 0, n = 0, k = 0;
  AccumType accum = AccumType::kF32;
  ActFormat act = ActFormat::kAuto;
};

// One per op kind that has attributes; std::monostate for the declared-only kinds.
using OpAttrs = std::variant<std::monostate, MatMulAttrs>;

struct Op {
  OpKind kind = OpKind::kMatMul;
  std::string name;              // optional, for diagnostics ("blk.0.attn_q")
  std::vector<ValueId> inputs;   // matmul: {x, w} or {x, w, bias}
  ValueId output = kNoValue;     // destination value (role kResult)
  OpAttrs attrs;
};

// Values and ops in execution order. Construction does not validate: verify()
// is the single gate (so malformed programs can be built and rejected).
class Program {
 public:
  ValueId add_value(Value v);  // assigns v.id
  int32_t add_op(Op op);       // returns the op index

  const std::vector<Value>& values() const { return values_; }
  const std::vector<Op>& ops() const { return ops_; }
  bool has_value(ValueId id) const { return id >= 0 && static_cast<size_t>(id) < values_.size(); }
  const Value& value(ValueId id) const { return values_[static_cast<size_t>(id)]; }

 private:
  std::vector<Value> values_;
  std::vector<Op> ops_;
};

// Deterministic text form (one line per value and op), for tests and dumps.
std::string to_text(const Program& p);

// --- verification -------------------------------------------------------------------

struct Diagnostic {
  int32_t op = -1;          // op index, -1 for a value-level problem
  ValueId value = kNoValue;  // offending value, if any
  std::string message;      // "op 1 (matmul 'q'): x has 64 columns but w has 128"
  bool unsupported = false;  // a well-formed op of a kind v0 does not define
};

// Every structural, shape, dtype, layout and dependency error (empty = valid).
// A program that verifies can be handed to any backend's check(); it is never
// repaired.
std::vector<Diagnostic> verify_all(const Program& p);
// OK, or kInvalidArgument with the first diagnostic (kUnsupported when every
// problem is an op kind v0 does not define).
Status verify(const Program& p);

// Smallest alignment a pointer to `t` data must have for the kernels' typed
// loads: the element size of scalar types, the alignment of the block's
// scale fields for quantized ones (fp16 = 2, Q8_K's float = 4).
int64_t natural_alignment(DType t);

// --- builder ----------------------------------------------------------------------

// Options of ProgramBuilder::matmul / matmul_into. At namespace scope: as a
// nested struct with default member initializers it cannot be the `= {}`
// default argument of its enclosing class's members (GCC and Clang reject it;
// CWG 1397).
struct MatMulOptions {
  ActFormat act = ActFormat::kAuto;
  AccumType accum = AccumType::kF32;
  std::string name;
};

// Typed construction of well-formed programs: dense row-major values, result
// values sized from the operands. Mistakes still go through verify() in
// finish(), which returns the verified program or the first diagnostic.
class ProgramBuilder {
 public:
  ValueId activation(std::string name, DType dtype, int64_t rows, int64_t cols, int64_t alignment = 0);
  ValueId weight(std::string name, DType dtype, int64_t rows, int64_t cols, int64_t alignment = 0);
  ValueId bias(std::string name, int64_t n);
  // Any value with an explicit layout (strided views, rank 1).
  ValueId value(std::string name, ValueRole role, TensorLayout layout, int64_t alignment = 0);

  using MatMulOptions = dk::MatMulOptions;
  // y = x . w^T (+ bias) into a new dense fp32 [M, N] result value.
  ValueId matmul(ValueId x, ValueId w, ValueId bias = kNoValue, MatMulOptions opts = {});
  // The same into an existing destination (e.g. a strided result value).
  void matmul_into(ValueId x, ValueId w, ValueId bias, ValueId y, MatMulOptions opts = {});

  Result<Program> finish();

 private:
  Program p_;
  Status status_;  // first construction error (unknown id, bad layout)
};

}  // namespace dynacore::dk
