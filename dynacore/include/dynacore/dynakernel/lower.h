#pragma once

// Lowering from DynaCore IR (graph level, dynacore/ir) to DynaKernel programs
// (kernel level, program.h). docs/dynakernel.md, "Lowering from graph IR".
//
// Lowering is all-or-nothing: either every op of the graph has an exact
// DynaKernel equivalent and one verified Program comes back, or nothing does
// and every reason is reported. Nothing is approximated, cast or reordered.
//
// v0 lowers matmul and quantized_matmul:
//   graph   y[M,N] = x[M,K] . w[N,K]^T (+ bias[N]), result dense f32
//   kernel  one OpKind::kMatMul per graph op, in graph order
// with these exact-representation rules (anything else is rejected):
//   - x is an f32 tensor (an input, or the result of an earlier matmul);
//   - w is a weight leaf: f32/f16/bf16 row-major, or a block-quantized dtype
//     in its GGUF block layout;
//   - bias is an f32 leaf (input or weight) of shape [N];
//   - tensors are row-major, or strided (rows `param` elements apart);
//   - symbolic dimensions are bound by LowerOptions::symbols (an unbound
//     symbol is an error, never a default);
//   - leaves other than tensor inputs and weights (KV caches, index arrays,
//     constants) may exist only if no lowered op reads them.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/dynakernel/program.h"
#include "dynacore/ir/ir.h"

namespace dynacore::dk {

struct LowerOptions {
  std::map<std::string, int64_t> symbols;  // e.g. {"M", 4}
  ActFormat act = ActFormat::kAuto;        // activation contract for every lowered matmul
};

enum class LowerIssueKind : uint8_t {
  kInvalidGraph,     // the graph does not pass ir::verify
  kUnsupportedOp,    // an op kind with no DynaKernel lowering
  kUnsupportedType,  // dtype, value kind, layout or device the kernel IR cannot represent exactly
  kUnboundSymbol,    // a symbolic dimension without a value in LowerOptions::symbols
  kInternal,         // the lowered program failed dk::verify (a lowering bug)
};
std::string_view lower_issue_kind_name(LowerIssueKind k);

struct LowerIssue {
  LowerIssueKind kind = LowerIssueKind::kUnsupportedOp;
  int32_t graph_op = -1;                 // op index in the graph, -1 if none
  ir::ValueId graph_value = ir::kNoValue;  // offending graph value, if any
  std::string message;
};

struct Lowered {
  Program program;
  // Graph value -> kernel value, for every graph value the program uses.
  std::map<ir::ValueId, ValueId> values;
  // Graph op index -> kernel op index (leaves have none).
  std::map<int32_t, int32_t> ops;

  // The kernel value for a graph value, or kNoValue.
  ValueId kernel_value(ir::ValueId g) const;
};

struct LowerResult {
  std::vector<LowerIssue> issues;  // empty on success
  Lowered lowered;                 // valid only when issues is empty

  bool ok() const { return issues.empty(); }
  // OK; kUnsupported when every issue is an unsupported op or type;
  // otherwise kInvalidArgument (invalid graph, unbound symbol, internal). The
  // message is the first issue, with a count of the rest.
  Status status() const;
};

LowerResult lower(const ir::Graph& g, const LowerOptions& opts = {});

}  // namespace dynacore::dk
