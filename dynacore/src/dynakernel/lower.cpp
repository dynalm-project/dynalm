#include "dynacore/dynakernel/lower.h"

#include <algorithm>

#include "dynacore/ir/verifier.h"

namespace dynacore::dk {

std::string_view lower_issue_kind_name(LowerIssueKind k) {
  switch (k) {
    case LowerIssueKind::kInvalidGraph: return "invalid_graph";
    case LowerIssueKind::kUnsupportedOp: return "unsupported_op";
    case LowerIssueKind::kUnsupportedType: return "unsupported_type";
    case LowerIssueKind::kUnboundSymbol: return "unbound_symbol";
    case LowerIssueKind::kInternal: return "internal";
  }
  return "?";
}

ValueId Lowered::kernel_value(ir::ValueId g) const {
  const auto it = values.find(g);
  return it == values.end() ? kNoValue : it->second;
}

Status LowerResult::status() const {
  if (issues.empty()) return Status::Ok();
  const bool unsupported = std::all_of(issues.begin(), issues.end(), [](const LowerIssue& i) {
    return i.kind == LowerIssueKind::kUnsupportedOp || i.kind == LowerIssueKind::kUnsupportedType;
  });
  std::string m = "lower: " + issues.front().message;
  if (issues.size() > 1) m += " (+" + std::to_string(issues.size() - 1) + " more)";
  return unsupported ? Unsupported(m) : InvalidArgument(m);
}

namespace {

class Lowerer {
 public:
  Lowerer(const ir::Graph& g, const LowerOptions& o) : g_(g), o_(o) {}

  LowerResult run() {
    // The graph layer's own rules first: lowering assumes a well-formed graph.
    for (const std::string& e : ir::verify_all(g_)) issue(LowerIssueKind::kInvalidGraph, -1, ir::kNoValue, e);
    if (!r_.issues.empty()) return std::move(r_);

    for (int32_t i = 0; i < static_cast<int32_t>(g_.ops().size()); ++i) {
      const ir::Op& op = g_.op(i);
      switch (op.kind) {
        case ir::OpKind::kInput:
        case ir::OpKind::kWeight:
        case ir::OpKind::kKvCache:
        case ir::OpKind::kConstant:
          break;  // leaves: lowered on first use (unused leaves need no binding)
        case ir::OpKind::kMatMul:
        case ir::OpKind::kQuantizedMatMul:
          lower_matmul(i);
          break;
        default:
          issue(LowerIssueKind::kUnsupportedOp, i, op.result,
                label(i) + ": no DynaKernel lowering for '" + std::string(ir::op_name(op.kind)) + "' in v0");
      }
    }
    if (!r_.issues.empty()) {
      r_.lowered = Lowered{};  // all-or-nothing: no partial program
      return std::move(r_);
    }
    // The kernel verifier is the final gate; failing it here is a lowering bug.
    for (const Diagnostic& d : verify_all(r_.lowered.program)) {
      issue(LowerIssueKind::kInternal, d.op >= 0 ? graph_op_of(d.op) : -1, ir::kNoValue,
            "lowered program does not verify: " + d.message);
    }
    if (!r_.issues.empty()) r_.lowered = Lowered{};
    return std::move(r_);
  }

 private:
  void issue(LowerIssueKind k, int32_t op, ir::ValueId v, std::string m) { r_.issues.push_back({k, op, v, std::move(m)}); }

  std::string label(int32_t i) const {
    const ir::Op& op = g_.op(i);
    std::string s = "graph op " + std::to_string(i) + " (" + std::string(ir::op_name(op.kind));
    const std::string& n = g_.value(op.result).name;
    if (!n.empty()) s += " '" + n + "'";
    return s + ")";
  }

  std::string vname(ir::ValueId v) const {
    const std::string& n = g_.value(v).name;
    return "%" + std::to_string(v) + (n.empty() ? "" : " '" + n + "'");
  }

  int32_t graph_op_of(int32_t kernel_op) const {
    for (const auto& [gop, kop] : r_.lowered.ops) {
      if (kop == kernel_op) return gop;
    }
    return -1;
  }

  // Concrete size of a graph dimension, or -1 with an issue.
  int64_t dim(const ir::Dim& d, int32_t op, ir::ValueId v) {
    if (d.known()) return d.size;
    const auto it = o_.symbols.find(d.sym);
    if (it == o_.symbols.end()) {
      issue(LowerIssueKind::kUnboundSymbol, op, v,
            label(op) + ": dimension '" + d.sym + "' of " + vname(v) + " has no value (set LowerOptions::symbols)");
      return -1;
    }
    if (it->second <= 0) {
      issue(LowerIssueKind::kUnboundSymbol, op, v, label(op) + ": symbol '" + d.sym + "' is " + std::to_string(it->second));
      return -1;
    }
    return it->second;
  }

  // The exact kernel layout of a graph value, or nothing (with an issue).
  bool layout_of(ir::ValueId v, int32_t op, TensorLayout& out) {
    const ir::Type& t = g_.value(v).type;
    if (t.device != DeviceType::kCpu) {
      issue(LowerIssueKind::kUnsupportedType, op, v, label(op) + ": " + vname(v) + " is not on the CPU");
      return false;
    }
    std::vector<int64_t> dims;
    for (const ir::Dim& d : t.shape) {
      const int64_t n = dim(d, op, v);
      if (n < 0) return false;
      dims.push_back(n);
    }
    auto shape = TensorShape::from(dims);
    if (!shape.ok()) {
      issue(LowerIssueKind::kUnsupportedType, op, v, label(op) + ": " + vname(v) + ": " + shape.status().message());
      return false;
    }
    auto dense = TensorLayout::contiguous(t.dtype, *shape);
    if (!dense.ok()) {
      issue(LowerIssueKind::kUnsupportedType, op, v, label(op) + ": " + vname(v) + ": " + dense.status().message());
      return false;
    }
    out = *dense;
    switch (t.layout.kind) {
      case ir::LayoutKind::kRowMajor:
        // Dense rows; for a quantized dtype the same bytes as kBlocked (whole
        // GGUF blocks per row), which weight_type() would have written.
        return true;
      case ir::LayoutKind::kBlocked:
        // GGUF blocks along the innermost dimension: exactly the dense layout.
        if (t.quantized() && t.layout.param == dtype_block_elems(t.dtype)) return true;
        break;
      case ir::LayoutKind::kStrided:
        // Rows `param` elements apart (a column slice of a wider buffer).
        if (!t.quantized() && dims.size() == 2 && t.layout.param >= dims[1]) {
          out.strides[0] = t.layout.param * dtype_block_bytes(t.dtype);
          return true;
        }
        break;
      default:
        break;
    }
    issue(LowerIssueKind::kUnsupportedType, op, v,
          label(op) + ": " + vname(v) + " has layout " + std::string(ir::layout_kind_name(t.layout.kind)) + "(" +
              std::to_string(t.layout.param) + "), which DynaKernel v0 cannot represent");
    return false;
  }

  // Maps graph value `v` to a kernel value with `role`, creating it on first use.
  ValueId map_value(ir::ValueId v, ValueRole role, int32_t op) {
    if (const ValueId k = r_.lowered.kernel_value(v); k != kNoValue) {
      if (r_.lowered.program.value(k).role != role) {
        issue(LowerIssueKind::kUnsupportedType, op, v,
              label(op) + ": " + vname(v) + " is used as both " +
                  std::string(value_role_name(r_.lowered.program.value(k).role)) + " and " +
                  std::string(value_role_name(role)));
        return kNoValue;
      }
      return k;
    }
    TensorLayout l;
    if (!layout_of(v, op, l)) return kNoValue;
    Value kv;
    kv.name = g_.value(v).name;
    kv.role = role;
    kv.layout = l;
    const ValueId k = r_.lowered.program.add_value(std::move(kv));
    r_.lowered.values[v] = k;
    return k;
  }

  void lower_matmul(int32_t i) {
    const ir::Op& op = g_.op(i);
    const size_t before = r_.issues.size();
    const ir::ValueId x = op.inputs[0], w = op.inputs[1];
    const ir::ValueId bias = op.inputs.size() > 2 ? op.inputs[2] : ir::kNoValue;
    const ir::Type &xt = g_.value(x).type, &wt = g_.value(w).type;
    auto producer_kind = [&](ir::ValueId v) { return g_.op(g_.value(v).producer).kind; };
    auto unsupported = [&](ir::ValueId v, std::string m) {
      issue(LowerIssueKind::kUnsupportedType, i, v, label(i) + ": " + m);
    };

    // x: f32 activations (CpuDevice::matmul reads x as float).
    const bool x_is_result = producer_kind(x) == ir::OpKind::kMatMul || producer_kind(x) == ir::OpKind::kQuantizedMatMul;
    if (xt.kind != ir::ValueKind::kTensor) unsupported(x, "x " + vname(x) + " must be a tensor");
    if (xt.dtype != DType::kF32) {
      unsupported(x, "x " + vname(x) + " is " + std::string(dtype_name(xt.dtype)) +
                         "; DynaKernel matmul takes f32 activations (an f16/bf16 x needs an explicit cast op)");
    }
    if (!x_is_result && producer_kind(x) != ir::OpKind::kInput) {
      unsupported(x, "x " + vname(x) + " comes from '" + std::string(ir::op_name(producer_kind(x))) +
                         "', which is neither an input nor a lowered matmul");
    }
    // w: a weight leaf.
    if (wt.kind != ir::ValueKind::kWeight || producer_kind(w) != ir::OpKind::kWeight) {
      unsupported(w, "w " + vname(w) + " must be a weight leaf (a computed right-hand side is not a kernel weight)");
    }
    // bias: an f32 leaf [N].
    if (bias != ir::kNoValue) {
      const ir::Type& bt = g_.value(bias).type;
      const ir::OpKind bk = producer_kind(bias);
      if (bt.dtype != DType::kF32) {
        unsupported(bias, "bias " + vname(bias) + " is " + std::string(dtype_name(bt.dtype)) + "; the kernel adds f32 bias");
      }
      if (bk != ir::OpKind::kInput && bk != ir::OpKind::kWeight) {
        unsupported(bias, "bias " + vname(bias) + " must be an input or weight leaf");
      }
    }
    if (r_.issues.size() != before) return;

    const ValueId kx = map_value(x, x_is_result ? ValueRole::kResult : ValueRole::kActivation, i);
    const ValueId kw = map_value(w, ValueRole::kWeight, i);
    const ValueId kb = bias != ir::kNoValue ? map_value(bias, ValueRole::kBias, i) : kNoValue;
    const ValueId ky = map_value(op.result, ValueRole::kResult, i);
    if (r_.issues.size() != before) return;

    const TensorLayout& xl = r_.lowered.program.value(kx).layout;
    Op k;
    k.kind = OpKind::kMatMul;
    k.name = g_.value(op.result).name;
    k.inputs = {kx, kw};
    if (kb != kNoValue) k.inputs.push_back(kb);
    k.output = ky;
    MatMulAttrs a;
    a.m = xl.shape[0];
    a.k = xl.shape[1];
    a.n = r_.lowered.program.value(kw).layout.shape[0];
    a.act = o_.act;
    k.attrs = a;
    r_.lowered.ops[i] = r_.lowered.program.add_op(std::move(k));
  }

  const ir::Graph& g_;
  const LowerOptions& o_;
  LowerResult r_;
};

}  // namespace

LowerResult lower(const ir::Graph& g, const LowerOptions& opts) { return Lowerer(g, opts).run(); }

}  // namespace dynacore::dk
