#include "dynacore/ir/verifier.h"

#include "dynacore/ir/text.h"

namespace dynacore::ir {
namespace {

int64_t concrete_bytes(const Type& t, int64_t symbol_size) {
  Type c = t;
  for (Dim& d : c.shape) {
    if (!d.known()) d = Dim::of(symbol_size);
  }
  const int64_t b = c.bytes();
  return b < 0 ? 0 : b;
}

}  // namespace

std::vector<std::string> verify_all(const Graph& g, const VerifyOptions& opts) {
  std::vector<std::string> errors;
  std::vector<char> defined(g.values().size(), 0);
  const auto nvalues = static_cast<ValueId>(g.values().size());
  int64_t static_bytes = 0, peak_activation = 0;

  for (size_t i = 0; i < g.ops().size(); ++i) {
    const Op& op = g.op(static_cast<int32_t>(i));
    const std::string where = "op " + std::to_string(i) + " (" + std::string(op_name(op.kind)) + ")";
    if (op.result < 0 || op.result >= nvalues) {
      errors.push_back(where + ": result is not a value");
      continue;
    }
    const Value& r = g.value(op.result);
    if (defined[static_cast<size_t>(op.result)]) errors.push_back(where + ": %" + std::to_string(r.id) + " defined twice");
    bool operands_ok = true;
    std::vector<Type> types;
    for (ValueId in : op.inputs) {
      if (in < 0 || in >= nvalues || !defined[static_cast<size_t>(in)]) {
        errors.push_back(where + ": operand %" + std::to_string(in) + " is used before it is defined");
        operands_ok = false;
        continue;
      }
      types.push_back(g.value(in).type);
    }
    switch (op.kind) {
      case OpKind::kInput:
        if (r.type.kind != ValueKind::kTensor && r.type.kind != ValueKind::kIndex) {
          errors.push_back(where + ": inputs are tensors or index arrays");
        }
        break;
      case OpKind::kWeight:
        if (r.type.kind != ValueKind::kWeight) errors.push_back(where + ": weight must have a weight type");
        static_bytes += concrete_bytes(r.type, opts.symbol_size);
        break;
      case OpKind::kKvCache:
        if (r.type.kind != ValueKind::kKv || r.type.rank() != 4) {
          errors.push_back(where + ": kv_cache must be kv<dtype>[blocks, kv_heads, block_size, head_dim]");
        }
        // K and V.
        static_bytes += 2 * concrete_bytes(r.type, opts.symbol_size);
        break;
      case OpKind::kConstant: {
        const ConstData* c = g.const_data(r.id);
        const bool ints = c != nullptr && r.type.kind == ValueKind::kIndex &&
                          r.type.numel() == static_cast<int64_t>(c->i32.size());
        const bool floats = c != nullptr && r.type.kind == ValueKind::kTensor && r.type.dtype == DType::kF32 &&
                            r.type.numel() == static_cast<int64_t>(c->f32.size());
        if (!ints && !floats) errors.push_back(where + ": constant data does not match its type");
        break;
      }
      default: {
        if (!operands_ok) break;
        Result<Type> t = infer_type(op.kind, types, op.attrs);
        if (!t.ok()) {
          errors.push_back(where + ": " + t.status().message());
        } else if (!result_type_matches(*t, r.type)) {
          errors.push_back(where + ": result type " + type_to_string(r.type) + " but operands give " +
                           type_to_string(*t));
        }
        if (op_writes_in_place(op.kind) && !op.inputs.empty() && g.value(op.inputs[0]).buffer != r.buffer) {
          errors.push_back(where + ": in-place op must share its first operand's buffer");
        }
        if (r.type.kind == ValueKind::kTensor) {
          peak_activation = std::max(peak_activation, concrete_bytes(r.type, opts.symbol_size));
        }
      }
    }
    if (r.producer != static_cast<int32_t>(i)) errors.push_back(where + ": value producer index is stale");
    defined[static_cast<size_t>(op.result)] = 1;
  }
  if (opts.memory_budget > 0 && static_bytes + peak_activation > opts.memory_budget) {
    errors.push_back("graph needs about " + std::to_string((static_bytes + peak_activation) >> 20) +
                     " MiB (weights + KV + largest activation), budget is " +
                     std::to_string(opts.memory_budget >> 20) + " MiB");
  }
  return errors;
}

Status verify(const Graph& g, const VerifyOptions& opts) {
  const std::vector<std::string> errors = verify_all(g, opts);
  if (errors.empty()) return Status::Ok();
  std::string msg = "IR verification failed: " + errors.front();
  if (errors.size() > 1) msg += " (+" + std::to_string(errors.size() - 1) + " more)";
  return InvalidArgument(msg);
}

}  // namespace dynacore::ir
