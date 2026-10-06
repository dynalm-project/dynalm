#pragma once

// IR verifier: a graph that passes can be executed or lowered without
// further checks. Run after building, after parsing and after every pass.
//
// Checks: operands are defined before use (op order is execution order),
// every result type equals what infer_type derives from its operands
// (dtype, shape, layout, quantization block alignment, attention head
// grouping, KV cache geometry), leaves have the kind their op says, in-place
// ops really share their operand's buffer, and optionally that the graph's
// static memory fits a budget.

#include <cstdint>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/ir/ir.h"

namespace dynacore::ir {

struct VerifyOptions {
  // Bytes of weights + KV caches + live activations the target may hold;
  // 0 = do not check. Symbolic dimensions count as `symbol_size`.
  int64_t memory_budget = 0;
  int64_t symbol_size = 1;
};

// OK, or the first error with the op index and its printed form.
Status verify(const Graph& g, const VerifyOptions& opts = {});

// Every error, for tools that report all problems at once.
std::vector<std::string> verify_all(const Graph& g, const VerifyOptions& opts = {});

}  // namespace dynacore::ir
