#pragma once

// The DynaCore inference language (DD-073): a small, explicit language for
// inference graphs, compiled to DynaCore IR.
//
//   graph decoder_layer(M = 1) {
//     config hidden = 1536, heads = 12, kv_heads = 2, head_dim = 128
//     input  x     : tensor<f32>[M, hidden]
//     input  pos   : index[M]
//     input  seq   : index[M]
//     weight wq    : q4_K[hidden, hidden]
//     kv     cache : kv<f16>[64, kv_heads, 16, head_dim]
//
//     q     = rope(x @ wq, pos, heads, head_dim)
//     cache = kv_write(cache, k, v, pos, seq)
//     a     = attention(q, cache, pos, seq, heads, kv_heads, head_dim)   // -> gqa
//     y     = swiglu(x @ w_gate, x @ w_up) @ w_down
//     output y
//
//     schedule { fuse gated; group shared_input }
//   }
//
// The language exposes inference semantics directly: quantized weight types,
// paged KV caches, grouped-query attention, RoPE, gated MLPs, and schedule
// permissions the optimizer honors. It has no loops, classes or general
// control flow: a program is a dataflow graph, and the compiler (not the
// author) chooses kernels, groups and fused steps. Syntax: docs/dynacore-language.md.

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/ir/ir.h"
#include "dynacore/ir/passes.h"

namespace dynacore::lang {

struct Program {
  ir::Graph graph;
  std::vector<ir::ValueId> outputs;
  std::map<std::string, int64_t> symbols;  // graph parameters with their defaults (M = 1)
  ir::FusionOptions fusion;                // from the schedule block
};

// Compiles source text to IR. Errors carry "file:line:col: message".
// `graph_name` picks one graph when the file holds several (empty = first).
Result<Program> compile(std::string_view source, std::string_view file = "<input>",
                        std::string_view graph_name = {});

}  // namespace dynacore::lang
