#pragma once

// Text form of the DynaCore IR, for debugging, golden tests and dynacorec.
//
//   graph @decode {
//     %0 = input "x" : tensor<f32>[M,1536]
//     %1 = weight "blk.0.attn_q" : weight<q4_K>[1536,1536] blocked(256)
//     %2 = kv_cache "blk.0" : kv<f16>[64,2,16,128] paged(16)
//     %3 = constant "positions" [7] : index<i32>[1]
//     %4 = qmatmul %0, %1 : tensor<f32>[M,1536]        // kernel=cpu.int8_dot
//     %5 = gqa %4, %2, %3, %6 {heads=12, head_dim=128, kv_heads=2, scale=0.0883883} : tensor<f32>[M,1536]
//   }
//
// Values are `%name` (the printer uses value ids). Result types are printed
// for readability; the parser re-infers them and rejects a mismatch. Ops that
// write in place (rope, scale, softcap, fill, kv_write, scatter_add) share
// their first operand's buffer; `alias=N` marks any other aliasing. `//`
// starts a comment (the printer uses it for pass annotations).

#include <string>
#include <string_view>

#include "dynacore/base/status.h"
#include "dynacore/ir/ir.h"

namespace dynacore::ir {

struct PrintOptions {
  bool annotations = true;  // kernel / fusion group comments
  bool names = true;        // debug names of computed values as comments
};

std::string print_graph(const Graph& g, const PrintOptions& opts = {});
Result<Graph> parse_graph(std::string_view text);

// True for ops whose result reuses operand 0's storage by definition.
bool op_writes_in_place(OpKind k);

}  // namespace dynacore::ir
