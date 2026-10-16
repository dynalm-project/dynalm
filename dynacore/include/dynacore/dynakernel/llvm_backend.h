#pragma once

// LlvmInt8Backend (prototype, opt-in): DynaKernel matmuls compiled to native
// code by LLVM at run time (llvm_jit.h).
//
// prepare() lowers each verified matmul of a program to an INT8 Q4_K
// specialization, emits its LLVM IR, verifies, optimizes and JIT-compiles it
// for this CPU; run() prepares what is missing and executes. Execution reuses
// GeneratedInt8Backend unchanged: the compiled functions have the generated
// INT8 kernels' ABI, so weight packing (BlockQ4_Kx8 copies owned by the
// backend), activation quantization (DynaCore's quantize_act_sx), the
// 8-row-group work split, binding validation and the packed-source
// fingerprint checks are the existing, tested code. Only the dot-product
// kernel is LLVM-generated.
//
// Contract (as GeneratedInt8Backend): matmul, Q4_K weight prepacked on this
// backend, activation contract auto or int8, f32 accumulation, M 1..8,
// K % 256 == 0, N % 8 == 0. Requires the AVX2 tier (the quantizer is an AVX2
// kernel) and the LLVM C API library; kAvxVnni also requires AVX-VNNI.
// Anything else is kUnsupported from check()/prepare()/run() before any
// output is written; the caller keeps ReferenceBackend. Nothing in DynaCore
// or DynaLM constructs this backend.

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/dynakernel/generated_backend.h"
#include "dynacore/dynakernel/llvm_jit.h"

namespace dynacore::dk {

// Target a lowering compiles for: llvm_tier_target(), and for kAvx2 with
// AVX-VNNI disabled so LLVM cannot substitute VNNI instructions (the kAvx2
// code then also runs on AVX2 CPUs without VNNI).
LlvmTarget llvm_lowering_target(LlvmLowering l);
// OK when kernels of lowering `l` can be compiled and run here.
Status llvm_lowering_runnable(LlvmLowering l);

class LlvmInt8Backend final : public Backend {
 public:
  // `pool` must outlive the backend. The JIT is created on first prepare().
  LlvmInt8Backend(ThreadPool& pool, LlvmLowering lowering = LlvmLowering::kAvx2);
  // EXPERIMENTAL: code-generation options for every kernel this backend
  // compiles; options.layout also selects the layout prepack_weight builds.
  LlvmInt8Backend(ThreadPool& pool, LlvmLowering lowering, const LlvmKernelOptions& options);
  ~LlvmInt8Backend() override;

  TargetDesc target() const override;
  // verify + contract + availability; does not compile.
  Status check(const Program& p) const override;
  // Compiles every op's specialization that is not compiled yet. On failure
  // nothing that failed is registered and the status names the op.
  Status prepare(const Program& p);
  Status run(const Program& p, const Bindings& b) override { return run(p, b, RunOptions{}); }
  Status run(const Program& p, const Bindings& b, const RunOptions& opts);

  Status runnable() const;
  LlvmLowering lowering() const { return lowering_; }
  const LlvmKernelOptions& options() const { return options_; }

  struct CompiledKernel {
    Q4KMatmulSpec spec;
    std::string symbol;
    LlvmCompileStats stats;
  };
  const std::deque<CompiledKernel>& compiled() const { return compiled_; }
  // The executable entry for an op, or nullptr when not compiled.
  const GeneratedInt8Kernel* find(const Op& op) const { return exec_.find(op); }
  // Target assembly of an op's kernel (inspection; compiles nothing in).
  Result<std::string> assembly(const Q4KMatmulSpec& s);

  Result<PackedWeight> prepack_weight(const TensorView& w, int64_t capacity_bytes = -1) {
    return exec_.prepack_weight(w, capacity_bytes);
  }
  bool release_packed(const void* data) { return exec_.release_packed(data); }
  size_t packed_count() const { return exec_.packed_count(); }

 private:
  Status ensure_jit();

  LlvmLowering lowering_;
  LlvmKernelOptions options_;
  std::unique_ptr<LlvmJit> jit_;            // owns the compiled code: destroyed last
  std::deque<CompiledKernel> compiled_;     // stable symbol strings
  std::vector<GeneratedInt8Kernel> table_;  // what exec_ runs
  GeneratedInt8Backend exec_;
};

}  // namespace dynacore::dk
