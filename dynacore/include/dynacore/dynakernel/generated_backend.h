#pragma once

// GeneratedBackend: runs DynaKernel programs on kernels emitted by the code
// generator (codegen.h). It owns no Device and calls no DynaCore matmul
// kernel: an op either has a generated specialization and runs on it, or
// check() rejects it (kUnsupported) before anything runs. The reference
// backend stays the correctness oracle and fallback, chosen by the caller.
//
// A backend runs one table, scalar or AVX2 (codegen.h); an AVX2 table runs
// only where the AVX2 tier is selected (runnable()).
//
// Supported per op (in addition to dk::verify):
//   - matmul with a Q4_K weight, dense rows (GGUF layout, not prepacked);
//   - x and y fp32 with contiguous rows (any row stride that is whole
//     elements; the verifier guarantees innermost contiguity);
//   - an exact (M, N, K, bias) specialization in the kernel table;
//   - activation contract f32 or auto: the kernels keep fp32 activations,
//     which satisfies both (auto permits reduced precision, never requires it).

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "dynacore/dynakernel/backend.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/compact_layout.h"
#include "dynacore/execution/thread_pool.h"

namespace dynacore::dk {

class GeneratedBackend final : public Backend {
 public:
  // `pool` and the kernel table (a generated unit's dk_generated_q4k_kernels)
  // must outlive the backend.
  GeneratedBackend(ThreadPool& pool, std::span<const GeneratedKernel> kernels);

  TargetDesc target() const override;
  Status check(const Program& p) const override;
  Status run(const Program& p, const Bindings& b) override;

  // The specialization an op runs on, or nullptr.
  const GeneratedKernel* find(const Op& op) const;
  // OK when this table may run in this process: one ISA variant, and for
  // AVX2 kernels the AVX2 tier selected (select_isa). check() and run() fail
  // with this status otherwise; nothing falls back silently.
  Status runnable() const;

 private:
  ThreadPool& pool_;
  std::span<const GeneratedKernel> kernels_;
  CodegenIsa isa_ = CodegenIsa::kScalar;
  bool mixed_ = false;
};

// GeneratedInt8Backend (prototype): matmuls with sub-scaled int8 activations
// on the interleaved Q4_K layout, computed by generated INT8 kernels
// (codegen.h, generate_q4k_int8_source).
//
//   - Activations are quantized per call with DynaCore's quantize_act_sx
//     (the reference path's own quantizer, DD-077), so codes are identical;
//     the dot product is generated code only.
//   - Weights must be prepacked on this backend (prepack_weight): it builds
//     and owns a BlockQ4_Kx8 copy with repack_q4_K_x8 (DD-078), fingerprints
//     the source, and frees the copy on release_packed. An op whose bound
//     weight is not registered is rejected, never run another way.
//   - Contract auto or int8 (reduced-precision activations allowed); f32 is
//     rejected. AVX2 tier only (runnable()); elsewhere nothing runs and the
//     caller keeps the reference backend.
class GeneratedInt8Backend final : public Backend {
 public:
  GeneratedInt8Backend(ThreadPool& pool, std::span<const GeneratedInt8Kernel> kernels);

  TargetDesc target() const override;
  Status check(const Program& p) const override;
  Status run(const Program& p, const Bindings& b) override { return run(p, b, RunOptions{}); }
  // RunOptions::check_packed re-fingerprints registered sources (default on).
  Status run(const Program& p, const Bindings& b, const RunOptions& opts);

  Status runnable() const;
  const GeneratedInt8Kernel* find(const Op& op) const;
  // Replaces the kernel table (same lifetime rule as the constructor's); used
  // by LlvmInt8Backend, whose table grows as it compiles kernels. Packed
  // weights are kept.
  void set_kernels(std::span<const GeneratedInt8Kernel> kernels) { kernels_ = kernels; }
  // EXPERIMENTAL (Phase 4): the layout prepack_weight builds (default kX8,
  // BlockQ4_Kx8). The kernel table must read that layout: the generated C++
  // kernels read kX8 only; LlvmInt8Backend compiles kernels for its layout.
  // Only while nothing is packed (kInvalidArgument otherwise).
  Status set_packed_layout(PackedLayout l);
  PackedLayout packed_layout() const { return layout_; }
  Result<PackedWeight> prepack_weight(const TensorView& w, int64_t capacity_bytes = -1);
  bool release_packed(const void* data);
  size_t packed_count() const { return packed_.size(); }

 private:
  struct Packed {
    PackedWeight info;
    std::vector<uint8_t> bytes;  // N / 8 groups x K / 256 blocks of layout_
  };
  const Packed* find_packed(const TensorView& w) const;

  ThreadPool& pool_;
  std::span<const GeneratedInt8Kernel> kernels_;
  CpuKernels k_;  // AVX2 table, for quantize_act_sx only (set when runnable)
  std::vector<Packed> packed_;
  PackedLayout layout_ = PackedLayout::kX8;
  std::vector<ActBlockQ8> act_;  // quantized activations of the current op (run() is not thread-safe)
};

}  // namespace dynacore::dk
