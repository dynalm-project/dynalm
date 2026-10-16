#pragma once

// DynaKernel -> LLVM: native CPU code generation (prototype; docs/dynakernel.md
// "LLVM native backend").
//
// Two parts with different dependencies:
//
//  1. IR emission (emit_q4k_int8_llvm_ir): lowers one INT8 Q4_K matmul
//     specialization to textual LLVM IR. Plain C++; needs no LLVM to build,
//     run or test.
//  2. Native compilation (LlvmJit): parses that IR, verifies it
//     (LLVMVerifyModule), optimizes it (new pass manager, "default<O3>") and
//     compiles it with ORC LLJIT for the host, through LLVM's C API. The C API
//     library (LLVM-C.dll / libLLVM.so) is loaded at run time, never linked:
//     the build has no LLVM dependency, and without the library every compile
//     returns kUnsupported so callers keep the reference backend.
//
// Kernel family (the one the C++ generator's INT8 unit covers, codegen.h):
//   y[M, N] = x[M, K] . w[N, K]^T (+ bias[N]), x sub-scaled int8 (ActSuperQ8,
//   quantize_act_sx, DD-077), w interleaved Q4_K x 8 rows (BlockQ4_Kx8,
//   DD-078), M 1..8, K % 256 == 0, N % 8 == 0. The compiled function has the
//   GeneratedQ4KInt8Fn ABI, so GeneratedInt8Backend can execute it.
//
// Numerics: every lowering forms the same exact int32 per-half-super-block
// sums (only integer add order differs) and the same fp32 operation sequence
// as the generated AVX2 C++ kernel (explicit llvm.fma, no fast-math, no
// contraction), so its results are bit-identical to that kernel's.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "dynacore/base/status.h"
#include "dynacore/dynakernel/codegen.h"
#include "dynacore/dynakernel/compact_layout.h"

namespace dynacore::dk {

// How the integer dot product is expressed in IR.
//   kPortable: target-independent vector IR (zext/sext, mul, shuffles, add);
//              compiles for any x86-64 feature set, instruction choice is LLVM's.
//   kAvx2:     llvm.x86.avx2.pmadd.ub.sw + llvm.x86.avx2.pmadd.wd (the C++
//              kernel's maddubs/madd); needs AVX2 in the target features.
//   kAvxVnni:  llvm.x86.avx512.vpdpbusd.256 (VEX-encoded with +avxvnni);
//              needs AVX-VNNI in the target features.
enum class LlvmLowering : uint8_t { kPortable, kAvx2, kAvxVnni };
std::string_view llvm_lowering_name(LlvmLowering l);  // "portable" | "avx2" | "avxvnni"

// EXPERIMENTAL code-generation options (Phase 3, bandwidth). The defaults
// generate exactly the IR and symbols of the plain functions.
struct LlvmKernelOptions {
  // > 0: at the start of every super-block, prefetch (llvm.prefetch, read,
  // locality 3) the 19 cache lines of the weight block this many bytes ahead
  // in the packed stream. A multiple of 64, at most 1 MiB. Prefetches never
  // fault, so reading past the end of the weights is harmless.
  int32_t prefetch_bytes = 0;
  // EXPERIMENTAL (Phase 4): packed weight layout the kernel reads
  // (compact_layout.h). Same values, so the same results; fp16 layouts need
  // F16C in the target (llvm_kernel_compatible).
  PackedLayout layout = PackedLayout::kX8;
};

// e.g. "dk_llvm_q4k_sx8_avx2_m8_n1536_k1536_b0"; the call counter is
// "<symbol>_calls". Non-default options append suffixes ("_pf2432", "_lhs").
std::string llvm_kernel_symbol(const Q4KMatmulSpec& s, LlvmLowering l, const LlvmKernelOptions& o = {});

// Textual LLVM IR (opaque pointers, LLVM >= 17 syntax) defining
//   void <symbol>(ptr w, ptr x, ptr bias, ptr y, i64 y_stride, i64 g0, i64 g1)
//   i64  <symbol>_calls()   (number of calls, counted by the native code)
// Deterministic. kUnsupported for specs outside the family
// (validate_int8_spec).
Result<std::string> emit_q4k_int8_llvm_ir(const Q4KMatmulSpec& s, LlvmLowering l, const LlvmKernelOptions& o = {});

// --- runtime ----------------------------------------------------------------------------------

// The LLVM C API library, loaded once per process on first use and never
// unloaded (LLVM's global state does not support unloading). Search order:
// DYNACORE_LLVM_C (full path), then the platform default name
// (LLVM-C.dll on Windows, libLLVM.so elsewhere) on the loader's search path.
struct LlvmRuntimeInfo {
  bool available = false;
  std::string library;        // path or name that was loaded
  unsigned major = 0, minor = 0, patch = 0;
  std::string host_triple;    // LLVMGetDefaultTargetTriple
  std::string host_cpu;       // LLVMGetHostCPUName, e.g. "alderlake"
  std::string host_features;  // LLVMGetHostCPUFeatures, "+avx2,+fma,-avx512f,..."
  std::string error;          // why not available
};
const LlvmRuntimeInfo& llvm_runtime();
// OK when the library is loaded and usable (x86-64 host, LLVM >= 17, every
// required entry point present); kUnsupported naming the reason otherwise.
Status llvm_available();

// What code is generated for. cpu empty = the host CPU and features (as
// reported by LLVM). Explicit values let tests compile, e.g., for "x86-64"
// with no extra features (SSE2 baseline).
struct LlvmTarget {
  std::string cpu;
  std::string features;
};
// The target a backend uses on this machine: host CPU and features when the
// process selected the AVX2 tier (select_isa, honoring DYNACORE_ISA);
// otherwise baseline "x86-64" with no extra features, so generated code never
// uses an instruction set the process did not select.
LlvmTarget llvm_tier_target();
// Whether `features` (LLVM syntax) enables `name` ("avx2", "avxvnni", ...).
bool llvm_feature_enabled(std::string_view features, std::string_view name);
// OK when code of lowering `l` may be compiled for `resolved` (a target as
// LlvmJit::target() reports it, features listed explicitly): kAvx2 needs
// +avx2, kAvxVnni +avx2 and +avxvnni; kPortable compiles for any x86-64.
// LLVM terminates the process (fatal error, not a recoverable status) when IR
// uses target intrinsics the target lacks, so this is checked before such IR
// reaches LLVM (LlvmJit::compile_kernel does).
Status llvm_lowering_compatible(const LlvmTarget& resolved, LlvmLowering l);
// llvm_lowering_compatible plus the options' needs (fp16 layouts: +f16c).
Status llvm_kernel_compatible(const LlvmTarget& resolved, LlvmLowering l, const LlvmKernelOptions& o);

struct LlvmCompileStats {
  double parse_ms = 0;     // IR text -> module
  double verify_ms = 0;    // LLVMVerifyModule
  double optimize_ms = 0;  // default<O3>
  double codegen_ms = 0;   // instruction selection, emission and in-memory linking (LLJIT lookup)
  double total_ms = 0;
  size_t ir_bytes = 0;
};

// One ORC LLJIT instance. All code it compiled lives exactly as long as the
// LlvmJit: function addresses must not be used after destruction. compile()
// is not thread-safe (call it from one thread); compiled functions may be
// called from any thread.
class LlvmJit {
 public:
  // kUnsupported when LLVM is unavailable or the target cannot be created.
  static Result<std::unique_ptr<LlvmJit>> create(const LlvmTarget& target = llvm_tier_target());
  ~LlvmJit();
  LlvmJit(const LlvmJit&) = delete;
  LlvmJit& operator=(const LlvmJit&) = delete;

  struct Compiled {
    void* fn = nullptr;     // the kernel
    void* calls = nullptr;  // uint64_t (*)()
  };
  // Parse, verify, optimize and compile `ir`, which must define `symbol` and
  // `<symbol>_calls`. Nothing is added to the JIT unless every step succeeds;
  // a symbol may be compiled once per LlvmJit. The IR must not use target
  // intrinsics this JIT's target lacks (see llvm_lowering_compatible).
  Result<Compiled> compile(const std::string& ir, const std::string& symbol, LlvmCompileStats* stats = nullptr);
  // The safe entry for DynaKernel kernels: llvm_lowering_compatible, then
  // emit_q4k_int8_llvm_ir, then compile under llvm_kernel_symbol(s, l).
  Result<Compiled> compile_kernel(const Q4KMatmulSpec& s, LlvmLowering l, LlvmCompileStats* stats = nullptr,
                                  const LlvmKernelOptions& o = {});
  // The optimized kernel as target assembly (for inspecting instruction
  // selection), with the same compatibility check; compiles nothing into the JIT.
  Result<std::string> kernel_assembly(const Q4KMatmulSpec& s, LlvmLowering l, const LlvmKernelOptions& o = {});

  const LlvmTarget& target() const { return target_; }  // resolved cpu and features
  // LlvmJit instances alive in this process (resource-lifetime tests).
  static int live_instances();

 private:
  LlvmJit() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  LlvmTarget target_;
};

}  // namespace dynacore::dk
