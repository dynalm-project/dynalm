#pragma once

// DynaKernel code generator, stage 0 (docs/dynakernel.md, "Generated C++
// backend"): emits C++ source for one constrained kernel family and nothing
// else. It is not a compiler for arbitrary programs and takes no source text.
//
// Family: y[M, N] = x[M, K] . w[N, K]^T (+ bias[N]) with
//   - w: Q4_K in its GGUF block layout, dense rows (dtype_row_bytes(Q4_K, K) apart);
//   - x: fp32, innermost contiguous, row stride given at run time (elements);
//   - y: fp32, innermost contiguous, row stride given at run time (elements);
//   - bias: optional fp32 [N] (part of the specialization);
//   - M in 1..8, K a positive multiple of 256, N > 0, all fixed per kernel;
//   - fp32 activations and fp32 accumulation, one weight value at a time:
//     w = (d * sc) * q - (dmin * m) in fp32, exactly the loader's
//     dequantize_q4_K arithmetic, then acc += w * x in k order.
// Each kernel computes output columns [n0, n1), so a caller can split N
// across threads. The generated code is portable scalar C++ (no intrinsics):
// a new implementation, not a wrapper around a DynaCore kernel, and not
// tuned; it makes no performance claim.
//
// Build flow (dynacore/CMakeLists.txt): the dkgen tool calls
// generate_q4k_source() at build time and writes
// <build>/dynacore/dynakernel_gen/dk_generated_q4k.cpp, which is compiled
// into the test-only library dynakernel_generated. Nothing in DynaCore or
// DynaLM links it.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/quantization/repack.h"

namespace dynacore::dk {

inline constexpr uint32_t kCodegenVersion = 2;  // 2: ISA variants (scalar, avx2)

// Instruction set a generated unit targets.
//   kScalar: portable C++, compiled with the project's baseline flags.
//   kAvx2:   AVX2 + FMA intrinsics; the unit must be compiled with the
//            per-file AVX2 flags (engine_set_isa) and its kernels may only
//            run where the AVX2 tier is selected (GeneratedBackend checks).
enum class CodegenIsa : uint8_t { kScalar, kAvx2 };
std::string_view codegen_isa_name(CodegenIsa isa);  // "scalar" | "avx2"

struct Q4KMatmulSpec {
  int32_t m = 1;
  int64_t n = 0, k = 0;
  bool bias = false;
  friend bool operator==(const Q4KMatmulSpec&, const Q4KMatmulSpec&) = default;
};

// OK if the generator supports `s` (M 1..8, K % 256 == 0, K > 0, N > 0, and
// sizes that keep the generated loops in int64 range).
Status validate_spec(const Q4KMatmulSpec& s);

// Stable C identifier of a specialization, e.g. "dk_q4k_m4_n24_k1280_b1"
// (scalar) or "dk_q4k_avx2_m4_n24_k1280_b1".
std::string kernel_symbol(const Q4KMatmulSpec& s, CodegenIsa isa = CodegenIsa::kScalar);

// ABI between generated code and its users (no other dependency).
//   w: Q4_K rows; x, y: fp32 rows `x_stride` / `y_stride` elements apart;
//   bias: [N] or nullptr; computes y[r][j] for r < M, n0 <= j < n1 <= N.
using GeneratedQ4KFn = void (*)(const void* w, const float* x, int64_t x_stride, const float* bias, float* y,
                                int64_t y_stride, int64_t n0, int64_t n1);

struct GeneratedKernel {
  Q4KMatmulSpec spec;
  const char* symbol;
  GeneratedQ4KFn fn;
  uint64_t (*calls)();  // number of fn invocations (proof of execution)
  CodegenIsa isa;       // instruction set the kernel was generated for
};

struct GeneratedSource {
  std::string text;          // one translation unit: kernels + registry
  uint64_t fingerprint = 0;  // FNV-1a of the text before the fingerprint function
};

// Deterministic: the same specs (in the same order) and kCodegenVersion give
// the same text. Rejects duplicates and invalid specs.
//
// The unit defines, with C linkage (<p> = "dk_generated_q4k" for scalar,
// "dk_generated_q4k_avx2" for AVX2):
//   const dynacore::dk::GeneratedKernel* <p>_kernels(size_t* count);
//   uint64_t <p>_fingerprint();
//   uint32_t <p>_version();
//
// AVX2 variant: per output column, each 32-value Q4_K sub-block is widened
// to 4 x 8 fp32 lanes, w = fma(d*sc, q, -(dmin*m)) (one rounding instead of
// two), and every row accumulates fma(w, x, acc) in 8 lanes; the lanes are
// summed in a fixed tree at the end. Same contract and bound as the scalar
// kernels, different summation order.
// `unit` names the registry functions (dk_generated_<unit>[_avx2]_*), so
// several generated units can link into one program.
Result<GeneratedSource> generate_q4k_source(std::span<const Q4KMatmulSpec> specs, CodegenIsa isa = CodegenIsa::kScalar,
                                            std::string_view unit = "q4k");

// The specializations the test build generates: M = 1..8 for each of several
// (N, K) shapes, with and without bias.
std::vector<Q4KMatmulSpec> q4k_test_specs();

// --- INT8 family (prototype; docs/dynakernel.md "Generated INT8 kernel") ---------------------
//
// y[M, N] = x[M, K] . w[N, K]^T (+ bias) with sub-scaled int8 activations
// (DD-077: x already quantized by quantize_act_sx, one ActSuperQ8 per 256
// values, rows K / 256 super-blocks apart) and the interleaved BlockQ4_Kx8
// weight layout (DD-078: group g holds rows 8g..8g+7, K / 256 blocks). Each
// call computes 8-row groups [g0, g1) for all M rows in one pass over the
// weights; y rows are `y_stride` floats apart. AVX2 only.
using GeneratedQ4KInt8Fn = void (*)(const BlockQ4_Kx8* w, const ActSuperQ8* x, const float* bias, float* y,
                                    int64_t y_stride, int64_t g0, int64_t g1);

struct GeneratedInt8Kernel {
  Q4KMatmulSpec spec;
  const char* symbol;
  GeneratedQ4KInt8Fn fn;
  uint64_t (*calls)();
};

// validate_spec plus N % 8 == 0 (whole 8-row groups).
Status validate_int8_spec(const Q4KMatmulSpec& s);
// e.g. "dk_q4k_sx8_avx2_m8_n1536_k1536_b0".
std::string int8_kernel_symbol(const Q4KMatmulSpec& s);
// One AVX2 unit; registry functions dk_generated_<unit>_int8_avx2_{kernels,
// fingerprint,version}. Deterministic like generate_q4k_source.
Result<GeneratedSource> generate_q4k_int8_source(std::span<const Q4KMatmulSpec> specs, std::string_view unit = "q4k");
// M = 1..8 for (N, K) in {(8,256), (16,768), (24,1280), (24,1280)+bias, (64,2304)}.
std::vector<Q4KMatmulSpec> q4k_int8_test_specs();

// The specializations the microbenchmark generates (unit "q4k_bench"):
// M = 1, 2, 4, 8 for Qwen2.5-1.5B-sized projections, N x K =
// 1536 x 1536 (attention q/o), 8960 x 1536 (FFN up/gate), 1536 x 8960 (FFN down).
std::vector<Q4KMatmulSpec> q4k_bench_specs();

}  // namespace dynacore::dk
