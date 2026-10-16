#pragma once

// CPU matmul kernel registry (DynaLM Next).
//
// y[M, N] = x[M, K] . w[N, K]^T (+ bias): which CPU kernel runs is decided
// here, by select_matmul(), a pure function of the call's shape, the step's
// KernelPlan and what the backend can offer for this weight (MatmulCaps).
// CpuDevice::matmul fills the caps, asks for a selection and executes it; the
// selection order is the order of the if-chain it replaced, so default
// behaviour is unchanged (test_matmul_registry checks this against a copy of
// the old chain). Every kernel has a static descriptor (accuracy class,
// evidence, default state) and every selection can list the higher-priority
// kernels it passed over and why.
//
// The registry is backend metadata: it never touches tensor data, memory or
// threads, so it can be tested without a device or a model.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/tensor/dtype.h"

namespace dynacore {

// Priority order: earlier ids win when several are eligible.
enum class MatmulKernelId : uint8_t {
  kExternal,              // a registered external kernel (KernelPlan::external_matmul)
  kInt8PrefillX8,         // int8 sub-scaled on the interleaved Q4_K copy above decode sizes (DD-078; fails DD-053, off)
  kInt8SubscaledX8Rows8,  // int8 sub-scaled, interleaved copy, 8-row tiles (experimental, off)
  kInt8SubscaledX8,       // int8 sub-scaled on the interleaved Q4_K copy (DD-077/078)
  kInt8Subscaled,         // int8 sub-scaled row kernel (DD-077)
  kInt8Superblock,        // int8 super-block activations (DD-075; fails DD-053, off)
  kInt8Per32,             // int8 per-32-block activations (DD-053)
  kInt16SplitPrefillX8,   // split-byte int16 tiles on the interleaved Q4_K copy (DynaLM Next; opt-in candidate)
  kInt16PrefillX8,        // int16 8-row tiles on the interleaved copy (Phase 6; opt-in)
  kInt16X8,               // int16 on the interleaved copy, decode sizes (DD-081/082)
  kInt16Rows,             // int16 row kernel (DD-076)
  kF32ExpandGemm,         // fp32 expand panels + register-blocked GEMM (reference prefill)
  kF32RowDot,             // fused dequant-dot, or per-row expand + dot (reference fallback)
  kCount
};
inline constexpr size_t kMatmulKernelCount = static_cast<size_t>(MatmulKernelId::kCount);
inline constexpr uint32_t matmul_kernel_bit(MatmulKernelId id) { return uint32_t{1} << static_cast<uint32_t>(id); }

enum class MatmulAccuracy : uint8_t {
  kReference,     // fp32 activations: the reference arithmetic
  kValidated,     // passed the DD-053 contract (perplexity <= +1%, mean KL <= 0.0025) on the reference models
  kExact,         // bit-identical to a validated kernel (tested), so it inherits that validation
  kFailsDd053,    // measured outside the DD-053 contract: never default
  kExperimental,  // identical results by construction, speed unproven
  kExternal,      // supplied at run time; validation is the provider's
};
std::string_view matmul_accuracy_name(MatmulAccuracy a);

struct MatmulKernelDesc {
  MatmulKernelId id;
  std::string_view name;         // stable: env lists, /metrics labels, logs
  std::string_view weights;      // weight types it accepts
  std::string_view rows;         // row rule (documents the predicate in select_matmul)
  std::string_view isa;          // instruction-set requirement
  std::string_view packing;      // weight layout it reads
  std::string_view activations;  // activation arithmetic
  MatmulAccuracy accuracy;
  bool default_on;               // can be selected under KernelPlan::defaults()
  std::string_view evidence;     // design decision / phase reference
};
std::span<const MatmulKernelDesc> matmul_kernels();
const MatmulKernelDesc& matmul_kernel_desc(MatmulKernelId id);
std::string_view matmul_kernel_name(MatmulKernelId id);
// Kernel by name; false if unknown.
bool parse_matmul_kernel(std::string_view name, MatmulKernelId& out);
// Comma-separated kernel names -> bitmask. "all-accelerated" = every kernel
// except the two fp32 ones (the reference arithmetic). Unknown names fail.
Status parse_matmul_kernel_list(std::string_view list, uint32_t& mask);

// --- the operation contract ---
struct MatmulShape {
  int64_t m = 0, n = 0, k = 0;
  DType weight = DType::kF32;
  bool x_dense = true;  // x rows are contiguous (row stride == k floats)
};
// m, n, k > 0; k a multiple of the weight type's block; a weight type the CPU
// backend can multiply. Device and DynaIR verifier use the same check.
Status validate_matmul_shape(const MatmulShape& s);

// What the backend offers for one call: kernel entries present for this
// weight type on this tier, the interleaved copies of this weight, and the
// read-once environment switches. Filled by CpuDevice per call; tests fill it
// freely.
struct MatmulCaps {
  bool int8_accelerated = false;  // SIMD integer kernels on this tier (AVX2)
  bool packed_q4 = false;         // interleaved Q4_K copy of this weight exists
  bool packed_q6 = false;         // interleaved Q6_K copy exists
  bool q16_packed_enabled = true;  // DYNACORE_Q16_PACKED (default on)
  bool q6_packed_enabled = false;  // DYNACORE_Q6_PACKED (default off)
  // Kernel-table entries (non-null for this weight type):
  bool quantize_act = false, quantize_act_sb = false, quantize_act_sx = false, quantize_act16 = false;
  bool dot_q8_rows = false, dot_q8_sb_rows = false, dot_q8_sx_rows = false;
  bool dot_q8_sx_x8_rows8 = false;
  bool dot_q16_x8_q4 = false, dot_q16_x8_q6 = false;
  bool dot_q16_x8_q4_rows8 = false, dot_q16_x8_q6_rows8 = false;
  bool dot_q16s_x8_q4_rows8 = false, split_act16 = false;
  bool dot_q16_rows = false;
  bool external_eligible = false;  // some registered external kernel accepts this call
};

enum class MatmulReject : uint8_t {
  kDisabled,    // in KernelPlan::matmul_disabled
  kNotEnabled,  // its opt-in plan switch is off
  kRows,        // row count outside its rule
  kAlignment,   // k (or n) not a multiple it needs
  kWeightType,  // weight type not supported by it
  kNoKernel,    // the tier has no implementation
  kNoPacked,    // needs an interleaved copy this weight does not have
  kSizeGate,    // weight too small for it (N, N*K gates)
  kNotDense,    // needs contiguous x rows
};
std::string_view matmul_reject_name(MatmulReject r);

enum class MatmulActFormat : uint8_t { kF32, kInt8Per32, kInt8Superblock, kInt8Subscaled, kInt16, kInt16Split };

struct MatmulSelection {
  MatmulKernelId id = MatmulKernelId::kF32RowDot;
  MatmulActFormat act = MatmulActFormat::kF32;
  bool packed = false;  // reads the interleaved copy
  // Higher-priority kernels passed over, in priority order (filled only when
  // asked: select_matmul(..., explain = true)).
  uint8_t n_rejected = 0;
  std::array<std::pair<MatmulKernelId, MatmulReject>, kMatmulKernelCount> rejected{};
  std::string explain() const;  // "kernel (passed over: a=reason, b=reason)"
};

// The kernel for this call. Deterministic; never fails (kF32RowDot accepts
// every valid shape).
MatmulSelection select_matmul(const MatmulShape& s, const KernelPlan& plan, const MatmulCaps& caps, bool explain = false);

// The int8 family alone (for matmul_many / matmul_gated, which choose
// between int8 and their own fp32 paths): kInt8* or kF32RowDot = none.
MatmulSelection select_matmul_int8(const MatmulShape& s, const KernelPlan& plan, const MatmulCaps& caps, bool explain = false);

// --- diagnostics ---
struct MatmulKernelStats {
  MatmulKernelId id;
  uint64_t calls = 0, rows = 0, macs = 0, ns = 0;  // ns only while timing is on
};
struct MatmulShapeStats {
  MatmulKernelId id;
  DType weight;
  int64_t n = 0, k = 0, m = 0;
  uint64_t calls = 0, ns = 0;
};

}  // namespace dynacore
