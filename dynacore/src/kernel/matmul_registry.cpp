#include "dynacore/kernel/matmul_registry.h"

#include <algorithm>
#include <iterator>

#include "dynacore/cpu/cpu_kernels.h"

namespace dynacore {
namespace {

using Id = MatmulKernelId;
using Acc = MatmulAccuracy;

// One row per MatmulKernelId, in enum order.
constexpr MatmulKernelDesc kKernels[] = {
    {Id::kExternal, "external", "any (provider)", "provider predicate", "provider", "provider", "provider", Acc::kExternal,
     false, "DynaLM Next extension point; KernelPlan::external_matmul"},
    {Id::kQmmGemm, "qmm_gemm", "Q4_0/1, Q5_0/1, Q8_0, IQ4_NL, Q3_K-Q6_K, IQ4_XS", ">= qmm_gemm_min_rows, <= the op's int8 / int16 rows",
     "NEON i8mm", "original (unpacked per 4-row group)", "int8 per-32 / int16 per-32", Acc::kValidated, true,
     "qmm.h: activation arithmetic of int8_per32 (DD-053) / int16_rows (DD-076); test_qmm"},
    {Id::kQmmGemv, "qmm_gemv", "Q4_0/1, Q5_0/1, Q8_0, IQ4_NL, Q3_K-Q6_K, IQ4_XS", "<= the op's int8 / int16 rows",
     "NEON SDOT (int16: i8mm USDOT)", "original", "int8 per-32 / int16 per-32", Acc::kValidated, true,
     "qmm.h: activation arithmetic of int8_per32 (DD-053) / int16_rows (DD-076); test_qmm"},
    {Id::kSmeGemm, "sme_f32_gemm", "Q4_0/1, Q5_0/1, Q8_0, IQ4_NL, Q3_K-Q6_K, IQ4_XS", ">= sme_min_rows",
     "SME (FMOPA, streaming mode)", "fp32 panels dequantized per call (packed copy where present)", "fp32",
     Acc::kReference, true, "qmm.h: exact weights, fp32 activations; test_qmm vs a double-precision product"},
    {Id::kInt8PrefillX8, "int8_prefill_x8", "Q4_K", "above decode rows, <= q4_repack_max_rows; N >= q4_repack_min_out",
     "AVX2", "interleaved Q4_Kx8", "int8 sub-scaled", Acc::kFailsDd053, false,
     "DD-078: Qwen3-4B prefill perplexity +2.9-3.4%"},
    {Id::kInt8SubscaledX8Rows8, "int8_subscaled_x8_rows8", "Q4_K", "5..8 rows with packed_int8_8rows; 9..16 with wide decode (wide_rows8)", "AVX2 (not the VNNI x8)",
     "interleaved Q4_Kx8", "int8 sub-scaled", Acc::kExperimental, false,
     "identical to int8_subscaled_x8; slower at 5-8 rows (pk8 validation), faster at 9-16 rows (wide-decode profile)"},
    {Id::kInt8SubscaledX8, "int8_subscaled_x8", "Q4_K", "int8_subscale_min_rows..max(int8 rows, 8), >= q4_repack_min_rows",
     "AVX2 (VNNI with DYNACORE_VNNI=1)", "interleaved Q4_Kx8", "int8 sub-scaled", Acc::kValidated, true, "DD-077, DD-078"},
    {Id::kInt8Subscaled, "int8_subscaled", "K-quants", "int8_subscale_min_rows..max(int8 rows, 8)", "AVX2", "original",
     "int8 sub-scaled", Acc::kValidated, true, "DD-077"},
    {Id::kInt8Superblock, "int8_superblock", "K-quants", "int8_superblock_min_rows..8", "AVX2", "original",
     "int8 super-block", Acc::kFailsDd053, false, "DD-075: mean KL 0.021 (1.5B)"},
    {Id::kInt8Per32, "int8_per32", "types with an int8 row kernel", "<= int8_decode_max_rows", "AVX2 or NEON SDOT", "original",
     "int8 per-32", Acc::kValidated, true, "DD-053"},
    {Id::kInt16SplitPrefillX8, "int16_split_prefill_x8", "Q4_K", "9..int16_prefill_max_rows; N >= 1024, N*K >= 4M",
     "AVX-VNNI", "interleaved Q4_Kx8", "int16 (split bytes)", Acc::kExact, false,
     "DynaLM Next: bit-identical to int16_prefill_x8"},
    {Id::kInt16PrefillX8, "int16_prefill_x8", "Q4_K (Q6_K with DYNACORE_Q6_PACKED=1)", "9..int16_prefill_max_rows; N >= 1024, N*K >= 4M",
     "AVX2 (vpdpwssd where AVX-VNNI)", "interleaved x8", "int16", Acc::kValidated, false,
     "Phase 6/7, V0, Phase 10 B-G1 (DD-053 pass)"},
    {Id::kInt16X8, "int16_x8", "Q4_K (Q6_K with DYNACORE_Q6_PACKED=1)", "q4_repack_min_rows..max(int16 rows, 8)", "AVX2",
     "interleaved x8", "int16", Acc::kValidated, true, "DD-081, DD-082 (FFN down at decode sizes)"},
    {Id::kInt16Rows, "int16_rows", "types with an int16 row kernel", "<= int16_decode_max_rows", "AVX2 or NEON SDOT", "original", "int16",
     Acc::kValidated, true, "DD-076 (FFN down)"},
    {Id::kF32ExpandGemm, "f32_expand_gemm", "all", ">= expand_min_rows, dense x", "any", "original", "fp32",
     Acc::kReference, true, "DD-036 (reference prefill path)"},
    {Id::kF32RowDot, "f32_row_dot", "all", "any", "any", "original", "fp32", Acc::kReference, true,
     "reference fallback"},
};
static_assert(std::size(kKernels) == kMatmulKernelCount);

constexpr bool kernels_in_order() {
  for (size_t i = 0; i < std::size(kKernels); ++i) {
    if (static_cast<size_t>(kKernels[i].id) != i) return false;
  }
  return true;
}
static_assert(kernels_in_order());

// Collects rejections in priority order when asked.
struct Selector {
  const KernelPlan& plan;
  bool explain;
  MatmulSelection sel;
  // false (and recorded) when the kernel is disabled by the plan.
  bool allowed(Id id) {
    if ((plan.matmul_disabled & matmul_kernel_bit(id)) == 0) return true;
    reject(id, MatmulReject::kDisabled);
    return false;
  }
  void reject(Id id, MatmulReject r) {
    if (!explain) return;
    for (uint8_t i = 0; i < sel.n_rejected; ++i) {
      if (sel.rejected[i].first == id) return;  // first reason only
    }
    if (sel.n_rejected < sel.rejected.size()) sel.rejected[sel.n_rejected++] = {id, r};
  }
  MatmulSelection pick(Id id, MatmulActFormat act, bool packed) {
    sel.id = id;
    sel.act = act;
    sel.packed = packed;
    return sel;
  }
};

// The int8 family, in the order of CpuDevice::int8_kernel before the registry
// (DD-053 / 075 / 077 / 078): true and s.sel filled when one applies.
bool select_int8(const MatmulShape& s, const MatmulCaps& c, Selector& s8) {
  const KernelPlan& p = s8.plan;
  const DType wt = s.weight;
  if (p.int8_decode_max_rows <= 0 || !c.int8_accelerated || s.k % kActQ8Block != 0) {
    const MatmulReject why = p.int8_decode_max_rows <= 0 ? MatmulReject::kNotEnabled
                             : !c.int8_accelerated   ? MatmulReject::kNoKernel
                                                     : MatmulReject::kAlignment;
    for (Id id : {Id::kInt8PrefillX8, Id::kInt8SubscaledX8, Id::kInt8Subscaled, Id::kInt8Superblock, Id::kInt8Per32}) {
      s8.reject(id, why);
    }
    return false;
  }
  const int64_t m = s.m;
  // Super-block family rows (K-quants, k a multiple of 256).
  const bool sb_rows = s.k % 256 == 0 && m <= std::max(p.int8_decode_max_rows, p.int8_superblock_max_rows);
  // 8-row tiles on the interleaved copy: everywhere above 4 rows with
  // packed_int8_8rows, or above 8 rows (only reachable with wide decode) with
  // wide_rows8. Same per-row arithmetic as the 4-row tiles.
  const bool rows8 = c.dot_q8_sx_x8_rows8 &&
                     ((p.packed_int8_8rows && m > kDotRowsMax) || (p.wide_rows8 && m > kPackedRowsMax));

  // Prefill-sized batches on the interleaved Q4_K copy (DD-078, opt-in).
  if (!sb_rows) {
    if (p.q4_repack_max_rows <= 0) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kNotEnabled);
    } else if (wt != DType::kQ4_K) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kWeightType);
    } else if (s.k % 256 != 0) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kAlignment);
    } else if (m > p.q4_repack_max_rows || p.q4_repack_min_rows <= 0 || p.int8_subscale_min_rows <= 0 ||
               m < p.int8_subscale_min_rows) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kRows);
    } else if (s.n < p.q4_repack_min_out) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kSizeGate);
    } else if (!c.quantize_act_sx || !c.dot_q8_sx_rows) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kNoKernel);
    } else if (!c.packed_q4) {
      s8.reject(Id::kInt8PrefillX8, MatmulReject::kNoPacked);
    } else if (s8.allowed(Id::kInt8PrefillX8)) {
      return s8.pick(Id::kInt8PrefillX8, MatmulActFormat::kInt8Subscaled, true), true;
    }
  } else {
    s8.reject(Id::kInt8PrefillX8, MatmulReject::kRows);
  }

  // Sub-scaled activations (DD-077), on the interleaved copy where present (DD-078).
  if (!sb_rows) {
    s8.reject(Id::kInt8SubscaledX8, MatmulReject::kRows);
    s8.reject(Id::kInt8Subscaled, MatmulReject::kRows);
  } else if (p.int8_subscale_min_rows <= 0) {
    s8.reject(Id::kInt8SubscaledX8, MatmulReject::kNotEnabled);
    s8.reject(Id::kInt8Subscaled, MatmulReject::kNotEnabled);
  } else if (m < p.int8_subscale_min_rows) {
    s8.reject(Id::kInt8SubscaledX8, MatmulReject::kRows);
    s8.reject(Id::kInt8Subscaled, MatmulReject::kRows);
  } else if (!c.quantize_act_sx || !c.dot_q8_sx_rows) {
    s8.reject(Id::kInt8SubscaledX8, MatmulReject::kNoKernel);
    s8.reject(Id::kInt8Subscaled, MatmulReject::kNoKernel);
  } else {
    const bool want_packed = wt == DType::kQ4_K && p.q4_repack_min_rows > 0 && m >= p.q4_repack_min_rows;
    if (want_packed && c.packed_q4) {
      if (rows8 && s8.allowed(Id::kInt8SubscaledX8Rows8)) {
        return s8.pick(Id::kInt8SubscaledX8Rows8, MatmulActFormat::kInt8Subscaled, true), true;
      }
      if (s8.allowed(Id::kInt8SubscaledX8)) return s8.pick(Id::kInt8SubscaledX8, MatmulActFormat::kInt8Subscaled, true), true;
    } else {
      s8.reject(Id::kInt8SubscaledX8, wt != DType::kQ4_K ? MatmulReject::kWeightType
                                      : !want_packed     ? MatmulReject::kRows
                                                         : MatmulReject::kNoPacked);
    }
    if (s8.allowed(Id::kInt8Subscaled)) return s8.pick(Id::kInt8Subscaled, MatmulActFormat::kInt8Subscaled, false), true;
  }

  // Super-block activations (DD-075, opt-in).
  if (!sb_rows) {
    s8.reject(Id::kInt8Superblock, MatmulReject::kRows);
  } else if (p.int8_superblock_min_rows <= 0) {
    s8.reject(Id::kInt8Superblock, MatmulReject::kNotEnabled);
  } else if (m < p.int8_superblock_min_rows) {
    s8.reject(Id::kInt8Superblock, MatmulReject::kRows);
  } else if (!c.quantize_act_sb || !c.dot_q8_sb_rows) {
    s8.reject(Id::kInt8Superblock, MatmulReject::kNoKernel);
  } else if (s8.allowed(Id::kInt8Superblock)) {
    return s8.pick(Id::kInt8Superblock, MatmulActFormat::kInt8Superblock, false), true;
  }

  // Per-32 activations (DD-053).
  if (m > p.int8_decode_max_rows) {
    s8.reject(Id::kInt8Per32, MatmulReject::kRows);
    return false;
  }
  if (!c.quantize_act || !c.dot_q8_rows) {
    s8.reject(Id::kInt8Per32, MatmulReject::kNoKernel);
    return false;
  }
  if (!s8.allowed(Id::kInt8Per32)) return false;
  return s8.pick(Id::kInt8Per32, MatmulActFormat::kInt8Per32, false), true;
}

}  // namespace

std::string_view matmul_accuracy_name(MatmulAccuracy a) {
  switch (a) {
    case Acc::kReference: return "reference";
    case Acc::kValidated: return "validated";
    case Acc::kExact: return "exact";
    case Acc::kFailsDd053: return "fails_dd053";
    case Acc::kExperimental: return "experimental";
    case Acc::kExternal: return "external";
  }
  return "?";
}

std::string_view matmul_reject_name(MatmulReject r) {
  switch (r) {
    case MatmulReject::kDisabled: return "disabled";
    case MatmulReject::kNotEnabled: return "not_enabled";
    case MatmulReject::kRows: return "rows";
    case MatmulReject::kAlignment: return "alignment";
    case MatmulReject::kWeightType: return "weight_type";
    case MatmulReject::kNoKernel: return "no_kernel";
    case MatmulReject::kNoPacked: return "no_packed";
    case MatmulReject::kSizeGate: return "size_gate";
    case MatmulReject::kNotDense: return "not_dense";
  }
  return "?";
}

std::span<const MatmulKernelDesc> matmul_kernels() { return kKernels; }
const MatmulKernelDesc& matmul_kernel_desc(MatmulKernelId id) { return kKernels[static_cast<size_t>(id)]; }
std::string_view matmul_kernel_name(MatmulKernelId id) {
  return static_cast<size_t>(id) < kMatmulKernelCount ? kKernels[static_cast<size_t>(id)].name : "?";
}

bool parse_matmul_kernel(std::string_view name, MatmulKernelId& out) {
  for (const MatmulKernelDesc& d : kKernels) {
    if (d.name == name) {
      out = d.id;
      return true;
    }
  }
  return false;
}

Status parse_matmul_kernel_list(std::string_view list, uint32_t& mask) {
  uint32_t m = 0;
  size_t p = 0;
  while (p <= list.size()) {
    const size_t q = std::min(list.find(',', p), list.size());
    std::string_view t = list.substr(p, q - p);
    while (!t.empty() && t.front() == ' ') t.remove_prefix(1);
    while (!t.empty() && t.back() == ' ') t.remove_suffix(1);
    if (t == "all-accelerated") {
      for (const MatmulKernelDesc& d : kKernels) {
        if (d.accuracy != Acc::kReference) m |= matmul_kernel_bit(d.id);
      }
    } else if (!t.empty()) {
      MatmulKernelId id;
      if (!parse_matmul_kernel(t, id)) return InvalidArgument("unknown matmul kernel '" + std::string(t) + "'");
      if (matmul_kernel_desc(id).accuracy == Acc::kReference) {
        return InvalidArgument("the fp32 reference kernel '" + std::string(t) + "' cannot be disabled");
      }
      m |= matmul_kernel_bit(id);
    }
    p = q + 1;
  }
  mask = m;
  return Status::Ok();
}

Status validate_matmul_shape(const MatmulShape& s) {
  if (s.m <= 0 || s.n <= 0 || s.k <= 0) {
    return InvalidArgument("matmul dims must be positive (m " + std::to_string(s.m) + ", n " + std::to_string(s.n) + ", k " +
                           std::to_string(s.k) + ")");
  }
  if (s.weight == DType::kI8 || s.weight == DType::kI16 || s.weight == DType::kI32 || s.weight == DType::kCount) {
    return Unsupported("matmul weight type " + std::string(dtype_name(s.weight)) + " is not supported");
  }
  const int32_t block = dtype_block_elems(s.weight);
  if (block > 1 && s.k % block != 0) {
    return InvalidArgument("matmul k " + std::to_string(s.k) + " is not a multiple of the " + std::string(dtype_name(s.weight)) +
                           " block (" + std::to_string(block) + ")");
  }
  return Status::Ok();
}

MatmulSelection select_matmul_int8(const MatmulShape& s, const KernelPlan& plan, const MatmulCaps& caps, bool explain) {
  Selector sel{plan, explain, {}};
  if (select_int8(s, caps, sel)) return sel.sel;
  sel.sel.id = Id::kF32RowDot;  // "none" for the int8-only query
  return sel.sel;
}

MatmulSelection select_matmul(const MatmulShape& s, const KernelPlan& p, const MatmulCaps& c, bool explain) {
  Selector sel{p, explain, {}};
  const DType wt = s.weight;
  const int64_t m = s.m, n = s.n, k = s.k;

  // External kernels first, only when the plan asks for them.
  if (!p.external_matmul) {
    sel.reject(Id::kExternal, MatmulReject::kNotEnabled);
  } else if (!c.external_eligible) {
    sel.reject(Id::kExternal, MatmulReject::kNoKernel);
  } else if (sel.allowed(Id::kExternal)) {
    return sel.pick(Id::kExternal, MatmulActFormat::kF32, false);
  }

  // SME fp32 panel gemm (qmm.h): fp32 activations, so no op-level precision
  // rule limits it; only the batch size.
  if (c.sme_f32 && p.sme_min_rows > 0) {
    if (m < p.sme_min_rows) {
      sel.reject(Id::kSmeGemm, MatmulReject::kRows);
    } else if (sel.allowed(Id::kSmeGemm)) {
      return sel.pick(Id::kSmeGemm, MatmulActFormat::kF32, false);
    }
  } else {
    sel.reject(Id::kSmeGemm, p.sme_min_rows > 0 ? MatmulReject::kNoKernel : MatmulReject::kNotEnabled);
  }

  // Native qmm kernels (qmm.h): the activation precision the op allows.
  {
    MatmulReject why = MatmulReject::kNotEnabled;
    MatmulActFormat act = MatmulActFormat::kF32;
    if (!p.qmm) {
      why = MatmulReject::kNotEnabled;
    } else if (p.int8_decode_max_rows > 0) {
      // The op's int8 row limit applies as to every int8 kernel (the engine
      // lifts it on CPUs with qmm kernels: their per-row arithmetic does not
      // depend on the batch).
      act = c.qmm_int8 && m <= p.int8_decode_max_rows ? MatmulActFormat::kInt8Per32 : MatmulActFormat::kF32;
      why = c.qmm_int8 ? MatmulReject::kRows : MatmulReject::kNoKernel;
    } else if (p.int16_decode_max_rows > 0) {
      act = c.qmm_int16 && m <= p.int16_decode_max_rows ? MatmulActFormat::kInt16 : MatmulActFormat::kF32;
      why = c.qmm_int16 ? MatmulReject::kRows : MatmulReject::kNoKernel;
    }
    if (act == MatmulActFormat::kF32) {
      sel.reject(Id::kQmmGemm, why);
      sel.reject(Id::kQmmGemv, why);
    } else {
      if (m < p.qmm_gemm_min_rows) {
        sel.reject(Id::kQmmGemm, MatmulReject::kRows);
      } else if (!c.qmm_gemm) {
        sel.reject(Id::kQmmGemm, MatmulReject::kNoKernel);
      } else if (sel.allowed(Id::kQmmGemm)) {
        return sel.pick(Id::kQmmGemm, act, false);
      }
      if (sel.allowed(Id::kQmmGemv)) return sel.pick(Id::kQmmGemv, act, false);
    }
  }

  if (select_int8(s, c, sel)) return sel.sel;

  // int16 prefill tiles (Phase 6, opt-in) and the split-byte variant (opt-in).
  const bool prefill_gate = p.int16_prefill_max_rows > 0 && m > kPackedRowsMax && m <= p.int16_prefill_max_rows &&
                            n >= p.q4_repack_min_out && n * k >= p.int16_prefill_min_weights && c.int8_accelerated &&
                            k % 256 == 0 && c.quantize_act16 && c.q16_packed_enabled;
  if (!prefill_gate) {
    const MatmulReject why = p.int16_prefill_max_rows <= 0                                  ? MatmulReject::kNotEnabled
                             : (m <= kPackedRowsMax || m > p.int16_prefill_max_rows)         ? MatmulReject::kRows
                             : (n < p.q4_repack_min_out || n * k < p.int16_prefill_min_weights) ? MatmulReject::kSizeGate
                             : k % 256 != 0                                                  ? MatmulReject::kAlignment
                                                                                             : MatmulReject::kNoKernel;
    sel.reject(Id::kInt16SplitPrefillX8, why);
    sel.reject(Id::kInt16PrefillX8, why);
  } else if (wt == DType::kQ4_K && c.dot_q16_x8_q4_rows8) {
    if (!c.packed_q4) {
      sel.reject(Id::kInt16SplitPrefillX8, MatmulReject::kNoPacked);
      sel.reject(Id::kInt16PrefillX8, MatmulReject::kNoPacked);
    } else {
      if (!p.int16_prefill_split) {
        sel.reject(Id::kInt16SplitPrefillX8, MatmulReject::kNotEnabled);
      } else if (!c.dot_q16s_x8_q4_rows8 || !c.split_act16) {
        sel.reject(Id::kInt16SplitPrefillX8, MatmulReject::kNoKernel);
      } else if (sel.allowed(Id::kInt16SplitPrefillX8)) {
        return sel.pick(Id::kInt16SplitPrefillX8, MatmulActFormat::kInt16Split, true);
      }
      if (sel.allowed(Id::kInt16PrefillX8)) return sel.pick(Id::kInt16PrefillX8, MatmulActFormat::kInt16, true);
    }
  } else if (wt == DType::kQ6_K && c.dot_q16_x8_q6_rows8 && c.q6_packed_enabled) {
    sel.reject(Id::kInt16SplitPrefillX8, MatmulReject::kWeightType);
    if (!c.packed_q6) {
      sel.reject(Id::kInt16PrefillX8, MatmulReject::kNoPacked);
    } else if (sel.allowed(Id::kInt16PrefillX8)) {
      return sel.pick(Id::kInt16PrefillX8, MatmulActFormat::kInt16, true);
    }
  } else {
    sel.reject(Id::kInt16SplitPrefillX8, wt == DType::kQ4_K ? MatmulReject::kNoKernel : MatmulReject::kWeightType);
    sel.reject(Id::kInt16PrefillX8,
               (wt == DType::kQ4_K || wt == DType::kQ6_K) ? MatmulReject::kNoKernel : MatmulReject::kWeightType);
  }

  // int16 on the interleaved copy at decode sizes (DD-081/082).
  const bool try16 = p.int16_decode_max_rows > 0 && p.q4_repack_min_rows > 0 && m >= p.q4_repack_min_rows && c.q16_packed_enabled;
  bool have16 = false;
  if (try16 && wt == DType::kQ4_K && c.dot_q16_x8_q4) {
    have16 = c.packed_q4;
  } else if (try16 && wt == DType::kQ6_K && c.dot_q16_x8_q6 && c.q6_packed_enabled) {
    have16 = c.packed_q6;
  }
  if (!try16) {
    sel.reject(Id::kInt16X8, p.int16_decode_max_rows <= 0 ? MatmulReject::kNotEnabled : MatmulReject::kRows);
  } else if (!have16) {
    sel.reject(Id::kInt16X8, MatmulReject::kNoPacked);
  } else if (m > std::max(p.int16_decode_max_rows, p.int8_superblock_max_rows)) {
    sel.reject(Id::kInt16X8, MatmulReject::kRows);
  } else if (!c.int8_accelerated || k % 256 != 0 || !c.quantize_act16) {
    sel.reject(Id::kInt16X8, !c.int8_accelerated ? MatmulReject::kNoKernel
                             : k % 256 != 0      ? MatmulReject::kAlignment
                                                 : MatmulReject::kNoKernel);
  } else if (sel.allowed(Id::kInt16X8)) {
    return sel.pick(Id::kInt16X8, MatmulActFormat::kInt16, true);
  }

  // int16 row kernel (DD-076).
  if (m > p.int16_decode_max_rows) {
    sel.reject(Id::kInt16Rows, p.int16_decode_max_rows <= 0 ? MatmulReject::kNotEnabled : MatmulReject::kRows);
  } else if (!c.int8_accelerated || !c.quantize_act16 || !c.dot_q16_rows) {
    sel.reject(Id::kInt16Rows, MatmulReject::kNoKernel);
  } else if (k % kActQ8Block != 0) {
    sel.reject(Id::kInt16Rows, MatmulReject::kAlignment);
  } else if (sel.allowed(Id::kInt16Rows)) {
    return sel.pick(Id::kInt16Rows, MatmulActFormat::kInt16, false);
  }

  // fp32: expand panels + GEMM for many dense rows, else the row kernel.
  const bool expand = wt != DType::kF32 && m >= p.expand_min_rows;
  const bool f32_panel = wt == DType::kF32 && m >= p.expand_min_rows;
  if (!(expand || f32_panel)) {
    sel.reject(Id::kF32ExpandGemm, MatmulReject::kRows);
  } else if (!s.x_dense) {
    sel.reject(Id::kF32ExpandGemm, MatmulReject::kNotDense);
  } else {
    return sel.pick(Id::kF32ExpandGemm, MatmulActFormat::kF32, false);
  }
  return sel.pick(Id::kF32RowDot, MatmulActFormat::kF32, false);
}

std::string MatmulSelection::explain() const {
  std::string s(matmul_kernel_name(id));
  if (n_rejected > 0) {
    s += " (passed over:";
    for (uint8_t i = 0; i < n_rejected; ++i) {
      s += ' ';
      s += matmul_kernel_name(rejected[i].first);
      s += '=';
      s += matmul_reject_name(rejected[i].second);
    }
    s += ')';
  }
  return s;
}

}  // namespace dynacore
