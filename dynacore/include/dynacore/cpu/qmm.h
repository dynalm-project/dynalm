#pragma once

// Native quantized matmul kernels (DynaKernel CPU "qmm" family).
//
// y[M, N] = x[M, K] . w[N, K]^T (+ bias) for block-quantized weights, with
// activations quantized per 32 values. Two activation modes:
//
//   kInt8   x ~= d * q, |q| <= 127, d = amax / 127, round to nearest even.
//           The arithmetic of the validated int8_per32 path (DD-053).
//   kInt16  x ~= d * q, |q| <= 32767, d = amax / 32767 (the DD-076 class, for
//           outlier-heavy inputs such as the FFN down projection). Kernels
//           run it as two byte planes, q = 256 * hi + lo (hi signed, lo
//           unsigned): the integer dot equals the int16 dot exactly.
//
// Weights are read in their original file layout (no repacked copy). Every
// format is first unpacked, per 32-value sub-block, into signed int8 codes,
// a float scale per 16 values and a float "min" coefficient per 32 values:
//
//   w[i] = s[i / 16] * q[i] - mn[i / 32]                     (exact, all formats)
//   y    = sum_sub ad_sub * sum_i s * q * aq   -   sum_sub mn_sub * c_sub
//
// with ad the activation block scale and c = ad * sum(aq) per block. Integer
// dot products are exact (int32); only the fp32 sum order differs between
// tiers, so all tiers agree with the scalar reference within fp32 rounding.
//
// Two kernel shapes:
//   gemv  up to kGemvMaxRows activation rows against weight rows [n0, n1):
//         each weight block is unpacked once per call and reused for the rows.
//         For decode (memory bound).
//   gemm  all M rows (activations in pair-interleaved layout) against weight
//         rows [n0, n1): weight rows are unpacked once per 4-row group across
//         all of K, then multiplied with the int8 matrix instructions where
//         the tier has them (ARM FEAT_I8MM SMMLA/USMMLA). For prefill.
//
// The tables are pure functions of their arguments (no global state, no
// threads); the caller partitions [n0, n1) across threads.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dynacore/tensor/dtype.h"

namespace dynacore::qmm {

enum class ActMode : uint8_t { kInt8, kInt16 };

inline constexpr int kBlock = 32;        // activation block (values per scale)
inline constexpr int kGemvMaxRows = 8;   // rows per gemv call
inline constexpr int kGemmRowGroup = 4;  // weight rows unpacked together by gemm

// Quantized activations: m rows of k values (k % 32 == 0).
//
// Row layout (paired == false): row r's codes at hi + r * k (and lo + r * k
// for kInt16), its block scales at d + r * kb and c + r * kb (kb = k / 32).
//
// Pair layout (paired == true, for gemm): rows 2p and 2p+1 interleaved in
// 8-byte chunks: chunk j of the pair is 16 bytes at hi + p * 2k + 16 j, row 2p
// first. Scales likewise: d + (p * kb + b) * 2 + (r & 1). An odd m is padded
// with a zero row (scales 0), never stored by the kernels.
struct Acts {
  ActMode mode = ActMode::kInt8;
  bool paired = false;
  int64_t m = 0, k = 0;
  const int8_t* hi = nullptr;
  const uint8_t* lo = nullptr;  // kInt16 only
  const float* d = nullptr;
  const float* c = nullptr;
};

// Owns the storage of an Acts (reused across calls; grows, never shrinks).
class ActBuffer {
 public:
  // Lays out storage for m rows of k values; contents undefined until
  // quantize_rows. k must be a multiple of 32.
  void prepare(ActMode mode, int64_t m, int64_t k, bool paired);
  // Quantizes rows [r0, r1) of x (row stride x_stride floats) into the
  // buffer with the given tier's quantizer. Disjoint row ranges may run
  // concurrently (pairs are written as 8-byte chunks of one row).
  void quantize_rows(const float* x, int64_t x_stride, int64_t r0, int64_t r1, const struct Kernels& kern);
  const Acts& acts() const { return acts_; }

 private:
  Acts acts_;
  std::vector<std::byte> storage_;
};

// Per-thread scratch for gemm (unpacked weight groups). Never shared.
struct GemmScratch {
  std::vector<std::byte> bytes;
};

// --- Packed (row-interleaved) weights for decode -------------------------
//
// A weight can be repacked once at load into groups of 4 rows interleaved in
// 4-byte runs, so one 16-byte vector holds 4 rows x 4 codes and a lane dot
// product (SDOT by element) yields 4 row results with no horizontal
// reduction, every activation load serving 4 rows. Scales are stored per
// (sub-block, row) so 4 rows' scales are one vector. Same arithmetic as the
// original layout (per-32 activation scales, exact integer sub-block dots);
// test_qmm checks packed against unpacked.
//
// Layout per 4-row group and 256-value super-block (Q4_K, 592 bytes):
//   uint16 d[4], dmin[4]; uint8 sc[8][4], mn[8][4];
//   qs[jp 0..3][i4 0..7][row 0..3][4 bytes]   (bytes 4 i4 .. 4 i4 + 3 of the
//   row's qs[32 jp ..]: low nibbles = sub-block 2jp, high = 2jp + 1).
// Groups are stored [group][super-block]. Rows past n % 4 are not packed
// (the caller runs them on the original layout).
bool packable(DType wt, int64_t k);
// Bytes of the packed copy of rows [0, n / 4 * 4).
size_t packed_bytes(DType wt, int64_t n, int64_t k);
// Packs 4-row groups [g0, g1) of w (row stride w_row_bytes) into `out`
// (the whole packed buffer). Disjoint ranges may run concurrently.
void pack_groups(DType wt, const void* w, int64_t w_row_bytes, int64_t k, int64_t g0, int64_t g1, void* out);

struct Kernels {
  const char* name = "none";
  bool gemm_matrix = false;  // gemm uses int8 matrix instructions (else a gemv loop)
  bool int16 = false;        // ActMode::kInt16 is supported
  // One activation row: codes at hi (and lo), chunk j of 8 codes at
  // hi + j * chunk_stride; block scales at d[b * ds], c[b * ds].
  void (*quantize_row)(const float* x, int64_t k, ActMode mode, int8_t* hi, uint8_t* lo, int64_t chunk_stride,
                       float* d, float* c, int64_t ds) = nullptr;
  // y[(r0 + r) * y_stride + n] for r < rows (<= kGemvMaxRows), n in [n0, n1).
  void (*gemv)(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, int64_t r0, int rows,
               float* y, int64_t y_stride, const float* bias) = nullptr;
  // y[r * y_stride + n] for every row r < a.m, n in [n0, n1); a.paired.
  void (*gemm)(DType wt, const void* w, int64_t w_row_bytes, int64_t n0, int64_t n1, const Acts& a, float* y,
               int64_t y_stride, const float* bias, GemmScratch& scratch) = nullptr;
  // Packed weights (pack_groups): rows [4 g0, 4 g1) of the packed copy `p`.
  // gemv: activation rows [r0, r0 + rows), rows <= kGemvMaxRows, row layout.
  void (*gemv_packed)(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, int64_t r0, int rows,
                      float* y, int64_t y_stride, const float* bias) = nullptr;
  // gemm: every activation row (pair layout).
  void (*gemm_packed)(DType wt, const void* p, int64_t k, int64_t g0, int64_t g1, const Acts& a, float* y, int64_t y_stride,
                      const float* bias, GemmScratch& scratch) = nullptr;
};

// --- SME fp32 panel gemm (prefill) ------------------------------------------
//
// For CPUs with the Scalable Matrix Extension (FEAT_SME): y = x . w^T with x
// in fp32 (NOT quantized) and w dequantized exactly to fp32, so the result is
// the fp32 reference up to summation order (more accurate than the int8 /
// int16 activation paths it replaces for prefill-sized batches).
//
// Both operands are laid out as k-major panels of kSmePanel rows:
// panel[kk * kSmePanel + i] = row (r0 + i), column kk; rows past the matrix are
// zero. block() multiplies one x panel by one w panel with outer products into
// the four 16 x 16 fp32 ZA tiles (512-bit streaming vectors) and stores the
// rows x cols corner of the 32 x 32 result.
inline constexpr int kSmePanel = 32;

struct SmeGemm {
  const char* name = "none";
  // Packs rows [m0, m0 + 32) of x (m rows, row stride x_stride floats) into panel.
  void (*pack_x)(const float* x, int64_t x_stride, int64_t m, int64_t k, int64_t m0, float* panel) = nullptr;
  // y[r * y_stride + c] = sum_kk xp[kk * 32 + r] * wp[kk * 32 + c], r < rows, c < cols (both <= 32).
  void (*block)(const float* xp, const float* wp, int64_t k, float* y, int64_t y_stride, int rows, int cols) = nullptr;
  // The same, accumulating: y[r * y_stride + c] += sum_kk ... (y read first).
  void (*block_acc)(const float* xp, const float* wp, int64_t k, float* y, int64_t y_stride, int rows, int cols) = nullptr;
};

// The SME tier when compiled in and the CPU has SME (else nullptr).
const SmeGemm* sme_gemm();
// Dequantizes weight rows [n0, n0 + 32) (rows >= n are zero) into a k-major
// panel, exactly (every qmm format). `packed`: the pack_groups copy of w (or
// nullptr): its rows are read from it, so the original pages need not be
// resident; rows past the packed groups come from w.
void weight_panel_f32(DType wt, const void* w, int64_t w_row_bytes, const void* packed, int64_t n, int64_t k, int64_t n0,
                      float* panel);

// Weight types with qmm kernels (every tier supports the same set).
bool supports(DType wt);

// Portable scalar reference: the arithmetic every tier is tested against.
const Kernels& reference_kernels();
// Best tier for this CPU and build (ARM: dot product, then i8mm); the
// reference where nothing better is compiled in.
const Kernels& best_kernels();
// True when best_kernels() is a SIMD tier (not the scalar reference).
bool accelerated();
// A specific tier when compiled and supported here, else nullptr
// ("reference", "neon_dot", "neon_i8mm").
const Kernels* kernels_by_name(const char* name);

// Internal registration (one per tier translation unit).
const Kernels* neon_dot_kernels();   // nullptr when not compiled
const Kernels* neon_i8mm_kernels();  // nullptr when not compiled
const SmeGemm* sme_kernels();        // nullptr when not compiled (no runtime check)

}  // namespace dynacore::qmm
