// qmm kernels (dynacore/cpu/qmm.h) of every tier built into this binary.
//
//  1. The reference tier against an independent oracle: sum of dequantized
//     weights (dequantize_row, quant_formats.cpp) times the reconstructed
//     activations d * q, in double. Checks unpacking, scales and minimum terms
//     of every format (relative 2e-5 of the sum of magnitudes: fp32 rounding).
//  2. Quantization: |x - d q| <= d / 2, c = d * sum(q), int16 byte planes
//     q = 256 hi + lo; every tier bit-identical to the reference.
//  3. Every tier's gemv and gemm against the reference (same arithmetic, other
//     fp32 summation order: relative 1e-5), with odd row counts, weight row
//     tails, legacy formats with k % 256 != 0, strided outputs and bias.
// No model files needed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "dynacore/cpu/qmm.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/memory/mapped_file.h"
#include "dynacore/quantization/dequant.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;

namespace {

int g_checks = 0;
#define CHECK(c)                                                                     \
  do {                                                                               \
    ++g_checks;                                                                      \
    if (!(c)) {                                                                      \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

std::mt19937 g_rng(4321);

// Byte offsets of the fp16 scale fields in one block (random bytes there
// could be Inf / NaN).
std::vector<int> fp16_fields(DType t) {
  switch (t) {
    case DType::kQ8_0: case DType::kQ4_0: case DType::kQ5_0: case DType::kIQ4_NL: case DType::kIQ4_XS: return {0};
    case DType::kQ4_1: case DType::kQ5_1: case DType::kQ4_K: case DType::kQ5_K: return {0, 2};
    case DType::kQ3_K: return {108};
    case DType::kQ6_K: return {208};
    default: return {};
  }
}

std::vector<uint8_t> random_rows(DType t, int64_t n, int64_t k) {
  const int64_t rb = dtype_row_bytes(t, k);
  std::vector<uint8_t> w(static_cast<size_t>(n * rb));
  for (uint8_t& b : w) b = static_cast<uint8_t>(g_rng());
  std::uniform_real_distribution<float> sd(0.001f, 0.02f);
  const int64_t be = dtype_block_elems(t), bb = dtype_block_bytes(t);
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t b = 0; b < k / be; ++b) {
      for (int off : fp16_fields(t)) {
        const uint16_t h = fp32_to_fp16(sd(g_rng));
        std::memcpy(w.data() + r * rb + b * bb + off, &h, 2);
      }
    }
  }
  return w;
}

std::vector<float> random_acts(int64_t m, int64_t k, int64_t stride) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(m * stride), 0.0f);
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t i = 0; i < k; ++i) x[static_cast<size_t>(r * stride + i)] = nd(g_rng);
    // Outlier channels, as in FFN activations.
    for (int o = 0; o < 3; ++o) x[static_cast<size_t>(r * stride + (g_rng() % k))] = 40.0f * nd(g_rng);
    // An all-zero block (d = 0) now and then.
    if (k >= 64 && r % 3 == 1) std::memset(&x[static_cast<size_t>(r * stride + 32)], 0, 32 * sizeof(float));
  }
  return x;
}

int32_t code_at(const qmm::Acts& a, int64_t r, int64_t i) {
  const int64_t at = a.paired ? (r / 2) * 2 * a.k + (i / 8) * 16 + (r & 1) * 8 + i % 8 : r * a.k + i;
  return a.mode == qmm::ActMode::kInt8 ? a.hi[at] : 256 * a.hi[at] + a.lo[at];
}
float scale_at(const float* s, const qmm::Acts& a, int64_t r, int64_t b) {
  const int64_t kb = a.k / qmm::kBlock;
  return s[a.paired ? ((r / 2) * kb + b) * 2 + (r & 1) : r * kb + b];
}

const char* mode_name(qmm::ActMode m) { return m == qmm::ActMode::kInt8 ? "int8" : "int16"; }

void check_quantization(const qmm::Kernels& tier, qmm::ActMode mode, bool paired) {
  const int64_t m = 5, k = 320;
  const std::vector<float> x = random_acts(m, k, k);
  qmm::ActBuffer ref, got;
  ref.prepare(mode, m, k, paired);
  got.prepare(mode, m, k, paired);
  ref.quantize_rows(x.data(), k, 0, m, qmm::reference_kernels());
  got.quantize_rows(x.data(), k, 0, m, tier);
  const qmm::Acts& a = ref.acts();
  const qmm::Acts& b = got.acts();
  const int32_t qmax = mode == qmm::ActMode::kInt8 ? 127 : 32767;
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t blk = 0; blk < k / 32; ++blk) {
      const float d = scale_at(a.d, a, r, blk);
      CHECK(d == scale_at(b.d, b, r, blk));
      CHECK(scale_at(a.c, a, r, blk) == scale_at(b.c, b, r, blk));
      int64_t sum = 0;
      for (int i = 0; i < 32; ++i) {
        const int64_t idx = blk * 32 + i;
        const int32_t q = code_at(a, r, idx);
        CHECK(q == code_at(b, r, idx));
        CHECK(q >= -qmax && q <= qmax);
        // d/2 plus fp32 rounding of x * (1/d) and of d * q.
        const float xv = x[static_cast<size_t>(r * k + idx)];
        CHECK(std::fabs(xv - d * static_cast<float>(q)) <= d * 0.5f + 1e-6f * std::fabs(xv) + 1e-30f);
        sum += q;
      }
      CHECK(std::fabs(scale_at(a.c, a, r, blk) - d * static_cast<float>(sum)) <= 1e-6f * std::fabs(d * static_cast<float>(sum)) + 1e-30f);
    }
  }
}

// Reference tier vs dequantized weights times reconstructed activations.
void check_reference_oracle(DType t, qmm::ActMode mode, int64_t k) {
  const int64_t n = 6, m = 3;
  const std::vector<uint8_t> w = random_rows(t, n, k);
  const std::vector<float> x = random_acts(m, k, k);
  qmm::ActBuffer buf;
  buf.prepare(mode, m, k, false);
  buf.quantize_rows(x.data(), k, 0, m, qmm::reference_kernels());
  const qmm::Acts& a = buf.acts();
  std::vector<float> y(static_cast<size_t>(m * n));
  const int64_t rb = dtype_row_bytes(t, k);
  qmm::reference_kernels().gemv(t, w.data(), rb, 0, n, a, 0, static_cast<int>(m), y.data(), n, nullptr);
  std::vector<float> wd(static_cast<size_t>(k));
  for (int64_t j = 0; j < n; ++j) {
    CHECK(dequantize_row(t, w.data() + j * rb, wd.data(), k));
    for (int64_t r = 0; r < m; ++r) {
      double want = 0, mag = 0;
      for (int64_t i = 0; i < k; ++i) {
        const double xv = static_cast<double>(scale_at(a.d, a, r, i / 32)) * code_at(a, r, i);
        want += static_cast<double>(wd[static_cast<size_t>(i)]) * xv;
        mag += std::fabs(static_cast<double>(wd[static_cast<size_t>(i)]) * xv);
      }
      const double got = y[static_cast<size_t>(r * n + j)];
      if (std::fabs(got - want) > 2e-5 * mag + 1e-6) {
        std::fprintf(stderr, "reference %s %s k=%lld row %lld col %lld: got %.7g want %.7g (mag %.3g)\n",
                     std::string(dtype_name(t)).c_str(), mode_name(mode), static_cast<long long>(k),
                     static_cast<long long>(r), static_cast<long long>(j), got, want, mag);
      }
      CHECK(std::fabs(got - want) <= 2e-5 * mag + 1e-6);
    }
  }
}

double row_mag(DType t, const uint8_t* wrow, const qmm::Acts& a, int64_t r) {
  std::vector<float> wd(static_cast<size_t>(a.k));
  dequantize_row(t, wrow, wd.data(), a.k);
  double mag = 0;
  for (int64_t i = 0; i < a.k; ++i) {
    mag += std::fabs(static_cast<double>(wd[static_cast<size_t>(i)]) * scale_at(a.d, a, r, i / 32) * code_at(a, r, i));
  }
  return mag;
}

// A tier's gemv and gemm against the reference gemv.
void check_tier(const qmm::Kernels& tier, DType t, qmm::ActMode mode, int64_t m, int64_t n, int64_t k) {
  const std::vector<uint8_t> w = random_rows(t, n, k);
  const int64_t rb = dtype_row_bytes(t, k);
  const std::vector<float> x = random_acts(m, k, k + 7);  // strided input rows
  std::vector<float> bias(static_cast<size_t>(n));
  for (float& b : bias) b = std::uniform_real_distribution<float>(-1, 1)(g_rng);
  const int64_t ys = n + 3;  // strided output rows
  qmm::ActBuffer rows_buf;
  rows_buf.prepare(mode, m, k, false);
  rows_buf.quantize_rows(x.data(), k + 7, 0, m, qmm::reference_kernels());
  std::vector<float> want(static_cast<size_t>(m * ys), 0.0f);
  for (int64_t r0 = 0; r0 < m; r0 += qmm::kGemvMaxRows) {
    const int rows = static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, m - r0));
    qmm::reference_kernels().gemv(t, w.data(), rb, 0, n, rows_buf.acts(), r0, rows, want.data(), ys, bias.data());
  }
  std::vector<double> mag(static_cast<size_t>(m * n));
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t j = 0; j < n; ++j) mag[static_cast<size_t>(r * n + j)] = row_mag(t, w.data() + j * rb, rows_buf.acts(), r);
  }
  auto compare = [&](const std::vector<float>& got, const char* what) {
    for (int64_t r = 0; r < m; ++r) {
      for (int64_t j = 0; j < n; ++j) {
        const double g = got[static_cast<size_t>(r * ys + j)], e = want[static_cast<size_t>(r * ys + j)];
        const double tol = 1e-5 * mag[static_cast<size_t>(r * n + j)] + 1e-5;
        if (std::fabs(g - e) > tol) {
          std::fprintf(stderr, "%s %s %s %s m=%lld n=%lld k=%lld [%lld,%lld]: got %.7g want %.7g\n", tier.name, what,
                       std::string(dtype_name(t)).c_str(), mode_name(mode), static_cast<long long>(m),
                       static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(r),
                       static_cast<long long>(j), g, e);
        }
        CHECK(std::fabs(g - e) <= tol);
      }
      for (int64_t j = n; j < ys; ++j) CHECK(got[static_cast<size_t>(r * ys + j)] == -7.0f);  // padding untouched
    }
  };
  // gemv, in two column ranges (as two threads would).
  {
    qmm::ActBuffer buf;
    buf.prepare(mode, m, k, false);
    buf.quantize_rows(x.data(), k + 7, 0, m, tier);
    std::vector<float> got(static_cast<size_t>(m * ys), -7.0f);
    const int64_t split = n / 2;
    for (int64_t r0 = 0; r0 < m; r0 += qmm::kGemvMaxRows) {
      const int rows = static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, m - r0));
      tier.gemv(t, w.data(), rb, 0, split, buf.acts(), r0, rows, got.data(), ys, bias.data());
      tier.gemv(t, w.data(), rb, split, n, buf.acts(), r0, rows, got.data(), ys, bias.data());
    }
    compare(got, "gemv");
  }
  // Packed copy (4-row groups): gemv and gemm against the same reference.
  if (qmm::packable(t, k) && tier.gemv_packed != nullptr) {
    const int64_t groups = n / 4;
    std::vector<std::byte> packed(qmm::packed_bytes(t, n, k));
    qmm::pack_groups(t, w.data(), rb, k, 0, groups / 2, packed.data());  // two ranges, as two threads
    qmm::pack_groups(t, w.data(), rb, k, groups / 2, groups, packed.data());
    auto packed_rows_ok = [&](const std::vector<float>& got, const char* what) {
      for (int64_t r = 0; r < m; ++r) {
        for (int64_t j = 0; j < groups * 4; ++j) {
          const double g = got[static_cast<size_t>(r * ys + j)], e = want[static_cast<size_t>(r * ys + j)];
          const double tol = 1e-5 * mag[static_cast<size_t>(r * n + j)] + 1e-5;
          if (std::fabs(g - e) > tol) {
            std::fprintf(stderr, "%s %s %s %s m=%lld n=%lld k=%lld [%lld,%lld]: got %.7g want %.7g\n", tier.name, what,
                         std::string(dtype_name(t)).c_str(), mode_name(mode), static_cast<long long>(m),
                         static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(r),
                         static_cast<long long>(j), g, e);
          }
          CHECK(std::fabs(g - e) <= tol);
        }
      }
    };
    {
      qmm::ActBuffer buf;
      buf.prepare(mode, m, k, false);
      buf.quantize_rows(x.data(), k + 7, 0, m, tier);
      std::vector<float> got(static_cast<size_t>(m * ys), -7.0f);
      for (int64_t r0 = 0; r0 < m; r0 += qmm::kGemvMaxRows) {
        const int rows = static_cast<int>(std::min<int64_t>(qmm::kGemvMaxRows, m - r0));
        tier.gemv_packed(t, packed.data(), k, 0, groups / 2, buf.acts(), r0, rows, got.data(), ys, bias.data());
        tier.gemv_packed(t, packed.data(), k, groups / 2, groups, buf.acts(), r0, rows, got.data(), ys, bias.data());
      }
      packed_rows_ok(got, "gemv_packed");
    }
    if (tier.gemm_packed != nullptr) {
      qmm::ActBuffer buf;
      buf.prepare(mode, m, k, true);
      buf.quantize_rows(x.data(), k + 7, 0, m, tier);
      std::vector<float> got(static_cast<size_t>(m * ys), -7.0f);
      qmm::GemmScratch scratch;
      tier.gemm_packed(t, packed.data(), k, 0, groups, buf.acts(), got.data(), ys, bias.data(), scratch);
      packed_rows_ok(got, "gemm_packed");
    }
  }
  // gemm on the pair layout, column ranges on row-group boundaries plus a tail.
  if (tier.gemm != nullptr) {
    qmm::ActBuffer buf;
    buf.prepare(mode, m, k, true);
    buf.quantize_rows(x.data(), k + 7, 0, m, tier);
    std::vector<float> got(static_cast<size_t>(m * ys), -7.0f);
    qmm::GemmScratch scratch;
    const int64_t split = (n / 2) / qmm::kGemmRowGroup * qmm::kGemmRowGroup;
    tier.gemm(t, w.data(), rb, 0, split, buf.acts(), got.data(), ys, bias.data(), scratch);
    tier.gemm(t, w.data(), rb, split, n, buf.acts(), got.data(), ys, bias.data(), scratch);
    compare(got, "gemm");
  }
}

}  // namespace

#if !defined(_WIN32)
// release_file_pages: only ranges inside a live MappedFile are released, and
// their bytes stay readable (they fault back in from the file).
void check_release_file_pages() {
  char path[] = "/tmp/dynalm_test_release_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  if (fd < 0) return;
  std::vector<uint8_t> bytes(size_t{1} << 20);
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 131 + 7);
  CHECK(::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));
  ::close(fd);
  {
    auto f = MappedFile::open(path);
    CHECK(f.ok());
    if (f.ok()) {
      const std::byte* d = (*f)->data();
      uint64_t sum = 0;
      for (size_t i = 0; i < bytes.size(); i += 4096) sum += static_cast<uint8_t>(d[i]);
      CHECK(sum > 0);
      const size_t released = release_file_pages(d + 1000, bytes.size() - 2000);
      CHECK(released > 0 && released <= bytes.size() - 2000);
      CHECK(std::memcmp(d, bytes.data(), bytes.size()) == 0);
      // Past the end of the mapping: refused.
      CHECK(release_file_pages(d + 4096, bytes.size()) == 0);
    }
  }
  // Owned host memory is never released.
  std::vector<uint8_t> heap(bytes);
  CHECK(release_file_pages(heap.data(), heap.size()) == 0);
  CHECK(heap == bytes);
  std::remove(path);
}
#endif

// weight_panel_f32: exactly dequantize_row, transposed into k-major panels;
// from the packed copy where given (Q4_K / Q6_K), zero rows past n.
void check_weight_panels(DType t, int64_t n, int64_t k) {
  const std::vector<uint8_t> w = random_rows(t, n, k);
  const int64_t rb = dtype_row_bytes(t, k);
  std::vector<std::byte> packed;
  if (qmm::packable(t, k) && n >= 4) {
    packed.resize(qmm::packed_bytes(t, n, k));
    qmm::pack_groups(t, w.data(), rb, k, 0, n / 4, packed.data());
  }
  std::vector<float> wd(static_cast<size_t>(k)), panel(static_cast<size_t>(k * qmm::kSmePanel));
  for (const void* pk : {static_cast<const void*>(nullptr), packed.empty() ? nullptr : static_cast<const void*>(packed.data())}) {
    for (int64_t n0 = 0; n0 < n; n0 += qmm::kSmePanel) {
      std::fill(panel.begin(), panel.end(), -1.0f);
      qmm::weight_panel_f32(t, w.data(), rb, pk, n, k, n0, panel.data());
      for (int j = 0; j < qmm::kSmePanel; ++j) {
        if (n0 + j < n) CHECK(dequantize_row(t, w.data() + (n0 + j) * rb, wd.data(), k));
        for (int64_t i = 0; i < k; ++i) {
          const float want = n0 + j < n ? wd[static_cast<size_t>(i)] : 0.0f;
          const float got = panel[static_cast<size_t>(i * qmm::kSmePanel + j)];
          if (got != want) {
            // Seen once on a macos-14 Rosetta CI runner and not reproduced:
            // say exactly which value differed, bit patterns included.
            uint32_t gb = 0, wb = 0;
            std::memcpy(&gb, &got, 4);
            std::memcpy(&wb, &want, 4);
            std::fprintf(stderr, "weight_panel_f32 %s n=%lld k=%lld packed=%d row=%lld k_index=%lld: got %.9g (0x%08x), want %.9g (0x%08x)\n",
                         std::string(dtype_name(t)).c_str(), static_cast<long long>(n), static_cast<long long>(k),
                         pk != nullptr ? 1 : 0, static_cast<long long>(n0 + j), static_cast<long long>(i), got, gb, want, wb);
          }
          CHECK(got == want);
        }
      }
    }
  }
}

// SME panel gemm vs a double-precision product of the same fp32 operands.
void check_sme_gemm(const qmm::SmeGemm& g, int64_t m, int64_t n, int64_t k) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(m * (k + 3))), w(static_cast<size_t>(n * k));
  for (float& v : x) v = nd(g_rng);
  for (float& v : w) v = nd(g_rng);
  const int64_t xs = k + 3, ys = n + 5;
  std::vector<float> y(static_cast<size_t>(m * ys), -7.0f);
  std::vector<float> xp(static_cast<size_t>(k * qmm::kSmePanel)), wp(static_cast<size_t>(k * qmm::kSmePanel));
  for (int64_t n0 = 0; n0 < n; n0 += qmm::kSmePanel) {
    for (int j = 0; j < qmm::kSmePanel; ++j) {
      for (int64_t i = 0; i < k; ++i) {
        wp[static_cast<size_t>(i * qmm::kSmePanel + j)] = n0 + j < n ? w[static_cast<size_t>((n0 + j) * k + i)] : 0.0f;
      }
    }
    for (int64_t m0 = 0; m0 < m; m0 += qmm::kSmePanel) {
      g.pack_x(x.data(), xs, m, k, m0, xp.data());
      // Odd k: the first half with block, the rest accumulated with block_acc.
      const int64_t k1 = k / 2;
      g.block(xp.data(), wp.data(), k1, y.data() + m0 * ys + n0, ys, static_cast<int>(std::min<int64_t>(qmm::kSmePanel, m - m0)),
              static_cast<int>(std::min<int64_t>(qmm::kSmePanel, n - n0)));
      g.block_acc(xp.data() + k1 * qmm::kSmePanel, wp.data() + k1 * qmm::kSmePanel, k - k1, y.data() + m0 * ys + n0, ys,
                  static_cast<int>(std::min<int64_t>(qmm::kSmePanel, m - m0)), static_cast<int>(std::min<int64_t>(qmm::kSmePanel, n - n0)));
    }
  }
  for (int64_t r = 0; r < m; ++r) {
    for (int64_t j = 0; j < ys; ++j) {
      const float got = y[static_cast<size_t>(r * ys + j)];
      if (j >= n) {
        CHECK(got == -7.0f);  // padding untouched
        continue;
      }
      double want = 0, mag = 0;
      for (int64_t i = 0; i < k; ++i) {
        const double p = static_cast<double>(x[static_cast<size_t>(r * xs + i)]) * w[static_cast<size_t>(j * k + i)];
        want += p;
        mag += std::fabs(p);
      }
      CHECK(std::fabs(got - want) <= 1e-5 * mag + 1e-6);
    }
  }
}

int main() {
  const DType formats[] = {DType::kQ8_0, DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ5_1, DType::kIQ4_NL,
                           DType::kQ3_K, DType::kQ4_K, DType::kQ5_K, DType::kQ6_K, DType::kIQ4_XS};
  for (DType t : formats) CHECK(qmm::supports(t));
  CHECK(!qmm::supports(DType::kF32) && !qmm::supports(DType::kQ2_K));

  std::vector<const qmm::Kernels*> tiers = {&qmm::reference_kernels()};
  for (const char* name : {"neon_dot", "neon_i8mm"}) {
    if (const qmm::Kernels* k = qmm::kernels_by_name(name)) tiers.push_back(k);
  }
  std::string names;
  for (const qmm::Kernels* t : tiers) names += std::string(names.empty() ? "" : ", ") + t->name;
  std::printf("tiers: %s (best: %s)\n", names.c_str(), qmm::best_kernels().name);

  for (const qmm::Kernels* tier : tiers) {
    for (qmm::ActMode mode : {qmm::ActMode::kInt8, qmm::ActMode::kInt16}) {
      if (mode == qmm::ActMode::kInt16 && !tier->int16) continue;
      check_quantization(*tier, mode, false);
      check_quantization(*tier, mode, true);
    }
  }
  for (DType t : formats) {
    const bool legacy = dtype_block_elems(t) == 32;
    for (qmm::ActMode mode : {qmm::ActMode::kInt8, qmm::ActMode::kInt16}) {
      check_reference_oracle(t, mode, 512);
      if (legacy) check_reference_oracle(t, mode, 352);  // k % 256 != 0
    }
  }
  for (const qmm::Kernels* tier : tiers) {
    if (tier == &qmm::reference_kernels()) continue;
    for (DType t : formats) {
      const bool legacy = dtype_block_elems(t) == 32;
      for (qmm::ActMode mode : {qmm::ActMode::kInt8, qmm::ActMode::kInt16}) {
        if (mode == qmm::ActMode::kInt16 && !tier->int16) continue;
        for (int64_t m : {1, 2, 3, 4, 5, 8, 9, 13}) check_tier(*tier, t, mode, m, 13, 512);
        check_tier(*tier, t, mode, 17, 36, 768);
        if (legacy) {
          check_tier(*tier, t, mode, 3, 9, 352);
          check_tier(*tier, t, mode, 11, 9, 352);
        }
      }
    }
  }
#if !defined(_WIN32)
  check_release_file_pages();
#endif
  for (DType t : formats) {
    check_weight_panels(t, 37, dtype_block_elems(t) == 32 ? 352 : 512);
    check_weight_panels(t, 8, 256);
  }
  if (const qmm::SmeGemm* g = qmm::sme_gemm()) {
    std::printf("sme tier: %s\n", g->name);
    for (int64_t m : {1, 5, 16, 17, 31, 32, 33, 70}) {
      for (int64_t n : {1, 15, 16, 17, 32, 45, 64}) check_sme_gemm(*g, m, n, 96);
    }
    check_sme_gemm(*g, 64, 96, 1536);
  } else {
    std::printf("sme tier: not available\n");
  }
  std::printf("all qmm tests passed (%d checks)\n", g_checks);
  return 0;
}
