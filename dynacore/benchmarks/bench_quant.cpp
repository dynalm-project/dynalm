// Phase 8 benchmark: reference dequantization throughput per block format
// (one 4096-element row, the unit a matmul converts per output feature).

#include <cstdio>
#include <random>
#include <vector>

#include "bench_harness.h"
#include "dynacore/tensor/fp16.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/quantization/dequant.h"

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  constexpr int64_t kN = 4096;
  std::mt19937 rng(1);
  std::vector<float> out(kN);
  std::printf("%-8s %10s %10s %10s %12s\n", "type", "p50 ns", "p99 ns", "ns/elem", "GB/s (src)");
  for (DType t : {DType::kF16, DType::kBF16, DType::kQ8_0, DType::kQ4_0, DType::kQ5_0, DType::kQ2_K, DType::kQ3_K,
                  DType::kQ4_K, DType::kQ5_K, DType::kQ6_K}) {
    const int64_t bytes = dtype_row_bytes(t, kN);
    std::vector<uint8_t> src(static_cast<size_t>(bytes));
    for (auto& b : src) b = static_cast<uint8_t>(rng());
    // Keep fp16 scale fields finite: overwrite every 2-byte-aligned word that
    // decodes to inf/nan (harmless for timing).
    for (size_t i = 0; i + 1 < src.size(); i += 2) {
      uint16_t h = static_cast<uint16_t>(src[i] | (src[i + 1] << 8));
      if ((h & 0x7C00) == 0x7C00) src[i + 1] &= 0xBB;
    }
    auto s = bench::run([&] {
      dequantize_row(t, src.data(), out.data(), kN);
      bench::do_not_optimize(out[0]);
    }, {.warmup_samples = 5, .samples = 200, .batch = 20});
    std::printf("%-8s %10.0f %10.0f %10.3f %12.2f\n", std::string(dtype_name(t)).c_str(), s.p50, s.p99,
                s.p50 / kN, bytes / s.p50);
  }
  return 0;
}
