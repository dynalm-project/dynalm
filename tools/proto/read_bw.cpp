// DRAM read-bandwidth ceiling for decode (compiler roadmap baseline).
//
// Streams a buffer far larger than the LLC with N threads using AVX2 loads
// and a trivial reduction (what a bandwidth-bound GEMV can at best do), and
// reports GB/s. Build with the msvc dev shell:
//   cl /O2 /arch:AVX2 /EHsc /std:c++20 tools\proto\read_bw.cpp
//   read_bw [threads=10] [MiB=1024] [reps=8]

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  const int threads = argc > 1 ? std::atoi(argv[1]) : 10;
  const size_t mib = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1024;
  const int reps = argc > 3 ? std::atoi(argv[3]) : 8;
  const size_t bytes = mib << 20;
  auto* buf = static_cast<float*>(_mm_malloc(bytes, 64));
  std::memset(buf, 1, bytes);  // fault in every page
  const size_t n = bytes / sizeof(float);
  std::vector<double> sink(static_cast<size_t>(threads) * 8);
  double best = 0;
  for (int r = 0; r < reps; ++r) {
    std::vector<std::thread> ts;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t) {
      ts.emplace_back([&, t] {
        const size_t lo = n / threads * t, hi = t + 1 == threads ? n : n / threads * (t + 1);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t i = lo; i + 32 <= hi; i += 32) {
          a0 = _mm256_add_ps(a0, _mm256_load_ps(buf + i));
          a1 = _mm256_add_ps(a1, _mm256_load_ps(buf + i + 8));
          a2 = _mm256_add_ps(a2, _mm256_load_ps(buf + i + 16));
          a3 = _mm256_add_ps(a3, _mm256_load_ps(buf + i + 24));
        }
        float out[8];
        _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        sink[static_cast<size_t>(t) * 8] = out[0];
      });
    }
    for (auto& th : ts) th.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    best = std::max(best, static_cast<double>(bytes) / s / 1e9);
  }
  std::printf("threads %d, %zu MiB: best %.1f GB/s (sink %g)\n", threads, mib, best, sink[0]);
  _mm_free(buf);
  return 0;
}
