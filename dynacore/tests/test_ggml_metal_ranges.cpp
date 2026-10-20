// ggml-metal's concurrency check (ggml_mem_ranges_check, pinned ggml with
// cmake/patches/ggml-metal-mem-ranges-touching.patch): which kernels may run
// together. Ranges are half-open, [data, data + size): two tensors that only
// touch do not overlap, one shared byte does. Before the patch, a source that
// ended exactly where an earlier destination began counted as overlapping and
// put a barrier between independent kernels. Host memory stands in for a Metal
// buffer (the check compares buffer ids and addresses only). Skips (exit 0)
// without Metal in the build.

#include <cstdio>
#include <cstdlib>

#if DYNACORE_HAS_METAL

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal-common.h"
#include "ggml.h"

namespace {

int checks = 0;
#define CHECK(c)                                                       \
  do {                                                                 \
    ++checks;                                                          \
    if (!(c)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

constexpr int64_t kN = 256;                          // f32 elements per tensor
constexpr size_t kBytes = kN * sizeof(float);        // 1 KiB

}  // namespace

int main() {
  ggml_init_params p{};
  p.mem_size = 64 * ggml_tensor_overhead();
  p.no_alloc = true;
  ggml_context* ctx = ggml_init(p);
  ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 16 * kBytes);
  CHECK(ctx != nullptr && buf != nullptr);
  auto* base = static_cast<char*>(ggml_backend_buffer_get_base(buf));
  // A tensor at byte `at` of the buffer (`n` elements), produced from `src` (or a leaf).
  auto at = [&](size_t off, ggml_tensor* src = nullptr, int64_t n = kN) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    CHECK(ggml_backend_tensor_alloc(buf, t, base + off) == GGML_STATUS_SUCCESS);
    t->src[0] = src;
    return t;
  };

  // Q = W_q x writes [kBytes, 2 kBytes); x lives at [0, kBytes).
  ggml_tensor* x = at(0);
  ggml_tensor* q = at(kBytes, x);
  ggml_mem_ranges_t mrs = ggml_mem_ranges_init(0);
  CHECK(ggml_mem_ranges_add(mrs, q));

  // K = W_k x reads x, which ends exactly where Q's output begins: concurrent
  // (this was the false barrier).
  CHECK(ggml_mem_ranges_check(mrs, at(2 * kBytes, x)));
  // A source that begins exactly where Q's output ends: concurrent.
  CHECK(ggml_mem_ranges_check(mrs, at(4 * kBytes, at(2 * kBytes))));
  // A destination right after Q's output: concurrent.
  CHECK(ggml_mem_ranges_check(mrs, at(2 * kBytes, at(8 * kBytes))));

  // Genuine overlaps are still caught:
  CHECK(!ggml_mem_ranges_check(mrs, at(4 * kBytes, q)));                       // reads Q's output
  CHECK(!ggml_mem_ranges_check(mrs, at(4 * kBytes, at(2 * kBytes - 4, nullptr, 1))));  // its last element
  CHECK(!ggml_mem_ranges_check(mrs, at(4 * kBytes, at(kBytes - 4, nullptr, 2))));       // straddles its start
  CHECK(!ggml_mem_ranges_check(mrs, at(kBytes + 512, at(8 * kBytes))));       // writes inside it
  CHECK(!ggml_mem_ranges_check(mrs, at(0, at(8 * kBytes))));                  // writes x, which Q reads
  CHECK(!ggml_mem_ranges_check(mrs, at(kBytes - 4, at(8 * kBytes), 1)));      // overwrites x's last element

  // Two sources may always overlap (reads only): part of x.
  CHECK(ggml_mem_ranges_check(mrs, at(4 * kBytes, at(kBytes / 2, nullptr, kN / 4))));

  ggml_mem_ranges_free(mrs);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  std::printf("all ggml-metal memory-range tests passed (%d checks)\n", checks);
  return 0;
}

#else

int main() {
  std::printf("no Metal in this build (ENABLE_METAL=OFF): nothing to test\n");
  return 0;
}

#endif
