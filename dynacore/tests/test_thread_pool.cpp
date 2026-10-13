// ThreadPool region ownership: reentrant and concurrent parallel_for are
// detected and serialized (never racing on the in-flight job), counted by
// misuse_count(); ordinary use is unaffected. Deterministic: concurrent cases
// hold a region open with a latch until the second caller has returned, so no
// timing assumption decides the outcome. Every wait has a timeout so a broken
// guard fails instead of hanging.

#include <algorithm>
#include <atomic>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

using namespace dynacore;

namespace {

int g_checks = 0;

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

// Waits (spinning with yields) until `flag` is set; false after `ms`.
bool wait_for(const std::atomic<bool>& flag, int ms = 10000) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() > end) return false;
    std::this_thread::yield();
  }
  return true;
}

// Every index of [0, n) visited exactly once.
bool exactly_once(const std::vector<std::atomic<int>>& hits) {
  for (const auto& h : hits) {
    if (h.load() != 1) return false;
  }
  return true;
}

void test_ordinary_use() {
  ThreadPool pool(4);
  for (int round = 0; round < 200; ++round) {
    const size_t n = 1000 + static_cast<size_t>(round);
    std::vector<std::atomic<int>> hits(n);
    pool.parallel_for(n, 7, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) hits[i].fetch_add(1);
    });
    CHECK(exactly_once(hits));
  }
  // Inline regions (n <= grain) and a 1-thread pool: not regions, never misuse.
  ThreadPool solo(1);
  std::atomic<int> sum{0};
  pool.parallel_for(5, 8, [&](size_t b, size_t e) { sum += static_cast<int>(e - b); });
  solo.parallel_for(100, 1, [&](size_t b, size_t e) { sum += static_cast<int>(e - b); });
  CHECK(sum == 105);
  // Sequential regions from different threads (joined in turn) are not misuse.
  for (int t = 0; t < 4; ++t) {
    std::thread th([&] { pool.parallel_for(64, 1, [](size_t, size_t) {}); });
    th.join();
  }
  CHECK(pool.misuse_count() == 0 && solo.misuse_count() == 0);
  std::printf("  ordinary use: 200 regions exact, inline and cross-thread sequential regions, misuse 0\n");
}

void test_reentrant() {
  ThreadPool pool(4);
  const size_t n = 64, inner_n = 50;
  std::vector<std::atomic<int>> outer(n);
  std::atomic<int> nested_calls{0};
  std::vector<std::atomic<int>> inner(n * inner_n);
  pool.parallel_for(n, 1, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      outer[i].fetch_add(1);
      if (i % 8 == 0) {  // fn calls parallel_for on its own pool: misuse
        nested_calls.fetch_add(1);
        pool.parallel_for(inner_n, 3, [&](size_t ib, size_t ie) {
          CHECK(ie - ib <= 3);  // the serialized fallback keeps the chunk bound
          for (size_t j = ib; j < ie; ++j) inner[i * inner_n + j].fetch_add(1);
        });
      }
    }
  });
  CHECK(exactly_once(outer));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < inner_n; ++j) CHECK(inner[i * inner_n + j].load() == (i % 8 == 0 ? 1 : 0));
  }
  CHECK(pool.misuse_count() == static_cast<uint64_t>(nested_calls.load()));
  CHECK(nested_calls.load() == 8);
  // The pool still runs ordinary regions afterwards.
  std::vector<std::atomic<int>> after(500);
  pool.parallel_for(500, 1, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) after[i].fetch_add(1);
  });
  CHECK(exactly_once(after) && pool.misuse_count() == 8);
  std::printf("  reentrant: 8 nested regions serialized and counted; outer region exact; pool healthy\n");
}

void test_concurrent() {
  ThreadPool pool(4);
  std::atomic<bool> holder_in{false}, second_done{false};
  const size_t n = 256;
  std::vector<std::atomic<int>> first(n), second(n);
  std::thread holder([&] {
    pool.parallel_for(n, 1, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) {
        first[i].fetch_add(1);
        if (i == 0) {
          // Keep this region in flight until the second caller is done.
          holder_in.store(true, std::memory_order_release);
          CHECK(wait_for(second_done));
        }
      }
    });
  });
  CHECK(wait_for(holder_in));
  // A second thread asks for a region while the first is in flight: it must
  // not touch the in-flight job, must not wait for it, and must still compute.
  pool.parallel_for(n, 4, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) second[i].fetch_add(1);
  });
  CHECK(exactly_once(second));
  CHECK(pool.misuse_count() == 1);
  second_done.store(true, std::memory_order_release);
  holder.join();
  CHECK(exactly_once(first));
  std::printf("  concurrent: second caller serialized without waiting; both regions exact; misuse 1\n");
}

// A DynaKernel backend sharing a pool whose region is in flight reports the
// violation through Status (results still complete).
void test_backend_reports_shared_pool() {
  ThreadPool pool(4);
  dk::ReferenceBackend be(pool, select_isa(cpu_info().features));
  dk::ProgramBuilder b;
  const dk::ValueId vx = b.activation("x", DType::kF32, 16, 64);
  const dk::ValueId vw = b.weight("w", DType::kF32, 512, 64);  // 128 panels: a real region, not inline
  const dk::ValueId vy = b.matmul(vx, vw);
  auto p = b.finish();
  CHECK(p.ok());
  std::vector<float> x(16 * 64, 1.0f), w(512 * 64, 0.5f), y(16 * 512, 0.0f);
  auto lay = [](int64_t r, int64_t c) { return *TensorLayout::contiguous(DType::kF32, TensorShape{r, c}); };
  dk::Bindings bi;
  bi.bind(vx, TensorView(x.data(), lay(16, 64)));
  bi.bind(vw, TensorView(w.data(), lay(512, 64)));
  bi.bind(vy, TensorView(y.data(), lay(16, 512)));
  CHECK(be.run(*p, bi).ok());  // pool idle: fine
  std::atomic<bool> holder_in{false}, run_done{false};
  std::thread holder([&] {
    pool.parallel_for(8, 1, [&](size_t b0, size_t) {
      if (b0 == 0) {
        holder_in.store(true, std::memory_order_release);
        CHECK(wait_for(run_done));
      }
    });
  });
  CHECK(wait_for(holder_in));
  std::fill(y.begin(), y.end(), 0.0f);
  const Status s = be.run(*p, bi);
  run_done.store(true, std::memory_order_release);
  holder.join();
  CHECK(s.code() == StatusCode::kInternal);
  CHECK(std::string(s.message()).find("thread pool contract violated") != std::string::npos);
  CHECK(std::all_of(y.begin(), y.end(), [](float v) { return v == 32.0f; }));  // 64 * 1 * 0.5, computed serially
  CHECK(be.run(*p, bi).ok());  // idle again: fine
  std::printf("  backend on a busy shared pool: kInternal \"%s\"\n", s.message().substr(0, 60).c_str());
}

}  // namespace

int main() {
  std::printf("test_thread_pool\n");
  test_ordinary_use();
  test_reentrant();
  test_concurrent();
  test_backend_reports_shared_pool();
  std::printf("all thread pool tests passed (%d checks)\n", g_checks);
  return 0;
}
