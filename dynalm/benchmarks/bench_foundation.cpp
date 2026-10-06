// Phase 0 microbenchmarks: costs of primitives that will sit on hot paths.
//   - now_ns():            every latency metric (TTFT/ITL) takes timestamps
//   - disabled LOG_DEBUG:  must be ~free so debug logging can stay in code
//   - Result<int> OK path: must not allocate

#include <cstdio>

#include "bench_harness.h"
#include "dynacore/base/status.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"

namespace {

engine::Result<int> make_result(int v) { return v; }

}  // namespace

int main() {
  using namespace engine;
  std::printf("cpu: %s\n\n", cpu_info().brand.c_str());
  bench::print_header();

  bench::print_row("now_ns()", bench::run([] { bench::do_not_optimize(now_ns()); }));

  log::set_level(log::Level::kInfo);
  int counter = 0;
  bench::print_row("LOG_DEBUG (disabled)", bench::run([&] { LOG_DEBUG("token {} logits", ++counter); }));

  int v = 0;
  bench::print_row("Result<int> ok path", bench::run([&] {
                     auto r = make_result(++v);
                     bench::do_not_optimize(r);
                   }));
  return 0;
}
