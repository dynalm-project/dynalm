// dynacorec: the DynaCore compiler driver (DD-073).
//
//   dynacorec <file.dyna | file.ir> [options]
//
//   --graph NAME          pick one graph of a .dyna file (default: the first)
//   --set SYM=N           bind a graph parameter (e.g. --set M=4); repeatable
//   --dump-ir             print the IR as written (before optimization)
//   --dump-optimized-ir   print the IR after canonicalization, kernel selection and fusion
//   --dump-kernels        print the execution plan: one line per device call
//   --memory              print the memory analysis
//   --emit FILE           write the optimized IR text to FILE
//   --benchmark           run the graph on this CPU: unplanned vs compiled plan,
//                         interleaved, with an exactness check of the outputs
//   --iters N             benchmark iterations per variant (default 50)
//   --threads N           compute threads (default: physical cores)
//   --no-fuse             disable fusion (overrides the program's schedule)
//
// Exit code: 0 ok, 1 compile/verify error, 2 usage, 3 benchmark mismatch.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/executor.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/text.h"
#include "dynacore/ir/verifier.h"
#include "dynacore/kernel/kernel_plan.h"
#include "dynacore/lang/lang.h"
#include "dynacore/version.h"

using namespace dynacore;

namespace {

int usage() {
  std::fprintf(stderr,
               "dynacorec %s - DynaCore inference compiler\n"
               "usage: dynacorec <file.dyna|file.ir> [--graph NAME] [--set SYM=N] [--dump-ir] [--dump-optimized-ir]\n"
               "                 [--dump-kernels] [--memory] [--emit FILE] [--benchmark] [--iters N] [--threads N]\n"
               "                 [--no-fuse]\n",
               kVersionString);
  return 2;
}

std::string plan_text(const ir::Graph& g, const ir::ExecPlan& plan) {
  std::string s;
  int n = 0;
  for (const ir::ExecStep& st : plan.steps) {
    char head[64];
    std::snprintf(head, sizeof head, "%4d  ", n++);
    std::string line = head;
    switch (st.kind) {
      case ir::ExecStep::Kind::kOp: line += "op          "; break;
      case ir::ExecStep::Kind::kMatmulGroup: line += "matmul_many "; break;
      case ir::ExecStep::Kind::kGatedMatmul: line += "matmul_gated"; break;
    }
    for (int32_t op : st.ops) {
      const ir::Op& o = g.op(op);
      line += " %" + std::to_string(o.result) + "=" + std::string(ir::op_name(o.kind));
      if (!o.kernel.empty()) line += "[" + o.kernel + "]";
    }
    s += line + "\n";
  }
  return s;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  std::string path, graph_name, emit;
  std::map<std::string, int64_t> sets;
  bool dump_ir = false, dump_opt = false, dump_kernels = false, memory = false, bench = false, no_fuse = false;
  int iters = 50, threads = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (a == "--graph") {
      const char* v = value();
      if (!v) return usage();
      graph_name = v;
    } else if (a == "--set") {
      const char* v = value();
      const char* eq = v ? std::strchr(v, '=') : nullptr;
      if (!eq) return usage();
      sets[std::string(v, eq)] = std::atoll(eq + 1);
    } else if (a == "--dump-ir") {
      dump_ir = true;
    } else if (a == "--dump-optimized-ir") {
      dump_opt = true;
    } else if (a == "--dump-kernels") {
      dump_kernels = true;
    } else if (a == "--memory") {
      memory = true;
    } else if (a == "--emit") {
      const char* v = value();
      if (!v) return usage();
      emit = v;
    } else if (a == "--benchmark") {
      bench = true;
    } else if (a == "--iters") {
      const char* v = value();
      if (!v) return usage();
      iters = std::max(1, std::atoi(v));
    } else if (a == "--threads") {
      const char* v = value();
      if (!v) return usage();
      threads = std::atoi(v);
    } else if (a == "--no-fuse") {
      no_fuse = true;
    } else if (!a.empty() && a[0] != '-' && path.empty()) {
      path = a;
    } else {
      return usage();
    }
  }
  if (path.empty()) return usage();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "dynacorec: cannot read %s\n", path.c_str());
    return 1;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string src = ss.str();

  lang::Program prog;
  if (path.size() > 3 && path.substr(path.size() - 3) == ".ir") {
    auto g = ir::parse_graph(src);
    if (!g.ok()) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), g.status().message().c_str());
      return 1;
    }
    prog.graph = std::move(*g);
  } else {
    auto p = lang::compile(src, path, graph_name);
    if (!p.ok()) {
      std::fprintf(stderr, "%s\n", p.status().message().c_str());
      return 1;
    }
    prog = std::move(*p);
  }
  for (const auto& [k, v] : sets) prog.symbols[k] = v;
  if (no_fuse) prog.fusion = ir::FusionOptions{false, false};

  const std::vector<std::string> errors = ir::verify_all(prog.graph);
  if (!errors.empty()) {
    for (const std::string& e : errors) std::fprintf(stderr, "%s: %s\n", path.c_str(), e.c_str());
    return 1;
  }
  if (dump_ir) std::printf("%s", ir::print_graph(prog.graph).c_str());

  // The pipeline compiled execution uses (passes.h).
  const int nthreads = threads > 0 ? threads : cpu_info().physical_cores;
  ir::Graph optimized = prog.graph;
  ir::CompileOptions co;
  co.verify = true;
  co.fusion = prog.fusion;
  co.cost = ir::CostModel::from(KernelPlan::defaults(), nthreads);
  ir::CompileReport report;
  auto plan = ir::compile_segment(optimized, co, &report);
  if (!plan.ok()) {
    std::fprintf(stderr, "%s: %s\n", path.c_str(), plan.status().message().c_str());
    return 1;
  }
  if (dump_opt) std::printf("%s", ir::print_graph(optimized).c_str());
  if (dump_kernels) std::printf("%s", plan_text(optimized, *plan).c_str());
  if (memory) {
    int64_t m = 1;
    if (auto it = prog.symbols.find("M"); it != prog.symbols.end()) m = it->second;
    const ir::MemoryReport r = ir::analyze_memory(optimized, m);
    std::printf("memory (M=%lld): %d activation values, %lld KiB; peak live %lld KiB; %d buffers %lld KiB; "
                "with reuse %lld KiB\n",
                static_cast<long long>(m), r.values, static_cast<long long>(r.activation_bytes >> 10),
                static_cast<long long>(r.peak_live_bytes >> 10), r.buffers, static_cast<long long>(r.buffer_bytes >> 10),
                static_cast<long long>(r.planned_bytes >> 10));
  }
  if (!emit.empty()) std::ofstream(emit, std::ios::binary) << ir::print_graph(optimized);
  std::fprintf(stderr, "%s: ok: %zu ops, %zu steps (%d groups, %d gated)\n", path.c_str(), optimized.ops().size(),
               plan->steps.size(), report.fusion.groups, report.fusion.gated);
  if (!bench) return 0;

  // Benchmark: unplanned (one call per op) vs the compiled plan, interleaved.
  ThreadPool pool(nthreads);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  ir::ExecutorOptions eo;
  eo.symbols = prog.symbols;
  auto ex = ir::GraphExecutor::create(optimized, cpu, eo);
  if (!ex.ok()) {
    std::fprintf(stderr, "dynacorec: %s\n", ex.status().message().c_str());
    return 1;
  }
  auto snapshot = [&]() {
    std::vector<std::vector<float>> out;
    for (ir::ValueId v : prog.outputs) {
      const TensorView t = (*ex)->view(v);
      const auto* p = t.data_as<const float>();
      out.emplace_back(p, p + t.numel());
    }
    return out;
  };
  if (!(*ex)->run(nullptr).ok()) return 1;
  const auto ref = snapshot();
  if (!(*ex)->run(&*plan).ok()) return 1;
  const bool exact = snapshot() == ref;
  std::vector<double> t_ref, t_plan;
  for (int i = 0; i < iters; ++i) {
    for (int v = 0; v < 2; ++v) {
      const bool planned = (v + i) % 2 == 1;
      const auto t0 = std::chrono::steady_clock::now();
      (void)(*ex)->run(planned ? &*plan : nullptr);
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      (planned ? t_plan : t_ref).push_back(us);
    }
  }
  const double a = median(t_ref), b = median(t_plan);
  std::printf("benchmark (%s, %d threads, %s): unplanned %.1f us, compiled %.1f us, %+.1f%%; outputs %s\n",
              std::string(cpu.name()).c_str(), nthreads,
              prog.symbols.empty() ? "no symbols"
                                   : ("M=" + std::to_string(prog.symbols.count("M") ? prog.symbols["M"] : 1)).c_str(),
              a, b, 100 * (b / a - 1), exact ? "bit-identical" : "DIFFER");
  return exact ? 0 : 3;
}
