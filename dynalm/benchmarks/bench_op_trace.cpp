// Compiler roadmap: the op-level profile of a real forward pass.
//
//   bench_op_trace <model.gguf> [threads] [steps=16] [--dump FILE]
//
// Runs the Transformer on a RecordingDevice in trace mode: every Device op is
// captured as DynaCore IR and timed individually. Prints time by op kind and
// the slowest individual ops of a decode step, and optionally writes the IR
// of the last decode step (bench_op_trace ... --dump step.ir). This is the
// measurement compiler decisions are made from.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/ir/text.h"
#include "dynacore/ir/verifier.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "common/core.h"

using namespace dynalm;
namespace ir = dynacore::ir;

namespace {

std::string op_label(const ir::Graph& g, const ir::Op& op) {
  std::string s(ir::op_name(op.kind));
  if ((op.kind == ir::OpKind::kQuantizedMatMul || op.kind == ir::OpKind::kMatMul) && op.inputs.size() >= 2) {
    const ir::Type& w = g.value(op.inputs[1]).type;
    s += "." + std::string(dtype_name(w.dtype)) + "[" + std::to_string(w.shape[0].size) + "x" +
         std::to_string(w.shape[1].size) + "]";
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_op_trace <model.gguf> [threads] [steps=16] [--dump FILE]\n");
    return 1;
  }
  std::string dump;
  std::vector<char*> pos_args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
    else pos_args.push_back(argv[i]);
  }
  const int threads = pos_args.size() > 1 ? std::atoi(pos_args[1]) : cpu_info().physical_cores;
  const int steps = pos_args.size() > 2 ? std::atoi(pos_args[2]) : 16;
  auto m = load_model(pos_args[0]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  ir::RecordingDevice rec(cpu, ir::RecordingDevice::Mode::kTrace);
  rec.set_timing(true);
  const int chunk = std::getenv("BENCH_TRACE_PREFILL") ? std::max(1, std::atoi(std::getenv("BENCH_TRACE_PREFILL"))) : 0;
  // BENCH_TRACE_CTX=n: tokens of context before the traced steps (default 256).
  const int ctx0 = std::getenv("BENCH_TRACE_CTX") ? std::max(1, std::atoi(std::getenv("BENCH_TRACE_CTX"))) : 256;
  const int64_t ctx_tokens = ctx0 + 512 + static_cast<int64_t>(steps) * std::max(chunk, 1);
  // BENCH_TRACE_USERS=n: n sequences decode together (batched decode steps).
  const int users = std::getenv("BENCH_TRACE_USERS") ? std::max(1, std::atoi(std::getenv("BENCH_TRACE_USERS"))) : 1;
  auto kv = KvBlockPool::create(kv_geometry_for(c, DType::kF16, 16, (ctx_tokens + 64) * users), rec);
  auto tf = Transformer::create(c, (*m)->weights, rec, std::max(256, chunk));
  if (!kv.ok() || !tf.ok()) return std::fprintf(stderr, "setup failed\n"), 1;
  // The engine's CPU plan: int8 activations at any row count where qmm runs
  // (BENCH_INT8_ROWS=n overrides; 0 = fp32 reference).
  {
    KernelPlan base = KernelPlan::defaults();
    const char* ir = std::getenv("BENCH_INT8_ROWS");
    base.int8_decode_max_rows = ir != nullptr && *ir != '\0' ? static_cast<int32_t>(std::atol(ir)) : 1 << 30;
    (*tf)->set_kernel_base(base);
  }
  const TensorRegistry& w = (*m)->weights;
  for (int r = 0; r < static_cast<int>(TensorRole::kCount); ++r) {
    const auto role = static_cast<TensorRole>(r);
    if (const Tensor* t = w.find(role, -1)) rec.set_name(t->data(), std::string(tensor_role_name(role)));
    for (int l = 0; l < w.num_layers(); ++l) {
      if (const Tensor* t = w.find(role, l)) {
        rec.set_name(t->data(), "blk." + std::to_string(l) + "." + std::string(tensor_role_name(role)));
      }
    }
  }

  std::vector<KvBlockTable> seqs;
  for (int u = 0; u < users; ++u) {
    seqs.emplace_back(**kv);
    if (!seqs.back().reserve(ctx_tokens).ok()) return 1;
  }
  KvBlockTable& seq = seqs.front();
  std::vector<float> logits(static_cast<size_t>(c.vocab_size) * users);
  for (int u = 0; u < users; ++u) {
    for (int p0 = 0; p0 < ctx0; p0 += 256) {  // prefill, pages weights in
      std::vector<TokenId> toks(static_cast<size_t>(std::min(256, ctx0 - p0)));
      for (size_t i = 0; i < toks.size(); ++i) toks[i] = static_cast<TokenId>(100 + (p0 + i + 7 * u) % 500);
      std::vector<int32_t> pos(toks.size());
      std::iota(pos.begin(), pos.end(), p0);
      if (!(*tf)->forward(toks, pos, **kv, seqs[static_cast<size_t>(u)].block_table(),
                          std::span<float>(logits).first(static_cast<size_t>(c.vocab_size))).ok()) {
        return 1;
      }
    }
  }

  std::map<std::string, std::pair<int64_t, int64_t>> by_kind;  // label -> (ns, count)
  std::vector<std::pair<int64_t, std::string>> slowest;
  int64_t total_ns = 0, wall_ns = 0;
  size_t ops_per_step = 0;
  for (int s = 0; s < steps; ++s) {
    rec.begin_graph(chunk > 0 ? "prefill" : "decode");
    std::vector<TokenId> ct(static_cast<size_t>(std::max(chunk, 1)), 100);
    std::vector<int32_t> cp(ct.size());
    for (size_t i = 0; i < cp.size(); ++i) cp[i] = ctx0 + s * static_cast<int32_t>(ct.size()) + static_cast<int32_t>(i);
    const auto t0 = std::chrono::steady_clock::now();
    if (users > 1) {
      std::vector<SeqBatch> batch(static_cast<size_t>(users));
      for (int u = 0; u < users; ++u) {
        batch[static_cast<size_t>(u)].tokens = ct;
        batch[static_cast<size_t>(u)].start_pos = cp.front();
        batch[static_cast<size_t>(u)].block_table = seqs[static_cast<size_t>(u)].block_table();
      }
      if (!(*tf)->forward_batch(batch, **kv, logits).ok()) return 1;
    } else if (!(*tf)->forward(ct, cp, **kv, seq.block_table(),
                               std::span<float>(logits).first(static_cast<size_t>(c.vocab_size))).ok()) {
      return 1;
    }
    wall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    const ir::Graph& g = rec.graph();
    if (s == 0) {
      if (Status st = ir::verify(g); !st.ok()) std::fprintf(stderr, "warning: %s\n", st.message().c_str());
    }
    ops_per_step = 0;
    for (size_t i = 0; i < g.ops().size(); ++i) {
      const ir::Op& op = g.op(static_cast<int32_t>(i));
      const int64_t ns = rec.op_ns()[i];
      if (ns == 0) continue;
      ++ops_per_step;
      const std::string label = op_label(g, op);
      by_kind[label].first += ns;
      by_kind[label].second += 1;
      total_ns += ns;
      if (s == steps - 1) slowest.emplace_back(ns, label + " " + g.value(op.result).name);
    }
    if (s == steps - 1 && !dump.empty()) {
      std::ofstream(dump) << ir::print_graph(g);
      std::printf("wrote %s (%zu ops)\n", dump.c_str(), g.ops().size());
    }
  }
  const double per_step = static_cast<double>(wall_ns) / steps / 1e6;
  const double in_ops = static_cast<double>(total_ns) / steps / 1e6;
  std::printf("model %s, %d threads, %d user(s), %d steps after %d tokens of context\n", pos_args[0], threads, users, steps,
              ctx0);
  std::printf("step %.2f ms wall, %.2f ms inside %zu ops (%.1f%%), %.2f ms between ops\n\n", per_step, in_ops,
              ops_per_step, 100 * in_ops / per_step, per_step - in_ops);
  std::vector<std::pair<int64_t, std::string>> kinds;
  for (const auto& [k, v] : by_kind) kinds.emplace_back(v.first, k);
  std::sort(kinds.rbegin(), kinds.rend());
  std::printf("%-36s %10s %8s %10s %7s\n", "op", "ms/step", "calls", "us/call", "share");
  for (const auto& [ns, k] : kinds) {
    const auto& v = by_kind[k];
    std::printf("%-36s %10.3f %8.1f %10.1f %6.1f%%\n", k.c_str(), static_cast<double>(ns) / steps / 1e6,
                static_cast<double>(v.second) / steps, static_cast<double>(ns) / static_cast<double>(v.second) / 1e3,
                100.0 * static_cast<double>(ns) / static_cast<double>(total_ns));
  }
  return 0;
}
