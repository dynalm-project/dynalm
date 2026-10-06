// Compiled execution A/B (DD-072): the same decode steps through
//   ref     the CPU device directly (reference C++ runtime)
//   ir      RecordingDevice, deferred, no fusion (recording + planning overhead)
//   group   + shared-input matmul groups (Q/K/V -> one matmul_many region)
//   gated   + gated MLP fusion (gate/up/act_mul -> matmul_gated)
//   all     group + gated
// Variants alternate within each repetition, order flipped every point, so
// clock and thermal drift hit all variants alike. An accuracy phase first
// compares every variant's logits with the first variant's.
//
//   bench_compiled <model.gguf> [threads=0] [context=256] [seqs=1,4] [reps=3] [variants=ref,ir,all] [steps=24]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "bench_harness.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/process_stats.h"
#include "dynacore/ir/passes.h"
#include "dynacore/ir/recording_device.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/sequence.h"
#include "common/core.h"

using namespace dynalm;
namespace ir = dynacore::ir;

namespace {

std::vector<std::string> split(const char* s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) out.push_back(t);
  return out;
}

struct Variant {
  std::string name;
  std::unique_ptr<ir::RecordingDevice> rec;  // null for ref
  std::unique_ptr<Transformer> tf;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bench_compiled <model.gguf> [threads=0] [context=256] [seqs=1,4] [reps=3] "
                 "[variants=ref,ir,group,gated,all] [steps=24]\n");
    return 1;
  }
  const int threads = argc > 2 && std::atoi(argv[2]) > 0 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  const int ctx = argc > 3 ? std::atoi(argv[3]) : 256;
  std::vector<int> seq_counts;
  for (const std::string& s : split(argc > 4 ? argv[4] : "1,4")) seq_counts.push_back(std::atoi(s.c_str()));
  const int reps = argc > 5 ? std::atoi(argv[5]) : 3;
  const std::vector<std::string> names = split(argc > 6 ? argv[6] : "ref,ir,all");
  const int steps = argc > 7 ? std::atoi(argv[7]) : 24;
  constexpr int kWarm = 2, kChunk = 256;

  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  const int max_seqs = std::max(2, *std::max_element(seq_counts.begin(), seq_counts.end()));
  const int per_seq = (ctx + kWarm + steps + 1 + 15) / 16 * 16;
  const int64_t nvar = static_cast<int64_t>(split(argc > 6 ? argv[6] : "ref,ir,all").size());
  const KvGeometry geom = kv_geometry_for(c, DType::kF16, 16, static_cast<int64_t>(per_seq) * (max_seqs * nvar + 1) + 64);
  auto kv = KvBlockPool::create(geom, cpu);
  if (!kv.ok()) return 1;

  std::vector<Variant> variants;
  for (const std::string& n : names) {
    if (n != "ref" && n != "ir" && n != "group" && n != "gated" && n != "all") {
      return std::fprintf(stderr, "unknown variant %s\n", n.c_str()), 1;
    }
    Variant v;
    v.name = n;
    Device* dev = &cpu;
    if (n != "ref") {
      v.rec = std::make_unique<ir::RecordingDevice>(cpu, ir::RecordingDevice::Mode::kDeferred);
      ir::CompileOptions co;
      co.cost = ir::CostModel::from(KernelPlan::defaults(), threads);
      co.fusion.group_shared_input = n == "group" || n == "all";
      co.fusion.gated_mlp = n == "gated" || n == "all";
      v.rec->set_planner(ir::make_planner(co));
      dev = v.rec.get();
    }
    auto tf = Transformer::create(c, (*m)->weights, *dev, kChunk);
    if (!tf.ok()) return std::fprintf(stderr, "%s\n", tf.status().to_string().c_str()), 1;
    v.tf = std::move(*tf);
    variants.push_back(std::move(v));
  }
  const auto vocab = static_cast<size_t>(c.vocab_size);

  // Source context, prefilled once through the first variant.
  KvBlockTable src(**kv);
  {
    std::vector<TokenId> toks(static_cast<size_t>(ctx));
    for (int i = 0; i < ctx; ++i) toks[static_cast<size_t>(i)] = 100 + (i * 37) % 5000;
    std::vector<int32_t> pos(static_cast<size_t>(ctx));
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits(vocab);
    if (!src.reserve(ctx).ok()) return 1;
    for (int p0 = 0; p0 < ctx; p0 += kChunk) {
      const int len = std::min(kChunk, ctx - p0);
      if (!variants[0].tf->forward({toks.data() + p0, static_cast<size_t>(len)},
                                   {pos.data() + p0, static_cast<size_t>(len)}, **kv, src.block_table(), logits)
               .ok()) {
        return 1;
      }
    }
  }

  const int bs = geom.block_size;
  auto fresh_tables = [&](int n, std::vector<KvBlockTable>& tables) -> bool {
    tables.clear();
    for (int s = 0; s < n; ++s) {
      tables.emplace_back(**kv);
      if (!tables.back().reserve(ctx + kWarm + steps + 1).ok()) return false;
      for (int b = 0; b < (ctx + bs - 1) / bs; ++b) {
        (*kv)->copy_block(tables.back().block_table()[static_cast<size_t>(b)],
                          src.block_table()[static_cast<size_t>(b)]);
      }
    }
    return true;
  };

  // Accuracy: one decode step of 2 sequences, every variant vs the first.
  {
    std::vector<std::vector<float>> outs;
    for (Variant& v : variants) {
      std::vector<KvBlockTable> tables;
      if (!fresh_tables(2, tables)) return std::fprintf(stderr, "KV exhausted\n"), 1;
      const TokenId t0 = 321, t1 = 654;
      std::vector<SeqBatch> batch = {{{&t0, 1}, ctx, tables[0].block_table(), true},
                                     {{&t1, 1}, ctx, tables[1].block_table(), true}};
      std::vector<float> out(vocab * 2);
      if (!v.tf->forward_batch(batch, **kv, out).ok()) return 1;
      outs.push_back(std::move(out));
    }
    for (size_t i = 0; i < variants.size(); ++i) {
      double d = 0;
      for (size_t j = 0; j < outs[i].size(); ++j) {
        d = std::max(d, static_cast<double>(std::fabs(outs[i][j] - outs[0][j])));
      }
      std::fprintf(stderr, "accuracy: %-6s max |logit - %s| = %.3g\n", variants[i].name.c_str(),
                   variants[0].name.c_str(), d);
    }
  }

  // Timing: variants interleave step by step (rotating which goes first), so
  // every variant samples the same clock and thermal conditions; each keeps
  // its own KV copies. One row per (round, seqs, variant).
  std::printf("variant,round,context,seqs,threads,step_p50_ms,step_mean_ms,agg_tok_s,vs_first_pct,"
              "record_us_step,plan_us_step,inner_calls_per_op,fallbacks\n");
  std::vector<float> logits(vocab * static_cast<size_t>(max_seqs));
  for (int round = 0; round < reps; ++round) {
    for (int n : seq_counts) {
      const size_t nv = variants.size();
      std::vector<std::vector<KvBlockTable>> tables(nv);
      for (size_t vi = 0; vi < nv; ++vi) {
        if (!fresh_tables(n, tables[vi])) return std::fprintf(stderr, "KV exhausted\n"), 1;
      }
      std::vector<ir::RecordingStats> before(nv);
      for (size_t vi = 0; vi < nv; ++vi) before[vi] = variants[vi].rec ? variants[vi].rec->stats() : ir::RecordingStats{};
      std::vector<std::vector<double>> ms(nv);
      std::vector<TokenId> step_tok(static_cast<size_t>(n));
      for (int step = 0; step < kWarm + steps; ++step) {
        for (int s = 0; s < n; ++s) step_tok[static_cast<size_t>(s)] = 200 + (s * 13 + step) % 3000;
        for (size_t k = 0; k < nv; ++k) {
          const size_t vi = (k + static_cast<size_t>(step)) % nv;
          std::vector<SeqBatch> batch;
          for (int s = 0; s < n; ++s) {
            batch.push_back({{&step_tok[static_cast<size_t>(s)], 1}, ctx + step,
                             tables[vi][static_cast<size_t>(s)].block_table(), true});
          }
          const Stopwatch sw;
          if (!variants[vi].tf->forward_batch(batch, **kv, {logits.data(), vocab * static_cast<size_t>(n)}).ok()) {
            return 1;
          }
          if (step >= kWarm) ms[vi].push_back(sw.elapsed_ms());
        }
      }
      double first_mean = 0;
      for (size_t vi = 0; vi < nv; ++vi) {
        Variant& v = variants[vi];
        const ir::RecordingStats after = v.rec ? v.rec->stats() : ir::RecordingStats{};
        std::vector<double>& t = ms[vi];
        std::sort(t.begin(), t.end());
        const double mean = std::accumulate(t.begin(), t.end(), 0.0) / static_cast<double>(t.size());
        if (vi == 0) first_mean = mean;
        const double calls_per_op = v.rec && after.ops > before[vi].ops
                                        ? static_cast<double>(after.planned_steps - before[vi].planned_steps) /
                                              static_cast<double>(after.ops - before[vi].ops)
                                        : 1.0;
        const double per_step = static_cast<double>(kWarm + steps);
        std::printf("%s,%d,%d,%d,%d,%.2f,%.2f,%.1f,%+.1f,%.0f,%.0f,%.3f,%lld\n", v.name.c_str(), round, ctx, n, threads,
                    t[t.size() / 2], mean, n * 1000.0 / mean, 100 * (mean / first_mean - 1),
                    static_cast<double>(after.record_ns - before[vi].record_ns) / per_step / 1e3,
                    static_cast<double>(after.plan_ns - before[vi].plan_ns) / per_step / 1e3, calls_per_op,
                    static_cast<long long>(after.fallbacks - before[vi].fallbacks));
        std::fflush(stdout);
      }
    }
  }
  return 0;
}
