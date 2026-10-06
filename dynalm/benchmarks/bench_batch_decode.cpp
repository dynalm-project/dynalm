// Decode throughput with N concurrent sequences at a given context, with an
// in-process A/B of kernel-plan variants.
//
// One source sequence is prefilled to the longest context once; each point
// clones its first `context` positions into N private sequences (block
// copies, so every sequence reads its own KV memory), then times decode steps
// that run one token for every sequence in one batched forward pass. A few
// extra steps run with per-op profiling for the attention time.
//
// With variants "old,new" each point runs both, alternating which goes first
// from repetition to repetition so thermal or frequency drift does not favour
// either: old = per-head attention (grouped_attention off), new = grouped GQA
// attention (DD-066). Output is CSV, one row per (rep, context, seqs, variant).
//
// Memory safety: model, KV and RSS are reported to stderr; before every
// point available memory is checked and the run stops cleanly (exit code 3,
// reason on stderr, rows so far kept) when it is below DYNALM_BENCH_MIN_FREE_MB
// (default 1500). A stopped run is incomplete and must be treated as such.
//
// Usage: bench_batch_decode <model.gguf> [threads=0] [contexts=128] [seqs=1,2,4,8,16,32]
//                           [reps=1] [variants=new] [kv=f16|f32] [steps=24]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "dynacore/hardware/process_stats.h"
#include "runtime/sequence.h"
#include "common/core.h"

namespace {

std::vector<int> parse_ints(const char* s) {
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
  return v;
}

std::vector<std::string> parse_words(const char* s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(t);
  return v;
}

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p / 100.0 * static_cast<double>(v.size())))];
}

}  // namespace

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bench_batch_decode <model.gguf> [threads=0] [contexts=128] [seqs=1,2,4,8,16,32] [reps=1] "
                 "[variants=new] [kv=f16|f32] [steps=24]\n");
    return 1;
  }
  const int threads = argc > 2 && std::atoi(argv[2]) > 0 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  const std::vector<int> contexts = parse_ints(argc > 3 ? argv[3] : "128");
  const std::vector<int> seq_counts = parse_ints(argc > 4 ? argv[4] : "1,2,4,8,16,32");
  const int reps = argc > 5 ? std::atoi(argv[5]) : 1;
  const std::vector<std::string> variants = parse_words(argc > 6 ? argv[6] : "new");
  const std::string kv_name = argc > 7 ? argv[7] : "f16";
  const int steps = argc > 8 ? std::atoi(argv[8]) : 24;
  DType kv_dtype = DType::kF16;
  if (kv_name == "f32") {
    kv_dtype = DType::kF32;
  } else if (kv_name == "q8_0") {
    kv_dtype = DType::kQ8_0;
  } else if (kv_name != "f16") {
    std::fprintf(stderr, "unknown kv dtype %s\n", kv_name.c_str());
    return 1;
  }
  for (const auto& v : variants) {
    if (v != "old" && v != "new" && !(v.size() > 1 && v[0] == 'c' && std::atoi(v.c_str() + 1) > 0)) {
      std::fprintf(stderr, "unknown variant %s (old|new|cN)\n", v.c_str());
      return 1;
    }
  }

  auto m = load_model(argv[1]);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), 1;
  const ModelConfig& c = (*m)->config;
  ThreadPool pool(threads);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  constexpr int kWarm = 2, kProfiled = 12, kChunk = 256;
  const int max_ctx = *std::max_element(contexts.begin(), contexts.end());
  const int max_seqs = *std::max_element(seq_counts.begin(), seq_counts.end());
  // Tokens per sequence, rounded up to whole 16-token blocks (each sequence
  // reserves whole blocks).
  const int per_seq = (max_ctx + kWarm + steps + kProfiled + 1 + 15) / 16 * 16;
  const KvGeometry geom = kv_geometry_for(c, kv_dtype, 16, static_cast<int64_t>(per_seq) * (max_seqs + 1) + 64);
  auto kv = KvBlockPool::create(geom, be);
  auto tf = Transformer::create(c, (*m)->weights, be, kChunk);
  if (!kv.ok() || !tf.ok()) {
    std::fprintf(stderr, "setup failed (KV %.0f MiB)\n", static_cast<double>(geom.total_bytes()) / (1 << 20));
    return 1;
  }
  const int bs = geom.block_size;
  const char* min_free_env = std::getenv("DYNALM_BENCH_MIN_FREE_MB");
  const int64_t min_free = (min_free_env ? std::atoll(min_free_env) : 1500) << 20;
  auto mib = [](int64_t b) { return static_cast<double>(b) / (1 << 20); };
  std::fprintf(stderr, "memory: model %.0f MiB, KV pool %.0f MiB, RSS %.0f MiB, available %.0f MiB, floor %.0f MiB\n",
               mib((*m)->weight_bytes), mib(geom.total_bytes()), mib(process_rss_bytes()),
               mib(memory_info().available_bytes), mib(min_free));
  const auto vocab = static_cast<size_t>(c.vocab_size);

  // Source sequence: varied tokens so keys differ across positions.
  KvBlockTable src(**kv);
  {
    std::vector<TokenId> toks(static_cast<size_t>(max_ctx));
    for (int i = 0; i < max_ctx; ++i) toks[static_cast<size_t>(i)] = 100 + (i * 37) % 5000;
    std::vector<int32_t> pos(static_cast<size_t>(max_ctx));
    std::iota(pos.begin(), pos.end(), 0);
    std::vector<float> logits(vocab);
    if (!src.reserve(max_ctx).ok()) return 1;
    const Stopwatch sw;
    for (int p0 = 0; p0 < max_ctx; p0 += kChunk) {
      const int len = std::min(kChunk, max_ctx - p0);
      if (!(*tf)->forward({toks.data() + p0, static_cast<size_t>(len)}, {pos.data() + p0, static_cast<size_t>(len)},
                          **kv, src.block_table(), logits)
               .ok()) {
        return 1;
      }
    }
    std::fprintf(stderr, "prefilled %d tokens in %.1f s\n", max_ctx, sw.elapsed_ms() / 1e3);
  }

  std::printf("model,variant,rep,context,seqs,threads,kv,step_p50_ms,step_p90_ms,step_p99_ms,step_mean_ms,"
              "agg_tok_s,per_req_tok_s,attn_ms_step,profiled_step_ms,attn_share,cpu_util,tail_wait_share\n");
  KernelPlan plan_old = KernelPlan::defaults();
  plan_old.grouped_attention = false;
  KernelPlan plan_new = KernelPlan::defaults();
  plan_new.grouped_attention = true;
  std::vector<float> logits(vocab * static_cast<size_t>(max_seqs));

  int point = 0;
  for (int rep = 0; rep < reps; ++rep) {
    for (int ctx : contexts) {
      for (int n : seq_counts) {
        if (const int64_t avail = memory_info().available_bytes; avail < min_free) {
          std::fprintf(stderr, "STOPPED: available memory %.0f MiB < floor %.0f MiB before rep %d ctx %d seqs %d; "
                       "results are incomplete\n", mib(avail), mib(min_free), rep, ctx, n);
          return 3;
        }
        std::vector<std::string> order = variants;
        if ((rep + point) % 2 == 1) std::reverse(order.begin(), order.end());
        ++point;
        for (const std::string& variant : order) {
          KernelPlan plan = variant == "old" ? plan_old : plan_new;
          if (variant[0] == 'c') plan.matmul_chunks_per_thread = std::atoi(variant.c_str() + 1);  // cN: N chunks/thread
          (*tf)->set_kernel_base(plan);
          // N private copies of the source's first ctx positions.
          std::vector<KvBlockTable> tables;
          const int copy_blocks = (ctx + bs - 1) / bs;
          for (int s = 0; s < n; ++s) {
            tables.emplace_back(**kv);
            if (!tables.back().reserve(ctx + kWarm + steps + kProfiled + 1).ok()) {
              std::fprintf(stderr, "KV pool exhausted at context %d, %d seqs\n", ctx, n);
              return 1;
            }
            for (int b = 0; b < copy_blocks; ++b) {
              (*kv)->copy_block(tables.back().block_table()[static_cast<size_t>(b)], src.block_table()[static_cast<size_t>(b)]);
            }
          }
          std::vector<TokenId> step_tok(static_cast<size_t>(n));
          std::vector<double> ms;
          double cpu_s = 0, wall_s = 0;
          int64_t region_ns = 0, tail_ns = 0;
          pool.set_stats_enabled(true);
          for (int step = 0; step < kWarm + steps + kProfiled; ++step) {
            const bool profiled = step >= kWarm + steps;
            if (step == kWarm + steps) {
              (*tf)->reset_profile();
              (*tf)->set_profiling(true);
            }
            for (int s = 0; s < n; ++s) step_tok[static_cast<size_t>(s)] = 200 + (s * 13 + step) % 3000;
            std::vector<SeqBatch> batch;
            for (int s = 0; s < n; ++s) {
              batch.push_back({{&step_tok[static_cast<size_t>(s)], 1}, ctx + step,
                               tables[static_cast<size_t>(s)].block_table(), true});
            }
            const double cpu0 = process_cpu_seconds();
            const ThreadPoolStats ps0 = pool.stats();
            const Stopwatch sw;
            if (!(*tf)->forward_batch(batch, **kv, {logits.data(), vocab * static_cast<size_t>(n)}).ok()) return 1;
            const double t = sw.elapsed_ms();
            if (step >= kWarm && !profiled) {
              ms.push_back(t);
              cpu_s += process_cpu_seconds() - cpu0;
              const ThreadPoolStats ps1 = pool.stats();
              region_ns += ps1.region_ns - ps0.region_ns;
              tail_ns += ps1.tail_wait_ns - ps0.tail_wait_ns;
              wall_s += t / 1e3;
            }
          }
          (*tf)->set_profiling(false);
          const ForwardProfile& pr = (*tf)->profile();
          const double attn = static_cast<double>(pr.ns[static_cast<size_t>(ForwardOp::kAttention)]) / 1e6 / kProfiled;
          if (std::getenv("DYNALM_BENCH_OPS")) {  // full per-op breakdown (ms per step, share of profiled step)
            const double tot = static_cast<double>(pr.total_ns()) / 1e6 / kProfiled;
            for (size_t op = 0; op < static_cast<size_t>(ForwardOp::kCount); ++op) {
              const double v = static_cast<double>(pr.ns[op]) / 1e6 / kProfiled;
              if (v <= 0) continue;
              std::fprintf(stderr, "ops,%s,%d,%d,%s,%.3f,%.3f\n", variant.c_str(), ctx, n,
                           std::string(forward_op_name(static_cast<ForwardOp>(op))).c_str(), v, tot > 0 ? v / tot : 0.0);
            }
          }
          const double prof_total = static_cast<double>(pr.total_ns()) / 1e6 / kProfiled;
          const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
          const double p50 = pct(ms, 50);
          std::printf("%s,%s,%d,%d,%d,%d,%s,%.2f,%.2f,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f\n",
                      (*m)->config.name.empty() ? "model" : (*m)->config.name.c_str(), variant.c_str(), rep, ctx, n,
                      threads, kv_name.c_str(), p50, pct(ms, 90), pct(ms, 99), mean, n * 1000.0 / mean, 1000.0 / mean,
                      attn, prof_total, prof_total > 0 ? attn / prof_total : 0.0,
                      wall_s > 0 ? cpu_s / wall_s / threads : 0.0,
                      region_ns > 0 ? static_cast<double>(tail_ns) / static_cast<double>(region_ns) : 0.0);
          std::fflush(stdout);
        }
      }
    }
  }
  std::fprintf(stderr, "memory at end: RSS %.0f MiB, peak RSS %.0f MiB, available %.0f MiB\n",
               mib(process_rss_bytes()), mib(process_peak_rss_bytes()), mib(memory_info().available_bytes));
  return 0;
}
