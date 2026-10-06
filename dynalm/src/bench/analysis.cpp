#include "bench/analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

#include "dynacore/base/timer.h"
#include "common/core.h"

namespace dynalm::bench {

double measure_read_bandwidth_gbs(ThreadPool& pool, size_t buffer_bytes) {
  const size_t n = buffer_bytes / sizeof(uint64_t);
  std::unique_ptr<uint64_t[]> buf(new uint64_t[n]);
  // Distinct values so the pages are really committed (not shared zero pages).
  for (size_t i = 0; i < n; ++i) buf[i] = i * 0x9E3779B97F4A7C15ull;
  const size_t chunks = static_cast<size_t>(pool.size()) * 16;
  const size_t per = (n + chunks - 1) / chunks;
  std::vector<uint64_t> sink(chunks);
  double best = 0;
  for (int pass = 0; pass < 4; ++pass) {
    const int64_t t0 = now_ns();
    pool.parallel_for(chunks, 1, [&](size_t b, size_t e) {
      for (size_t c = b; c < e; ++c) {
        const uint64_t* p = buf.get() + c * per;
        const size_t len = std::min(per, n - std::min(n, c * per));
        // Four independent accumulators keep the loads, not the adds, as the limit.
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        size_t i = 0;
        for (; i + 4 <= len; i += 4) {
          a0 ^= p[i];
          a1 ^= p[i + 1];
          a2 ^= p[i + 2];
          a3 ^= p[i + 3];
        }
        for (; i < len; ++i) a0 ^= p[i];
        sink[c] = a0 ^ a1 ^ a2 ^ a3;
      }
    });
    const double s = static_cast<double>(now_ns() - t0) * 1e-9;
    if (pass > 0) best = std::max(best, static_cast<double>(n * sizeof(uint64_t)) / s / 1e9);  // pass 0 warms up
  }
  uint64_t x = 0;
  for (uint64_t v : sink) x ^= v;
  if (x == 0x1234567) std::fputc(' ', stderr);  // keep the reads observable
  return best;
}

double decode_weight_bytes_per_step(const LoadedModel& model, double rows) {
  const TensorRegistry& w = model.weights;
  double total = static_cast<double>(w.total_bytes());
  const ModelConfig& c = model.config;
  // An untied input embedding is only gathered row by row.
  if (const Tensor* emb = w.find(TensorRole::kTokenEmbedding); emb && w.has(TensorRole::kOutput)) {
    const double row_bytes = static_cast<double>(emb->view().span_bytes()) / static_cast<double>(c.vocab_size);
    total -= static_cast<double>(emb->view().span_bytes()) - row_bytes * rows;
  }
  if (c.moe.num_experts > 0 && c.moe.experts_per_token > 0) {
    const double p = static_cast<double>(c.moe.experts_per_token) / static_cast<double>(c.moe.num_experts);
    const double touched = 1.0 - std::pow(1.0 - p, std::max(1.0, rows));
    for (int l = 0; l < c.num_layers; ++l) {
      for (TensorRole r : {TensorRole::kFfnGateExperts, TensorRole::kFfnUpExperts, TensorRole::kFfnDownExperts}) {
        if (const Tensor* t = w.find(r, l)) total -= static_cast<double>(t->view().span_bytes()) * (1.0 - touched);
      }
    }
  }
  return total;
}

double kv_bytes_per_token_read(int32_t num_layers, int32_t kv_heads, int32_t head_dim, int32_t head_dim_v,
                               int32_t kv_dtype_bytes, double ctx) {
  return static_cast<double>(num_layers) * kv_heads * (head_dim + head_dim_v) * kv_dtype_bytes * ctx;
}

std::string_view bottleneck_name(Bottleneck b) {
  switch (b) {
    case Bottleneck::kComputeBound: return "COMPUTE_BOUND";
    case Bottleneck::kMemoryBound: return "MEMORY_BOUND";
    case Bottleneck::kCacheBound: return "CACHE_BOUND";
    case Bottleneck::kSynchronizationBound: return "SYNCHRONIZATION_BOUND";
    case Bottleneck::kDispatchBound: return "DISPATCH_BOUND";
    case Bottleneck::kLoadImbalanced: return "LOAD_IMBALANCED";
    case Bottleneck::kIoBound: return "IO_BOUND";
    case Bottleneck::kMixed: return "MIXED";
  }
  return "MIXED";
}

namespace {
std::string fmt(const char* f, double a, double b = 0) {
  char buf[200];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}
}  // namespace

Classification classify(const ClassifierInputs& in) {
  Classification c;
  auto hit = [&](Bottleneck b, std::string why) {
    if (std::find(c.all.begin(), c.all.end(), b) == c.all.end()) c.all.push_back(b);
    c.evidence.push_back(std::string(bottleneck_name(b)) + ": " + std::move(why));
  };
  const double fwd = std::max(in.forward_s, 1e-9);
  const double bw_ratio = in.est_decode_bw_gbs > 0 && in.peak_bw_gbs > 0 ? in.est_decode_bw_gbs / in.peak_bw_gbs : -1;
  const double decode_share = in.decode_forward_s / fwd;
  const double prefill_share = in.prefill_forward_s / fwd;

  // Rules in priority order; the first that fires is the primary class.
  if (in.major_fault_mb_s >= 50) {
    hit(Bottleneck::kIoBound, fmt("page faults read %.0f MB/s from disk", in.major_fault_mb_s));
  }
  bool sync_like = false;
  if (in.pool_region_s > 0 && in.pool_tail_wait_s >= 0 && in.pool_regions > 0) {
    const double tail_share = in.pool_tail_wait_s / fwd;
    const double mean_us = in.pool_region_s * 1e6 / in.pool_regions;
    if (tail_share >= 0.25) {
      sync_like = true;
      if (mean_us < 50) {
        hit(Bottleneck::kSynchronizationBound,
            fmt("the caller idles %.0f%% of forward time at region ends; regions average %.1f us", 100 * tail_share,
                mean_us));
      } else {
        hit(Bottleneck::kLoadImbalanced,
            fmt("the caller idles %.0f%% of forward time waiting for slower workers; regions average %.0f us",
                100 * tail_share, mean_us));
      }
    }
    if (in.regions_per_step >= 150 && mean_us < 20) {
      hit(Bottleneck::kDispatchBound,
          fmt("%.0f parallel regions per step averaging %.1f us: fork/join cost is a large share", in.regions_per_step,
              mean_us));
    }
  }
  if (in.host_s >= 0 && in.host_s / (in.host_s + fwd) >= 0.2) {
    hit(Bottleneck::kDispatchBound,
        fmt("scheduler work outside the forward pass takes %.0f%% of step time", 100 * in.host_s / (in.host_s + fwd)));
  }
  if (decode_share >= 0.5 && bw_ratio >= 0.6) {
    hit(Bottleneck::kMemoryBound, fmt("decode moves an estimated %.1f GB/s, %.0f%% of the measured ceiling",
                                      in.est_decode_bw_gbs, 100 * bw_ratio));
  }
  if (in.llc_miss_ratio >= 0.3 && bw_ratio < 0.6) {
    hit(Bottleneck::kCacheBound, fmt("%.0f%% of last-level cache references miss while DRAM bandwidth is at %.0f%%",
                                     100 * in.llc_miss_ratio, 100 * std::max(0.0, bw_ratio)));
  }
  if ((prefill_share >= 0.5 && (bw_ratio < 0.5) && !sync_like) || in.ipc >= 2.0) {
    hit(Bottleneck::kComputeBound,
        in.ipc >= 2.0 ? fmt("IPC %.2f: the cores are busy retiring instructions", in.ipc)
                      : fmt("prompt processing is %.0f%% of forward time and memory traffic is low", 100 * prefill_share));
  } else if (decode_share >= 0.5 && bw_ratio >= 0 && bw_ratio < 0.35 && !sync_like && c.all.empty()) {
    // Decode with DRAM mostly idle and no hand-off problem: the time goes to
    // instructions (dequantization, arithmetic), not to moving bytes.
    hit(Bottleneck::kComputeBound, fmt("decode uses only %.0f%% of the bandwidth ceiling (%.1f GB/s) with no "
                                       "synchronization limit: dequantization/arithmetic dominates",
                                       100 * bw_ratio, in.est_decode_bw_gbs));
  }
  if (c.all.empty()) {
    c.primary = Bottleneck::kMixed;
    c.evidence.push_back(fmt("MIXED: no single limit dominates (decode %.0f%% of forward time, bandwidth use %.0f%%)",
                             100 * decode_share, 100 * std::max(0.0, bw_ratio)));
  } else {
    c.primary = c.all.front();
  }
  return c;
}

}  // namespace dynalm::bench
