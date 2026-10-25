// CpuDevice::attention for decode shapes (1 query row, GQA): per-call time vs
// threads and context, fp16 head-major KV. No model files.
//
//   bench_attention            (Qwen2.5-1.5B: 12 heads, 2 KV heads)
//   BENCH_ATTN_HEADS=32,8 bench_attention   (other head counts, e.g. Qwen3-4B)
//   BENCH_ATTN_LAYERS=36: rotate over that many KV copies (cold, as a decode step)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "bench_harness.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/tensor/fp16.h"

using namespace dynacore;

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0, 1);
  int32_t heads = 12, kvh = 2;
  const int32_t hd = 128;
  if (const char* e = std::getenv("BENCH_ATTN_HEADS")) std::sscanf(e, "%d,%d", &heads, &kvh);
  for (int ctx : {256, 1024, 2048, 4096}) {
    KvGeometry g;
    g.num_layers = 1;
    g.num_kv_heads = kvh;
    g.head_dim = hd;
    g.head_dim_v = hd;
    g.block_size = 16;
    g.num_blocks = ctx / 16 + 1;
    // BENCH_ATTN_F32=1: fp32 KV (no conversion) instead of fp16.
    const bool f32 = std::getenv("BENCH_ATTN_F32") != nullptr;
    g.dtype = f32 ? DType::kF32 : DType::kF16;
    const int layers = std::getenv("BENCH_ATTN_LAYERS") ? std::max(1, std::atoi(std::getenv("BENCH_ATTN_LAYERS"))) : 1;
    const size_t layer_elems = static_cast<size_t>(g.k_block_elems() * g.num_blocks) * (f32 ? 2 : 1);
    std::vector<uint16_t> k(layer_elems * static_cast<size_t>(layers)), v(k.size());
    if (f32) {
      for (size_t i = 0; i < k.size() / 2; ++i) reinterpret_cast<float*>(k.data())[i] = nd(rng);
      for (size_t i = 0; i < v.size() / 2; ++i) reinterpret_cast<float*>(v.data())[i] = nd(rng);
    } else {
      for (auto& e : k) e = fp32_to_fp16(nd(rng));
      for (auto& e : v) e = fp32_to_fp16(nd(rng));
    }
    std::vector<int32_t> table(static_cast<size_t>(g.num_blocks));
    for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<int32_t>(i);
    std::vector<KvLayerView> lvs;
    for (int l = 0; l < layers; ++l) {
      lvs.push_back({k.data() + layer_elems * static_cast<size_t>(l), v.data() + layer_elems * static_cast<size_t>(l), &g, table});
    }
    auto q = Tensor::empty(DType::kF32, {1, heads * hd});
    auto o = Tensor::empty(DType::kF32, {1, heads * hd});
    for (int64_t i = 0; i < heads * hd; ++i) q->data_as<float>()[i] = nd(rng);
    const std::vector<int32_t> pos = {ctx - 1}, seq = {0};
    for (int t : {1, 8}) {
      ThreadPool pool(t);
      CpuDevice dev(pool, select_best_isa(cpu_info().features));
      AttentionParams p;
      p.q = *q;
      p.out = *o;
      p.positions = pos;
      p.row_seq = seq;
      p.kv = {&lvs[0], 1};
      p.num_heads = heads;
      p.scale = 0.088f;
      KernelPlan plan = plan_kernels(StepShape{1, ctx, kvh, 0, false}, HardwareProfile::detect(t, "neon"), KernelPlan::defaults());
      dev.set_kernel_plan(plan);
      size_t next = 0;
      const auto s = bench::run(
          [&] {
            p.kv = {&lvs[next], 1};
            next = (next + 1) % lvs.size();
            dev.attention(p);
          },
          {.warmup_samples = 5, .samples = 200, .batch = 1});
      std::printf("ctx %5d threads %2d: %7.1f us p50 (%s)\n", ctx, t, s.p50 * 1e-3,
                  std::string(attention_strategy_name(plan.attention_full)).c_str());
    }
  }
  return 0;
}
