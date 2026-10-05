// Performance program P3: decode-shaped matmuls on a real model's weights.
//
//   bench_decode_matmul <model.gguf> [threads]
//
// For each distinct (weight type, shape) of layer 0 plus the LM head, times
// y[M, n] = x[M, k] . w^T for M = 1..32 on each matmul path the planner can
// choose (KernelPlan): `fused` = per-row dequantize-dot from the packed
// weights; `expand` = dequantize weight panels to fp32 once, then the GEMM
// micro-kernel over all M rows; `int8` = int8 activations + integer dot
// against the packed weights (DD-053). Reports p50 time, GFLOP/s and the effective
// weight bandwidth, so the M at which each path wins is measured, not assumed.
// Each iteration uses the next layer's copy of the tensor, so the working set
// is the whole model's weights for that role (far larger than the LLC), as in
// a real decode step; timing one hot tensor would measure cache, not DRAM.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.h"
#include "bench_harness.h"
#include "loader/model_loader.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

int main(int argc, char** argv) {
  using namespace engine;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_decode_matmul <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  auto m = load_model(argv[1]);
  if (!m.ok()) {
    std::fprintf(stderr, "%s\n", m.status().to_string().c_str());
    return 1;
  }
  const TensorRegistry& reg = (*m)->weights;
  struct Case {
    std::string name;
    std::vector<const Tensor*> layers;  // same role, type and shape in every layer
  };
  std::vector<Case> cases;
  std::set<std::string> seen;
  const int num_layers = (*m)->config.num_layers;
  auto add = [&](const char* name, TensorRole role, int layer) {
    const Tensor* t = reg.find(role, layer);
    if (!t) return;
    const std::string key = std::string(dtype_name(t->dtype())) + t->shape().to_string();
    if (!seen.insert(key).second) return;
    Case c{name, {}};
    for (int l = layer < 0 ? -1 : 0; l < (layer < 0 ? 0 : num_layers); ++l) {
      const Tensor* u = reg.find(role, l);
      if (u && u->dtype() == t->dtype() && u->shape().to_string() == t->shape().to_string()) c.layers.push_back(u);
    }
    cases.push_back(std::move(c));
  };
  add("attn_q", TensorRole::kAttnQ, 0);
  add("attn_k", TensorRole::kAttnK, 0);
  add("attn_out", TensorRole::kAttnOutput, 0);
  add("ffn_gate", TensorRole::kFfnGate, 0);
  add("ffn_up", TensorRole::kFfnUp, 0);
  add("ffn_down", TensorRole::kFfnDown, 0);
  add("ffn_down", TensorRole::kFfnDown, 1);  // Q4_K_M mixes Q4_K / Q6_K across layers
  add("lm_head", reg.has(TensorRole::kOutput) ? TensorRole::kOutput : TensorRole::kTokenEmbedding, -1);

  ThreadPool pool(threads);
  CpuBackend be(pool, select_best_isa(cpu_info().features));
  std::printf("cpu: %s, %d threads, %s\nmodel: %s\n\n", cpu_info().brand.c_str(), threads,
              std::string(be.name()).c_str(), argv[1]);
  std::printf("%-9s %-6s %-14s %3s | %9s %7s | %9s %7s | %9s %7s | %s\n", "tensor", "type", "shape", "M", "fused ms",
              "GB/s", "expand ms", "GB/s", "int8 ms", "GB/s", "best");
  std::mt19937 rng(3);
  for (const Case& c : cases) {
    const Tensor& w0 = *c.layers.front();
    const int64_t n = w0.shape()[0], k = w0.shape()[1];
    const double wbytes = static_cast<double>(w0.view().span_bytes());
    for (int64_t rows : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}) {
      auto x = Tensor::empty(DType::kF32, {rows, k});
      auto y = Tensor::empty(DType::kF32, {rows, n});
      for (int64_t i = 0; i < rows * k; ++i) x->data_as<float>()[i] = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;
      double ms[3];
      for (int path = 0; path < 3; ++path) {
        KernelPlan kp = KernelPlan::defaults();
        kp.expand_min_rows = path == 1 ? 1 : 1 << 30;  // 0: fused, 1: expand, 2: int8
        kp.int8_decode_max_rows = path == 2 ? 64 : 0;
        be.set_kernel_plan(kp);
        size_t next = 0;
        const auto s = bench::run([&] { be.matmul(*x, *c.layers[next++ % c.layers.size()], nullptr, *y); },
                                  {.warmup_samples = 3, .samples = 2 * static_cast<int>(c.layers.size()), .batch = 1});
        ms[path] = s.p50 * 1e-6;
      }
      const double flop = 2.0 * static_cast<double>(rows) * n * k;
      (void)flop;
      const int best = ms[0] <= ms[1] ? (ms[0] <= ms[2] ? 0 : 2) : (ms[1] <= ms[2] ? 1 : 2);
      static constexpr const char* kNames[] = {"fused", "expand", "int8"};
      std::printf("%-9s %-6s %-14s %3lld | %9.3f %7.1f | %9.3f %7.1f | %9.3f %7.1f | %s\n", c.name.c_str(),
                  std::string(dtype_name(w0.dtype())).c_str(), w0.shape().to_string().c_str(),
                  static_cast<long long>(rows), ms[0], wbytes / ms[0] * 1e-6, ms[1], wbytes / ms[1] * 1e-6, ms[2],
                  wbytes / ms[2] * 1e-6, kNames[best]);
    }
  }
  return 0;
}
