// CpuDevice::attention: the SME prompt-chunk path (attention_sme, CPUs with
// FEAT_SME) against the per-row path (KernelPlan::sme_attention_min_rows = 0)
// on the same KV cache: prompt runs starting mid-block next to decode rows of
// other sequences, sliding windows, softcap, head sizes 64 / 128 / 256, query
// groups 1..8 and f16 / f32 / q8_0 KV. Without SME both runs take the per-row
// path (the test then only checks determinism). Also the grouped per-row
// kernels (all heads of a KV head per K/V load, fp16 read directly) against
// one head at a time (KernelPlan::grouped_attention = false).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/cpu/qmm.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/memory/storage.h"

using namespace dynacore;

namespace {

int g_checks = 0;
#define CHECK(c)                                                                     \
  do {                                                                               \
    ++g_checks;                                                                      \
    if (!(c)) {                                                                      \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

std::mt19937 g_rng(99);

Tensor host(DType t, std::initializer_list<int64_t> shape) {
  auto r = Tensor::empty(t, shape);
  CHECK(r.ok());
  return *r;
}

void randomize(Tensor& t, float scale) {
  std::normal_distribution<float> nd(0.0f, scale);
  float* p = t.data_as<float>();
  for (int64_t i = 0; i < t.numel(); ++i) p[i] = nd(g_rng);
}

void check_case(CpuDevice& dev, DType kv_dtype, int32_t hd, int32_t nkv, int32_t heads, int32_t window, float softcap) {
  KvGeometry g;
  g.num_layers = 1;
  g.num_kv_heads = nkv;
  g.head_dim = hd;
  g.head_dim_v = hd;
  g.block_size = 16;
  g.num_blocks = 80;
  g.dtype = kv_dtype;
  g.layout = KvLayout::kHeadMajor;
  const auto layer_bytes = static_cast<size_t>(g.k_block_bytes() * g.num_blocks);
  auto ks = Storage::allocate_host(layer_bytes), vs = Storage::allocate_host(layer_bytes);
  CHECK(ks.ok() && vs.ok());
  // Three sequences in scattered blocks: 60, 300 and 130 positions.
  const std::vector<std::vector<int32_t>> tables = {
      {7, 2, 30, 41},
      {1, 40, 3, 9, 22, 11, 5, 60, 33, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
      {50, 51, 52, 53, 54, 55, 56, 57, 58}};
  const std::vector<int32_t> lens = {60, 300, 130};
  std::vector<KvLayerView> kv;
  for (const auto& t : tables) kv.push_back({(*ks)->data(), (*vs)->data(), &g, t});
  std::vector<int32_t> pos, seq;
  for (size_t s = 0; s < lens.size(); ++s) {
    for (int32_t p = 0; p < lens[s]; ++p) {
      pos.push_back(p);
      seq.push_back(static_cast<int32_t>(s));
    }
  }
  Tensor k = host(DType::kF32, {static_cast<int64_t>(pos.size()), nkv * hd});
  Tensor v = host(DType::kF32, {static_cast<int64_t>(pos.size()), nkv * hd});
  randomize(k, 1.0f);
  randomize(v, 1.0f);
  dev.kv_store(k, v, pos, seq, kv);
  // Batch: a decode row of sequence 0, a prompt chunk of sequence 1 starting
  // mid-block (rows 77..286: several 32-row blocks and a partial one), a
  // short run of sequence 2 (under the SME minimum), another decode row.
  std::vector<int32_t> qpos = {59}, qseq = {0};
  for (int32_t p = 77; p < 287; ++p) {
    qpos.push_back(p);
    qseq.push_back(1);
  }
  for (int32_t p = 120; p < 128; ++p) {
    qpos.push_back(p);
    qseq.push_back(2);
  }
  qpos.push_back(129);
  qseq.push_back(2);
  const auto m = static_cast<int64_t>(qpos.size());
  Tensor q = host(DType::kF32, {m, heads * hd});
  randomize(q, 1.0f);
  Tensor out_sme = host(DType::kF32, {m, heads * hd}), out_row = host(DType::kF32, {m, heads * hd});
  AttentionParams p;
  p.q = q;
  p.positions = qpos;
  p.row_seq = qseq;
  p.kv = kv;
  p.num_heads = heads;
  p.scale = 1.0f / std::sqrt(static_cast<float>(hd));
  p.softcap = softcap;
  p.sliding_window = window;
  KernelPlan plan = KernelPlan::defaults();
  plan.sme_attention_min_rows = 16;
  dev.set_kernel_plan(plan);
  p.out = out_sme;
  dev.attention(p);
  plan.sme_attention_min_rows = 0;
  dev.set_kernel_plan(plan);
  p.out = out_row;
  dev.attention(p);
  Tensor out_head = host(DType::kF32, {m, heads * hd});
  plan.grouped_attention = false;
  dev.set_kernel_plan(plan);
  p.out = out_head;
  dev.attention(p);
  {
    const float* r = out_row.data_as<float>();
    const float* h = out_head.data_as<float>();
    double worst = 0;
    for (int64_t i = 0; i < m * heads * hd; ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(r[i]) - h[i]) / (std::fabs(static_cast<double>(h[i])) + 0.05));
    }
    if (worst > 1e-5) std::fprintf(stderr, "grouped vs per-head: max rel diff %.3g\n", worst);
    CHECK(worst <= 1e-5);
  }
  const float* a = out_sme.data_as<float>();
  const float* b = out_row.data_as<float>();
  double worst = 0;
  for (int64_t i = 0; i < m * heads * hd; ++i) {
    CHECK(std::isfinite(a[i]));
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / (std::fabs(static_cast<double>(b[i])) + 0.05));
  }
  if (worst > 1e-4) {
    std::fprintf(stderr, "attention kv %d hd %d heads %d/%d window %d softcap %.1f: max rel diff %.3g\n",
                 static_cast<int>(kv_dtype), hd, heads, nkv, window, softcap, worst);
  }
  CHECK(worst <= 1e-4);
}

}  // namespace

int main() {
  ThreadPool pool(std::max(2, cpu_info().physical_cores));
  CpuDevice dev(pool, select_best_isa(cpu_info().features));
  std::printf("device %s, sme %s\n", std::string(dev.name()).c_str(), qmm::sme_gemm() ? "on" : "not available");
  for (DType t : {DType::kF16, DType::kF32, DType::kQ8_0}) {
    for (auto [hd, nkv, heads] : {std::tuple{128, 2, 12}, std::tuple{64, 4, 4}, std::tuple{128, 8, 32}, std::tuple{256, 1, 8},
                                  std::tuple{128, 2, 16}}) {
      check_case(dev, t, hd, nkv, heads, 0, 0.0f);
    }
  }
  check_case(dev, DType::kF16, 128, 2, 8, 40, 0.0f);    // sliding window shorter than the run
  check_case(dev, DType::kF16, 128, 2, 8, 100, 50.0f);  // window + softcap (Gemma 2)
  std::printf("all cpu attention tests passed (%d checks)\n", g_checks);
  return 0;
}
