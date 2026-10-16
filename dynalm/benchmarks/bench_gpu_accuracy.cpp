// GPU accuracy against the CPU fp32 reference (docs/gpu-backend.md, section 8).
//
//   bench_gpu_accuracy <model.gguf> [threads]
//
// Runs the same model on the CPU (fp32 activations: the DD-053 reference
// arithmetic) and on the GPU device (ggml, through the RecordingDevice as the
// engine runs it) and compares next-token distributions:
//   1. decode shape: the fixed text one token at a time (M = 1);
//   2. prefill shapes: the text in chunks of 32 and 64 tokens, every position;
//   3. greedy continuations of several prompts: first diverging position;
//   4. GPU determinism: the decode pass twice, logits bit-identical;
//   5. two sequences in one batch with separate block tables (paged KV,
//      different positions): logits equal to each sequence run alone.
// Also prints the CPU's own summation-order noise floor (fp32 fused dot vs
// fp32 expand + GEMM) for scale. Exit code 0 only if the DD-053 contract
// holds on every comparison (perplexity change <= +1%, mean KL <= 0.0025
// nats) and checks 4 and 5 pass.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/core.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/recording_device.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/engine.h"

namespace {

constexpr const char* kText =
    "The river ran quietly through the valley, carrying the last of the autumn leaves toward the sea. "
    "On its banks the old mill still stood, its wheel long silent, its windows dark. Children from the "
    "village would sometimes dare each other to walk inside, but none of them stayed for long. The miller "
    "had left decades ago, and nobody remembered exactly why. Some said he had gone to the city to make "
    "his fortune; others insisted he had simply walked into the forest one winter morning and never come "
    "back. Whatever the truth, the mill had become part of the landscape, as natural as the hills and the "
    "water. In spring the meadow around it filled with flowers, and in summer the swallows nested under "
    "its roof. Every evening the light turned the stones gold, and for a few minutes the place looked "
    "almost alive again, as if the wheel might begin to turn and the miller might step out to greet the "
    "travelers on the road.";

constexpr const char* kPrompts[] = {
    "Explain what a CPU cache is in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List three differences between TCP and UDP.",
    "Translate to French: The weather is nice today and we are going to the park.",
};

std::vector<double> log_softmax(const float* l, int64_t n) {
  double mx = -INFINITY;
  for (int64_t i = 0; i < n; ++i) mx = std::max(mx, static_cast<double>(l[i]));
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += std::exp(l[i] - mx);
  const double lse = mx + std::log(s);
  std::vector<double> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = l[i] - lse;
  return out;
}

int64_t argmax(const float* v, int64_t n) { return std::max_element(v, v + n) - v; }

struct Stats {
  double nll_ref = 0, nll_alt = 0, kl_sum = 0, kl_max = 0, dmax = 0;
  int agree = 0, n = 0;
  void add(const float* ref, const float* alt, int64_t vocab, int64_t next) {
    const auto a = log_softmax(ref, vocab), b = log_softmax(alt, vocab);
    nll_ref -= a[static_cast<size_t>(next)];
    nll_alt -= b[static_cast<size_t>(next)];
    double kl = 0;
    for (int64_t i = 0; i < vocab; ++i) {
      kl += std::exp(a[static_cast<size_t>(i)]) * (a[static_cast<size_t>(i)] - b[static_cast<size_t>(i)]);
      dmax = std::max(dmax, std::abs(static_cast<double>(ref[i]) - alt[i]));
    }
    kl_sum += kl;
    kl_max = std::max(kl_max, kl);
    agree += argmax(ref, vocab) == argmax(alt, vocab) ? 1 : 0;
    ++n;
  }
  double ppl_change() const { return 100.0 * (std::exp(nll_alt / n) / std::exp(nll_ref / n) - 1.0); }
  bool pass() const { return ppl_change() <= 1.0 && kl_sum / n <= 0.0025; }
  void print(const char* label) const {
    std::printf("\n[%s]\n", label);
    std::printf("positions: %d\n", n);
    std::printf("perplexity ref %.4f | alt %.4f | change %+.3f%%\n", std::exp(nll_ref / n), std::exp(nll_alt / n),
                ppl_change());
    std::printf("KL(ref || alt) mean %.6f max %.6f nats\n", kl_sum / n, kl_max);
    std::printf("top-1 agreement %.2f%% | max |logit diff| %.4f\n", 100.0 * agree / n, dmax);
    std::printf("DD-053 (ppl <= +1%%, mean KL <= 0.0025): %s\n", pass() ? "PASS" : "FAIL");
  }
};

}  // namespace

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 2) {
    std::fprintf(stderr, "usage: bench_gpu_accuracy <model.gguf> [threads]\n");
    return 1;
  }
  const int threads = argc > 2 ? std::atoi(argv[2]) : cpu_info().physical_cores;
  constexpr int kMaxRows = 64;
  auto lm = load_model(argv[1]);
  if (!lm.ok()) {
    std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
    return 1;
  }
  const ModelConfig& c = (*lm)->config;
  const int64_t vocab = c.vocab_size;

  // CPU reference.
  ThreadPool pool(threads);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  auto tf_cpu = Transformer::create(c, (*lm)->weights, cpu, kMaxRows);
  // GPU, wrapped as the engine wraps it.
  // BENCH_GPU=metal|vulkan|cuda (default: Metal, else Vulkan, else CUDA).
  dynacore::DeviceType gpu_type = dynacore::DeviceType::kMetal;
  if (!dynacore::ggml_gpu_compiled(gpu_type)) gpu_type = dynacore::DeviceType::kVulkan;
  if (!dynacore::ggml_gpu_compiled(gpu_type)) gpu_type = dynacore::DeviceType::kCuda;
  if (const char* g = std::getenv("BENCH_GPU"); g != nullptr && *g != '\0') {
    const std::string_view v(g);
    gpu_type = v == "vulkan" ? dynacore::DeviceType::kVulkan
               : v == "cuda" ? dynacore::DeviceType::kCuda
                             : dynacore::DeviceType::kMetal;
  }
  auto gpu = dynacore::create_ggml_gpu_device(gpu_type);
  if (!tf_cpu.ok() || !gpu.ok()) {
    std::fprintf(stderr, "%s\n", (!tf_cpu.ok() ? tf_cpu.status() : gpu.status()).to_string().c_str());
    return 1;
  }
  dynacore::ir::RecordingDevice rec(**gpu, dynacore::ir::RecordingDevice::Mode::kDeferred);
  auto tf_gpu = Transformer::create(c, (*lm)->weights, rec, kMaxRows);
  if (!tf_gpu.ok()) {
    std::fprintf(stderr, "GPU: %s\n", tf_gpu.status().to_string().c_str());
    return 1;
  }
  std::printf("model: %s (%s), reference %s (%d threads), GPU %s, ggml %s\n", argv[1], (*lm)->quantization.c_str(),
              std::string(cpu.name()).c_str(), threads, std::string((*gpu)->name()).c_str(),
              dynacore::ggml_revision().c_str());

  const std::vector<TokenId> text = (*lm)->tokenizer->encode(kText, true, false);
  const int32_t ctx = 2 * (static_cast<int32_t>(text.size()) + 160);
  // Each device's preferred KV layout, as the engine creates it; on the GPU,
  // BENCH_KV_LAYOUT=head selects the gather path instead (A/B).
  const char* kl = std::getenv("BENCH_KV_LAYOUT");
  const char* kd = std::getenv("BENCH_KV_DTYPE");  // f16 (default) | f32 | q8_0
  DType kv_dtype = DType::kF16;
  if (kd && std::string_view(kd) == "f32") kv_dtype = DType::kF32;
  if (kd && std::string_view(kd) == "q8_0") kv_dtype = DType::kQ8_0;
  auto make_kv = [&](Device& dev) -> std::unique_ptr<KvBlockPool> {
    KvGeometry geo = kv_geometry_for(c, kv_dtype, 16, ctx);
    geo.layout = dev.preferred_kv_layout();
    if (kl && std::string_view(kl) == "head") geo.layout = KvLayout::kHeadMajor;
    auto p = KvBlockPool::create(geo, dev);
    if (!p.ok()) throw std::runtime_error("KV: " + p.status().to_string());
    return std::move(*p);
  };
  // `expand`: the CPU fp32 expand + GEMM path (noise floor); otherwise the
  // fused fp32 path. int8/int16 activations are off on the reference.
  auto run = [&](Transformer& tf, KvBlockPool& kv, std::span<const SeqBatch> batch, std::vector<float>& out,
                 bool expand = false) {
    ExecutionPlan plan = tf.planner().plan(batch);
    plan.kernels.int8_decode_max_rows = 0;
    plan.kernels.int16_ffn_down = false;
    plan.kernels.int16_prefill_max_rows = 0;
    plan.kernels.q4_repack_max_rows = 0;
    plan.kernels.expand_min_rows = expand ? 1 : 1 << 30;
    size_t rows = 0;
    for (const SeqBatch& b : batch) rows += b.want_logits ? static_cast<size_t>(b.logits_last) : 0;
    out.resize(rows * static_cast<size_t>(vocab));
    if (const Status st = tf.forward_batch(batch, kv, out, &plan); !st.ok()) {
      // Throw rather than exit(): the GPU device must be destroyed before the
      // process ends (ggml-metal asserts on live residency sets at exit).
      throw std::runtime_error("forward: " + st.to_string());
    }
  };
  auto step = [&](Transformer& tf, KvBlockPool& kv, KvBlockTable& tab, TokenId t, int32_t pos, std::vector<float>& out,
                  bool expand = false) {
    if (!tab.reserve(pos + 1).ok()) std::abort();
    const SeqBatch b{std::span<const TokenId>(&t, 1), pos, tab.block_table(), true};
    run(tf, kv, {&b, 1}, out, expand);
  };
  bool ok = true;
  try {

  // 1. Decode shape, and 4. determinism (the GPU pass twice).
  {
    auto kv_ref = make_kv(cpu), kv_floor = make_kv(cpu), kv_g1 = make_kv(rec), kv_g2 = make_kv(rec);
    KvBlockTable t_ref(*kv_ref), t_floor(*kv_floor), t_g1(*kv_g1), t_g2(*kv_g2);
    std::vector<float> lr, lf, l1, l2;
    Stats floor, gpu_stats;
    bool identical = true;
    for (size_t p = 0; p + 1 < text.size(); ++p) {
      const auto pos = static_cast<int32_t>(p);
      step(**tf_cpu, *kv_ref, t_ref, text[p], pos, lr);
      step(**tf_cpu, *kv_floor, t_floor, text[p], pos, lf, /*expand=*/true);
      step(**tf_gpu, *kv_g1, t_g1, text[p], pos, l1);
      step(**tf_gpu, *kv_g2, t_g2, text[p], pos, l2);
      floor.add(lr.data(), lf.data(), vocab, text[p + 1]);
      gpu_stats.add(lr.data(), l1.data(), vocab, text[p + 1]);
      identical = identical && l1 == l2;
    }
    floor.print("noise floor: CPU fp32 fused vs CPU fp32 expand (decode, M = 1)");
    gpu_stats.print("GPU vs CPU fp32 reference: decode (M = 1)");
    std::printf("\n[GPU determinism: decode pass twice] logits %s\n", identical ? "bit-identical: PASS" : "differ: FAIL");
    ok = ok && gpu_stats.pass() && identical;
  }

  // 2. Prefill shapes.
  for (const int chunk : {32, 64}) {
    auto kv_ref = make_kv(cpu), kv_gpu = make_kv(rec);
    KvBlockTable t_ref(*kv_ref), t_gpu(*kv_gpu);
    std::vector<float> lr, lg;
    Stats s;
    const auto total = static_cast<int32_t>(text.size()) - 1;
    for (int32_t start = 0; start < total; start += chunk) {
      const int32_t len = std::min(chunk, total - start);
      const std::span<const TokenId> toks(text.data() + start, static_cast<size_t>(len));
      if (!t_ref.reserve(start + len).ok() || !t_gpu.reserve(start + len).ok()) std::abort();
      SeqBatch br{toks, start, t_ref.block_table(), true}, bg{toks, start, t_gpu.block_table(), true};
      br.logits_last = bg.logits_last = len;
      run(**tf_cpu, *kv_ref, {&br, 1}, lr);
      run(**tf_gpu, *kv_gpu, {&bg, 1}, lg);
      for (int32_t r = 0; r < len; ++r) {
        s.add(lr.data() + static_cast<size_t>(r) * vocab, lg.data() + static_cast<size_t>(r) * vocab, vocab,
              text[static_cast<size_t>(start + r + 1)]);
      }
    }
    const std::string label = "GPU vs CPU fp32 reference: prefill chunks of " + std::to_string(chunk);
    s.print(label.c_str());
    ok = ok && s.pass();
  }

  // 3. Greedy continuations: chat-templated prompts, prefill then 64 tokens.
  std::printf("\n[greedy continuations, 64 tokens, chat template]\n");
  int identical_prompts = 0;
  for (const char* prompt : kPrompts) {
    std::vector<ChatMessage> msgs{{"user", prompt}};
    std::string templated = prompt;
    if ((*lm)->chat_template) {
      auto t = (*lm)->chat_template->apply(msgs, true);
      if (t.ok()) templated = *t;
    }
    std::vector<TokenId> toks;
    if (!(*lm)->tokenizer->encode(templated, true, true, toks).ok()) std::abort();
    std::vector<TokenId> out[2];
    for (int path = 0; path < 2; ++path) {
      Transformer& tf = path == 0 ? **tf_cpu : **tf_gpu;
      auto kv = make_kv(path == 0 ? static_cast<Device&>(cpu) : static_cast<Device&>(rec));
      KvBlockTable tab(*kv);
      std::vector<float> l;
      // Prefill in chunks of 32, as the scheduler would.
      int32_t pos = 0;
      while (pos < static_cast<int32_t>(toks.size())) {
        const int32_t len = std::min<int32_t>(32, static_cast<int32_t>(toks.size()) - pos);
        if (!tab.reserve(pos + len).ok()) std::abort();
        SeqBatch b{std::span<const TokenId>(toks.data() + pos, static_cast<size_t>(len)), pos, tab.block_table(), true};
        run(tf, *kv, {&b, 1}, l);
        pos += len;
      }
      TokenId t = static_cast<TokenId>(argmax(l.data(), vocab));
      for (int i = 0; i < 64; ++i) {
        out[path].push_back(t);
        step(tf, *kv, tab, t, pos++, l);
        t = static_cast<TokenId>(argmax(l.data(), vocab));
      }
    }
    size_t d = 0;
    while (d < out[0].size() && out[0][d] == out[1][d]) ++d;
    identical_prompts += d == out[0].size() ? 1 : 0;
    std::printf("  %-72.72s identical for %zu of %zu tokens\n", prompt, d, out[0].size());
  }
  std::printf("  greedy: %d of %zu prompts identical over 64 tokens (reported, not gated: DD-053 does not require it)\n",
              identical_prompts, std::size(kPrompts));

  // 5. Two sequences in one batch, separate block tables (the second shifted
  // by one block so physical blocks differ), against each run alone.
  {
    auto kv_alone = make_kv(rec), kv_pair = make_kv(rec);
    KvBlockTable a1(*kv_alone), pa(*kv_pair), pad(*kv_pair), pb(*kv_pair);
    if (!pad.reserve(16).ok()) std::abort();  // occupy a block so B's blocks differ from A's
    // 40 + 20 rows: within one forward pass (kMaxRows), different lengths and positions.
    const auto half = static_cast<int32_t>(std::min<size_t>(text.size(), 40));
    const std::span<const TokenId> sa(text.data(), static_cast<size_t>(half));
    const std::span<const TokenId> sb(text.data() + 10, static_cast<size_t>(half - 20));
    std::vector<float> la, lb, lp;
    // Alone: A, then B (fresh tables in the same pool).
    if (!a1.reserve(half).ok()) std::abort();
    SeqBatch ba{sa, 0, a1.block_table(), true};
    ba.logits_last = 4;
    run(**tf_gpu, *kv_alone, {&ba, 1}, la);
    a1.release();
    if (!a1.reserve(static_cast<int64_t>(sb.size())).ok()) std::abort();
    SeqBatch bb{sb, 0, a1.block_table(), true};
    bb.logits_last = 4;
    run(**tf_gpu, *kv_alone, {&bb, 1}, lb);
    // Together.
    if (!pa.reserve(half).ok() || !pb.reserve(static_cast<int64_t>(sb.size())).ok()) std::abort();
    SeqBatch pair[2] = {{sa, 0, pa.block_table(), true}, {sb, 0, pb.block_table(), true}};
    pair[0].logits_last = pair[1].logits_last = 4;
    run(**tf_gpu, *kv_pair, pair, lp);
    double dmax = 0;
    for (size_t i = 0; i < la.size(); ++i) dmax = std::max(dmax, std::abs(static_cast<double>(la[i]) - lp[i]));
    for (size_t i = 0; i < lb.size(); ++i) dmax = std::max(dmax, std::abs(static_cast<double>(lb[i]) - lp[la.size() + i]));
    // Batch invariance is not promised on the GPU (kernels may change with
    // the row count); the check is that a multi-sequence batch is correct,
    // i.e. within the decode-vs-reference error scale, not bit-identical.
    const bool pass = dmax < 0.5;
    std::printf("\n[two sequences in one batch vs each alone (GPU)] max |logit diff| %.5f: %s\n", dmax,
                pass ? "PASS (< 0.5)" : "FAIL");
    ok = ok && pass;
  }

  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    std::printf("\nRESULT: FAIL (%s)\n", ex.what());
    return 1;
  }

  // 6. Prefix sharing and copy-on-write on the GPU: fork a 40-token sequence
  // (the fork shares two full blocks and the partial third), continue the fork
  // with different tokens (copy-on-write of the shared partial block), and
  // compare with the same tokens computed from scratch; the original sequence
  // must be unaffected.
  try {
    auto kv = make_kv(rec);
    const auto n0 = static_cast<int32_t>(std::min<size_t>(text.size(), 40));
    std::vector<TokenId> a_toks(text.begin(), text.begin() + n0), b_extra(text.begin() + 50, text.begin() + 62);
    KvBlockTable a(*kv);
    if (!a.reserve(n0).ok()) std::abort();
    std::vector<float> la, lb, lref, la2, la2_ref;
    SeqBatch ba{a_toks, 0, a.block_table(), true};
    run(**tf_gpu, *kv, {&ba, 1}, la);
    KvBlockTable b = a.clone();  // shares all of A's blocks
    const auto nb = static_cast<int32_t>(b_extra.size());
    if (!b.reserve(n0 + nb).ok() || !b.make_writable(n0, n0 + nb).ok()) std::abort();  // copy-on-write
    SeqBatch bb{b_extra, n0, b.block_table(), true};
    bb.logits_last = nb;
    run(**tf_gpu, *kv, {&bb, 1}, lb);
    // Reference: A's prefix + B's tokens from scratch in a fresh table.
    std::vector<TokenId> full = a_toks;
    full.insert(full.end(), b_extra.begin(), b_extra.end());
    KvBlockTable r(*kv);
    if (!r.reserve(static_cast<int64_t>(full.size())).ok()) std::abort();
    SeqBatch br{full, 0, r.block_table(), true};
    br.logits_last = nb;
    run(**tf_gpu, *kv, {&br, 1}, lref);
    // A continues after B wrote its blocks: same as continuing without B.
    const TokenId next = text[static_cast<size_t>(n0)];
    if (!a.reserve(n0 + 1).ok()) std::abort();
    SeqBatch a2{std::span<const TokenId>(&next, 1), n0, a.block_table(), true};
    run(**tf_gpu, *kv, {&a2, 1}, la2);
    KvBlockTable r2(*kv);
    std::vector<TokenId> a_full = a_toks;
    a_full.push_back(next);
    if (!r2.reserve(n0 + 1).ok()) std::abort();
    SeqBatch br2{a_full, 0, r2.block_table(), true};
    run(**tf_gpu, *kv, {&br2, 1}, la2_ref);
    double db = 0, da = 0;
    for (size_t i = 0; i < lb.size(); ++i) db = std::max(db, std::abs(static_cast<double>(lb[i]) - lref[i]));
    for (size_t i = 0; i < la2.size(); ++i) da = std::max(da, std::abs(static_cast<double>(la2[i]) - la2_ref[i]));
    // Fork vs scratch differ only in how the prefix rows were batched (one
    // 40-row chunk vs one 52-row chunk): same arithmetic class, small tolerance.
    const bool pass = db < 0.5 && da < 0.5;
    std::printf("\n[prefix sharing + copy-on-write (GPU)] fork vs scratch max |logit diff| %.5f; original after fork %.5f: %s\n",
                db, da, pass ? "PASS (< 0.5)" : "FAIL");
    ok = ok && pass;
  } catch (const std::exception& ex) {
    std::printf("\n[prefix sharing + copy-on-write (GPU)] FAIL: %s\n", ex.what());
    ok = false;
  }

  if (auto* exec = rec.inner().segment_executor()) {
    const SegmentStats s = exec->segment_stats();
    std::printf("\n[GPU segments] %lld run, %lld graphs built, %lld reused, %lld errors\n",
                static_cast<long long>(s.segments), static_cast<long long>(s.graph_builds),
                static_cast<long long>(s.graph_reuses), static_cast<long long>(s.errors));
  }
  std::printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
