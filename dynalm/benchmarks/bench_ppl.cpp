// Perplexity over a text file, scored exactly as llama.cpp's llama-perplexity
// does, so the two numbers compare directly (same file, same context):
//
//   bench_ppl <model.gguf> <text file> [ctx 512] [chunks 0 = all] [threads]
//   BENCH_DEVICE=native (default) | ggml-cpu | metal | vulkan | cuda
//   BENCH_UBATCH=n: score each chunk in batches of n tokens (llama-perplexity -ub n)
//   BENCH_FRESH_KV=0: reuse one KV pool for all chunks (default: a new pool per
//     chunk, as llama-perplexity clears its cache; a reused pool hands blocks back
//     newest first, so attention adds the same cells in another order)
//   BENCH_TOKENS=1: print the text's token ids instead (tokenizer comparisons)
//   BENCH_KLD_BASE=<file>: also write DynaLM's scored log-probabilities in
//     llama-perplexity's --kl-divergence-base format, so that
//       llama-perplexity -m <model.gguf> -c <ctx> --kl-divergence-base <file> --kl-divergence
//     runs llama.cpp on the same token ids (read from the file) and reports
//     its KL divergence, "Same top p" (top-1 agreement) and delta-p against DynaLM.
//
//   llama-perplexity -m <model.gguf> -f <text file> -c <ctx> [--chunks N]
//
// The text is tokenized once (with BOS when the model adds one) and cut into
// chunks of `ctx` tokens; each chunk starts with BOS (when the model uses one)
// and runs as one batch from position 0; the tokens of its second half are
// scored (logits at j predict token j + 1, j from ctx/2 to ctx - 2). The
// device runs its default plan, as the engine does (on the native CPU device:
// DynaCore's DD-053 kernels). Prints the running estimate per chunk, as
// llama-perplexity does, and the final estimate with its standard error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

int main(int argc, char** argv) {
  using namespace dynalm;
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench_ppl <model.gguf> <text file> [ctx 512] [chunks 0 = all] [threads]\n");
    return 1;
  }
  const int32_t ctx = argc > 3 ? std::atoi(argv[3]) : 512;
  const int max_chunks = argc > 4 ? std::atoi(argv[4]) : 0;
  const int threads = argc > 5 ? std::atoi(argv[5]) : cpu_info().physical_cores;
  if (ctx < 16) {
    std::fprintf(stderr, "ctx must be >= 16\n");
    return 1;
  }
  std::ifstream f(argv[2], std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  auto lm = load_model(argv[1]);
  if (!lm.ok()) {
    std::fprintf(stderr, "%s\n", lm.status().to_string().c_str());
    return 1;
  }
  const ModelConfig& c = (*lm)->config;
  const int64_t vocab = c.vocab_size;
  // BENCH_TOKENS=1: print the file's token ids (no BOS) and exit, to compare
  // tokenizers (llama-server: POST /tokenize {"content": ..., "add_special": false}).
  if (const char* t = std::getenv("BENCH_TOKENS"); t != nullptr && t[0] == '1') {
    for (TokenId id : (*lm)->tokenizer->encode(text, /*add_special=*/false, false)) std::printf("%d ", id);
    std::printf("\n");
    return 0;
  }

  ThreadPool pool(threads);
  std::unique_ptr<Device> owned;  // the ggml device, when one is used
  std::unique_ptr<dynacore::ir::RecordingDevice> rec;
  std::unique_ptr<CpuDevice> cpu;
  Device* dev = nullptr;
  const std::string_view which = std::getenv("BENCH_DEVICE") != nullptr ? std::getenv("BENCH_DEVICE") : "native";
  if (which == "native") {
    cpu = std::make_unique<CpuDevice>(pool, select_best_isa(cpu_info().features));
    dev = cpu.get();
  } else {
    const dynacore::DeviceType t = which == "ggml-cpu" ? dynacore::DeviceType::kCpu
                                   : which == "vulkan" ? dynacore::DeviceType::kVulkan
                                   : which == "cuda"   ? dynacore::DeviceType::kCuda
                                                       : dynacore::DeviceType::kMetal;
    dynacore::GgmlDeviceOptions go;
    go.threads = cpu_info().performance_cores > 0 ? cpu_info().performance_cores : threads;  // as the engine
    auto g = dynacore::create_ggml_gpu_device(t, go);
    if (!g.ok()) {
      std::fprintf(stderr, "%s\n", g.status().to_string().c_str());
      return 1;
    }
    owned = std::move(*g);
    rec = std::make_unique<dynacore::ir::RecordingDevice>(*owned, dynacore::ir::RecordingDevice::Mode::kDeferred);
    dev = rec.get();
  }
  int rc = 0;
  try {
    auto tf = Transformer::create(c, (*lm)->weights, *dev, ctx, ctx);
    if (!tf.ok()) throw std::runtime_error(tf.status().to_string());

    std::vector<TokenId> tokens = (*lm)->tokenizer->encode(text, /*add_special=*/true, false);
    const TokenId bos = (*lm)->tokenizer->bos();
    const bool add_bos = !tokens.empty() && bos != kNoToken && tokens.front() == bos;
    int n_chunk = static_cast<int>(tokens.size() / static_cast<size_t>(ctx));
    if (max_chunks > 0) n_chunk = std::min(n_chunk, max_chunks);
    if (n_chunk == 0) throw std::runtime_error("text shorter than one chunk");
    // BENCH_KLD_REF=<file>: a llama-perplexity --kl-divergence-base file. Its token
    // ids are scored (teacher-forced: the same input as llama.cpp's), DynaLM's own
    // tokenization of the text is only compared with them, and each scored
    // position is compared with llama.cpp's log-probabilities; PARITY {json} at the end.
    std::ifstream ref;
    int64_t tok_mismatch = -1;  // first index where DynaLM's tokenizer differs from the file's
    if (const char* rp = std::getenv("BENCH_KLD_REF"); rp != nullptr && *rp != '\0') {
      ref.open(rp, std::ios::binary);
      char magic[8] = {};
      int32_t hdr[3] = {};
      ref.read(magic, 8);
      ref.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
      if (!ref || std::memcmp(magic, "_logits_", 8) != 0) throw std::runtime_error(std::string(rp) + ": not a llama-perplexity file");
      if (hdr[0] != ctx || hdr[1] != vocab) {
        throw std::runtime_error("reference ctx " + std::to_string(hdr[0]) + " vocab " + std::to_string(hdr[1]) +
                                 " != ctx " + std::to_string(ctx) + " vocab " + std::to_string(vocab));
      }
      std::vector<TokenId> rt(static_cast<size_t>(hdr[0]) * static_cast<size_t>(hdr[2]));
      ref.read(reinterpret_cast<char*>(rt.data()), static_cast<std::streamsize>(rt.size() * sizeof(TokenId)));
      if (!ref) throw std::runtime_error("reference tokens truncated");
      for (size_t k = 0; k < rt.size(); ++k) {
        if (k >= tokens.size() || tokens[k] != rt[k]) {
          tok_mismatch = static_cast<int64_t>(k);
          break;
        }
      }
      tokens = std::move(rt);
      n_chunk = max_chunks > 0 ? std::min(hdr[2], max_chunks) : hdr[2];
    }
    std::printf("model: %s (%s), device %s, %zu tokens, %d chunks of %d (BOS %s)\n", argv[1],
                (*lm)->quantization.c_str(), std::string(dev->name()).c_str(), tokens.size(), n_chunk, ctx,
                add_bos ? "yes" : "no");

    KvGeometry geo = kv_geometry_for(c, DType::kF16, 16, ctx);
    geo.layout = dev->preferred_kv_layout();
    auto kv = KvBlockPool::create(geo, *dev);
    if (!kv.ok()) throw std::runtime_error("KV: " + kv.status().to_string());
    std::vector<float> logits(static_cast<size_t>(ctx) * static_cast<size_t>(vocab));
    // BENCH_KLD_BASE: header, the evaluated tokens, then per chunk one row per
    // scored position: as llama-perplexity's log_softmax(), the log-probabilities
    // within 16 of the maximum as uint16 steps above the minimum (scale, min
    // log-prob as two leading floats).
    std::ofstream kld;
    const int64_t nv = 2 * ((vocab + 1) / 2) + 4;
    std::vector<uint16_t> row(static_cast<size_t>(nv));
    if (const char* kp = std::getenv("BENCH_KLD_BASE"); kp != nullptr && *kp != '\0') {
      kld.open(kp, std::ios::binary);
      if (!kld) throw std::runtime_error(std::string("cannot write ") + kp);
      const int32_t hdr[3] = {ctx, static_cast<int32_t>(vocab), n_chunk};
      kld.write("_logits_", 8);
      kld.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
      kld.write(reinterpret_cast<const char*>(tokens.data()),
                static_cast<std::streamsize>(static_cast<size_t>(n_chunk) * ctx * sizeof(TokenId)));
    }
    auto write_row = [&](const float* l) {
      float mx = l[0], mn = l[0];
      for (int64_t v = 1; v < vocab; ++v) mx = std::max(mx, l[v]), mn = std::min(mn, l[v]);
      mn = std::max(mn, mx - 16);
      double sum = 0;
      for (int64_t v = 0; v < vocab; ++v) sum += expf(l[v] - mx);
      const float lse = static_cast<float>(std::log(sum));
      const float scale = (mx - mn) / 65535.f;
      const float head[2] = {scale, mn - mx - lse};
      std::fill(row.begin(), row.end(), uint16_t{0});
      std::memcpy(row.data(), head, sizeof(head));
      if (scale != 0) {
        const float inv = 1 / scale;
        for (int64_t v = 0; v < vocab; ++v) {
          row[static_cast<size_t>(4 + v)] = l[v] > mn ? static_cast<uint16_t>(std::lround(inv * (l[v] - mn))) : 0;
        }
      }
      kld.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(uint16_t)));
    };
    // Parity with the reference (BENCH_KLD_REF), per scored position.
    struct Parity {
      int64_t n = 0, agree = 0, top5 = 0, near_ties = 0;  // near tie: llama.cpp's top-2 gap < 0.1 nats
      double kl = 0, kl_max = 0, dp_max = 0, dlp_max = 0;
      int64_t first_chunk = -1, first_pos = -1, first_ref = -1, first_dyn = -1;
      double first_margin = 0;
      std::vector<double> margins;  // llama.cpp's top-2 gap where the top-1 differs
    } par;
    std::vector<uint16_t> rrow(static_cast<size_t>(nv));
    std::vector<double> lp(static_cast<size_t>(vocab));
    auto compare_row = [&](const float* l, TokenId next, int chunk, int pos) {
      ref.read(reinterpret_cast<char*>(rrow.data()), static_cast<std::streamsize>(rrow.size() * sizeof(uint16_t)));
      if (!ref) throw std::runtime_error("reference log-probabilities truncated");
      float head[2];
      std::memcpy(head, rrow.data(), sizeof(head));
      const uint16_t* q = rrow.data() + 4;
      auto rlp = [&](int64_t v) { return static_cast<double>(head[0]) * q[v] + head[1]; };  // as llama-perplexity
      double mx = l[0];
      for (int64_t v = 1; v < vocab; ++v) mx = std::max(mx, static_cast<double>(l[v]));
      double sum = 0;
      for (int64_t v = 0; v < vocab; ++v) sum += std::exp(l[v] - mx);
      const double lse = mx + std::log(sum);
      for (int64_t v = 0; v < vocab; ++v) lp[static_cast<size_t>(v)] = l[v] - lse;
      std::vector<int64_t> rd(5), dd(5);
      auto top5 = [&](auto&& score, std::vector<int64_t>& out) {
        std::vector<std::pair<double, int64_t>> best;
        for (int64_t v = 0; v < vocab; ++v) {
          const double s = score(v);
          if (best.size() < 5 || s > best.back().first) {
            best.emplace_back(s, v);
            std::sort(best.begin(), best.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (best.size() > 5) best.pop_back();
          }
        }
        for (size_t k = 0; k < 5; ++k) out[k] = best[k].second;
        return best[0].first - best[1].first;
      };
      const double margin = top5([&](int64_t v) { return rlp(v); }, rd);
      top5([&](int64_t v) { return lp[static_cast<size_t>(v)]; }, dd);
      for (int64_t a : rd) par.top5 += std::count(dd.begin(), dd.end(), a);
      if (rd[0] == dd[0]) {
        ++par.agree;
      } else {
        par.margins.push_back(margin);
        par.near_ties += margin < 0.1;
        if (par.first_chunk < 0) {
          par.first_chunk = chunk, par.first_pos = pos, par.first_ref = rd[0], par.first_dyn = dd[0];
          par.first_margin = margin;
        }
      }
      double kl = 0;  // KL(llama.cpp || DynaLM), as llama-perplexity sums it (stored log-probs above -16)
      for (int64_t v = 0; v < vocab; ++v) {
        const double r = rlp(v);
        if (r > -16) kl += std::exp(r) * (r - lp[static_cast<size_t>(v)]);
      }
      par.kl += kl;
      par.kl_max = std::max(par.kl_max, kl);
      par.dp_max = std::max(par.dp_max, std::abs(std::exp(lp[static_cast<size_t>(next)]) - std::exp(rlp(next))));
      par.dlp_max = std::max(par.dlp_max, std::abs(lp[static_cast<size_t>(rd[0])] - rlp(rd[0])));
      ++par.n;
    };
    double nll = 0, nll2 = 0;
    int64_t count = 0;
    const bool fresh_kv = [] {
      const char* v = std::getenv("BENCH_FRESH_KV");
      return v == nullptr || std::string_view(v) != "0";
    }();
    std::printf("scoring: ubatch %d, KV %s\n", [&] {
      const char* v = std::getenv("BENCH_UBATCH");
      const int32_t n = v != nullptr ? std::atoi(v) : 0;
      return n > 0 && n < ctx ? n : ctx;
    }(), fresh_kv ? "fresh pool per chunk" : "one pool reused across chunks");
    for (int i = 0; i < n_chunk; ++i) {
      if (fresh_kv && i > 0) {
        kv = KvBlockPool::create(geo, *dev);
        if (!kv.ok()) throw std::runtime_error("KV: " + kv.status().to_string());
      }
      std::vector<TokenId> chunk(tokens.begin() + static_cast<ptrdiff_t>(i) * ctx,
                                 tokens.begin() + static_cast<ptrdiff_t>(i + 1) * ctx);
      if (add_bos) chunk[0] = bos;
      KvBlockTable tab(**kv);
      if (!tab.reserve(ctx).ok()) throw std::runtime_error("KV reserve");
      // BENCH_UBATCH=n: the chunk in consecutive batches of n tokens (as llama.cpp's -ub n)
      // instead of one batch of ctx; kernels can depend on the batch size.
      static const int32_t ub = [&] {
        const char* v = std::getenv("BENCH_UBATCH");
        const int32_t n = v != nullptr ? std::atoi(v) : 0;
        return n > 0 && n < ctx ? n : ctx;
      }();
      for (int32_t s = 0; s < ctx; s += ub) {
        const int32_t n = std::min(ub, ctx - s);
        SeqBatch b{std::span<const TokenId>(chunk).subspan(static_cast<size_t>(s), static_cast<size_t>(n)), s,
                   tab.block_table(), true};
        b.logits_last = n;
        ExecutionPlan plan = (*tf)->planner().plan({&b, 1});
        std::span<float> out(logits.data() + static_cast<size_t>(s) * static_cast<size_t>(vocab),
                             static_cast<size_t>(n) * static_cast<size_t>(vocab));
        if (Status st = (*tf)->forward_batch({&b, 1}, **kv, out, &plan); !st.ok()) {
          throw std::runtime_error("forward: " + st.to_string());
        }
      }
      for (int32_t j = ctx / 2; j < ctx - 1; ++j) {
        const float* l = logits.data() + static_cast<size_t>(j) * static_cast<size_t>(vocab);
        if (kld.is_open()) write_row(l);
        if (ref.is_open()) compare_row(l, chunk[static_cast<size_t>(j + 1)], i, j);
        float mx = l[0];
        for (int64_t v = 1; v < vocab; ++v) mx = std::max(mx, l[v]);
        double sum = 0;
        for (int64_t v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(l[v]) - mx);
        const double x = -(static_cast<double>(l[chunk[static_cast<size_t>(j + 1)]]) - mx - std::log(sum));
        nll += x;
        nll2 += x * x;
        ++count;
      }
      std::printf("[%d]%.4f,", i + 1, std::exp(nll / static_cast<double>(count)));
      std::fflush(stdout);
    }
    const double mean = nll / static_cast<double>(count);
    const double var = nll2 / static_cast<double>(count) - mean * mean;
    const double se = std::sqrt(std::max(0.0, var) / static_cast<double>(count - 1));
    std::printf("\nFinal estimate: PPL = %.4f +/- %.5f (%lld scored tokens)\n", std::exp(mean), std::exp(mean) * se,
                static_cast<long long>(count));
    if (ref.is_open() && par.n > 0) {
      std::sort(par.margins.begin(), par.margins.end());
      const double n = static_cast<double>(par.n);
      std::printf(
          "PARITY {\"positions\": %lld, \"top1\": %.4f, \"top5_overlap\": %.4f, \"kl_mean\": %.8f, \"kl_max\": %.6f, "
          "\"dp_max\": %.6f, \"dlogp_top1_max\": %.6f, \"ppl_dynalm\": %.6f, \"disagreements\": %lld, "
          "\"near_ties\": %lld, \"median_margin\": %.6f, \"first_div\": {\"chunk\": %lld, \"pos\": %lld, \"llama\": %lld, "
          "\"dynalm\": %lld, \"margin\": %.6f}, \"tokenizer_first_mismatch\": %lld}\n",
          static_cast<long long>(par.n), 100.0 * static_cast<double>(par.agree) / n,
          100.0 * static_cast<double>(par.top5) / (5 * n), par.kl / n, par.kl_max, par.dp_max, par.dlp_max,
          std::exp(mean), static_cast<long long>(par.n - par.agree),
          static_cast<long long>(par.near_ties), par.margins.empty() ? 0.0 : par.margins[par.margins.size() / 2],
          static_cast<long long>(par.first_chunk), static_cast<long long>(par.first_pos),
          static_cast<long long>(par.first_ref), static_cast<long long>(par.first_dyn), par.first_margin,
          static_cast<long long>(tok_mismatch));
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    rc = 1;
  }
  return rc;  // the ggml device is destroyed before exit (ggml-metal asserts otherwise)
}
