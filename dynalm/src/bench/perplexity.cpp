#include "bench/perplexity.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/recording_device.h"
#include "kv_cache/kv_cache.h"
#include "loader/model_loader.h"
#include "model/transformer.h"
#include "runtime/engine.h"

namespace dynalm {

std::string parity_json(const PerplexityResult& r) {
  const PerplexityParity& p = r.parity;
  char buf[1024];
  std::snprintf(
      buf, sizeof(buf),
      "{\"positions\": %lld, \"top1\": %.4f, \"top5_overlap\": %.4f, \"kl_mean\": %.8f, \"kl_max\": %.6f, "
      "\"dp_max\": %.6f, \"dlogp_top1_max\": %.6f, \"ppl_dynalm\": %.6f, \"disagreements\": %lld, "
      "\"near_ties\": %lld, \"median_margin\": %.6f, \"first_div\": {\"chunk\": %lld, \"pos\": %lld, \"llama\": %lld, "
      "\"dynalm\": %lld, \"margin\": %.6f}, \"tokenizer_first_mismatch\": %lld}",
      static_cast<long long>(p.positions), p.top1_pct(), p.top5_pct(), p.kl_mean, p.kl_max, p.dp_max,
      p.dlogp_top1_max, r.ppl, static_cast<long long>(p.positions - p.agree), static_cast<long long>(p.near_ties),
      p.median_margin, static_cast<long long>(p.first_chunk), static_cast<long long>(p.first_pos),
      static_cast<long long>(p.first_ref), static_cast<long long>(p.first_dyn), p.first_margin,
      static_cast<long long>(p.tokenizer_first_mismatch));
  return buf;
}

Result<PerplexityResult> run_perplexity(const PerplexityOptions& o) {
  std::FILE* log = o.log;
  const int32_t ctx = o.ctx;
  const int max_chunks = o.max_chunks;
  const int threads = o.threads > 0 ? o.threads : cpu_info().physical_cores;
  if (ctx < 16) return InvalidArgument("ctx must be >= 16");
  std::ifstream f(o.text_path, std::ios::binary);
  if (!f) return NotFound("cannot read " + o.text_path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  auto lm = load_model(o.model_path);
  if (!lm.ok()) return lm.status();
  const ModelConfig& c = (*lm)->config;
  const int64_t vocab = c.vocab_size;

  ThreadPool pool(threads);
  std::unique_ptr<Device> owned;  // the ggml device, when one is used
  std::unique_ptr<dynacore::ir::RecordingDevice> rec;
  std::unique_ptr<CpuDevice> cpu;
  Device* dev = nullptr;
  const std::string_view which = o.device;
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
    if (!g.ok()) return g.status();
    owned = std::move(*g);
    rec = std::make_unique<dynacore::ir::RecordingDevice>(*owned, dynacore::ir::RecordingDevice::Mode::kDeferred);
    dev = rec.get();
  }
  PerplexityResult result;
  try {
    auto tf = Transformer::create(c, (*lm)->weights, *dev, ctx, ctx);
    if (!tf.ok()) throw std::runtime_error(tf.status().to_string());
    // int8 activation row limit of the native CPU plan, as the engine sets it
    // (0 = fp32 everywhere, the reference arithmetic).
    if (o.int8_rows >= 0) {
      KernelPlan base = KernelPlan::defaults();
      base.int8_decode_max_rows = o.int8_rows;
      (*tf)->set_kernel_base(base);
      if (log) std::fprintf(log, "kernel plan: int8 rows %d, qmm %s\n", base.int8_decode_max_rows, base.qmm ? "on" : "off");
    }

    std::vector<TokenId> tokens = (*lm)->tokenizer->encode(text, /*add_special=*/true, false);
    const TokenId bos = (*lm)->tokenizer->bos();
    const bool add_bos = !tokens.empty() && bos != kNoToken && tokens.front() == bos;
    int n_chunk = static_cast<int>(tokens.size() / static_cast<size_t>(ctx));
    if (max_chunks > 0) n_chunk = std::min(n_chunk, max_chunks);
    if (n_chunk == 0) throw std::runtime_error("text shorter than one chunk");
    // kld_ref: a llama-perplexity --kl-divergence-base file. Its token ids are
    // scored (teacher-forced: the same input as llama.cpp's), DynaLM's own
    // tokenization of the text is only compared with them, and each scored
    // position is compared with llama.cpp's log-probabilities.
    std::ifstream ref;
    int64_t tok_mismatch = -1;  // first index where DynaLM's tokenizer differs from the file's
    if (!o.kld_ref.empty()) {
      ref.open(o.kld_ref, std::ios::binary);
      char magic[8] = {};
      int32_t hdr[3] = {};
      ref.read(magic, 8);
      ref.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
      if (!ref || std::memcmp(magic, "_logits_", 8) != 0) throw std::runtime_error(o.kld_ref + ": not a llama-perplexity file");
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
    if (log) {
      std::fprintf(log, "model: %s (%s), device %s, %zu tokens, %d chunks of %d (BOS %s)\n", o.model_path.c_str(),
                   (*lm)->quantization.c_str(), std::string(dev->name()).c_str(), tokens.size(), n_chunk, ctx,
                   add_bos ? "yes" : "no");
    }

    KvGeometry geo = kv_geometry_for(c, DType::kF16, 16, ctx);
    geo.layout = dev->preferred_kv_layout();
    auto kv = KvBlockPool::create(geo, *dev);
    if (!kv.ok()) throw std::runtime_error("KV: " + kv.status().to_string());
    std::vector<float> logits(static_cast<size_t>(ctx) * static_cast<size_t>(vocab));
    // kld_base: header, the evaluated tokens, then per chunk one row per
    // scored position: as llama-perplexity's log_softmax(), the log-probabilities
    // within 16 of the maximum as uint16 steps above the minimum (scale, min
    // log-prob as two leading floats).
    std::ofstream kld;
    const int64_t nv = 2 * ((vocab + 1) / 2) + 4;
    std::vector<uint16_t> row(static_cast<size_t>(nv));
    if (!o.kld_base.empty()) {
      kld.open(o.kld_base, std::ios::binary);
      if (!kld) throw std::runtime_error("cannot write " + o.kld_base);
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
    // Parity with the reference (kld_ref), per scored position.
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
    const int32_t ub = o.ubatch > 0 && o.ubatch < ctx ? o.ubatch : ctx;
    if (log) std::fprintf(log, "scoring: ubatch %d, KV %s\n", ub, o.fresh_kv ? "fresh pool per chunk" : "one pool reused across chunks");
    for (int i = 0; i < n_chunk; ++i) {
      if (o.fresh_kv && i > 0) {
        kv = KvBlockPool::create(geo, *dev);
        if (!kv.ok()) throw std::runtime_error("KV: " + kv.status().to_string());
      }
      std::vector<TokenId> chunk(tokens.begin() + static_cast<ptrdiff_t>(i) * ctx,
                                 tokens.begin() + static_cast<ptrdiff_t>(i + 1) * ctx);
      if (add_bos) chunk[0] = bos;
      KvBlockTable tab(**kv);
      if (!tab.reserve(ctx).ok()) throw std::runtime_error("KV reserve");
      // ubatch: the chunk in consecutive batches of n tokens (as llama.cpp's -ub n)
      // instead of one batch of ctx; kernels can depend on the batch size.
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
      if (log) {
        std::fprintf(log, "[%d]%.4f,", i + 1, std::exp(nll / static_cast<double>(count)));
        std::fflush(log);
      }
    }
    const double mean = nll / static_cast<double>(count);
    const double var = nll2 / static_cast<double>(count) - mean * mean;
    const double se = std::sqrt(std::max(0.0, var) / static_cast<double>(count - 1));
    result.ppl = std::exp(mean);
    result.ppl_se = std::exp(mean) * se;
    result.scored = count;
    result.chunks = n_chunk;
    result.device_name = std::string(dev->name());
    result.quantization = (*lm)->quantization;
    if (log) {
      std::fprintf(log, "\nFinal estimate: PPL = %.4f +/- %.5f (%lld scored tokens)\n", result.ppl, result.ppl_se,
                   static_cast<long long>(count));
    }
    if (ref.is_open() && par.n > 0) {
      std::sort(par.margins.begin(), par.margins.end());
      const double n = static_cast<double>(par.n);
      PerplexityParity& p = result.parity;
      result.has_parity = true;
      p.positions = par.n;
      p.agree = par.agree;
      p.top5 = par.top5;
      p.near_ties = par.near_ties;
      p.kl_mean = par.kl / n;
      p.kl_max = par.kl_max;
      p.dp_max = par.dp_max;
      p.dlogp_top1_max = par.dlp_max;
      p.median_margin = par.margins.empty() ? 0.0 : par.margins[par.margins.size() / 2];
      p.first_chunk = par.first_chunk;
      p.first_pos = par.first_pos;
      p.first_ref = par.first_ref;
      p.first_dyn = par.first_dyn;
      p.first_margin = par.first_margin;
      p.tokenizer_first_mismatch = tok_mismatch;
      if (log) std::fprintf(log, "PARITY %s\n", parity_json(result).c_str());
    }
  } catch (const std::exception& ex) {
    return Internal(ex.what());
  }
  return result;  // the ggml device is destroyed before the caller continues (ggml-metal asserts otherwise)
}

}  // namespace dynalm
