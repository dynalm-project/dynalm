// Perplexity over a text file, scored exactly as llama.cpp's llama-perplexity
// does, so the two numbers compare directly (same file, same context):
//
//   bench_ppl <model.gguf> <text file> [ctx 512] [chunks 0 = all] [threads]
//   BENCH_DEVICE=native (default) | ggml-cpu | metal | vulkan | cuda
//   BENCH_TOKENS=1: print the text's token ids instead (tokenizer comparisons)
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    std::printf("model: %s (%s), device %s, %zu tokens, %d chunks of %d (BOS %s)\n", argv[1],
                (*lm)->quantization.c_str(), std::string(dev->name()).c_str(), tokens.size(), n_chunk, ctx,
                add_bos ? "yes" : "no");

    KvGeometry geo = kv_geometry_for(c, DType::kF16, 16, ctx);
    geo.layout = dev->preferred_kv_layout();
    auto kv = KvBlockPool::create(geo, *dev);
    if (!kv.ok()) throw std::runtime_error("KV: " + kv.status().to_string());
    std::vector<float> logits(static_cast<size_t>(ctx) * static_cast<size_t>(vocab));
    double nll = 0, nll2 = 0;
    int64_t count = 0;
    for (int i = 0; i < n_chunk; ++i) {
      std::vector<TokenId> chunk(tokens.begin() + static_cast<ptrdiff_t>(i) * ctx,
                                 tokens.begin() + static_cast<ptrdiff_t>(i + 1) * ctx);
      if (add_bos) chunk[0] = bos;
      KvBlockTable tab(**kv);
      if (!tab.reserve(ctx).ok()) throw std::runtime_error("KV reserve");
      SeqBatch b{chunk, 0, tab.block_table(), true};
      b.logits_last = ctx;
      ExecutionPlan plan = (*tf)->planner().plan({&b, 1});
      if (Status st = (*tf)->forward_batch({&b, 1}, **kv, logits, &plan); !st.ok()) {
        throw std::runtime_error("forward: " + st.to_string());
      }
      for (int32_t j = ctx / 2; j < ctx - 1; ++j) {
        const float* l = logits.data() + static_cast<size_t>(j) * static_cast<size_t>(vocab);
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
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    rc = 1;
  }
  return rc;  // the ggml device is destroyed before exit (ggml-metal asserts otherwise)
}
