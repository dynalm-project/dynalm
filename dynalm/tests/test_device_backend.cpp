// Phase 28: GPU-backend architecture (DD-045).
//
// GuardedBackend models a discrete device. Its memory is host pages that are
// mprotect(PROT_NONE) except while one of ITS OWN ops runs, and it reports
// host_accessible() == false. Running whole models through it proves the
// runtime (Transformer, KV cache, prefix cache, scheduler, speculative
// decoding) never dereferences device memory directly: any such access is
// a SIGSEGV. Results must equal the plain CPU backend bit for bit (same
// kernels, same arithmetic order).

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "dynacore/device/device_registry.h"
#include "dynacore/cpu/cpu_device.h"
#include "loader/model_loader.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "runtime/generator.h"
#include "runtime/speculative.h"
#include "scheduler/scheduler.h"
#include "common/core.h"

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace dynalm {
namespace {

TEST(BackendRegistry, CpuBuiltGpuKindsReportedClearly) {
  ThreadPool pool(1);
  auto cpu = create_device(DeviceKind::kCpu, pool);
  ASSERT_TRUE(cpu.ok());
  EXPECT_TRUE((*cpu)->host_accessible());
  for (const char* name : {"cuda", "hip", "metal", "vulkan"}) {
    auto kind = parse_device_kind(name);
    ASSERT_TRUE(kind.ok());
    auto b = create_device(*kind, pool);
    ASSERT_FALSE(b.ok());
    EXPECT_EQ(b.status().code(), StatusCode::kUnsupported);
    EXPECT_NE(b.status().message().find(name), std::string::npos);
  }
  EXPECT_FALSE(parse_device_kind("tpu").ok());
  EXPECT_EQ(compiled_devices(), std::vector<DeviceKind>{DeviceKind::kCpu});
}

#if defined(__linux__)

std::string data(const std::string& f) { return std::string(ENGINE_TEST_DATA_DIR) + "/" + f; }

class GuardedBackend final : public Device {
 public:
  explicit GuardedBackend(ThreadPool& pool) : cpu_(pool, select_best_isa(cpu_info().features)) {}
  ~GuardedBackend() override {
    // Outstanding storages keep their pages (process exit reclaims them).
  }

  std::string_view name() const override { return "guarded"; }
  DeviceLoc device() const override { return {DeviceType::kSimulated, 0}; }
  bool host_accessible() const override { return false; }
  bool supports_weight_type(DType t) const override { return cpu_.supports_weight_type(t); }
  void synchronize() override {}
  uint64_t ops() const { return ops_; }

  Result<std::shared_ptr<Storage>> allocate(size_t bytes) override {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t len = std::max(page, (bytes + page - 1) / page * page);
    void* p = mmap(nullptr, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return OutOfMemory("guarded allocation failed");
    {
      std::lock_guard<std::mutex> lock(mu_);
      regions_[p] = len;
    }
    auto keep = std::shared_ptr<const void>(p, [this, len](const void* q) {
      std::lock_guard<std::mutex> lock(mu_);
      regions_.erase(const_cast<void*>(q));
      munmap(const_cast<void*>(q), len);
    });
    return Storage::borrow(p, bytes, device(), std::move(keep));
  }
  void copy(void* dst, const void* src, size_t bytes) override {
    Open o(*this);
    cpu_.copy(dst, src, bytes);
  }
  Result<Tensor> upload(const Tensor& host) override {
    ENGINE_ASSIGN_OR_RETURN(TensorLayout layout, TensorLayout::contiguous(host.dtype(), host.shape()));
    ENGINE_ASSIGN_OR_RETURN(auto storage, allocate(static_cast<size_t>(layout.span_bytes())));
    {
      Open o(*this);
      std::memcpy(storage->data(), host.data(), static_cast<size_t>(layout.span_bytes()));
    }
    return Tensor::from_storage(storage, TensorView(storage->data(), layout, device()));
  }
  void download(const TensorView& src, std::span<float> dst) override {
    Open o(*this);
    cpu_.download(src, dst);
  }

  void embedding(const TensorView& t, std::span<const int32_t> ids, const TensorView& out) override {
    Open o(*this);
    cpu_.embedding(t, ids, out);
  }
  void matmul(const TensorView& x, const TensorView& w, const TensorView* b, const TensorView& y) override {
    Open o(*this);
    cpu_.matmul(x, w, b, y);
  }
  void matmul_many(std::span<const MatmulJob> jobs) override {
    Open o(*this);
    cpu_.matmul_many(jobs);
  }
  void rms_norm(const TensorView& x, const TensorView& w, float eps, const TensorView& y) override {
    Open o(*this);
    cpu_.rms_norm(x, w, eps, y);
  }
  void layer_norm(const TensorView& x, const TensorView& w, const TensorView* b, float eps,
                  const TensorView& y) override {
    Open o(*this);
    cpu_.layer_norm(x, w, b, eps, y);
  }
  void rope(const TensorView& x, int32_t heads, int32_t hd, std::span<const int32_t> pos, const RopeConfig& r,
            const float* ff) override {
    Open o(*this);
    cpu_.rope(x, heads, hd, pos, r, ff);
  }
  void kv_store(const TensorView& k, const TensorView& v, std::span<const int32_t> pos,
                std::span<const int32_t> row_seq, std::span<const KvLayerView> kv) override {
    Open o(*this);
    cpu_.kv_store(k, v, pos, row_seq, kv);
  }
  void attention(const AttentionParams& p) override {
    Open o(*this);
    cpu_.attention(p);
  }
  void act_mul(Activation a, const TensorView& g, const TensorView& u, const TensorView& out) override {
    Open o(*this);
    cpu_.act_mul(a, g, u, out);
  }
  void activation(Activation a, const TensorView& x, const TensorView& out) override {
    Open o(*this);
    cpu_.activation(a, x, out);
  }
  void add(const TensorView& a, const TensorView& b, const TensorView& y) override {
    Open o(*this);
    cpu_.add(a, b, y);
  }
  void scale(const TensorView& x, float s) override {
    Open o(*this);
    cpu_.scale(x, s);
  }
  void softcap(const TensorView& x, float cap) override {
    Open o(*this);
    cpu_.softcap(x, cap);
  }
  void fill(const TensorView& x, float v) override {
    Open o(*this);
    cpu_.fill(x, v);
  }
  void gather_rows(const TensorView& src, std::span<const int32_t> rows, const TensorView& dst) override {
    Open o(*this);
    cpu_.gather_rows(src, rows, dst);
  }
  void scatter_add_rows(const TensorView& src, std::span<const int32_t> rows, std::span<const float> w,
                        const TensorView& dst) override {
    Open o(*this);
    cpu_.scatter_add_rows(src, rows, w, dst);
  }

  // Touches the first byte of a device allocation from the host (for the
  // death test that shows the guard is armed).
  static void poke(const Storage& s) { *static_cast<volatile char*>(s.data()) = 1; }

 private:
  // Unprotects every device region for the duration of one op.
  struct Open {
    explicit Open(GuardedBackend& b) : b_(b) {
      std::lock_guard<std::mutex> lock(b_.mu_);
      ++b_.ops_;
      for (const auto& [p, len] : b_.regions_) mprotect(p, len, PROT_READ | PROT_WRITE);
    }
    ~Open() {
      std::lock_guard<std::mutex> lock(b_.mu_);
      for (const auto& [p, len] : b_.regions_) mprotect(p, len, PROT_NONE);
    }
    GuardedBackend& b_;
  };

  CpuDevice cpu_;
  std::mutex mu_;
  std::map<void*, size_t> regions_;
  uint64_t ops_ = 0;
};

TEST(GuardedBackend, HostAccessToDeviceMemoryFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ThreadPool pool(1);
  GuardedBackend g(pool);
  auto s = g.allocate(4096);
  ASSERT_TRUE(s.ok());
  EXPECT_DEATH(GuardedBackend::poke(**s), "");
}

// One model on a backend: KV pool + transformer, everything through `be`.
struct Stack {
  std::unique_ptr<KvBlockPool> kv;
  std::unique_ptr<Transformer> tf;
};

Stack build(const LoadedModel& m, Device& be, int32_t blocks = 32) {
  const ModelConfig& c = m.config;
  KvGeometry g{c.num_layers, c.num_kv_heads, c.head_dim, c.head_dim_v, 16, blocks, DType::kF16};
  auto kv = KvBlockPool::create(g, be);
  auto tf = Transformer::create(c, m.weights, be, 32);
  EXPECT_TRUE(kv.ok() && tf.ok()) << (tf.ok() ? "" : tf.status().to_string());
  return {std::move(*kv), std::move(*tf)};
}

class DeviceArch : public ::testing::TestWithParam<std::string> {};

TEST_P(DeviceArch, ForwardAndGenerationMatchCpu) {
  auto m = load_model(data("tiny_" + GetParam() + ".gguf"));
  ASSERT_TRUE(m.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  GuardedBackend dev(pool);
  Stack a = build(**m, cpu), b = build(**m, dev);

  const std::vector<TokenId> prompt = {5, 17, 99, 3, 200, 42, 7, 128, 64, 11, 250, 33, 5, 17, 99};
  GenerateOptions o;
  o.max_new_tokens = 24;
  o.stop_at_eog = false;
  std::vector<TokenId> out_cpu, out_dev;
  Generator gc(*a.tf, *a.kv, *(*m)->tokenizer), gd(*b.tf, *b.kv, *(*m)->tokenizer);
  ASSERT_TRUE(gc.generate(prompt, o, [&](TokenId t) { out_cpu.push_back(t); return true; }).ok());
  ASSERT_TRUE(gd.generate(prompt, o, [&](TokenId t) { out_dev.push_back(t); return true; }).ok());
  EXPECT_EQ(out_dev, out_cpu);
  EXPECT_GT(dev.ops(), 100u);

  // Logits bit for bit.
  KvBlockTable ta(*a.kv), tb(*b.kv);
  ASSERT_TRUE(ta.reserve(static_cast<int64_t>(prompt.size())).ok() && tb.reserve(static_cast<int64_t>(prompt.size())).ok());
  std::vector<int32_t> pos(prompt.size());
  std::iota(pos.begin(), pos.end(), 0);
  std::vector<float> la(static_cast<size_t>((*m)->config.vocab_size)), lb(la.size());
  ASSERT_TRUE(a.tf->forward(prompt, pos, *a.kv, ta.block_table(), la).ok());
  ASSERT_TRUE(b.tf->forward(prompt, pos, *b.kv, tb.block_table(), lb).ok());
  EXPECT_EQ(la, lb);
}

INSTANTIATE_TEST_SUITE_P(All, DeviceArch,
                         ::testing::Values("llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3", "mixtral",
                                           "qwen2moe", "qwen3moe", "granitemoe"),
                         [](const ::testing::TestParamInfo<std::string>& i) { return i.param; });

TEST(GuardedBackend, SchedulerPrefixCacheAndSpeculationStayOnDevice) {
  // Continuous batching with shared prefixes (prefix cache + copy-on-write
  // KV blocks) and speculative decoding (multi-row logits + KV rollback).
  auto m = load_model(data("tiny_llama.gguf"));
  ASSERT_TRUE(m.ok());
  ThreadPool pool(2);
  CpuDevice cpu(pool, select_best_isa(cpu_info().features));
  GuardedBackend dev(pool);
  auto run_scheduler = [&](Device& be) {
    Stack s = build(**m, be, 64);
    Scheduler sched(*s.tf, *s.kv, *(*m)->tokenizer, SchedulerConfig{});
    std::map<uint64_t, std::vector<TokenId>> out;
    std::vector<TokenId> shared(40);
    std::iota(shared.begin(), shared.end(), 10);
    for (int i = 0; i < 6; ++i) {
      Request r;
      r.prompt = shared;
      r.prompt.push_back(static_cast<TokenId>(100 + i));
      if (i % 2) r.prompt.resize(37);  // partial-block prefix match -> copy-on-write
      r.stop = StopParams{12, false};
      r.on_event = [&out](const RequestEvent& ev) {
        if (!ev.finished) out[ev.request_id].push_back(ev.token);
      };
      sched.submit(std::move(r));
      sched.step();  // staggered: later requests find cached prefixes
    }
    sched.run_until_idle();
    std::vector<std::vector<TokenId>> v;
    for (auto& [id, toks] : out) v.push_back(toks);
    return v;
  };
  EXPECT_EQ(run_scheduler(dev), run_scheduler(cpu));

  auto run_spec = [&](Device& be) {
    Stack s = build(**m, be, 64);
    NgramDrafter ngram;
    SpeculativeGenerator g(*s.tf, *s.kv, *(*m)->tokenizer, ngram);
    GenerateOptions o;
    o.max_new_tokens = 30;
    o.stop_at_eog = false;
    std::vector<TokenId> out;
    const std::vector<TokenId> prompt = {5, 17, 99, 3, 5, 17, 99, 3, 5, 17};
    EXPECT_TRUE(g.generate(prompt, o, SpeculativeOptions{}, [&](TokenId t) { out.push_back(t); return true; }).ok());
    return out;
  };
  EXPECT_EQ(run_spec(dev), run_spec(cpu));
}

#endif  // __linux__

}  // namespace
}  // namespace dynalm
