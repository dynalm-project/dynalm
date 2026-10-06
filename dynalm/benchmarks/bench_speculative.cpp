// Phase 27: speculative decoding speed-up (single sequence, greedy).
//
//   bench_speculative <target.gguf> [draft.gguf] [threads] [k]
//
// For two chat prompts — one that quotes its input (prompt lookup's best
// case) and an open-ended one — compares plain greedy decoding with the
// n-gram drafter and (if given) a draft model sharing the target's
// vocabulary. Reports decode tok/s, acceptance, tokens per target pass, and
// whether the output equals plain greedy.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/base/timer.h"
#include "loader/model_loader.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "runtime/generator.h"
#include "runtime/sequence.h"
#include "runtime/speculative.h"
#include "common/core.h"

using namespace dynalm;

namespace {

struct Model {
  std::unique_ptr<LoadedModel> m;
  std::unique_ptr<KvBlockPool> kv;
  std::unique_ptr<Transformer> tf;
};

bool load(const std::string& path, CpuDevice& be, Model& out) {
  auto m = load_model(path);
  if (!m.ok()) return std::fprintf(stderr, "%s\n", m.status().to_string().c_str()), false;
  out.m = std::move(*m);
  auto kv = KvBlockPool::create(kv_geometry_for(out.m->config, DType::kF16, 16, 4096), be);
  auto tf = Transformer::create(out.m->config, out.m->weights, be, 64);
  if (!kv.ok() || !tf.ok()) return false;
  out.kv = std::move(*kv);
  out.tf = std::move(*tf);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: bench_speculative <target.gguf> [draft.gguf] [threads] [k]\n"), 1;
  const int threads = argc > 3 ? std::atoi(argv[3]) : cpu_info().physical_cores;
  const int k = argc > 4 ? std::atoi(argv[4]) : 4;
  ThreadPool pool(threads);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  Model target, draft;
  if (!load(argv[1], be, target)) return 1;
  const bool have_draft = argc > 2 && std::string(argv[2]) != "-" && load(argv[2], be, draft);

  const std::string paragraph =
      "The library opens at nine in the morning and closes at six in the evening. Visitors may borrow up to five "
      "books at a time for three weeks. Late returns are charged ten cents per day, and lost books must be "
      "replaced or paid for at the current price.";
  const std::pair<const char*, std::string> prompts[] = {
      {"quote", "Repeat the following text exactly, then stop:\n\n" + paragraph},
      {"open", "Write a short story about a lighthouse keeper who finds a message in a bottle."},
  };
  constexpr int32_t kTokens = 96;
  std::printf("target %s, draft %s, %d threads, k = %d, %d new tokens (greedy)\n\n", argv[1],
              have_draft ? argv[2] : "(none)", threads, k, kTokens);
  std::printf("%-6s %-8s %9s %9s %11s %10s %7s\n", "prompt", "drafter", "tok/s", "speedup", "acceptance", "tok/pass",
              "exact");

  for (const auto& [name, text] : prompts) {
    const ChatMessage msgs[] = {{"user", text}};
    auto chat = target.m->chat_template->apply(msgs, true);
    if (!chat.ok()) return 1;
    const std::vector<TokenId> prompt = target.m->tokenizer->encode(*chat, true, true);
    GenerateOptions o;
    o.max_new_tokens = kTokens;
    o.stop_at_eog = false;

    Generator plain(*target.tf, *target.kv, *target.m->tokenizer);
    std::vector<TokenId> ref;
    (void)plain.generate(prompt, o, [&](TokenId t) { ref.push_back(t); return true; });  // warm-up
    ref.clear();
    GenerationStats gs;
    (void)plain.generate(prompt, o, [&](TokenId t) { ref.push_back(t); return true; }, &gs);
    const double base = gs.decode_tok_per_s();
    std::printf("%-6s %-8s %9.1f %9s %11s %10s %7s\n", name, "none", base, "1.00x", "-", "1.00", "-");

    auto run = [&](Drafter& d) {
      SpeculativeGenerator sg(*target.tf, *target.kv, *target.m->tokenizer, d);
      SpeculativeOptions so;
      so.draft_tokens = k;
      std::vector<TokenId> out;
      SpeculativeStats st;
      int64_t t_first = 0;
      const int64_t t0 = now_ns();
      (void)sg.generate(prompt, o, so,
                        [&](TokenId t) {
                          if (out.empty()) t_first = now_ns();
                          out.push_back(t);
                          return true;
                        },
                        &st);
      const double secs = static_cast<double>(now_ns() - (t_first ? t_first : t0)) * 1e-9;
      const double tps = secs > 0 ? static_cast<double>(out.size() - 1) / secs : 0;
      std::printf("%-6s %-8s %9.1f %8.2fx %10.0f%% %10.2f %7s\n", name, std::string(d.name()).c_str(), tps, tps / base,
                  100 * st.acceptance(), st.tokens_per_pass(), out == ref ? "yes" : "NO");
    };
    NgramDrafter ngram;
    run(ngram);
    if (have_draft) {
      auto md = ModelDrafter::create(*draft.tf, *draft.kv, *draft.m->tokenizer, *target.m->tokenizer);
      if (!md.ok()) return std::fprintf(stderr, "%s\n", md.status().to_string().c_str()), 1;
      run(**md);
    }
  }
  return 0;
}
