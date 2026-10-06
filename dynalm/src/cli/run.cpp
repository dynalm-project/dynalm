// `dynalm run <model> -p <prompt> [options]`: streamed generation through the
// Engine (the same path the HTTP server uses). Without -p (or as `dynalm chat`)
// it starts an interactive chat (cli/chat.cpp).

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <chrono>
#include <string>
#include <thread>

#include "dynacore/device/device_registry.h"
#include "cli/chat.h"
#include "cli/commands.h"
#include "config/config.h"
#include "dynacore/base/timer.h"
#include "logging/log.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/hardware/thread_qos.h"
#include "runtime/engine.h"
#include "runtime/speculative.h"
#include "runtime/text_stream.h"
#include "common/core.h"

namespace dynalm::cli {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

void usage() {
  std::fprintf(stderr,
               "usage: dynalm run <model> -p <prompt> [options]   one answer, then exit\n"
               "       dynalm run <model> [options]               interactive chat (same as dynalm chat)\n"
               "  -p, --prompt TEXT     prompt text\n"
               "  -n, --max-tokens N    tokens to generate (default 128; chat: 2048 per reply)\n"
               "  --chat                wrap the prompt in the model's chat template (default)\n"
               "  --raw                 use the prompt as-is (no chat template)\n"
               "  --system TEXT         system message (chat mode)\n"
               "  --stop TEXT           stop string (repeatable; never printed)\n"
               "  -t, --threads N       worker threads (default: physical cores)\n"
               "  -c, --ctx N           KV cache capacity in tokens (default 4096)\n"
               "  --batch N             max tokens per forward pass (default 256)\n"
               "  --kv f16|f32          KV cache dtype (default f16)\n"
               "  --int8-decode N       int8 activations for matmuls of <= N rows (default 4, 0 = off)\n"
               "  --backend cpu         compute backend (GPU backends are not built yet)\n"
               "  --no-stream           print only the final text\n"
               "sampling (default: greedy; chat: temp 0.8, top-k 40, top-p 0.9, repeat-penalty 1.1):\n"
               "  --temp T              temperature (0 = greedy)\n"
               "  --top-k K, --top-p P, --min-p P\n"
               "  --repeat-penalty R, --presence-penalty P, --frequency-penalty F, --repeat-last-n N\n"
               "  --seed S              RNG seed (reproducible output)\n"
               "speculative decoding (single sequence, DD-044):\n"
               "  --spec ngram|DRAFT    prompt-lookup drafter, or a draft model sharing the vocabulary\n"
               "  --spec-k K            most tokens drafted per step (default 4)\n"
               "  --spec-fixed          always draft K; by default k is chosen from {0, 3, K} by measured\n"
               "                        tokens/s, and speculation turns off when it is slower (DD-062)\n");
}

bool parse_int(std::string_view s, int& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc() && p == s.data() + s.size();
}

bool parse_float(std::string_view s, float& out) {
  try {
    size_t used = 0;
    const std::string str(s);
    out = std::stof(str, &used);
    return used == str.size();
  } catch (...) {
    return false;
  }
}

struct SpecModel {
  std::unique_ptr<LoadedModel> model;
  std::unique_ptr<KvBlockPool> kv;
  std::unique_ptr<Transformer> tf;
};

Status load_spec_model(const std::string& path, CpuDevice& be, DType kv_dtype, int ctx, int batch, SpecModel& out) {
  ENGINE_ASSIGN_OR_RETURN(out.model, load_model(path));
  ENGINE_ASSIGN_OR_RETURN(out.kv, KvBlockPool::create(kv_geometry_for(out.model->config, kv_dtype, 16, ctx), be));
  ENGINE_ASSIGN_OR_RETURN(out.tf, Transformer::create(out.model->config, out.model->weights, be, batch));
  return Status::Ok();
}

// `dynalm run --spec ...`: the single-sequence speculative path (the Engine's
// continuous-batching scheduler does not host drafters yet).
int run_speculative(const std::string& path, const std::string& spec, int k, bool adaptive,
                    const std::string& prompt,
                    const std::string& system, bool chat, bool stream, int threads, int ctx, int batch,
                    const EngineOptions& opts, const GenerateParams& params) {
  request_full_speed_process();  // as the Engine does (DD-052): this thread runs kernel chunks too
  request_full_speed_thread();
  ThreadPool pool(threads > 0 ? threads : cpu_info().physical_cores);
  CpuDevice be(pool, select_best_isa(cpu_info().features));
  SpecModel target, draft;
  if (Status st = load_spec_model(path, be, opts.kv_dtype, ctx, batch, target); !st.ok()) {
    std::fprintf(stderr, "run: %s\n", st.to_string().c_str());
    return 1;
  }
  const Tokenizer& tok = *target.model->tokenizer;
  std::unique_ptr<Drafter> drafter;
  if (spec == "ngram") {
    drafter = std::make_unique<NgramDrafter>();
  } else {
    if (Status st = load_spec_model(spec, be, opts.kv_dtype, ctx, batch, draft); !st.ok()) {
      std::fprintf(stderr, "run: draft: %s\n", st.to_string().c_str());
      return 1;
    }
    auto md = ModelDrafter::create(*draft.tf, *draft.kv, *draft.model->tokenizer, tok);
    if (!md.ok()) {
      std::fprintf(stderr, "run: %s\n", md.status().to_string().c_str());
      return 1;
    }
    drafter = std::move(*md);
  }
  std::vector<TokenId> ids;
  if (chat) {
    if (!target.model->chat_template) {
      std::fprintf(stderr, "run: model has no chat template (use --raw)\n");
      return 1;
    }
    std::vector<ChatMessage> msgs;
    if (!system.empty()) msgs.push_back({"system", system});
    msgs.push_back({"user", prompt});
    auto text = target.model->chat_template->apply(msgs, true);
    if (!text.ok()) return 1;
    ids = tok.encode(*text, true, true);
  } else {
    ids = tok.encode(prompt, true, false);
  }

  SpeculativeGenerator gen(*target.tf, *target.kv, tok, *drafter);
  GenerateOptions go;
  go.max_new_tokens = params.max_tokens;
  go.stop_at_eog = params.stop_at_eog;
  SpeculativeOptions so;
  so.draft_tokens = k;
  so.adaptive = adaptive;
  so.sampling = params.sampling;
  TextStreamer streamer(tok, params.stop);
  std::string text;
  SpeculativeStats st;
  const Stopwatch timer;
  double first_ms = -1;
  const Status s = gen.generate(ids, go, so, [&](TokenId t) {
    if (first_ms < 0) first_ms = timer.elapsed_ms();
    if (tok.is_eog(t) && go.stop_at_eog) return false;
    const TextStreamer::Delta d = streamer.push(t);
    if (stream) {
      std::fwrite(d.text.data(), 1, d.text.size(), stdout);
      std::fflush(stdout);
    } else {
      text += d.text;
    }
    return !d.stopped;
  }, &st);
  if (!streamer.stopped()) {
    const std::string rest = streamer.finish();
    if (stream) std::fwrite(rest.data(), 1, rest.size(), stdout); else text += rest;
  }
  std::printf(stream ? "\n" : "%s\n", text.c_str());
  if (!s.ok()) {
    std::fprintf(stderr, "run: %s\n", s.to_string().c_str());
    return 1;
  }
  const double decode_s = (timer.elapsed_ms() - std::max(first_ms, 0.0)) / 1e3;
  const std::string adaptive_note =
      adaptive ? "adaptive: best k " + std::to_string(st.final_k) + ", " + std::to_string(st.plain_passes) + "/" +
                     std::to_string(st.target_passes) + " passes without drafts"
               : std::string("fixed k");
  std::fprintf(stderr,
               "\n[stats] prompt %zu tok | TTFT %.1f ms | generated %d tok, %.1f tok/s | drafter %s, k %d | "
               "acceptance %.0f%% | %.2f tokens per target pass | %s\n",
               ids.size(), first_ms, st.generated, decode_s > 0 ? (st.generated - 1) / decode_s : 0.0,
               std::string(drafter->name()).c_str(), k, 100 * st.acceptance(), st.tokens_per_pass(),
               adaptive_note.c_str());
  return 0;
}

}  // namespace

// Options `run` takes from the config file and DYNALM_* (config/config.h).
constexpr OptionSpec kRunConfigOptions[] = {
    {"model"}, {"threads"}, {"ctx"}, {"batch"}, {"kv"}, {"backend"}, {"int8-decode"}, {"temperature"}, {"max-tokens"},
};

int cmd_run(std::span<const std::string_view> raw_args) {
  auto merged = merge_config(raw_args, kRunConfigOptions);
  if (!merged.ok()) {
    std::fprintf(stderr, "run: %s\n", merged.status().message().c_str());
    return 1;
  }
  const std::vector<std::string_view> arg_views(merged->begin(), merged->end());
  const std::span<const std::string_view> args(arg_views);
  bool max_set = false, sampling_set = false;
  std::string path, prompt, system, config_model;
  int max_tokens = 128, threads = 0, ctx = 4096, batch = 256;
  bool chat = true, stream = true;
  std::string spec;
  int spec_k = 4;
  bool spec_fixed = false;
  EngineOptions opts;
  GenerateParams params;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    auto value = [&]() -> std::string_view { return i + 1 < args.size() ? args[++i] : std::string_view(); };
    bool ok = true;
    if (a == "-p" || a == "--prompt") prompt = value();
    else if (a == "--system") system = value();
    else if (a == "--stop") params.stop.emplace_back(value());
    else if (a == "-n" || a == "--max-tokens") ok = parse_int(value(), max_tokens), max_set = true;
    else if (a == "-t" || a == "--threads") ok = parse_int_or_auto(value(), threads);
    else if (a == "-c" || a == "--ctx") ok = parse_int_or_auto(value(), ctx);
    else if (a == "--batch") ok = parse_int_or_auto(value(), batch);
    else if (a == "--model") config_model = value();
    else if (a == "--int8-decode") ok = parse_int(value(), opts.int8_decode_rows) && opts.int8_decode_rows >= 0;
    else if (a == "--chat") chat = true;
    else if (a == "--raw") chat = false;
    else if (a == "--no-stream") stream = false;
    else if (a == "--temp" || a == "--temperature") ok = parse_float(value(), params.sampling.temperature);
    else if (a == "--spec") spec = value();
    else if (a == "--spec-k") ok = parse_int(value(), spec_k) && spec_k >= 1;
    else if (a == "--spec-fixed") spec_fixed = true;
    else if (a == "--top-k") ok = parse_int(value(), params.sampling.top_k);
    else if (a == "--top-p") ok = parse_float(value(), params.sampling.top_p);
    else if (a == "--min-p") ok = parse_float(value(), params.sampling.min_p);
    else if (a == "--repeat-penalty") ok = parse_float(value(), params.sampling.repetition_penalty);
    else if (a == "--presence-penalty") ok = parse_float(value(), params.sampling.presence_penalty);
    else if (a == "--frequency-penalty") ok = parse_float(value(), params.sampling.frequency_penalty);
    else if (a == "--repeat-last-n") ok = parse_int(value(), params.sampling.penalty_last_n);
    else if (a == "--seed") {
      int seed = 0;
      ok = parse_int(value(), seed);
      params.sampling.seed = static_cast<uint64_t>(seed);
      params.sampling.has_seed = true;
    }
    else if (a == "--backend") {
      auto k = parse_device_kind(value());
      ok = k.ok();
      if (ok) opts.backend = *k;
    } else if (a == "--kv") {
      const std::string_view v = value();
      ok = v == "f16" || v == "f32";
      opts.kv_dtype = v == "f32" ? DType::kF32 : DType::kF16;
    } else if (path.empty() && !a.starts_with("-")) path = a;
    else ok = false;
    if (a.starts_with("--temp") || a.starts_with("--top-") || a == "--min-p" || a.ends_with("-penalty") ||
        a == "--repeat-last-n" || a == "--seed") {
      sampling_set = true;
    }
    if (!ok) {
      std::fprintf(stderr, "run: invalid argument '%.*s'\n", static_cast<int>(a.size()), a.data());
      usage();
      return 1;
    }
  }
  if (path.empty()) path = config_model;  // model.path from the config file
  if (ctx == 0) ctx = 4096;               // auto
  if (batch == 0) batch = 256;            // auto
  const bool interactive = prompt.empty();
  if (interactive && (!chat || !spec.empty())) {
    std::fprintf(stderr, "run: interactive chat needs chat mode (no --raw, no --spec); pass -p for one prompt\n");
    return 1;
  }
  if (interactive) {
    if (!max_set) max_tokens = 2048;
    if (!sampling_set) {  // conversational defaults: greedy decoding tends to loop in long replies
      params.sampling.temperature = 0.8f;
      params.sampling.top_k = 40;
      params.sampling.top_p = 0.9f;
      params.sampling.repetition_penalty = 1.1f;
    }
  }
  if (path.empty() || max_tokens <= 0 || threads < 0 || ctx <= 0 || batch <= 0) {
    usage();
    return 1;
  }
  if (auto resolved = ensure_model(path, true); resolved.ok()) {
    path = *resolved;
  } else {
    std::fprintf(stderr, "run: %s\n", resolved.status().message().c_str());
    return 1;
  }
  if (!spec.empty() && spec != "ngram") {
    if (auto draft = ensure_model(spec, true); draft.ok()) {
      spec = *draft;
    } else {
      std::fprintf(stderr, "run: draft model: %s\n", draft.status().message().c_str());
      return 1;
    }
  }
  if (Status st = params.sampling.validate(); !st.ok()) {
    std::fprintf(stderr, "run: %s\n", st.to_string().c_str());
    return 1;
  }

  if (!spec.empty()) {
    params.max_tokens = max_tokens;
    return run_speculative(path, spec, spec_k, !spec_fixed, prompt, system, chat, stream, threads, ctx, batch, opts, params);
  }

  const Stopwatch load_timer;
  opts.model_path = path;
  opts.threads = threads;
  opts.kv_tokens = ctx;
  opts.max_batch_tokens = batch;
  auto eng = Engine::create(opts);
  if (!eng.ok()) {
    std::fprintf(stderr, "run: %s\n", eng.status().to_string().c_str());
    return 1;
  }
  Engine& e = **eng;
  const LoadedModel& lm = e.model();
  const ModelConfig& cfg = lm.config;
  LOG_INFO("Model: {} ({}, {})", cfg.name.empty() ? cfg.architecture : cfg.name, lm.architecture->name(),
           lm.quantization);
  LOG_INFO("Device: {}, threads: {}", e.backend_name(), e.threads());
  LOG_INFO("Weights: {:.1f} MiB mmapped, KV cache: {:.1f} MiB ({} tokens, {})", lm.weight_bytes / kMiB,
           e.kv_geometry().total_bytes() / kMiB, ctx, dtype_name(opts.kv_dtype));
  LOG_INFO("Load time: {:.1f} ms", load_timer.elapsed_ms());

  params.max_tokens = max_tokens;
  if (interactive) {
    ChatSettings cs;
    cs.system = system;
    cs.params = params;
    cs.context = ctx;
    return run_chat_session(e, std::move(cs));
  }
  const Stopwatch request_timer;
  Result<std::shared_ptr<RequestStream>> s = InvalidArgument("unreachable");
  if (chat) {
    std::vector<ChatMessage> msgs;
    if (!system.empty()) msgs.push_back({"system", system});
    msgs.push_back({"user", prompt});
    s = e.generate_chat(msgs, params);
  } else {
    s = e.generate_text(prompt, /*parse_special=*/false, params);
  }
  if (!s.ok()) {
    std::fprintf(stderr, "run: %s\n", s.status().to_string().c_str());
    return 1;
  }

  std::string text;
  StreamEvent ev;
  double ttft_ms = -1, last_ms = 0;
  std::vector<double> gaps;
  while ((*s)->next(ev)) {
    if (!ev.text.empty()) {
      const double now = request_timer.elapsed_ms();
      if (ttft_ms < 0) {
        ttft_ms = now;
      } else {
        gaps.push_back(now - last_ms);
      }
      last_ms = now;
      if (stream) {
        std::fwrite(ev.text.data(), 1, ev.text.size(), stdout);
        std::fflush(stdout);
      } else {
        text += ev.text;
      }
    }
    if (ev.done) break;
  }
  if (stream) {
    std::printf("\n");
  } else {
    std::printf("%s\n", text.c_str());
  }
  if (ev.finish == StreamFinish::kError) {
    std::fprintf(stderr, "run: %s\n", ev.error.to_string().c_str());
    return 1;
  }

  const double total_ms = request_timer.elapsed_ms();
  std::sort(gaps.begin(), gaps.end());
  auto pct = [&](double p) {
    return gaps.empty() ? 0.0 : gaps[std::min(gaps.size() - 1, static_cast<size_t>(p / 100.0 * gaps.size()))];
  };
  const double decode_s = (total_ms - std::max(ttft_ms, 0.0)) / 1e3;
  std::fprintf(stderr,
               "\n[stats] prompt %d tok | TTFT %.1f ms | generated %d tok, %.1f tok/s | finish %s | "
               "inter-delta p50 %.1f p90 %.1f p99 %.1f ms\n",
               ev.prompt_tokens, ttft_ms, ev.completion_tokens,
               decode_s > 0 ? (ev.completion_tokens - 1) / decode_s : 0.0,
               std::string(stream_finish_name(ev.finish)).c_str(), pct(50), pct(90), pct(99));
  // Engine-side view of the same run: mean forward time of a decode step.
  for (int i = 0; i < 100 && e.stats().scheduler.running != 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if (const SchedulerStats ss = e.stats().scheduler; ss.steps_decode_only > 0) {
    std::fprintf(stderr, "[engine] prefill %.1f ms (forward, %llu steps) | decode step %.1f ms (forward, %llu steps)\n",
                 ss.forward_prefill_only_ms + ss.forward_mixed_ms,
                 static_cast<unsigned long long>(ss.steps_prefill_only + ss.steps_mixed),
                 ss.forward_decode_only_ms / static_cast<double>(ss.steps_decode_only),
                 static_cast<unsigned long long>(ss.steps_decode_only));
  }
  return 0;
}

}  // namespace dynalm::cli
