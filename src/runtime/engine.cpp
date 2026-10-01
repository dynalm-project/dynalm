#include "runtime/engine.h"

#include <algorithm>

#include "logging/log.h"
#include "runtime/sequence.h"
#include "runtime/text_stream.h"

namespace engine {

std::string_view stream_finish_name(StreamFinish f) {
  switch (f) {
    case StreamFinish::kNone: return "none";
    case StreamFinish::kStop: return "stop";
    case StreamFinish::kLength: return "length";
    case StreamFinish::kCancelled: return "cancelled";
    case StreamFinish::kError: return "error";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// RequestStream

void RequestStream::push(StreamEvent ev) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (final_pushed_) return;  // nothing after the final event
    final_pushed_ = ev.done;
    events_.push_back(std::move(ev));
  }
  cv_.notify_one();
}

bool RequestStream::next(StreamEvent& out, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mu_);
  if (final_popped_) return false;
  if (!cv_.wait_for(lock, timeout, [&] { return !events_.empty(); })) return false;
  out = std::move(events_.front());
  events_.pop_front();
  if (out.done) final_popped_ = true;
  return true;
}

bool RequestStream::finished() const {
  std::lock_guard<std::mutex> lock(mu_);
  return final_pushed_;
}

void RequestStream::cancel() {
  if (cancel_fn_) cancel_fn_();
}

// ---------------------------------------------------------------------------
// Engine

namespace {

// Per-request streaming state, owned by the scheduler callback.
struct RequestState {
  std::shared_ptr<RequestStream> stream;
  TextStreamer streamer;
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
  bool ended = false;  // final event already pushed (e.g. a stop string matched)

  RequestState(const Tokenizer& tok, std::vector<std::string> stops) : streamer(tok, std::move(stops)) {}
};

StreamFinish map_finish(const RequestEvent& ev) {
  switch (ev.status) {
    case SequenceStatus::kFinished: return ev.reason == FinishReason::kLength ? StreamFinish::kLength : StreamFinish::kStop;
    case SequenceStatus::kCancelled: return StreamFinish::kCancelled;
    default: return StreamFinish::kError;
  }
}

}  // namespace

Result<std::unique_ptr<Engine>> Engine::create(EngineOptions opts) {
  std::unique_ptr<Engine> e(new Engine());
  ENGINE_ASSIGN_OR_RETURN(e->model_, load_model(opts.model_path));
  const ModelConfig& c = e->model_->config;
  const int threads = opts.threads > 0 ? opts.threads : cpu_info().physical_cores;
  e->pool_ = std::make_unique<ThreadPool>(threads);
  e->backend_ = std::make_unique<CpuBackend>(*e->pool_, select_best_isa(cpu_info().features));
  const int64_t kv_tokens = opts.kv_tokens > 0 ? opts.kv_tokens : std::min<int64_t>(c.context_length, 16384);
  ENGINE_ASSIGN_OR_RETURN(e->kv_, KvBlockPool::create(kv_geometry_for(c, opts.kv_dtype, 16, kv_tokens), *e->backend_));
  ENGINE_ASSIGN_OR_RETURN(e->transformer_,
                          Transformer::create(c, e->model_->weights, *e->backend_, opts.max_batch_tokens));
  e->scheduler_ = std::make_unique<Scheduler>(*e->transformer_, *e->kv_, *e->model_->tokenizer, opts.scheduler);
  e->core_ = std::make_shared<Core>();
  e->core_->sched = e->scheduler_.get();
  e->thread_ = std::thread([eng = e.get()] { eng->loop(); });
  return e;
}

Engine::~Engine() {
  {
    std::lock_guard<std::mutex> lock(wake_mu_);
    stop_ = true;
  }
  wake_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  {
    std::lock_guard<std::mutex> lock(core_->mu);
    core_->sched = nullptr;  // streams' cancel() becomes a no-op
  }
  // Requests still in flight end as cancelled; the scheduler's destructor
  // releases their KV.
  for (auto& weak : live_) {
    if (auto st = weak.lock()) {
      StreamEvent ev;
      ev.done = true;
      ev.finish = StreamFinish::kCancelled;
      st->push(std::move(ev));
    }
  }
}

void Engine::loop() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(wake_mu_);
      wake_cv_.wait(lock, [&] { return stop_ || !scheduler_->idle(); });
      if (stop_) return;
    }
    scheduler_->step();
    EngineStats st;
    st.scheduler = scheduler_->stats();
    if (const PrefixCache* pc = scheduler_->prefix_cache()) st.prefix = pc->stats();
    st.kv_blocks_total = kv_->num_blocks();
    st.kv_blocks_used = kv_->used_blocks();
    std::lock_guard<std::mutex> lock(stats_mu_);
    stats_ = st;
  }
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(stats_mu_);
  return stats_;
}

Result<std::shared_ptr<RequestStream>> Engine::submit(std::vector<TokenId> tokens, const GenerateParams& params) {
  if (params.max_tokens <= 0) return InvalidArgument("max_tokens must be > 0");
  auto stream = std::make_shared<RequestStream>();
  auto state = std::make_shared<RequestState>(*model_->tokenizer, params.stop);
  state->stream = stream;
  state->prompt_tokens = static_cast<int32_t>(tokens.size());
  std::weak_ptr<Core> weak_core = core_;

  Request r;
  r.prompt = std::move(tokens);
  r.stop = StopParams{params.max_tokens, params.stop_at_eog};
  r.priority = params.priority;
  r.timeout_ms = params.timeout_ms;
  // Runs on the scheduler thread.
  r.on_event = [state, weak_core](const RequestEvent& ev) {
    if (state->ended) return;
    if (!ev.finished) {
      ++state->completion_tokens;
      TextStreamer::Delta d = state->streamer.push(ev.token);
      if (!d.text.empty() || d.stopped) {
        StreamEvent out;
        out.text = std::move(d.text);
        if (d.stopped) {
          out.done = true;
          out.finish = StreamFinish::kStop;
          out.prompt_tokens = state->prompt_tokens;
          out.completion_tokens = state->completion_tokens;
          state->ended = true;
          if (auto core = weak_core.lock()) {
            std::lock_guard<std::mutex> lock(core->mu);
            if (core->sched) core->sched->cancel(ev.request_id);  // stop generating
          }
        }
        state->stream->push(std::move(out));
      }
      return;
    }
    StreamEvent out;
    out.text = state->streamer.finish();
    out.done = true;
    out.finish = map_finish(ev);
    out.error = ev.error;
    out.prompt_tokens = state->prompt_tokens;
    out.completion_tokens = state->completion_tokens;
    state->ended = true;
    state->stream->push(std::move(out));
  };

  const uint64_t id = scheduler_->submit(std::move(r));
  stream->set_id(id);
  stream->set_cancel_fn([weak_core, id] {
    if (auto core = weak_core.lock()) {
      std::lock_guard<std::mutex> lock(core->mu);
      if (core->sched) core->sched->cancel(id);  // the request is active, so the loop is awake
    }
  });
  {
    std::lock_guard<std::mutex> lock(wake_mu_);
    live_.erase(std::remove_if(live_.begin(), live_.end(), [](const auto& w) { return w.expired(); }), live_.end());
    live_.push_back(stream);
  }
  wake_cv_.notify_one();
  return stream;
}

Result<std::shared_ptr<RequestStream>> Engine::generate_text(std::string_view prompt, bool parse_special,
                                                             const GenerateParams& params) {
  std::vector<TokenId> tokens;
  ENGINE_RETURN_IF_ERROR(model_->tokenizer->encode(prompt, /*add_special=*/true, parse_special, tokens));
  return submit(std::move(tokens), params);
}

Result<std::shared_ptr<RequestStream>> Engine::generate_chat(std::span<const ChatMessage> messages,
                                                             const GenerateParams& params) {
  if (!model_->chat_template) return Unsupported("model has no recognized chat template");
  ENGINE_ASSIGN_OR_RETURN(std::string text, model_->chat_template->apply(messages, /*add_generation_prompt=*/true));
  return generate_text(text, /*parse_special=*/true, params);
}

}  // namespace engine
