#include "scheduler/scheduler.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "dynacore/base/env.h"
#include "dynacore/base/timer.h"
#include "logging/log.h"
#include "sampling/sampler.h"
#include "common/core.h"

namespace dynalm {

std::string_view scheduler_policy_name(SchedulerPolicy p) {
  switch (p) {
    case SchedulerPolicy::kBalanced: return "balanced";
    case SchedulerPolicy::kLatencyFirst: return "latency";
    case SchedulerPolicy::kThroughputFirst: return "throughput";
  }
  return "balanced";
}

std::string_view prefill_order_name(PrefillOrder o) {
  return o == PrefillOrder::kShortestRemaining ? "srpt" : "fifo";
}

bool parse_prefill_order(std::string_view s, PrefillOrder& out) {
  if (s == "fifo") out = PrefillOrder::kFifo;
  else if (s == "srpt") out = PrefillOrder::kShortestRemaining;
  else return false;
  return true;
}

bool parse_scheduler_policy(std::string_view s, SchedulerPolicy& out) {
  if (s == "balanced") out = SchedulerPolicy::kBalanced;
  else if (s == "latency") out = SchedulerPolicy::kLatencyFirst;
  else if (s == "throughput") out = SchedulerPolicy::kThroughputFirst;
  else return false;
  return true;
}

Scheduler::Scheduler(Transformer& model, KvBlockPool& kv, const Tokenizer& tokenizer, SchedulerConfig config)
    : model_(model), kv_(kv), tokenizer_(tokenizer), config_(config) {
  if (config_.policy == SchedulerPolicy::kLatencyFirst) config_.prefill_token_budget = 32;
  if (config_.policy == SchedulerPolicy::kThroughputFirst) config_.prefill_token_budget = 128;
  // Both budgets together must fit one forward pass.
  const int32_t cap = model_.max_batch_tokens();
  config_.decode_token_budget = std::clamp(config_.decode_token_budget, 1, cap - 1);
  config_.prefill_token_budget = std::clamp(config_.prefill_token_budget, 1, cap - config_.decode_token_budget);
  if (config_.policy == SchedulerPolicy::kThroughputFirst) config_.decode_protect = false;
  if (config_.policy == SchedulerPolicy::kLatencyFirst) {
    config_.protected_prefill_min = 16;
    config_.decode_latency_target = 4.0f;
  }
  // Environment overrides; empty or invalid values are ignored (with a warning)
  // instead of being read as 0, which would collapse the mixed-step prefill
  // budget to one row (a target of 0 halves it on every mixed step).
  long lv = 0;
  double dv = 0;
  if (const char* dp = dynacore::env_value("DYNALM_DECODE_PROTECT")) {
    if (dynacore::parse_env_long(dp, lv)) {
      config_.decode_protect = lv != 0;
    } else {
      LOG_WARN("ignoring DYNALM_DECODE_PROTECT='{}' (expected 0 or 1)", dp);
    }
  }
  if (const char* dt = dynacore::env_value("DYNALM_DECODE_TARGET")) {
    if (dynacore::parse_env_double(dt, dv) && dv > 0) {
      config_.decode_latency_target = static_cast<float>(dv);
    } else {
      LOG_WARN("ignoring DYNALM_DECODE_TARGET='{}' (expected a number > 0)", dt);
    }
  }
  if (const char* pm = dynacore::env_value("DYNALM_PROTECT_MIN")) {
    if (dynacore::parse_env_long(pm, lv) && lv >= 1) {
      config_.protected_prefill_min = static_cast<int32_t>(lv);
    } else {
      LOG_WARN("ignoring DYNALM_PROTECT_MIN='{}' (expected an integer >= 1)", pm);
    }
  }
  config_.protected_prefill_min = std::clamp(config_.protected_prefill_min, 1, config_.prefill_token_budget);
  // A step without decode rows carries prefill rows only: the solo chunk may use the whole batch.
  config_.solo_prefill_chunk = std::clamp(config_.solo_prefill_chunk, 0, cap);
  stats_.solo_prefill_chunk = config_.solo_prefill_chunk;
  stats_.solo_prefill_split = config_.solo_prefill_split;
  mixed_prefill_budget_ = config_.prefill_token_budget;
  batch_.reserve(static_cast<size_t>(config_.max_running));
  batch_owner_.reserve(static_cast<size_t>(config_.max_running));
  if (config_.enable_prefix_cache) {
    prefix_cache_ = config_.prefix_cache_kind == PrefixCacheKind::kRadix
                         ? make_radix_prefix_cache(kv_, config_.prefix_cache_max_blocks)
                         : make_hash_prefix_cache(kv_, config_.prefix_cache_max_blocks);
  }
}

Scheduler::~Scheduler() = default;

uint64_t Scheduler::submit(Request request) {
  const uint64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
  Entry e;
  e.id = id;
  e.on_event = std::move(request.on_event);
  e.cancel_flag = std::make_shared<std::atomic<bool>>(false);
  // The KvBlockTable inside SequenceState only records the pool; no blocks
  // are taken until admission.
  e.seq = std::make_unique<SequenceState>(id, request.prompt, request.stop, kv_);
  e.priority = request.priority;
  if (!request.sampling.greedy() || request.sampling.has_penalties()) {
    e.sampler = std::make_unique<Sampler>(request.sampling);
  }
  e.arrival_ns = now_ns();
  e.deadline_ns = request.timeout_ms > 0 ? e.arrival_ns + request.timeout_ms * 1000000 : 0;
  e.trace.id = id;
  e.trace.prompt_tokens = static_cast<int32_t>(request.prompt.size());
  e.trace.submit_ns = e.arrival_ns;
  std::lock_guard<std::mutex> lock(in_mu_);
  incoming_.push_back(std::move(e));
  incoming_count_.fetch_add(1, std::memory_order_release);
  return id;
}

void Scheduler::cancel(uint64_t request_id) {
  std::lock_guard<std::mutex> lock(in_mu_);
  cancel_requests_.push_back(request_id);
  incoming_count_.fetch_add(1, std::memory_order_release);
}

bool Scheduler::idle() const {
  return waiting_.empty() && running_.empty() && incoming_count_.load(std::memory_order_acquire) == 0;
}

void Scheduler::emit_final(Entry& e) {
  RequestEvent ev;
  ev.request_id = e.id;
  ev.finished = true;
  ev.status = e.seq->status();
  ev.reason = e.seq->finish_reason();
  ev.error = e.seq->error();
  if (e.on_event) e.on_event(ev);
}

void Scheduler::retire(Entry& e) {
  switch (e.seq->status()) {
    case SequenceStatus::kFinished: ++stats_.completed; break;
    case SequenceStatus::kCancelled: ++stats_.cancelled; break;
    default:
      ++stats_.failed;
      if (e.seq->error().code() == StatusCode::kDeadlineExceeded) ++stats_.timed_out;
      break;
  }
  if (config_.trace) {
    e.trace.finish_ns = now_ns();
    e.trace.status = e.seq->status();
    e.trace.generated = e.seq->generated();
    e.trace.preemptions = e.seq->preemptions();
    std::lock_guard<std::mutex> lock(trace_mu_);
    request_trace_.push_back(e.trace);
  }
  emit_final(e);
}

void Scheduler::drain_incoming() {
  if (incoming_count_.load(std::memory_order_acquire) == 0) return;
  std::vector<Entry> fresh;
  std::vector<uint64_t> cancels;
  {
    std::lock_guard<std::mutex> lock(in_mu_);
    fresh.swap(incoming_);
    cancels.swap(cancel_requests_);
    incoming_count_.store(0, std::memory_order_release);
  }
  const ModelConfig& c = model_.config();
  for (Entry& e : fresh) {
    // Validate up front so one bad request never reaches the model.
    const SequenceState& s = *e.seq;
    Status bad;
    if (s.prompt_len() == 0) {
      bad = InvalidArgument("empty prompt");
    } else if (e.sampler && !e.sampler->params().validate().ok()) {
      bad = e.sampler->params().validate();
    } else if (s.prompt_len() + static_cast<int64_t>(s.stop().max_new_tokens) > c.context_length) {
      bad = InvalidArgument("prompt + max_new_tokens exceeds context length " + std::to_string(c.context_length));
    } else if (s.prompt_len() + static_cast<int64_t>(s.stop().max_new_tokens) >
               static_cast<int64_t>(kv_.num_blocks()) * kv_.geometry().block_size) {
      // Could never fit even alone; reject now instead of failing mid-generation.
      bad = InvalidArgument("prompt + max_new_tokens exceeds the KV cache capacity of " +
                            std::to_string(static_cast<int64_t>(kv_.num_blocks()) * kv_.geometry().block_size) +
                            " tokens");
    } else if (s.stop().max_new_tokens <= 0) {
      bad = InvalidArgument("max_new_tokens must be > 0");
    } else {
      for (TokenId t : s.tokens()) {
        if (t < 0 || t >= c.vocab_size) {
          bad = InvalidArgument("token id " + std::to_string(t) + " out of range");
          break;
        }
      }
    }
    if (!bad.ok()) {
      e.seq->fail(bad);
      retire(e);
      continue;
    }
    waiting_.push_back(std::move(e));
  }
  // Priority first, then arrival (FCFS). Preempted sequences keep their
  // original arrival time, so they return ahead of later arrivals.
  std::stable_sort(waiting_.begin(), waiting_.end(), [](const Entry& x, const Entry& y) {
    return x.priority != y.priority ? x.priority > y.priority : x.arrival_ns < y.arrival_ns;
  });
  for (uint64_t id : cancels) {
    for (Entry& e : running_) {
      if (e.id == id) e.cancel_flag->store(true, std::memory_order_relaxed);
    }
    for (Entry& e : waiting_) {
      if (e.id == id) e.cancel_flag->store(true, std::memory_order_relaxed);
    }
  }
}

void Scheduler::expire_deadlines(int64_t now) {
  auto expired = [now](const Entry& e) { return e.deadline_ns != 0 && now > e.deadline_ns; };
  for (Entry& e : running_) {
    if (!is_terminal(e.seq->status()) && expired(e)) {
      e.seq->fail(Status(StatusCode::kDeadlineExceeded, "request timed out"));
    }
  }
  for (auto it = waiting_.begin(); it != waiting_.end();) {
    if (expired(*it)) {
      it->seq->fail(Status(StatusCode::kDeadlineExceeded, "request timed out while queued"));
      retire(*it);
      it = waiting_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scheduler::admit(int64_t now) {
  const int32_t bs = kv_.geometry().block_size;
  while (!waiting_.empty() && static_cast<int32_t>(running_.size()) < config_.max_running) {
    Entry& e = waiting_.front();
    // Reuse a cached prefix (always leaving >= 1 token to compute: the last
    // prompt token's logits are needed).
    PrefixCache::Match match;
    if (prefix_cache_ && e.seq->num_computed() == 0) {
      const int64_t t0 = now_ns();
      match = prefix_cache_->lookup(e.seq->tokens(), static_cast<int32_t>(e.seq->tokens().size()) - 1);
      stats_.prefix_lookup_ms += static_cast<double>(now_ns() - t0) * 1e-6;
    }
    // Admission control: the remaining pending tokens plus one block of
    // headroom must fit now, so admitted sequences rarely need preemption.
    // Cached blocks can be evicted to make room.
    const int64_t pending_after = e.seq->pending() - match.tokens;
    const int64_t need_blocks = (pending_after + bs - 1) / bs + 1;
    if (kv_.free_blocks() < need_blocks && prefix_cache_) {
      prefix_cache_->evict(static_cast<int32_t>(need_blocks - kv_.free_blocks()));
    }
    if (kv_.free_blocks() < need_blocks && !running_.empty()) {
      for (int32_t b : match.blocks) kv_.release(b);  // give the references back
      break;
    }
    if (match.tokens > 0) {
      e.seq->adopt_prefix(match.blocks, match.tokens);
      // Only whole cached blocks count; a partial (radix) private copy is
      // offered to the cache once it fills up.
      e.cached_blocks = match.tokens / bs;
    }
    e.admit_order = ++admit_counter_;
    e.admit_ns = now;
    if (e.trace.admit_ns == 0) e.trace.admit_ns = now;
    if (e.seq->preemptions() == 0) {
      const double q = static_cast<double>(now - e.arrival_ns) * 1e-6;
      stats_.queue_ms_total += q;
      stats_.queue_ms_max = std::max(stats_.queue_ms_max, q);
      ++stats_.admitted;
    }
    running_.push_back(std::move(e));
    waiting_.pop_front();
  }
}

// Frees KV by sending the most recently admitted running sequence (other than
// `keep`) back to the waiting queue for recomputation.
bool Scheduler::preempt_one(const Entry* keep) {
  Entry* victim = nullptr;
  for (Entry& e : running_) {
    if (&e == keep || e.seq->block_table().empty()) continue;  // nothing to free
    if (!victim || e.admit_order > victim->admit_order) victim = &e;
  }
  if (!victim) return false;
  victim->seq->reset_for_recompute();
  victim->cached_blocks = 0;
  ++stats_.preemptions;
  LOG_DEBUG("preempted request {} (KV pressure)", victim->id);
  waiting_.push_front(std::move(*victim));
  running_.erase(running_.begin() + (victim - running_.data()));
  return true;
}

bool Scheduler::step() {
  const int64_t t_plan = now_ns();
  drain_incoming();
  const int64_t now = now_ns();
  expire_deadlines(now);

  // Apply cancellations and drop terminal sequences.
  for (auto it = running_.begin(); it != running_.end();) {
    if (it->cancel_flag->load(std::memory_order_relaxed)) it->seq->cancel();
    if (is_terminal(it->seq->status())) {
      retire(*it);
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = waiting_.begin(); it != waiting_.end();) {
    if (it->cancel_flag->load(std::memory_order_relaxed)) {
      it->seq->cancel();
      retire(*it);
      it = waiting_.erase(it);
    } else {
      ++it;
    }
  }

  admit(now);
  stats_.running = static_cast<int32_t>(running_.size());
  stats_.waiting = static_cast<int32_t>(waiting_.size());
  if (running_.empty()) return !waiting_.empty() || incoming_count_.load() != 0;

  // Plan the step: decode rows first (one per generating sequence, least
  // recently served first so a small decode budget still rotates fairly),
  // then prefill chunks by priority and admission order, each budget capped
  // separately. KV is reserved as the plan is built; if the pool runs dry,
  // the newest other sequence is preempted and planning restarts.
  const uint64_t step_no = stats_.steps + 1;
  auto is_decode_row = [](const Entry& e) {
    return e.seq->status() == SequenceStatus::kDecode && e.seq->pending() == 1;
  };
  std::vector<Entry*> order;
  bool solo = false, solo_skipped = false, srpt = false;
  int32_t traced_budget = 0, traced_chunk = 0, traced_prefilling = 0;
  for (int attempt = 0;; ++attempt) {
    batch_.clear();
    batch_owner_.clear();
    int32_t decode_rows = 0, prefill_rows = 0;
    bool restart = false;

    order.clear();
    for (Entry& e : running_) order.push_back(&e);
    // Decode-protected prefill (DD-079): smaller prefill share while streams decode.
    bool any_decode = false;
    for (const Entry* e : order) any_decode = any_decode || is_decode_row(*e);
    // Solo prefill (experimental, off when solo_prefill_chunk == 0): a step with
    // no decode rows takes larger chunks. FIFO (the default mode) only for a
    // lone prompt: with two or more sequences prefilling, the oldest would take
    // the whole solo budget and a short prompt would wait behind every large
    // chunk of a long one (Phase 9 V3), so such steps keep the regular chunk and
    // budget. Split mode shares the solo budget instead. Mixed steps never see
    // the solo budget; mixed_prefill_budget_ stays bounded by
    // prefill_token_budget (DD-079).
    int32_t prefilling = 0;
    for (const Entry* e : order) {
      prefilling += !is_terminal(e->seq->status()) && e->seq->pending() > 0 && !is_decode_row(*e) ? 1 : 0;
    }
    // Experimental prefill order (default FIFO); only matters with >= 2 prompts prefilling.
    srpt = config_.prefill_order == PrefillOrder::kShortestRemaining && prefilling > 1;
    const int64_t aging_ns = static_cast<int64_t>(config_.prefill_aging_ms) * 1000000;
    solo = config_.solo_prefill_chunk > 0 && !any_decode && (prefilling == 1 || config_.solo_prefill_split);
    solo_skipped = config_.solo_prefill_chunk > 0 && !any_decode && !solo;
    const int32_t prefill_budget = solo ? std::max(config_.prefill_token_budget, config_.solo_prefill_chunk)
                                   : config_.decode_protect && any_decode ? mixed_prefill_budget_
                                                                          : config_.prefill_token_budget;
    int32_t prefill_chunk = config_.max_prefill_chunk;  // 0 = up to the budget
    if (solo) {
      prefill_chunk = config_.solo_prefill_chunk;
      if (config_.solo_prefill_split && prefilling > 1) {
        prefill_chunk = std::max({config_.max_prefill_chunk, config_.solo_prefill_chunk / prefilling, 1});
      }
    }
    traced_budget = prefill_budget;
    traced_chunk = prefill_chunk;
    traced_prefilling = prefilling;
    std::stable_sort(order.begin(), order.end(), [&](const Entry* a, const Entry* b) {
      const bool ad = is_decode_row(*a), bd = is_decode_row(*b);
      if (ad != bd) return ad;
      if (ad) return a->last_step != b->last_step ? a->last_step < b->last_step : a->admit_order < b->admit_order;
      if (a->priority != b->priority) return a->priority > b->priority;
      if (srpt) {
        // Prompts waiting longer than prefill_aging_ms first (FIFO among them),
        // then the fewest prompt tokens left.
        const bool sa = now - a->admit_ns >= aging_ns, sb = now - b->admit_ns >= aging_ns;
        if (sa != sb) return sa;
        if (!sa && a->seq->pending() != b->seq->pending()) return a->seq->pending() < b->seq->pending();
      }
      return a->admit_order < b->admit_order;
    });

    for (Entry* e : order) {
      if (is_terminal(e->seq->status()) || e->seq->pending() == 0) continue;
      const bool decode = is_decode_row(*e);
      int32_t& used = decode ? decode_rows : prefill_rows;
      const int32_t budget = decode ? config_.decode_token_budget : prefill_budget;
      int32_t n = std::min(e->seq->pending(), budget - used);
      if (!decode && prefill_chunk > 0) n = std::min(n, prefill_chunk);
      if (n <= 0) continue;
      const int64_t t_kv = now_ns();
      const Status st = e->seq->reserve_kv(n);
      stats_.kv_reserve_ms += static_cast<double>(now_ns() - t_kv) * 1e-6;
      if (st.code() == StatusCode::kResourceExhausted) {
        // Prefer dropping cached (unused) prefix blocks over preempting work.
        if (attempt < 64 && prefix_cache_ && prefix_cache_->evict(4) > 0) {
          restart = true;
          break;
        }
        if (attempt < 64 && preempt_one(e)) {
          restart = true;  // running_ changed; rebuild the plan
          break;
        }
        e->seq->fail(ResourceExhausted("request needs more KV than the cache holds"));
        continue;
      }
      if (!st.ok()) {
        e->seq->fail(st);
        continue;
      }
      const int32_t start = e->seq->num_computed();
      // Logits only when this chunk completes the sequence's pending work.
      batch_.push_back({e->seq->tokens().subspan(static_cast<size_t>(start), static_cast<size_t>(n)), start,
                        e->seq->block_table(), n == e->seq->pending()});
      batch_owner_.push_back(e);
      used += n;
    }
    if (restart) continue;
    stats_.last_decode_rows = decode_rows;
    stats_.last_prefill_rows = prefill_rows;
    if (solo && prefill_rows > 0) ++stats_.steps_solo_prefill;
    if (solo_skipped && prefill_rows > 0) ++stats_.steps_solo_prefill_skipped;
    if (srpt && prefill_rows > 0) ++stats_.steps_prefill_srpt;
    break;
  }

  // Trace (statistics only): what each sequence computed, and who waited.
  StepTrace cur;
  if (config_.trace && !batch_.empty()) {
    cur.step = step_no;
    cur.plan_start_ns = t_plan;
    cur.prefill_rows = stats_.last_prefill_rows;
    cur.decode_rows = stats_.last_decode_rows;
    cur.prefill_budget = traced_budget;
    cur.prefill_chunk = traced_chunk;
    cur.prefilling = traced_prefilling;
    cur.waiting = static_cast<int32_t>(waiting_.size());
    cur.solo = solo;
    cur.solo_skipped = solo_skipped;
    cur.srpt = srpt;
    const int64_t planned = now_ns();  // after admission and planning
    for (size_t b = 0; b < batch_.size(); ++b) {
      Entry& e = *batch_owner_[b];
      const bool dec = is_decode_row(e);
      cur.rows.push_back({e.id, static_cast<int32_t>(batch_[b].tokens.size()), batch_[b].start_pos, dec});
      if (!dec) {
        if (e.trace.first_scheduled_ns == 0) e.trace.first_scheduled_ns = planned;
        ++e.trace.prefill_steps;
      }
    }
    for (Entry& e : running_) {
      if (is_terminal(e.seq->status()) || e.seq->pending() == 0 || is_decode_row(e)) continue;
      if (std::find(batch_owner_.begin(), batch_owner_.end(), &e) != batch_owner_.end()) continue;
      cur.prefill_unserved.push_back(e.id);
      ++e.trace.prefill_unserved_steps;
    }
  }

  if (batch_.empty()) {
    stats_.plan_ms += static_cast<double>(now_ns() - t_plan) * 1e-6;
    return true;
  }
  size_t n_logits = 0;
  for (const SeqBatch& b : batch_) n_logits += b.want_logits ? 1 : 0;
  const auto vocab = static_cast<size_t>(model_.config().vocab_size);
  logits_.resize(n_logits * vocab);
  // How this batch runs on this hardware (DD-051); part of planning time.
  const ExecutionPlan plan = model_.planner().plan(batch_);
  if (plan.kernels.attention_full == AttentionStrategy::kSplitK) ++stats_.steps_split_attention;
  const int64_t t_fwd = now_ns();
  stats_.plan_ms += static_cast<double>(t_fwd - t_plan) * 1e-6;
  const Status st = model_.forward_batch(batch_, kv_, logits_, &plan);
  const int64_t t_post = now_ns();
  const double fwd_ms = static_cast<double>(t_post - t_fwd) * 1e-6;
  stats_.forward_ms += fwd_ms;
  if (solo && stats_.last_prefill_rows > 0) {  // statistics only: forward time of solo-prefill steps
    stats_.forward_solo_prefill_ms += fwd_ms;
    stats_.solo_prefill_step_ms_max = std::max(stats_.solo_prefill_step_ms_max, fwd_ms);
  }
  if (stats_.last_prefill_rows == 0) {
    ++stats_.steps_decode_only;
    stats_.forward_decode_only_ms += fwd_ms;
    decode_step_ema_ms_ = decode_step_ema_ms_ == 0 ? fwd_ms : 0.8 * decode_step_ema_ms_ + 0.2 * fwd_ms;
  } else if (stats_.last_decode_rows == 0) {
    ++stats_.steps_prefill_only;
    stats_.forward_prefill_only_ms += fwd_ms;
  } else {
    ++stats_.steps_mixed;
    stats_.forward_mixed_ms += fwd_ms;
  }
  if (config_.decode_protect && stats_.last_prefill_rows > 0 && stats_.last_decode_rows > 0 && decode_step_ema_ms_ > 0) {
    // Mixed step (DD-079): steer the next mixed step's prefill share toward
    // decode_latency_target x the decode-only step time.
    const double target = config_.decode_latency_target * decode_step_ema_ms_;
    if (fwd_ms > target) {
      mixed_prefill_budget_ = std::max(config_.protected_prefill_min, mixed_prefill_budget_ / 2);
    } else if (fwd_ms < 0.5 * target) {
      mixed_prefill_budget_ = std::min(config_.prefill_token_budget, mixed_prefill_budget_ * 2);
    }
  }
  stats_.mixed_prefill_budget = mixed_prefill_budget_;
  stats_.decode_rows_total += static_cast<uint64_t>(stats_.last_decode_rows);
  stats_.prefill_rows_total += static_cast<uint64_t>(stats_.last_prefill_rows);
  ++stats_.steps;
  stats_.last_batch_seqs = static_cast<int32_t>(batch_.size());
  stats_.last_batch_rows = stats_.last_decode_rows + stats_.last_prefill_rows;
  stats_.tokens_computed += static_cast<uint64_t>(stats_.last_batch_rows);

  int64_t insert_ns = 0, emit_ns = 0;
  size_t li = 0;
  for (size_t b = 0; b < batch_.size(); ++b) {
    Entry& e = *batch_owner_[b];
    if (!st.ok()) {
      e.seq->fail(st);  // a model error fails this batch's sequences only
      continue;
    }
    const bool was_decode = is_decode_row(e);
    e.seq->mark_computed(static_cast<int32_t>(batch_[b].tokens.size()));
    if (was_decode) e.last_step = step_no;
    if (config_.trace && !was_decode) e.trace.prefill_forward_ms += fwd_ms;
    if (prefix_cache_) {
      // Offer newly completed (hence immutable) blocks for reuse.
      const int32_t full = e.seq->num_computed() / kv_.geometry().block_size;
      if (full > e.cached_blocks) {
        const int64_t t0 = now_ns();
        prefix_cache_->insert(e.seq->tokens(), e.seq->block_table(), e.cached_blocks, full);
        insert_ns += now_ns() - t0;
        e.cached_blocks = full;
      }
    }
    if (!batch_[b].want_logits) continue;
    const std::span<float> row(logits_.data() + li * vocab, vocab);
    ++li;
    if (e.seq->pending() != 0 || e.seq->status() != SequenceStatus::kDecode) continue;
    const TokenId next = e.sampler ? e.sampler->sample(row, e.seq->tokens()) : sample_greedy(row);
    e.seq->append_token(next, tokenizer_);
    if (config_.trace && e.trace.first_token_ns == 0) e.trace.first_token_ns = now_ns();
    ++stats_.tokens_generated;
    if (e.on_event) {
      const int64_t t0 = now_ns();
      RequestEvent ev;
      ev.request_id = e.id;
      ev.token = next;
      ev.status = e.seq->status();
      e.on_event(ev);
      emit_ns += now_ns() - t0;
    }
  }
  // Post-forward time = sampling + stop checks + cache inserts + callbacks.
  const int64_t post_ns = now_ns() - t_post;
  stats_.prefix_insert_ms += static_cast<double>(insert_ns) * 1e-6;
  stats_.emit_ms += static_cast<double>(emit_ns) * 1e-6;
  stats_.sample_ms += static_cast<double>(post_ns - insert_ns - emit_ns) * 1e-6;
  if (config_.trace) {
    cur.forward_start_ns = t_fwd;
    cur.forward_end_ns = t_post;
    std::lock_guard<std::mutex> lock(trace_mu_);
    step_trace_.push_back(std::move(cur));
  }
  return true;
}

std::vector<StepTrace> Scheduler::take_step_trace() {
  std::lock_guard<std::mutex> lock(trace_mu_);
  std::vector<StepTrace> out;
  out.swap(step_trace_);
  return out;
}

std::vector<RequestTrace> Scheduler::take_request_trace() {
  std::lock_guard<std::mutex> lock(trace_mu_);
  std::vector<RequestTrace> out;
  out.swap(request_trace_);
  return out;
}

void Scheduler::run_until_idle() {
  while (step() || !idle()) {
  }
}

}  // namespace dynalm
