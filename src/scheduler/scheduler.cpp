#include "scheduler/scheduler.h"

#include <algorithm>
#include <string>

#include "common/timer.h"
#include "logging/log.h"
#include "sampling/sampler.h"

namespace engine {

Scheduler::Scheduler(Transformer& model, KvBlockPool& kv, const Tokenizer& tokenizer, SchedulerConfig config)
    : model_(model), kv_(kv), tokenizer_(tokenizer), config_(config) {
  // Both budgets together must fit one forward pass.
  const int32_t cap = model_.max_batch_tokens();
  config_.decode_token_budget = std::clamp(config_.decode_token_budget, 1, cap - 1);
  config_.prefill_token_budget = std::clamp(config_.prefill_token_budget, 1, cap - config_.decode_token_budget);
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
  e.arrival_ns = now_ns();
  e.deadline_ns = request.timeout_ms > 0 ? e.arrival_ns + request.timeout_ms * 1000000 : 0;
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
      match = prefix_cache_->lookup(e.seq->tokens(), static_cast<int32_t>(e.seq->tokens().size()) - 1);
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
  for (int attempt = 0;; ++attempt) {
    batch_.clear();
    batch_owner_.clear();
    int32_t decode_rows = 0, prefill_rows = 0;
    bool restart = false;

    order.clear();
    for (Entry& e : running_) order.push_back(&e);
    std::stable_sort(order.begin(), order.end(), [&](const Entry* a, const Entry* b) {
      const bool ad = is_decode_row(*a), bd = is_decode_row(*b);
      if (ad != bd) return ad;
      if (ad) return a->last_step != b->last_step ? a->last_step < b->last_step : a->admit_order < b->admit_order;
      return a->priority != b->priority ? a->priority > b->priority : a->admit_order < b->admit_order;
    });

    for (Entry* e : order) {
      if (is_terminal(e->seq->status()) || e->seq->pending() == 0) continue;
      const bool decode = is_decode_row(*e);
      int32_t& used = decode ? decode_rows : prefill_rows;
      const int32_t budget = decode ? config_.decode_token_budget : config_.prefill_token_budget;
      int32_t n = std::min(e->seq->pending(), budget - used);
      if (!decode && config_.max_prefill_chunk > 0) n = std::min(n, config_.max_prefill_chunk);
      if (n <= 0) continue;
      const Status st = e->seq->reserve_kv(n);
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
    break;
  }

  if (batch_.empty()) return true;
  size_t n_logits = 0;
  for (const SeqBatch& b : batch_) n_logits += b.want_logits ? 1 : 0;
  const auto vocab = static_cast<size_t>(model_.config().vocab_size);
  logits_.resize(n_logits * vocab);
  const Status st = model_.forward_batch(batch_, kv_, logits_);
  ++stats_.steps;
  stats_.last_batch_seqs = static_cast<int32_t>(batch_.size());
  stats_.last_batch_rows = stats_.last_decode_rows + stats_.last_prefill_rows;
  stats_.tokens_computed += static_cast<uint64_t>(stats_.last_batch_rows);

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
    if (prefix_cache_) {
      // Offer newly completed (hence immutable) blocks for reuse.
      const int32_t full = e.seq->num_computed() / kv_.geometry().block_size;
      if (full > e.cached_blocks) {
        prefix_cache_->insert(e.seq->tokens(), e.seq->block_table(), e.cached_blocks, full);
        e.cached_blocks = full;
      }
    }
    if (!batch_[b].want_logits) continue;
    const std::span<const float> row(logits_.data() + li * vocab, vocab);
    ++li;
    if (e.seq->pending() != 0 || e.seq->status() != SequenceStatus::kDecode) continue;
    const TokenId next = sample_greedy(row);
    e.seq->append_token(next, tokenizer_);
    ++stats_.tokens_generated;
    if (e.on_event) {
      RequestEvent ev;
      ev.request_id = e.id;
      ev.token = next;
      ev.status = e.seq->status();
      e.on_event(ev);
    }
  }
  return true;
}

void Scheduler::run_until_idle() {
  while (step() || !idle()) {
  }
}

}  // namespace engine
