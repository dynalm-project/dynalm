#include "scheduler/scheduler.h"

#include <algorithm>
#include <string>

#include "logging/log.h"
#include "sampling/sampler.h"

namespace engine {

Scheduler::Scheduler(Transformer& model, KvBlockPool& kv, const Tokenizer& tokenizer, SchedulerConfig config)
    : model_(model), kv_(kv), tokenizer_(tokenizer), config_(config) {
  config_.max_batch_tokens = std::min(config_.max_batch_tokens, model_.max_batch_tokens());
  batch_.reserve(static_cast<size_t>(config_.max_running));
  batch_owner_.reserve(static_cast<size_t>(config_.max_running));
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
    default: ++stats_.failed; break;
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
  for (uint64_t id : cancels) {
    for (Entry& e : running_) {
      if (e.id == id) e.cancel_flag->store(true, std::memory_order_relaxed);
    }
    for (Entry& e : waiting_) {
      if (e.id == id) e.cancel_flag->store(true, std::memory_order_relaxed);
    }
  }
}

void Scheduler::admit() {
  const int32_t bs = kv_.geometry().block_size;
  while (!waiting_.empty() && static_cast<int32_t>(running_.size()) < config_.max_running) {
    Entry& e = waiting_.front();
    // Admission control: the whole pending prefix plus one block of headroom
    // must fit now, so admitted sequences rarely need preemption.
    const int64_t need_blocks = (e.seq->pending() + bs - 1) / bs + 1;
    if (kv_.free_blocks() < need_blocks && !running_.empty()) break;
    e.admit_order = ++admit_counter_;
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
  ++stats_.preemptions;
  LOG_DEBUG("preempted request {} (KV pressure)", victim->id);
  waiting_.push_front(std::move(*victim));
  running_.erase(running_.begin() + (victim - running_.data()));
  return true;
}

bool Scheduler::step() {
  drain_incoming();

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

  admit();
  stats_.running = static_cast<int32_t>(running_.size());
  stats_.waiting = static_cast<int32_t>(waiting_.size());
  if (running_.empty()) return !waiting_.empty() || incoming_count_.load() != 0;

  // Build the batch. Decode rows (1 pending token) first so ongoing
  // generations are never starved by large prompts; then prefill chunks in
  // admission order, each capped by the remaining token budget.
  std::stable_sort(running_.begin(), running_.end(), [](const Entry& a, const Entry& b) {
    const bool ad = a.seq->status() == SequenceStatus::kDecode, bd = b.seq->status() == SequenceStatus::kDecode;
    return ad != bd ? ad : a.admit_order < b.admit_order;
  });
  batch_.clear();
  batch_owner_.clear();
  int32_t budget = config_.max_batch_tokens;
  for (size_t i = 0; i < running_.size() && budget > 0;) {
    Entry& e = running_[i];
    const int32_t n = std::min(e.seq->pending(), budget);
    if (n <= 0) {
      ++i;
      continue;
    }
    Status st = e.seq->reserve_kv(n);
    if (st.code() == StatusCode::kResourceExhausted) {
      // Make room by preempting the newest other sequence and retry. Batch
      // entries already built only reference running_ by index, so rebuild.
      if (preempt_one(&e)) {
        batch_.clear();
        batch_owner_.clear();
        budget = config_.max_batch_tokens;
        i = 0;
        continue;
      }
      // Alone and still out of KV: this request cannot fit at all.
      st = ResourceExhausted("request needs more KV than the cache holds");
    }
    if (!st.ok()) {
      e.seq->fail(st);
      ++i;
      continue;
    }
    const int32_t start = e.seq->num_computed();
    // The last pending token needs logits only if it completes the sequence's
    // pending work (otherwise more chunks follow).
    const bool completes = n == e.seq->pending();
    batch_.push_back({e.seq->tokens().subspan(static_cast<size_t>(start), static_cast<size_t>(n)), start,
                      e.seq->block_table(), completes});
    batch_owner_.push_back(&e);
    budget -= n;
    ++i;
  }

  if (!batch_.empty()) {
    size_t n_logits = 0;
    for (const SeqBatch& b : batch_) n_logits += b.want_logits ? 1 : 0;
    const auto vocab = static_cast<size_t>(model_.config().vocab_size);
    logits_.resize(n_logits * vocab);
    const Status st = model_.forward_batch(batch_, kv_, logits_);
    ++stats_.steps;
    stats_.last_batch_seqs = static_cast<int32_t>(batch_.size());
    stats_.last_batch_rows = config_.max_batch_tokens - budget;
    stats_.tokens_computed += static_cast<uint64_t>(stats_.last_batch_rows);

    size_t li = 0;
    for (size_t b = 0; b < batch_.size(); ++b) {
      Entry& e = *batch_owner_[b];
      if (!st.ok()) {
        e.seq->fail(st);  // a model error fails this batch's sequences only
        continue;
      }
      e.seq->mark_computed(static_cast<int32_t>(batch_[b].tokens.size()));
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
  }
  return true;
}

void Scheduler::run_until_idle() {
  while (step() || !idle()) {
  }
}

}  // namespace engine
