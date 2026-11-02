#include "runtime/eagle3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "api/json.h"
#include "dynacore/quantization/dequant.h"
#include "loader/safetensors/safetensors.h"

namespace dynalm {
namespace {

namespace fs = std::filesystem;

TensorView rows_of(const Tensor& t, int64_t rows) {
  TensorLayout l = t.view().layout();
  l.shape = TensorShape{rows, t.shape()[1]};
  return TensorView(t.view().data(), l, t.view().device());
}

TensorView cols_of(const TensorView& v, int64_t c0, int64_t n) {
  TensorLayout l = v.layout();
  l.shape = TensorShape{v.dim(0), n};
  return TensorView(static_cast<float*>(v.data()) + c0, l, v.device());
}

}  // namespace

Result<Tensor> Eagle3Drafter::scratch(int64_t rows, int64_t cols) {
  ENGINE_ASSIGN_OR_RETURN(TensorLayout layout, TensorLayout::contiguous(DType::kF32, {rows, cols}));
  ENGINE_ASSIGN_OR_RETURN(auto storage, device_.allocate(static_cast<size_t>(layout.span_bytes())));
  ENGINE_ASSIGN_OR_RETURN(Tensor t, Tensor::from_storage(storage, TensorView(storage->data(), layout, device_.device())));
  device_.fill(t.view(), 0.0f);
  return t;
}

Result<TensorView> Eagle3Drafter::weight(const Tensor& host) {
  if (device_.host_accessible()) {
    owned_.push_back(host);
    return host.view();
  }
  ENGINE_ASSIGN_OR_RETURN(Tensor dev, device_.upload(host));
  owned_.push_back(dev);
  return dev.view();
}

Result<std::unique_ptr<Eagle3Drafter>> Eagle3Drafter::create(const std::string& path, const Transformer& target,
                                                             Device& device, int64_t kv_tokens, int32_t max_rows) {
  if (device.device().type != dynacore::DeviceType::kCpu) {
    return Unsupported("--spec-eagle: EAGLE-3 heads run on the CPU in this release");
  }
  const fs::path p(path);
  const fs::path st = p.extension() == ".safetensors" ? p : p / "model.safetensors";
  const fs::path cfg_path = st.parent_path() / "config.json";
  std::ifstream cf(cfg_path, std::ios::binary);
  if (!cf) return NotFound("--spec-eagle: no config.json in " + st.parent_path().string());
  const std::string text((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
  ENGINE_ASSIGN_OR_RETURN(json::Value cfg, json::parse(text));
  auto num = [&](const char* k, double def) {
    const json::Value* v = cfg.find(k);
    return v != nullptr && v->is_number() ? v->as_number() : def;
  };

  std::unique_ptr<Eagle3Drafter> d(new Eagle3Drafter(device));
  const ModelConfig& tc = target.config();
  d->H_ = static_cast<int64_t>(num("hidden_size", 0));
  d->heads_ = static_cast<int64_t>(num("num_attention_heads", 0));
  d->kv_heads_ = static_cast<int64_t>(num("num_key_value_heads", static_cast<double>(d->heads_)));
  d->head_dim_ = static_cast<int64_t>(num("head_dim", d->heads_ > 0 ? static_cast<double>(d->H_ / d->heads_) : 0));
  d->ff_ = static_cast<int64_t>(num("intermediate_size", 0));
  d->vocab_ = static_cast<int64_t>(num("vocab_size", 0));
  d->draft_vocab_ = static_cast<int64_t>(num("draft_vocab_size", static_cast<double>(d->vocab_)));
  d->eps_ = static_cast<float>(num("rms_norm_eps", 1e-6));
  if (const json::Value* layers = cfg.find("num_hidden_layers"); layers != nullptr && layers->is_number() && layers->as_number() != 1) {
    return Unsupported("--spec-eagle: heads with one layer only");
  }
  if (d->H_ != tc.hidden_size) {
    return InvalidArgument("--spec-eagle: the head's hidden size " + std::to_string(d->H_) + " is not the model's (" +
                           std::to_string(tc.hidden_size) + "): a head trained for another model");
  }
  if (d->vocab_ != tc.vocab_size) {
    return InvalidArgument("--spec-eagle: the head's vocabulary (" + std::to_string(d->vocab_) + ") is not the model's (" +
                           std::to_string(tc.vocab_size) + ")");
  }
  if (d->heads_ <= 0 || d->kv_heads_ <= 0 || d->head_dim_ <= 0 || d->ff_ <= 0 || d->draft_vocab_ <= 0) {
    return InvalidArgument("--spec-eagle: incomplete config.json");
  }
  d->rope_.style = RopeStyle::kHalfSplit;  // Llama layer (rotate_half)
  d->rope_.dim = static_cast<int32_t>(d->head_dim_);
  d->rope_.freq_base = static_cast<float>(num("rope_theta", 10000.0));
  const int32_t L = tc.num_layers;
  if (L < 5) return Unsupported("--spec-eagle: the model has too few layers for an EAGLE-3 head");
  d->capture_ = {2, L / 2, L - 3};
  d->max_rows_ = max_rows;
  d->embed_ = target.token_embedding();

  ENGINE_ASSIGN_OR_RETURN(auto file, safetensors::SafeTensorsFile::open(st.string()));
  auto matrix = [&](const char* name, int64_t rows, int64_t cols, TensorView& out) -> Status {
    const safetensors::TensorInfo* info = file->find(name);
    if (info == nullptr) return InvalidArgument(std::string("--spec-eagle: missing tensor ") + name);
    if (info->shape.rank() != 2 || info->shape[0] != rows || info->shape[1] != cols) {
      return InvalidArgument(std::string("--spec-eagle: unexpected shape of ") + name);
    }
    ENGINE_ASSIGN_OR_RETURN(Tensor t, file->load_tensor(*info));
    ENGINE_ASSIGN_OR_RETURN(out, d->weight(t));
    return Status::Ok();
  };
  auto vector = [&](const char* name, TensorView& out) -> Status {
    const safetensors::TensorInfo* info = file->find(name);
    if (info == nullptr || info->shape.numel() != d->H_) return InvalidArgument(std::string("--spec-eagle: missing ") + name);
    ENGINE_ASSIGN_OR_RETURN(Tensor t, file->load_tensor(*info));
    ENGINE_ASSIGN_OR_RETURN(Tensor f, Tensor::empty(DType::kF32, {d->H_}));
    if (!dequantize_row(t.dtype(), t.data(), f.data_as<float>(), d->H_)) return Unsupported(std::string("--spec-eagle: dtype of ") + name);
    ENGINE_ASSIGN_OR_RETURN(out, d->weight(f));
    return Status::Ok();
  };
  const int64_t H = d->H_, qd = d->heads_ * d->head_dim_, kd = d->kv_heads_ * d->head_dim_;
  ENGINE_RETURN_IF_ERROR(matrix("fc.weight", H, 3 * H, d->fc_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.self_attn.q_proj.weight", qd, 2 * H, d->wq_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.self_attn.k_proj.weight", kd, 2 * H, d->wk_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.self_attn.v_proj.weight", kd, 2 * H, d->wv_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.self_attn.o_proj.weight", H, qd, d->wo_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.mlp.gate_proj.weight", d->ff_, H, d->w_gate_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.mlp.up_proj.weight", d->ff_, H, d->w_up_));
  ENGINE_RETURN_IF_ERROR(matrix("midlayer.mlp.down_proj.weight", H, d->ff_, d->w_down_));
  ENGINE_RETURN_IF_ERROR(matrix("lm_head.weight", d->draft_vocab_, H, d->lm_head_));
  ENGINE_RETURN_IF_ERROR(vector("midlayer.hidden_norm.weight", d->norm_h_));
  ENGINE_RETURN_IF_ERROR(vector("midlayer.input_layernorm.weight", d->norm_in_));
  ENGINE_RETURN_IF_ERROR(vector("midlayer.post_attention_layernorm.weight", d->norm_post_));
  ENGINE_RETURN_IF_ERROR(vector("norm.weight", d->norm_));
  // d2t: draft id -> offset to the target id (int64). Absent: the same ids.
  d->d2t_.assign(static_cast<size_t>(d->draft_vocab_), 0);
  if (const safetensors::TensorInfo* info = file->find("d2t")) {
    if (info->dtype_name != "I64" || info->shape.numel() != d->draft_vocab_) {
      return InvalidArgument("--spec-eagle: d2t must be I64 [draft_vocab_size]");
    }
    std::memcpy(d->d2t_.data(), file->file().data() + info->offset, static_cast<size_t>(d->draft_vocab_) * 8);
  } else if (d->draft_vocab_ != d->vocab_) {
    return InvalidArgument("--spec-eagle: a reduced draft vocabulary needs d2t");
  }
  for (int64_t i = 0; i < d->draft_vocab_; ++i) {
    const int64_t t = i + d->d2t_[static_cast<size_t>(i)];
    if (t < 0 || t >= d->vocab_) return InvalidArgument("--spec-eagle: d2t maps outside the vocabulary");
  }

  KvGeometry g;
  g.num_layers = 1;
  g.num_kv_heads = static_cast<int32_t>(d->kv_heads_);
  g.head_dim = static_cast<int32_t>(d->head_dim_);
  g.head_dim_v = static_cast<int32_t>(d->head_dim_);
  g.block_size = 16;
  g.num_blocks = static_cast<int32_t>((kv_tokens + 15) / 16);
  g.dtype = DType::kF16;
  g.layout = device.preferred_kv_layout();
  ENGINE_ASSIGN_OR_RETURN(d->pool_, KvBlockPool::create(g, device));

  const int64_t R = max_rows;
  ENGINE_ASSIGN_OR_RETURN(d->f_, d->scratch(R, 3 * H));
  ENGINE_ASSIGN_OR_RETURN(d->h_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->e_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->cat_, d->scratch(R, 2 * H));
  ENGINE_ASSIGN_OR_RETURN(d->q_, d->scratch(R, qd));
  ENGINE_ASSIGN_OR_RETURN(d->k_, d->scratch(R, kd));
  ENGINE_ASSIGN_OR_RETURN(d->v_, d->scratch(R, kd));
  ENGINE_ASSIGN_OR_RETURN(d->a_, d->scratch(R, qd));
  ENGINE_ASSIGN_OR_RETURN(d->o_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->n2_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->g_, d->scratch(R, d->ff_));
  ENGINE_ASSIGN_OR_RETURN(d->u_, d->scratch(R, d->ff_));
  ENGINE_ASSIGN_OR_RETURN(d->act_, d->scratch(R, d->ff_));
  ENGINE_ASSIGN_OR_RETURN(d->d_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->hn_, d->scratch(R, H));
  ENGINE_ASSIGN_OR_RETURN(d->lg_, d->scratch(R, d->draft_vocab_));
  return d;
}

Eagle3Drafter::Stream& Eagle3Drafter::stream(uint64_t id) {
  auto it = streams_.find(id);
  if (it == streams_.end()) it = streams_.emplace(id, std::make_unique<Stream>(*pool_)).first;
  return *it->second;
}

void Eagle3Drafter::release(uint64_t id) { streams_.erase(id); }

Status Eagle3Drafter::run(std::span<const Row> rows, bool fc, std::span<const int32_t> logit_rows) {
  const auto m = static_cast<int64_t>(rows.size());
  const int64_t H = H_;
  ids_.clear();
  pos_.clear();
  row_seq_.clear();
  for (const Row& r : rows) {
    ids_.push_back(r.token);
    pos_.push_back(r.index);
    row_seq_.push_back(r.stream);
  }
  kv_views_.clear();
  for (Stream* s : pass_streams_) kv_views_.push_back(pool_->layer_view(0, s->kv.block_table()));

  const TensorView h = rows_of(h_, m), e = rows_of(e_, m), cat = rows_of(cat_, m);
  const TensorView q = rows_of(q_, m), k = rows_of(k_, m), v = rows_of(v_, m), a = rows_of(a_, m);
  const TensorView o = rows_of(o_, m), n2 = rows_of(n2_, m), gg = rows_of(g_, m), uu = rows_of(u_, m);
  const TensorView act = rows_of(act_, m), dd = rows_of(d_, m);
  if (fc) device_.matmul(rows_of(f_, m), fc_, nullptr, h);
  device_.embedding(embed_, ids_, e);
  device_.rms_norm(e, norm_in_, eps_, cols_of(cat, 0, H));
  device_.rms_norm(h, norm_h_, eps_, cols_of(cat, H, H));
  device_.matmul(cat, wq_, nullptr, q);
  device_.matmul(cat, wk_, nullptr, k);
  device_.matmul(cat, wv_, nullptr, v);
  device_.rope(q, static_cast<int32_t>(heads_), static_cast<int32_t>(head_dim_), pos_, rope_, nullptr);
  device_.rope(k, static_cast<int32_t>(kv_heads_), static_cast<int32_t>(head_dim_), pos_, rope_, nullptr);
  device_.kv_store(k, v, pos_, row_seq_, kv_views_);
  AttentionParams ap;
  ap.q = q;
  ap.out = a;
  ap.positions = pos_;
  ap.row_seq = row_seq_;
  ap.kv = kv_views_;
  ap.num_heads = static_cast<int32_t>(heads_);
  ap.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  device_.attention(ap);
  device_.matmul(a, wo_, nullptr, o);
  device_.add(h, o, dd);  // h' = h + attn (the residual is h, before its norm)
  device_.rms_norm(dd, norm_post_, eps_, n2);
  device_.matmul(n2, w_gate_, nullptr, gg);
  device_.matmul(n2, w_up_, nullptr, uu);
  device_.act_mul(Activation::kSilu, gg, uu, act);
  device_.matmul(act, w_down_, nullptr, o);
  device_.add(dd, o, dd);  // h'' (kept for the next draft)
  if (!logit_rows.empty()) {
    const auto n = static_cast<int64_t>(logit_rows.size());
    const TensorView sel = rows_of(o_, n), hn = rows_of(hn_, n), lg = rows_of(lg_, n);
    device_.gather_rows(dd, logit_rows, sel);
    device_.rms_norm(sel, norm_, eps_, hn);
    device_.matmul(hn, lm_head_, nullptr, lg);
  }
  ++passes_;
  return device_.take_error();
}

Status Eagle3Drafter::extend(std::span<const Extend> ext) {
  const int64_t H = H_, W = 3 * H;
  size_t i = 0;
  while (i < ext.size()) {
    // A pass: whole streams while they fit in max_rows (a longer one alone, in chunks).
    pass_streams_.clear();
    std::vector<Row> rows;
    std::vector<int32_t> last_rows;
    std::vector<std::pair<Stream*, size_t>> owners;  // per logit row: its stream and Extend
    host_.clear();
    while (i < ext.size()) {
      const Extend& x = ext[i];
      const auto n = static_cast<int64_t>(x.next_tokens.size());
      if (n == 0) {
        ++i;
        continue;
      }
      if (static_cast<int64_t>(x.features.size()) != n * W) return InvalidArgument("eagle3: features do not match rows");
      Stream& s = stream(x.id);
      // Where these rows go: after what the head holds; earlier = rollback or
      // a restarted stream; a gap (positions the head never saw) restarts it.
      if (s.base < 0 || x.start_pos < s.base || x.start_pos > s.base + s.len) {
        s.kv.release();
        s.base = x.start_pos;
        s.len = 0;
      } else {
        s.len = x.start_pos - s.base;
        s.kv.truncate(s.len);
      }
      const int64_t take = std::min<int64_t>(n, max_rows_ - static_cast<int64_t>(rows.size()));
      if (take <= 0) break;
      ENGINE_RETURN_IF_ERROR(s.kv.reserve(s.len + take));
      const auto si = static_cast<int32_t>(pass_streams_.size());
      pass_streams_.push_back(&s);
      for (int64_t r = 0; r < take; ++r) {
        rows.push_back({si, s.len + static_cast<int32_t>(r), x.next_tokens[static_cast<size_t>(r)]});
        host_.insert(host_.end(), x.features.begin() + r * W, x.features.begin() + (r + 1) * W);
      }
      s.len += static_cast<int32_t>(take);
      s.ready = false;
      if (take == n) {
        last_rows.push_back(static_cast<int32_t>(rows.size()) - 1);
        owners.emplace_back(&s, i);
        ++i;
      } else {
        // The rest of this stream next pass.
        Extend rest = x;
        rest.start_pos = x.start_pos + static_cast<int32_t>(take);
        rest.features = x.features.subspan(static_cast<size_t>(take * W));
        rest.next_tokens = x.next_tokens.subspan(static_cast<size_t>(take));
        std::vector<Extend> tail(ext.begin() + static_cast<ptrdiff_t>(i) + 1, ext.end());
        tail.insert(tail.begin(), rest);
        // Finish this pass, then continue with the remainder.
        auto* dst = static_cast<float*>(f_.view().data());
        std::copy(host_.begin(), host_.end(), dst);
        ENGINE_RETURN_IF_ERROR(run(rows, true, last_rows));
        ENGINE_RETURN_IF_ERROR(finish(owners, last_rows.size()));
        return extend(tail);
      }
    }
    if (rows.empty()) break;
    auto* dst = static_cast<float*>(f_.view().data());
    std::copy(host_.begin(), host_.end(), dst);
    ENGINE_RETURN_IF_ERROR(run(rows, true, last_rows));
    ENGINE_RETURN_IF_ERROR(finish(owners, last_rows.size()));
  }
  return Status::Ok();
}

Status Eagle3Drafter::finish(std::span<const std::pair<Stream*, size_t>> owners, size_t n) {
  device_.synchronize();
  const auto* lg = static_cast<const float*>(lg_.view().data());
  const auto* hid = static_cast<const float*>(o_.view().data());  // run() gathered the logit rows' h'' here
  for (size_t r = 0; r < n && r < owners.size(); ++r) {
    Stream& s = *owners[r].first;
    const float* row = lg + static_cast<int64_t>(r) * draft_vocab_;
    const auto best = static_cast<int64_t>(std::max_element(row, row + draft_vocab_) - row);
    s.next = static_cast<TokenId>(best + d2t_[static_cast<size_t>(best)]);
    s.hidden.assign(hid + static_cast<int64_t>(r) * H_, hid + static_cast<int64_t>(r + 1) * H_);
    s.ready = true;
  }
  return Status::Ok();
}

Status Eagle3Drafter::draft(std::span<const Request> requests) {
  struct Chain {
    Stream* s;
    std::vector<TokenId>* out;
    int32_t k;
    std::vector<float> hidden;
  };
  std::vector<Chain> chains;
  for (const Request& r : requests) {
    if (r.out == nullptr) continue;
    r.out->clear();
    auto it = streams_.find(r.id);
    if (r.k <= 0 || it == streams_.end() || !it->second->ready) continue;
    Stream& s = *it->second;
    r.out->push_back(s.next);
    if (r.k > 1) chains.push_back({&s, r.out, r.k, s.hidden});
  }
  const int64_t H = H_;
  for (int32_t j = 1;; ++j) {
    pass_streams_.clear();
    std::vector<Row> rows;
    std::vector<int32_t> all;
    std::vector<Chain*> live;
    auto* hdst = static_cast<float*>(h_.view().data());
    for (Chain& c : chains) {
      if (static_cast<int32_t>(c.out->size()) >= c.k || static_cast<int64_t>(rows.size()) >= max_rows_) continue;
      // Draft rows sit after the committed ones; the next extend() rolls them back.
      ENGINE_RETURN_IF_ERROR(c.s->kv.reserve(c.s->len + j));
      const auto si = static_cast<int32_t>(pass_streams_.size());
      pass_streams_.push_back(c.s);
      std::copy(c.hidden.begin(), c.hidden.end(), hdst + static_cast<int64_t>(rows.size()) * H);
      all.push_back(static_cast<int32_t>(rows.size()));
      rows.push_back({si, c.s->len + j - 1, c.out->back()});
      live.push_back(&c);
    }
    if (rows.empty()) break;
    ENGINE_RETURN_IF_ERROR(run(rows, false, all));
    device_.synchronize();
    const auto* lg = static_cast<const float*>(lg_.view().data());
    const auto* hid = static_cast<const float*>(o_.view().data());
    for (size_t r = 0; r < live.size(); ++r) {
      const float* row = lg + static_cast<int64_t>(r) * draft_vocab_;
      const auto best = static_cast<int64_t>(std::max_element(row, row + draft_vocab_) - row);
      live[r]->out->push_back(static_cast<TokenId>(best + d2t_[static_cast<size_t>(best)]));
      live[r]->hidden.assign(hid + static_cast<int64_t>(r) * H, hid + static_cast<int64_t>(r + 1) * H);
    }
  }
  return Status::Ok();
}

}  // namespace dynalm
