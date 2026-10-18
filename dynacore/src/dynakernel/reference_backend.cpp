#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/dynakernel/backend.h"

namespace dynacore::dk {

void Bindings::bind(ValueId v, const Tensor& t) {
  int64_t capacity = -1;
  if (const std::shared_ptr<Storage>& s = t.storage(); s != nullptr && s->data() != nullptr && t.data() != nullptr) {
    const auto base = reinterpret_cast<uintptr_t>(s->data()), at = reinterpret_cast<uintptr_t>(t.data());
    capacity = at >= base && at - base <= s->size() ? static_cast<int64_t>(s->size() - (at - base)) : 0;
  }
  entries_[v] = {t.view(), capacity};
}

const Bindings::Entry* Bindings::entry(ValueId v) const {
  const auto it = entries_.find(v);
  return it == entries_.end() ? nullptr : &it->second;
}

const TensorView* Bindings::find(ValueId v) const {
  const Entry* e = entry(v);
  return e == nullptr ? nullptr : &e->view;
}

ReferenceBackend::ReferenceBackend(ThreadPool& pool, CpuIsa isa)
    : pool_(pool), dev_(std::make_unique<CpuDevice>(pool, isa)), isa_(isa_name(dev_->kernels().isa)), threads_(pool.size()) {}

uint64_t fingerprint_bytes(const void* data, int64_t bytes) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64
  const auto* p = static_cast<const unsigned char*>(data);
  for (int64_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

Result<PackedWeight> ReferenceBackend::prepack_weight(const TensorView& w, int64_t capacity_bytes) {
  const std::string who = "prepack_weight: ";
  if (w.data() == nullptr) return InvalidArgument(who + "null data");
  if (w.dtype() != DType::kQ4_K) {
    return Unsupported(who + "only q4_K has an interleaved layout for this backend (got " + std::string(dtype_name(w.dtype())) +
                       "; packed q6_K serves int16 activations only, DD-082)");
  }
  if (w.rank() != 2 || !w.is_contiguous()) return InvalidArgument(who + "needs a dense rank-2 [N, K] view");
  const int64_t n = w.dim(0), k = w.dim(1);
  if (n % 8 != 0) return Unsupported(who + "N = " + std::to_string(n) + " is not a multiple of 8 (8-row groups)");
  if (reinterpret_cast<uintptr_t>(w.data()) % static_cast<uintptr_t>(natural_alignment(DType::kQ4_K)) != 0) {
    return InvalidArgument(who + "data is not " + std::to_string(natural_alignment(DType::kQ4_K)) + "-byte aligned");
  }
  const int64_t span = w.span_bytes();
  if (capacity_bytes >= 0 && span > capacity_bytes) {
    return InvalidArgument(who + "layout spans " + std::to_string(span) + " bytes but the buffer has " +
                           std::to_string(capacity_bytes));
  }
  for (const PackedWeight& p : packed_) {
    if (p.data == w.data()) {
      if (p.n == n && p.k == k) return p;  // already packed
      return InvalidArgument(who + "this data is already packed as a [" + std::to_string(p.n) + ", " +
                             std::to_string(p.k) + "] weight");
    }
  }
  if (std::find(retired_.begin(), retired_.end(), w.data()) != retired_.end()) {
    return Unsupported(who + "this address was packed and released earlier; the device still holds that copy "
                             "(copies are freed with the backend), so it cannot be packed again here");
  }
  if (dev_->kernels().dot_q8_sx_x8_q4_K == nullptr || dev_->kernels().isa != CpuIsa::kAvx2) {
    return Unsupported(who + "no interleaved q4_K kernel on the " + isa_ + " tier");
  }
  const Device::PrepackStats before = dev_->prepack_stats();
  dev_->prepack_weight(w, /*int16_activations=*/false);
  const Device::PrepackStats after = dev_->prepack_stats();
  if (after.tensors != before.tensors + 1) {
    return Unsupported(who + (after.skipped > before.skipped
                                  ? "the device's memory budget declined the copy (DD-078; DYNACORE_Q4_REPACK=always overrides)"
                                  : "the device made no copy (DYNACORE_Q4_REPACK=off or q4_repack_min_rows = 0)"));
  }
  PackedWeight pw;
  pw.data = w.data();
  pw.n = n;
  pw.k = k;
  pw.source_bytes = span;
  pw.packed_bytes = after.packed_bytes - before.packed_bytes;
  pw.fingerprint = fingerprint_bytes(w.data(), span);
  packed_.push_back(pw);
  return pw;
}

ReferenceBackend::~ReferenceBackend() = default;

TargetDesc ReferenceBackend::target() const {
  TargetDesc t;
  t.backend = "reference";
  t.isa = isa_;
  t.threads = threads_;
  // Mirrors CpuDevice: integer-dot kernels are accelerated on the AVX2 tier only.
  t.int8_dot = dev_->kernels().int8_simd;  // AVX2, or NEON with the dot-product instructions
  t.int8_subscaled = t.int8_dot && dev_->kernels().quantize_act_sx != nullptr;
  return t;
}

KernelPlan ReferenceBackend::plan_for(const MatMulAttrs& a) {
  KernelPlan k = KernelPlan::defaults();
  if (a.act == ActFormat::kF32) {
    // int8_decode_max_rows == 0 turns off every int8 activation path
    // (per-32, sub-scaled, super-block, packed Q4_K); int16 needs its own limit.
    k.int8_decode_max_rows = 0;
    k.int16_decode_max_rows = 0;
  }
  return k;
}

Status ReferenceBackend::check(const Program& p) const {
  // The checks below index operands and attributes: only valid programs.
  ENGINE_RETURN_IF_ERROR(verify(p));
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    const std::string where = "op " + std::to_string(i) + " (" + std::string(op_kind_name(op.kind)) +
                              (op.name.empty() ? "" : " '" + op.name + "'") + ")";
    if (op.kind != OpKind::kMatMul) return Unsupported(where + ": the reference backend runs matmul only");
    const auto& a = std::get<MatMulAttrs>(op.attrs);
    if (a.act == ActFormat::kInt8 || a.act == ActFormat::kInt16) {
      // Forcing a reduced-precision path outside the planner's verified row
      // limits would run a kernel on shapes it was never validated for.
      return Unsupported(where + ": activation format " + std::string(act_format_name(a.act)) +
                         " cannot be requested on the reference backend (use auto or f32)");
    }
    const DType wt = p.value(op.inputs[1]).layout.dtype;
    if (!dev_->supports_weight_type(wt)) {
      return Unsupported(where + ": no " + std::string(isa_) + " kernel for weight dtype " + std::string(dtype_name(wt)));
    }
  }
  return Status::Ok();
}

bool ReferenceBackend::release_packed(const void* data) {
  const auto it = std::find_if(packed_.begin(), packed_.end(), [&](const PackedWeight& p) { return p.data == data; });
  if (it == packed_.end()) return false;
  retired_.push_back(it->data);
  packed_.erase(it);
  return true;
}

Status validate_bindings(const Program& p, const Bindings& b) {
  struct Range {
    uintptr_t lo, hi;
    ValueId id;
  };
  std::vector<Range> ranges;
  for (const Value& v : p.values()) {
    const std::string who = "value %" + std::to_string(v.id) + (v.name.empty() ? "" : " '" + v.name + "'");
    const Bindings::Entry* e = b.entry(v.id);
    if (e == nullptr) return InvalidArgument(who + " is not bound");
    const TensorView* t = &e->view;
    if (t->data() == nullptr) return InvalidArgument(who + " is bound to null");
    const TensorLayout& want = v.layout;
    const TensorLayout& got = t->layout();
    bool same = got.dtype == want.dtype && got.shape == want.shape;
    for (int i = 0; same && i < want.shape.rank(); ++i) same = got.strides[i] == want.strides[i];
    if (!same) return InvalidArgument(who + ": bound view does not match the declared dtype, shape and strides");
    const auto addr = reinterpret_cast<uintptr_t>(t->data());
    const int64_t align = std::max(v.alignment, natural_alignment(v.layout.dtype));
    if (addr % static_cast<uintptr_t>(align) != 0) {
      return InvalidArgument(who + ": data is not " + std::to_string(align) + "-byte aligned");
    }
    if (e->capacity >= 0 && got.span_bytes() > e->capacity) {
      return InvalidArgument(who + ": layout spans " + std::to_string(got.span_bytes()) + " bytes but the buffer has " +
                             std::to_string(e->capacity));
    }
    ranges.push_back({addr, addr + static_cast<uintptr_t>(got.span_bytes()), v.id});
  }
  // Results are written while operands are read: no result may overlap any
  // other bound buffer (byte ranges, conservative for strided views).
  for (const Range& r : ranges) {
    if (p.value(r.id).role != ValueRole::kResult) continue;
    for (const Range& o : ranges) {
      if (o.id != r.id && r.lo < o.hi && o.lo < r.hi) {
        return InvalidArgument("value %" + std::to_string(r.id) + " (result) overlaps the buffer of %" +
                               std::to_string(o.id));
      }
    }
  }
  return Status::Ok();
}

Status ReferenceBackend::validate_bindings(const Program& p, const Bindings& b, const RunOptions& opts) const {
  ENGINE_RETURN_IF_ERROR(dk::validate_bindings(p, b));
  for (const Value& v : p.values()) {
    const std::string who = "value %" + std::to_string(v.id) + (v.name.empty() ? "" : " '" + v.name + "'");
    const TensorView* t = b.find(v.id);
    const TensorLayout& got = t->layout();
    // A prepacked source is used through its snapshot: it must be the same
    // weight, unchanged.
    for (const PackedWeight& pw : packed_) {
      if (pw.data != t->data()) continue;
      if (v.role != ValueRole::kWeight || got.dtype != pw.dtype || got.shape[0] != pw.n || got.shape[1] != pw.k) {
        return InvalidArgument(who + " is bound to the data of a prepacked [" + std::to_string(pw.n) + ", " +
                               std::to_string(pw.k) + "] q4_K weight with another role, dtype or shape");
      }
      if (opts.check_packed && fingerprint_bytes(t->data(), pw.source_bytes) != pw.fingerprint) {
        return InvalidArgument(who + ": weight bytes changed since prepack_weight (its interleaved copy is stale)");
      }
    }
  }
  return Status::Ok();
}

std::vector<std::vector<int32_t>> ReferenceBackend::plan_calls(const Program& p, const RunOptions& opts) {
  std::vector<std::vector<int32_t>> calls;
  std::vector<ValueId> written;  // results of the open group
  for (int32_t i = 0; i < static_cast<int32_t>(p.ops().size()); ++i) {
    const Op& op = p.ops()[static_cast<size_t>(i)];
    bool join = opts.group_independent && !calls.empty();
    if (join) {
      const Op& head = p.ops()[static_cast<size_t>(calls.back().front())];
      // One kernel plan per device call: the same activation contract.
      join = std::get<MatMulAttrs>(op.attrs).act == std::get<MatMulAttrs>(head.attrs).act;
      // matmul_many runs its jobs in one parallel region: no job may read
      // another's result.
      for (ValueId in : op.inputs) join = join && std::find(written.begin(), written.end(), in) == written.end();
    }
    if (!join) {
      calls.push_back({});
      written.clear();
    }
    calls.back().push_back(i);
    written.push_back(op.output);
  }
  return calls;
}

Status ReferenceBackend::run(const Program& p, const Bindings& b) { return run(p, b, RunOptions{}); }

Status ReferenceBackend::run(const Program& p, const Bindings& b, const RunOptions& opts) {
  ENGINE_RETURN_IF_ERROR(check(p));  // verifies first
  ENGINE_RETURN_IF_ERROR(validate_bindings(p, b, opts));
  const uint64_t misuse0 = pool_.misuse_count();
  // Every check is done: nothing below can fail, so no output is ever left
  // half-written by a rejected run.
  // Packed copies only for registered weights (validated above); every other
  // op runs with them disabled, which changes nothing else in the dispatch.
  auto packed_ok = [&](int32_t i) {
    const void* w = b.find(p.ops()[static_cast<size_t>(i)].inputs[1])->data();
    return std::any_of(packed_.begin(), packed_.end(), [&](const PackedWeight& pw) { return pw.data == w; });
  };
  std::vector<std::vector<int32_t>> calls;
  for (const std::vector<int32_t>& group : plan_calls(p, opts)) {  // one kernel plan per call
    for (size_t j = 0; j < group.size(); ++j) {
      if (j == 0 || packed_ok(group[j]) != packed_ok(group[j - 1])) calls.push_back({});
      calls.back().push_back(group[j]);
    }
  }
  std::vector<Device::MatmulJob> jobs;
  for (const std::vector<int32_t>& call : calls) {
    KernelPlan plan = plan_for(std::get<MatMulAttrs>(p.ops()[static_cast<size_t>(call.front())].attrs));
    if (!packed_ok(call.front())) plan.q4_repack_min_rows = 0;
    if (opts.experimental_packed_8rows) plan.packed_int8_8rows = true;
    dev_->set_kernel_plan(plan);
    jobs.clear();
    for (int32_t i : call) {
      const Op& op = p.ops()[static_cast<size_t>(i)];
      jobs.push_back({*b.find(op.inputs[0]), *b.find(op.inputs[1]), *b.find(op.output),
                      op.inputs.size() == 3 ? b.find(op.inputs[2]) : nullptr});
    }
    if (jobs.size() == 1) {
      dev_->matmul(jobs[0].x, jobs[0].w, jobs[0].bias, jobs[0].y);
    } else {
      dev_->matmul_many(jobs);
    }
  }
  dev_->synchronize();
  // The pool ran another region while this run was in flight (shared with a
  // second caller, or reentered): those regions were serialized, so results
  // are complete, but the ownership contract was broken.
  if (const uint64_t m = pool_.misuse_count() - misuse0; m != 0) {
    return Internal("thread pool contract violated during run(): " + std::to_string(m) +
                    " parallel region(s) found the pool busy (shared or reentered) and ran serially; "
                    "outputs are complete");
  }
  return Status::Ok();
}

}  // namespace dynacore::dk
