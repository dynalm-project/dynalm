#include "dynacore/dynakernel/generated_backend.h"

#include <algorithm>

#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

namespace dynacore::dk {

GeneratedBackend::GeneratedBackend(ThreadPool& pool, std::span<const GeneratedKernel> kernels)
    : pool_(pool), kernels_(kernels) {
  if (!kernels_.empty()) isa_ = kernels_.front().isa;
  for (const GeneratedKernel& k : kernels_) mixed_ = mixed_ || k.isa != isa_;
}

Status GeneratedBackend::runnable() const {
  if (mixed_) return InvalidArgument("generated backend: the kernel table mixes ISA variants");
  if (isa_ == CodegenIsa::kAvx2) {
    // The tier this process selected (CPU features, and DYNACORE_ISA when it
    // forces a lower tier): AVX2 code never runs where it was not selected.
    const CpuIsa tier = select_isa(cpu_info().features);
    if (tier != CpuIsa::kAvx2) {
      return Unsupported("generated avx2 kernels need the avx2 tier; this process selected " +
                         std::string(isa_name(tier)));
    }
  }
  return Status::Ok();
}

TargetDesc GeneratedBackend::target() const {
  TargetDesc t;
  t.backend = "generated";
  t.isa = std::string(codegen_isa_name(isa_));
  t.threads = pool_.size();
  t.int8_dot = false;
  return t;
}

const GeneratedKernel* GeneratedBackend::find(const Op& op) const {
  const auto* a = std::get_if<MatMulAttrs>(&op.attrs);
  if (a == nullptr || op.inputs.size() < 2) return nullptr;
  const Q4KMatmulSpec want{static_cast<int32_t>(a->m), a->n, a->k, op.inputs.size() == 3};
  for (const GeneratedKernel& g : kernels_) {
    if (g.spec == want && g.fn != nullptr) return &g;
  }
  return nullptr;
}

Status GeneratedBackend::check(const Program& p) const {
  ENGINE_RETURN_IF_ERROR(runnable());
  ENGINE_RETURN_IF_ERROR(verify(p));
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    const std::string where = "op " + std::to_string(i) + " (" + std::string(op_kind_name(op.kind)) +
                              (op.name.empty() ? "" : " '" + op.name + "'") + ")";
    if (op.kind != OpKind::kMatMul) return Unsupported(where + ": the generated backend runs matmul only");
    const auto& a = std::get<MatMulAttrs>(op.attrs);
    const DType wt = p.value(op.inputs[1]).layout.dtype;
    if (wt != DType::kQ4_K) {
      return Unsupported(where + ": generated kernels take q4_K weights, not " + std::string(dtype_name(wt)));
    }
    if (a.act != ActFormat::kAuto && a.act != ActFormat::kF32) {
      return Unsupported(where + ": activation format " + std::string(act_format_name(a.act)) +
                         " is not generated (kernels keep fp32 activations)");
    }
    if (a.accum != AccumType::kF32) return Unsupported(where + ": accumulation must be f32");
    if (find(op) == nullptr) {
      const Q4KMatmulSpec s{static_cast<int32_t>(a.m), a.n, a.k, op.inputs.size() == 3};
      const Status v = validate_spec(s);
      return Unsupported(where + ": no generated kernel " + kernel_symbol(s) +
                         (v.ok() ? " in this build's table (" + std::to_string(kernels_.size()) + " kernels)"
                                 : " (" + v.message() + ")"));
    }
  }
  return Status::Ok();
}

Status GeneratedBackend::run(const Program& p, const Bindings& b) {
  ENGINE_RETURN_IF_ERROR(check(p));
  ENGINE_RETURN_IF_ERROR(validate_bindings(p, b));
  const uint64_t misuse0 = pool_.misuse_count();
  for (const Op& op : p.ops()) {
    const GeneratedKernel* g = find(op);
    const TensorView& x = *b.find(op.inputs[0]);
    const TensorView& w = *b.find(op.inputs[1]);
    const TensorView* bias = op.inputs.size() == 3 ? b.find(op.inputs[2]) : nullptr;
    const TensorView& y = *b.find(op.output);
    // Strides are bytes in the IR and whole fp32 elements (verified).
    const int64_t xs = x.stride(0) / static_cast<int64_t>(sizeof(float));
    const int64_t ys = y.stride(0) / static_cast<int64_t>(sizeof(float));
    const auto* xp = x.data_as<const float>();
    auto* yp = y.data_as<float>();
    const float* bp = bias != nullptr ? bias->data_as<const float>() : nullptr;
    const int64_t n = g->spec.n;
    // Columns split across the pool, 8 per chunk: each output element is
    // computed by one thread in k order, so results do not depend on threads.
    constexpr int64_t kCols = 8;
    const size_t chunks = static_cast<size_t>((n + kCols - 1) / kCols);
    pool_.parallel_for(chunks, 1, [&](size_t c0, size_t c1) {
      const int64_t n0 = static_cast<int64_t>(c0) * kCols, n1 = std::min(n, static_cast<int64_t>(c1) * kCols);
      g->fn(w.data(), xp, xs, bp, yp, ys, n0, n1);
    });
  }
  if (const uint64_t m = pool_.misuse_count() - misuse0; m != 0) {
    return Internal("thread pool contract violated during run(): " + std::to_string(m) +
                    " parallel region(s) found the pool busy (shared or reentered) and ran serially; "
                    "outputs are complete");
  }
  return Status::Ok();
}

// --- GeneratedInt8Backend --------------------------------------------------------------------------

GeneratedInt8Backend::GeneratedInt8Backend(ThreadPool& pool, std::span<const GeneratedInt8Kernel> kernels)
    : pool_(pool), kernels_(kernels) {
  if (select_isa(cpu_info().features) == CpuIsa::kAvx2) k_ = make_cpu_kernels(CpuIsa::kAvx2);
}

Status GeneratedInt8Backend::runnable() const {
  const CpuIsa tier = select_isa(cpu_info().features);
  if (tier != CpuIsa::kAvx2) {
    return Unsupported("generated int8 kernels need the avx2 tier; this process selected " + std::string(isa_name(tier)));
  }
  if (k_.quantize_act_sx == nullptr) return Unsupported("generated int8 backend: no sub-scaled activation quantizer");
  return Status::Ok();
}

TargetDesc GeneratedInt8Backend::target() const {
  TargetDesc t;
  t.backend = "generated-int8";
  t.isa = "avx2";
  t.threads = pool_.size();
  t.int8_dot = true;
  return t;
}

const GeneratedInt8Kernel* GeneratedInt8Backend::find(const Op& op) const {
  const auto* a = std::get_if<MatMulAttrs>(&op.attrs);
  if (a == nullptr || op.inputs.size() < 2) return nullptr;
  const Q4KMatmulSpec want{static_cast<int32_t>(a->m), a->n, a->k, op.inputs.size() == 3};
  for (const GeneratedInt8Kernel& g : kernels_) {
    if (g.spec == want && g.fn != nullptr) return &g;
  }
  return nullptr;
}

Status GeneratedInt8Backend::check(const Program& p) const {
  ENGINE_RETURN_IF_ERROR(runnable());
  ENGINE_RETURN_IF_ERROR(verify(p));
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    const std::string where = "op " + std::to_string(i) + " (" + std::string(op_kind_name(op.kind)) +
                              (op.name.empty() ? "" : " '" + op.name + "'") + ")";
    if (op.kind != OpKind::kMatMul) return Unsupported(where + ": the generated int8 backend runs matmul only");
    const auto& a = std::get<MatMulAttrs>(op.attrs);
    const DType wt = p.value(op.inputs[1]).layout.dtype;
    if (wt != DType::kQ4_K) return Unsupported(where + ": generated int8 kernels take q4_K weights, not " + std::string(dtype_name(wt)));
    if (a.act != ActFormat::kAuto && a.act != ActFormat::kInt8) {
      return Unsupported(where + ": activation contract " + std::string(act_format_name(a.act)) +
                         " does not allow int8 activations");
    }
    if (a.accum != AccumType::kF32) return Unsupported(where + ": accumulation must be f32");
    if (find(op) == nullptr) {
      const Q4KMatmulSpec s{static_cast<int32_t>(a.m), a.n, a.k, op.inputs.size() == 3};
      const Status v = validate_int8_spec(s);
      return Unsupported(where + ": no generated int8 kernel " + int8_kernel_symbol(s) +
                         (v.ok() ? " in this build's table" : " (" + v.message() + ")"));
    }
  }
  return Status::Ok();
}

Status GeneratedInt8Backend::set_packed_layout(PackedLayout l) {
  if (!valid_packed_layout(l)) return InvalidArgument("set_packed_layout: unknown layout");
  if (!packed_.empty() && l != layout_) return InvalidArgument("set_packed_layout: weights are already packed in another layout");
  layout_ = l;
  return Status::Ok();
}

const GeneratedInt8Backend::Packed* GeneratedInt8Backend::find_packed(const TensorView& w) const {
  for (const Packed& pk : packed_) {
    if (pk.info.data == w.data() && pk.info.n == w.dim(0) && pk.info.k == w.dim(1)) return &pk;
  }
  return nullptr;
}

Result<PackedWeight> GeneratedInt8Backend::prepack_weight(const TensorView& w, int64_t capacity_bytes) {
  const std::string who = "prepack_weight: ";
  if (w.data() == nullptr) return InvalidArgument(who + "null data");
  if (w.dtype() != DType::kQ4_K) return Unsupported(who + "the generated int8 kernels take q4_K weights");
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
  for (const Packed& pk : packed_) {
    if (pk.info.data == w.data()) {
      if (pk.info.n == n && pk.info.k == k) return pk.info;
      return InvalidArgument(who + "this data is already packed with another shape");
    }
  }
  Packed pk;
  const int64_t sb = k / 256, groups = n / 8, row_bytes = dtype_row_bytes(DType::kQ4_K, k);
  const int64_t block = packed_layout_info(layout_).block_bytes;
  pk.bytes.resize(static_cast<size_t>(groups * sb * block));
  for (int64_t g = 0; g < groups; ++g) {
    repack_q4_K_x8_layout(layout_, w.data(), row_bytes, g * 8, k, pk.bytes.data() + g * sb * block);
  }
  pk.info.data = w.data();
  pk.info.n = n;
  pk.info.k = k;
  pk.info.source_bytes = span;
  pk.info.packed_bytes = static_cast<int64_t>(pk.bytes.size());
  pk.info.fingerprint = fingerprint_bytes(w.data(), span);
  packed_.push_back(std::move(pk));
  return packed_.back().info;
}

bool GeneratedInt8Backend::release_packed(const void* data) {
  const auto it = std::find_if(packed_.begin(), packed_.end(), [&](const Packed& p) { return p.info.data == data; });
  if (it == packed_.end()) return false;
  packed_.erase(it);  // the copy is owned here: freed now
  return true;
}

Status GeneratedInt8Backend::run(const Program& p, const Bindings& b, const RunOptions& opts) {
  ENGINE_RETURN_IF_ERROR(check(p));
  ENGINE_RETURN_IF_ERROR(validate_bindings(p, b));
  // Every op's weight must be a registered, unchanged packed source.
  std::vector<const Packed*> packed(p.ops().size(), nullptr);
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    const TensorView& w = *b.find(op.inputs[1]);
    packed[i] = find_packed(w);
    if (packed[i] == nullptr) {
      return InvalidArgument("op " + std::to_string(i) +
                             ": its weight is not prepacked on this backend (call prepack_weight first)");
    }
    if (opts.check_packed && fingerprint_bytes(w.data(), packed[i]->info.source_bytes) != packed[i]->info.fingerprint) {
      return InvalidArgument("op " + std::to_string(i) + ": weight bytes changed since prepack_weight");
    }
  }
  const uint64_t misuse0 = pool_.misuse_count();
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    const GeneratedInt8Kernel* g = find(op);
    const TensorView& x = *b.find(op.inputs[0]);
    const TensorView* bias = op.inputs.size() == 3 ? b.find(op.inputs[2]) : nullptr;
    const TensorView& y = *b.find(op.output);
    const int64_t m = g->spec.m, k = g->spec.k, nb = k / kActQ8Block;
    // Activations: DynaCore's sub-scaled quantizer, one row at a time (rows
    // may be strided); K / 256 ActSuperQ8 per row, contiguous.
    act_.resize(static_cast<size_t>(m * nb));
    auto row = [&](int64_t r) {
      return reinterpret_cast<const float*>(static_cast<const std::byte*>(x.data()) + r * x.stride(0));
    };
    const int64_t sb = k / 256, units = m * sb;  // super-blocks: quantize_act_sx treats each independently
    if (opts.experimental_parallel_quant && pool_.size() > 1 && units >= opts.parallel_quant_min_superblocks) {
      // EXPERIMENTAL: (row, super-block range) chunks across the pool, one
      // region before the kernel's. Same quantizer on the same super-blocks,
      // so the codes are identical to the serial loop below.
      const size_t grain = std::max<size_t>(4, static_cast<size_t>(units) / (2 * static_cast<size_t>(pool_.size())));
      pool_.parallel_for(static_cast<size_t>(units), grain, [&](size_t u0, size_t u1) {
        while (u0 < u1) {
          const int64_t r = static_cast<int64_t>(u0) / sb, s0 = static_cast<int64_t>(u0) % sb;
          const int64_t s1 = std::min<int64_t>(sb, s0 + static_cast<int64_t>(u1 - u0));
          k_.quantize_act_sx(row(r) + s0 * 256, act_.data() + r * nb + s0 * (256 / kActQ8Block), (s1 - s0) * 256);
          u0 += static_cast<size_t>(s1 - s0);
        }
      });
    } else {
      for (int64_t r = 0; r < m; ++r) k_.quantize_act_sx(row(r), act_.data() + r * nb, k);
    }
    const auto* xa = reinterpret_cast<const ActSuperQ8*>(act_.data());
    // The kernel ABI names BlockQ4_Kx8; kernels for other layouts read their own (same pointer).
    const auto* wp = reinterpret_cast<const BlockQ4_Kx8*>(packed[i]->bytes.data());
    const float* bp = bias != nullptr ? bias->data_as<const float>() : nullptr;
    auto* yp = y.data_as<float>();
    const int64_t ys = y.stride(0) / static_cast<int64_t>(sizeof(float));
    const size_t groups = static_cast<size_t>(g->spec.n / 8);
    // Same work split as CpuDevice's packed path: 8-row groups, ~32 chunks per thread.
    const size_t grain = std::max<size_t>(1, groups / (static_cast<size_t>(pool_.size()) * 32));
    pool_.parallel_for(groups, grain, [&](size_t g0, size_t g1) {
      g->fn(wp, xa, bp, yp, ys, static_cast<int64_t>(g0), static_cast<int64_t>(g1));
    });
  }
  if (const uint64_t mis = pool_.misuse_count() - misuse0; mis != 0) {
    return Internal("thread pool contract violated during run(): " + std::to_string(mis) +
                    " parallel region(s) found the pool busy (shared or reentered) and ran serially; "
                    "outputs are complete");
  }
  return Status::Ok();
}

}  // namespace dynacore::dk
