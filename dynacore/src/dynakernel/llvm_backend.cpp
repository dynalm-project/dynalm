#include "dynacore/dynakernel/llvm_backend.h"

#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

namespace dynacore::dk {

namespace {

Q4KMatmulSpec spec_of(const Op& op) {
  const auto& a = std::get<MatMulAttrs>(op.attrs);
  return {static_cast<int32_t>(a.m), a.n, a.k, op.inputs.size() == 3};
}

std::string where(const Op& op, size_t i) {
  return "op " + std::to_string(i) + " (" + std::string(op_kind_name(op.kind)) + (op.name.empty() ? "" : " '" + op.name + "'") +
         ")";
}

}  // namespace

LlvmTarget llvm_lowering_target(LlvmLowering l) {
  LlvmTarget t = llvm_tier_target();
  if (l == LlvmLowering::kAvx2 && t.cpu.empty()) t.features += std::string(t.features.empty() ? "" : ",") + "-avxvnni";
  return t;
}

Status llvm_lowering_runnable(LlvmLowering l) {
  ENGINE_RETURN_IF_ERROR(llvm_available());
  const CpuIsa tier = select_isa(cpu_info().features);
  if (tier != CpuIsa::kAvx2) {
    return Unsupported("llvm int8 backend needs the avx2 tier (DynaCore's sub-scaled quantizer); this process selected " +
                       std::string(isa_name(tier)));
  }
  const std::string features = llvm_runtime().host_features;
  if (!llvm_feature_enabled(features, "avx2") || !llvm_feature_enabled(features, "fma")) {
    return Unsupported("llvm backend: LLVM does not report avx2 and fma for this CPU");
  }
  if (l == LlvmLowering::kAvxVnni && !(cpu_info().features.avx_vnni && llvm_feature_enabled(features, "avxvnni"))) {
    return Unsupported("llvm backend: the avxvnni lowering needs AVX-VNNI, which this CPU does not report");
  }
  return Status::Ok();
}

LlvmInt8Backend::LlvmInt8Backend(ThreadPool& pool, LlvmLowering lowering) : LlvmInt8Backend(pool, lowering, {}) {}

LlvmInt8Backend::LlvmInt8Backend(ThreadPool& pool, LlvmLowering lowering, const LlvmKernelOptions& options)
    : lowering_(lowering), options_(options), exec_(pool, {}) {
  (void)exec_.set_packed_layout(valid_packed_layout(options.layout) ? options.layout : PackedLayout::kX8);
}

LlvmInt8Backend::~LlvmInt8Backend() = default;

Status LlvmInt8Backend::runnable() const {
  ENGINE_RETURN_IF_ERROR(llvm_lowering_runnable(lowering_));
  if (!valid_packed_layout(options_.layout)) return InvalidArgument("llvm int8 backend: unknown packed layout");
  if (packed_layout_info(options_.layout).fp16 && !llvm_feature_enabled(llvm_runtime().host_features, "f16c")) {
    return Unsupported("llvm int8 backend: the fp16 packed layouts need F16C");
  }
  return exec_.runnable();
}

TargetDesc LlvmInt8Backend::target() const {
  TargetDesc t = exec_.target();
  t.backend = "llvm-int8";
  t.isa = std::string(llvm_lowering_name(lowering_));
  return t;
}

Status LlvmInt8Backend::check(const Program& p) const {
  ENGINE_RETURN_IF_ERROR(runnable());
  ENGINE_RETURN_IF_ERROR(verify(p));
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    if (op.kind != OpKind::kMatMul) return Unsupported(where(op, i) + ": the llvm int8 backend compiles matmul only");
    const auto& a = std::get<MatMulAttrs>(op.attrs);
    const DType wt = p.value(op.inputs[1]).layout.dtype;
    if (wt != DType::kQ4_K) {
      return Unsupported(where(op, i) + ": llvm int8 kernels take q4_K weights, not " + std::string(dtype_name(wt)));
    }
    if (a.act != ActFormat::kAuto && a.act != ActFormat::kInt8) {
      return Unsupported(where(op, i) + ": activation contract " + std::string(act_format_name(a.act)) +
                         " does not allow int8 activations");
    }
    if (a.accum != AccumType::kF32) return Unsupported(where(op, i) + ": accumulation must be f32");
    if (const Status v = validate_int8_spec(spec_of(op)); !v.ok()) return Unsupported(where(op, i) + ": " + v.message());
  }
  return Status::Ok();
}

Status LlvmInt8Backend::ensure_jit() {
  if (jit_) return Status::Ok();
  auto j = LlvmJit::create(llvm_lowering_target(lowering_));
  if (!j.ok()) return j.status();
  jit_ = std::move(*j);
  return Status::Ok();
}

Status LlvmInt8Backend::prepare(const Program& p) {
  ENGINE_RETURN_IF_ERROR(check(p));
  ENGINE_RETURN_IF_ERROR(ensure_jit());
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    if (exec_.find(op) != nullptr) continue;
    const Q4KMatmulSpec s = spec_of(op);
    CompiledKernel ck{s, llvm_kernel_symbol(s, lowering_, options_), {}};
    auto c = jit_->compile_kernel(s, lowering_, &ck.stats, options_);
    if (!c.ok()) return Status(c.status().code(), where(op, i) + ": " + c.status().message());
    compiled_.push_back(std::move(ck));
    table_.push_back({s, compiled_.back().symbol.c_str(), reinterpret_cast<GeneratedQ4KInt8Fn>(c->fn),
                      reinterpret_cast<uint64_t (*)()>(c->calls)});
    exec_.set_kernels(table_);  // table_ may have reallocated
  }
  return Status::Ok();
}

Status LlvmInt8Backend::run(const Program& p, const Bindings& b, const RunOptions& opts) {
  // Every op already compiled: skip prepare(). The executor's run() still
  // verifies the program, checks the contract, tier and kernel table, and
  // validates the bindings; what prepare() adds (LLVM availability, target
  // features) held when the kernels were compiled and cannot change.
  bool compiled = !p.ops().empty();
  for (const Op& op : p.ops()) compiled = compiled && exec_.find(op) != nullptr;
  if (!compiled) ENGINE_RETURN_IF_ERROR(prepare(p));
  return exec_.run(p, b, opts);
}

Result<std::string> LlvmInt8Backend::assembly(const Q4KMatmulSpec& s) {
  ENGINE_RETURN_IF_ERROR(llvm_lowering_runnable(lowering_));
  ENGINE_RETURN_IF_ERROR(ensure_jit());
  return jit_->kernel_assembly(s, lowering_, options_);
}

}  // namespace dynacore::dk
