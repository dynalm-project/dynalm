#include "dynacore/dynakernel/program.h"

#include <algorithm>
#include <cstdio>

#include "dynacore/quantization/dequant.h"

namespace dynacore::dk {

std::string_view value_role_name(ValueRole r) {
  switch (r) {
    case ValueRole::kActivation: return "activation";
    case ValueRole::kWeight: return "weight";
    case ValueRole::kBias: return "bias";
    case ValueRole::kResult: return "result";
  }
  return "?";
}

std::string_view op_kind_name(OpKind k) {
  switch (k) {
    case OpKind::kMatMul: return "matmul";
    case OpKind::kQuantize: return "quantize";
    case OpKind::kDequantize: return "dequantize";
    case OpKind::kBiasAdd: return "bias_add";
    case OpKind::kActivation: return "activation";
    case OpKind::kRmsNorm: return "rms_norm";
    case OpKind::kAttention: return "attention";
    case OpKind::kFusedGroup: return "fused_group";
    case OpKind::kCount: break;
  }
  return "?";
}

std::string_view accum_type_name(AccumType a) {
  switch (a) {
    case AccumType::kF32: return "f32";
  }
  return "?";
}

std::string_view act_format_name(ActFormat a) {
  switch (a) {
    case ActFormat::kAuto: return "auto";
    case ActFormat::kF32: return "f32";
    case ActFormat::kInt8: return "int8";
    case ActFormat::kInt16: return "int16";
  }
  return "?";
}

int64_t natural_alignment(DType t) {
  if (!dtype_is_quantized(t)) return dtype_block_bytes(t);
  return t == DType::kQ8_K ? 4 : 2;  // BlockQ8_K starts with a float, the others with fp16 scales
}

ValueId Program::add_value(Value v) {
  v.id = static_cast<ValueId>(values_.size());
  values_.push_back(std::move(v));
  return values_.back().id;
}

int32_t Program::add_op(Op op) {
  ops_.push_back(std::move(op));
  return static_cast<int32_t>(ops_.size() - 1);
}

namespace {

// Byte stride of the innermost dimension of a dense row: 0 for block-quantized
// dtypes (whole packed blocks), else the element size (TensorLayout's rule).
int64_t inner_stride(DType t) { return dtype_is_quantized(t) ? 0 : dtype_block_bytes(t); }

std::string layout_text(const TensorLayout& l) {
  std::string s(dtype_name(l.dtype));
  s += l.shape.to_string();
  s += " strides(";
  for (int i = 0; i < l.shape.rank(); ++i) {
    if (i) s += ',';
    s += std::to_string(l.strides[i]);
  }
  s += ')';
  return s;
}

std::string op_label(const Program& p, int32_t i) {
  const Op& op = p.ops()[static_cast<size_t>(i)];
  std::string s = "op " + std::to_string(i) + " (" + std::string(op_kind_name(op.kind));
  if (!op.name.empty()) s += " '" + op.name + "'";
  return s + ")";
}

std::string value_label(const Program& p, ValueId v) {
  if (!p.has_value(v)) return "%" + std::to_string(v);
  const Value& x = p.value(v);
  return "%" + std::to_string(v) + (x.name.empty() ? "" : " '" + x.name + "'");
}

// Problems of one value on its own: shape, strides, alignment.
void check_value(const Value& v, std::vector<Diagnostic>& out) {
  auto fail = [&](std::string m) {
    out.push_back({-1, v.id, "value %" + std::to_string(v.id) + (v.name.empty() ? "" : " '" + v.name + "'") + ": " + m});
  };
  const TensorLayout& l = v.layout;
  if (static_cast<int>(l.dtype) < 0 || l.dtype >= DType::kCount) return fail("invalid dtype");
  const int r = l.shape.rank();
  if (r < 1 || r > kMaxDims) return fail("rank " + std::to_string(r) + " (needs 1.." + std::to_string(kMaxDims) + ")");
  for (int i = 0; i < r; ++i) {
    if (l.shape[i] <= 0) return fail("dimension " + std::to_string(i) + " is " + std::to_string(l.shape[i]));
    if (l.strides[i] < 0) return fail("negative stride");
  }
  if (l.strides[r - 1] != inner_stride(l.dtype)) {
    return fail("innermost stride " + std::to_string(l.strides[r - 1]) + " (must be " +
                std::to_string(inner_stride(l.dtype)) + ": innermost elements are contiguous)");
  }
  const int64_t row = dtype_row_bytes(l.dtype, l.shape[r - 1]);
  if (row < 0) {
    return fail("innermost dimension " + std::to_string(l.shape[r - 1]) + " is not a multiple of the " +
                std::string(dtype_name(l.dtype)) + " block size " + std::to_string(dtype_block_elems(l.dtype)));
  }
  // Outer strides must step over at least one whole inner extent (no overlap).
  int64_t extent = row;
  for (int i = r - 2; i >= 0; --i) {
    if (l.shape[i] > 1 && l.strides[i] < extent) {
      return fail("stride " + std::to_string(l.strides[i]) + " of dimension " + std::to_string(i) +
                  " overlaps the previous one (needs >= " + std::to_string(extent) + ")");
    }
    extent = std::max<int64_t>(extent, l.strides[i] * l.shape[i]);
  }
  // Kernels index scalar rows as T* + stride / sizeof(T): strides must be
  // whole elements.
  if (!dtype_is_quantized(l.dtype)) {
    const int64_t es = dtype_block_bytes(l.dtype);
    for (int i = 0; i < r - 1; ++i) {
      if (l.strides[i] % es != 0) {
        return fail("stride " + std::to_string(l.strides[i]) + " of dimension " + std::to_string(i) +
                    " is not a multiple of the " + std::to_string(es) + "-byte element");
      }
    }
  }
  if (v.alignment < 0 || (v.alignment & (v.alignment - 1)) != 0) {
    return fail("alignment " + std::to_string(v.alignment) + " is not a power of two");
  }
}

int64_t rows_of(const TensorLayout& l) { return l.shape.rank() == 1 ? 1 : l.shape[0]; }
int64_t cols_of(const TensorLayout& l) { return l.shape[l.shape.rank() - 1]; }

void check_matmul(const Program& p, int32_t i, std::vector<Diagnostic>& out) {
  const Op& op = p.ops()[static_cast<size_t>(i)];
  auto fail = [&](ValueId v, std::string m) { out.push_back({i, v, op_label(p, i) + ": " + m}); };
  const auto* a = std::get_if<MatMulAttrs>(&op.attrs);
  if (a == nullptr) return fail(kNoValue, "missing matmul attributes");
  if (op.inputs.size() != 2 && op.inputs.size() != 3) {
    return fail(kNoValue, "takes {x, w} or {x, w, bias}, got " + std::to_string(op.inputs.size()) + " operands");
  }
  const ValueId xi = op.inputs[0], wi = op.inputs[1], bi = op.inputs.size() == 3 ? op.inputs[2] : kNoValue;
  const ValueId yi = op.output;
  const Value& x = p.value(xi);
  const Value& w = p.value(wi);
  const Value& y = p.value(yi);
  const TensorLayout &xl = x.layout, &wl = w.layout, &yl = y.layout;

  // Roles.
  if (x.role != ValueRole::kActivation && x.role != ValueRole::kResult) {
    fail(xi, "x " + value_label(p, xi) + " is a " + std::string(value_role_name(x.role)) + ", not an activation");
  }
  if (w.role != ValueRole::kWeight) fail(wi, "w " + value_label(p, wi) + " must be a weight");
  if (y.role != ValueRole::kResult) fail(yi, "output " + value_label(p, yi) + " must be a result value");
  if (bi != kNoValue && p.value(bi).role != ValueRole::kBias) fail(bi, "bias " + value_label(p, bi) + " must be a bias");

  // Ranks and dtypes the matmul contract defines.
  if (xl.shape.rank() != 2) return fail(xi, "x must be rank 2 [M, K], is " + xl.shape.to_string());
  if (wl.shape.rank() != 2) return fail(wi, "w must be rank 2 [N, K], is " + wl.shape.to_string());
  if (yl.shape.rank() != 2) return fail(yi, "output must be rank 2 [M, N], is " + yl.shape.to_string());
  if (xl.dtype != DType::kF32) fail(xi, "x must be f32, is " + std::string(dtype_name(xl.dtype)));
  if (yl.dtype != DType::kF32) fail(yi, "output must be f32, is " + std::string(dtype_name(yl.dtype)));
  if (wl.dtype == DType::kI8 || wl.dtype == DType::kI16 || wl.dtype == DType::kI32 || !dequant_supported(wl.dtype)) {
    fail(wi, "weight dtype " + std::string(dtype_name(wl.dtype)) + " has no defined dequantization");
  }
  // Weights are read as whole dense rows (quantized blocks in GGUF order).
  if (!wl.is_contiguous()) fail(wi, "w must be dense row-major (" + layout_text(wl) + ")");

  // Shapes.
  const int64_t m = xl.shape[0], k = xl.shape[1], n = wl.shape[0];
  if (wl.shape[1] != k) {
    fail(wi, "x has " + std::to_string(k) + " columns but w has " + std::to_string(wl.shape[1]) + " (K mismatch)");
  }
  if (yl.shape[0] != m || yl.shape[1] != n) {
    fail(yi, "output is " + yl.shape.to_string() + ", expected [" + std::to_string(m) + ", " + std::to_string(n) + "]");
  }
  if (a->m != m || a->n != n || a->k != k) {
    fail(kNoValue, "attributes M,N,K = " + std::to_string(a->m) + "," + std::to_string(a->n) + "," +
                       std::to_string(a->k) + " disagree with operands " + std::to_string(m) + "," +
                       std::to_string(n) + "," + std::to_string(k));
  }
  if (bi != kNoValue) {
    const TensorLayout& bl = p.value(bi).layout;
    if (bl.dtype != DType::kF32 || bl.shape.rank() != 1 || bl.shape[0] != n || !bl.is_contiguous()) {
      fail(bi, "bias must be a dense f32 [" + std::to_string(n) + "] vector, is " + layout_text(bl));
    }
  }
  if (a->accum != AccumType::kF32) fail(kNoValue, "accumulation type must be f32");
  if (static_cast<int>(a->act) < 0 || a->act > ActFormat::kInt16) fail(kNoValue, "invalid activation format");
  // No in-place matmul: the destination may not be an operand.
  for (ValueId in : op.inputs) {
    if (in == yi) fail(yi, "output " + value_label(p, yi) + " is also an operand");
  }
}

}  // namespace

std::vector<Diagnostic> verify_all(const Program& p) {
  std::vector<Diagnostic> out;
  for (const Value& v : p.values()) check_value(v, out);
  // A value is available once it is a leaf, or after the op that writes it.
  std::vector<int32_t> producer(p.values().size(), -1);
  for (int32_t i = 0; i < static_cast<int32_t>(p.ops().size()); ++i) {
    const Op& op = p.ops()[static_cast<size_t>(i)];
    auto fail = [&](ValueId v, std::string m) { out.push_back({i, v, op_label(p, i) + ": " + m}); };
    if (op.kind != OpKind::kMatMul) {
      if (op.kind < OpKind::kCount) {
        out.push_back({i, kNoValue, op_label(p, i) + ": op kind is declared but not defined in DynaKernel v0", true});
        // Its destination counts as written, so the one real problem is not
        // buried under "result is never written" for the same op.
        if (p.has_value(op.output) && producer[static_cast<size_t>(op.output)] < 0) {
          producer[static_cast<size_t>(op.output)] = i;
        }
      } else {
        fail(kNoValue, "invalid op kind " + std::to_string(static_cast<int>(op.kind)));
      }
      continue;
    }
    bool refs_ok = true;
    for (size_t j = 0; j < op.inputs.size(); ++j) {
      const ValueId in = op.inputs[j];
      if (!p.has_value(in)) {
        fail(in, "operand " + std::to_string(j) + " references undefined value " + value_label(p, in));
        refs_ok = false;
      } else if (p.value(in).role == ValueRole::kResult && producer[static_cast<size_t>(in)] < 0) {
        fail(in, "operand " + std::to_string(j) + " " + value_label(p, in) + " is read before any op writes it");
        refs_ok = false;
      }
    }
    if (!p.has_value(op.output)) {
      fail(op.output, "output references undefined value " + value_label(p, op.output));
      refs_ok = false;
    } else if (producer[static_cast<size_t>(op.output)] >= 0) {
      fail(op.output, "output " + value_label(p, op.output) + " is already written by op " +
                          std::to_string(producer[static_cast<size_t>(op.output)]));
      refs_ok = false;
    }
    if (refs_ok) {
      check_matmul(p, i, out);
      producer[static_cast<size_t>(op.output)] = i;
    }
  }
  // Results nothing writes would be read as garbage by a caller.
  for (const Value& v : p.values()) {
    if (v.role == ValueRole::kResult && producer[static_cast<size_t>(v.id)] < 0) {
      bool reported = false;
      for (const Diagnostic& d : out) reported = reported || d.value == v.id;
      if (!reported) out.push_back({-1, v.id, "value " + value_label(p, v.id) + ": result is never written"});
    }
  }
  return out;
}

Status verify(const Program& p) {
  const std::vector<Diagnostic> d = verify_all(p);
  if (d.empty()) return Status::Ok();
  const bool only_kind = std::all_of(d.begin(), d.end(), [](const Diagnostic& x) { return x.unsupported; });
  std::string m = d.front().message;
  if (d.size() > 1) m += " (+" + std::to_string(d.size() - 1) + " more)";
  return only_kind ? Unsupported(m) : InvalidArgument(m);
}

std::string to_text(const Program& p) {
  std::string s;
  for (const Value& v : p.values()) {
    s += "%" + std::to_string(v.id) + " = " + std::string(value_role_name(v.role)) + " " + layout_text(v.layout);
    if (v.alignment) s += " align(" + std::to_string(v.alignment) + ")";
    if (!v.name.empty()) s += " \"" + v.name + "\"";
    s += '\n';
  }
  for (size_t i = 0; i < p.ops().size(); ++i) {
    const Op& op = p.ops()[i];
    s += std::string(op_kind_name(op.kind)) + " %" + std::to_string(op.output) + " <-";
    for (ValueId in : op.inputs) s += " %" + std::to_string(in);
    if (const auto* a = std::get_if<MatMulAttrs>(&op.attrs)) {
      s += " {m=" + std::to_string(a->m) + " n=" + std::to_string(a->n) + " k=" + std::to_string(a->k) +
           " accum=" + std::string(accum_type_name(a->accum)) + " act=" + std::string(act_format_name(a->act)) + "}";
    }
    if (!op.name.empty()) s += " \"" + op.name + "\"";
    s += '\n';
  }
  return s;
}

// --- builder ------------------------------------------------------------------------

namespace {

Result<TensorLayout> dense(DType dtype, int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0) {
    return InvalidArgument("non-positive shape [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
  }
  return TensorLayout::contiguous(dtype, TensorShape{rows, cols});
}

}  // namespace

ValueId ProgramBuilder::value(std::string name, ValueRole role, TensorLayout layout, int64_t alignment) {
  Value v;
  v.name = std::move(name);
  v.role = role;
  v.layout = layout;
  v.alignment = alignment;
  return p_.add_value(std::move(v));
}

ValueId ProgramBuilder::activation(std::string name, DType dtype, int64_t rows, int64_t cols, int64_t alignment) {
  auto l = dense(dtype, rows, cols);
  if (!l.ok()) {
    if (status_.ok()) status_ = InvalidArgument("activation '" + name + "': " + l.status().message());
    return kNoValue;
  }
  return value(std::move(name), ValueRole::kActivation, *l, alignment);
}

ValueId ProgramBuilder::weight(std::string name, DType dtype, int64_t rows, int64_t cols, int64_t alignment) {
  auto l = dense(dtype, rows, cols);
  if (!l.ok()) {
    if (status_.ok()) status_ = InvalidArgument("weight '" + name + "': " + l.status().message());
    return kNoValue;
  }
  return value(std::move(name), ValueRole::kWeight, *l, alignment);
}

ValueId ProgramBuilder::bias(std::string name, int64_t n) {
  auto l = n > 0 ? TensorLayout::contiguous(DType::kF32, TensorShape{n})
                 : Result<TensorLayout>(InvalidArgument("non-positive bias length"));
  if (!l.ok()) {
    if (status_.ok()) status_ = InvalidArgument("bias '" + name + "': " + l.status().message());
    return kNoValue;
  }
  return value(std::move(name), ValueRole::kBias, *l);
}

void ProgramBuilder::matmul_into(ValueId x, ValueId w, ValueId bias, ValueId y, MatMulOptions opts) {
  Op op;
  op.kind = OpKind::kMatMul;
  op.name = std::move(opts.name);
  op.inputs = {x, w};
  if (bias != kNoValue) op.inputs.push_back(bias);
  op.output = y;
  MatMulAttrs a;
  // Sizes from the operands when they exist; verify() reports the rest.
  if (p_.has_value(x) && p_.value(x).layout.shape.rank() == 2) {
    a.m = rows_of(p_.value(x).layout);
    a.k = cols_of(p_.value(x).layout);
  }
  if (p_.has_value(w) && p_.value(w).layout.shape.rank() == 2) a.n = rows_of(p_.value(w).layout);
  a.accum = opts.accum;
  a.act = opts.act;
  op.attrs = a;
  p_.add_op(std::move(op));
}

ValueId ProgramBuilder::matmul(ValueId x, ValueId w, ValueId bias, MatMulOptions opts) {
  if (!p_.has_value(x) || !p_.has_value(w)) {
    if (status_.ok()) status_ = InvalidArgument("matmul '" + opts.name + "': unknown operand");
    return kNoValue;
  }
  const int64_t m = rows_of(p_.value(x).layout), n = rows_of(p_.value(w).layout);
  auto l = dense(DType::kF32, m, n);
  if (!l.ok()) {
    if (status_.ok()) status_ = InvalidArgument("matmul '" + opts.name + "': " + l.status().message());
    return kNoValue;
  }
  const ValueId y = value(opts.name.empty() ? std::string() : opts.name + ".out", ValueRole::kResult, *l);
  matmul_into(x, w, bias, y, std::move(opts));
  return y;
}

Result<Program> ProgramBuilder::finish() {
  if (!status_.ok()) return status_;
  ENGINE_RETURN_IF_ERROR(verify(p_));
  return std::move(p_);
}

}  // namespace dynacore::dk
