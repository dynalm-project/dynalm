// DynaKernel INT8 Q4_K matmul -> textual LLVM IR (llvm_jit.h). No LLVM
// dependency: this file only writes text.
//
// The emitted function mirrors the generated AVX2 C++ kernel (codegen.cpp,
// q4k_sx_x8) operation for operation in its floating-point part, so results
// are bit-identical; the integer dot product is expressed per lowering:
//   portable: zext(w) * sext(x) in i16, summed over 8 chunks, pairs added (=
//             maddubs without saturation, which the invariants rule out), then
//             sext to i32, times sc * m, pairs added (= madd);
//   avx2:     llvm.x86.avx2.pmadd.ub.sw / llvm.x86.avx2.pmadd.wd;
//   avxvnni:  llvm.x86.avx512.vpdpbusd.256 (32 products per int32 lane), then
//             times sc * m in int32.
// All three give the same exact int32 per half super-block:
//   |lane| <= 32 * 15 * 127 * 63 * 128 per sub-block, 4 sub-blocks < 1.97e9.
//
// Structure: a loop over 8-row groups [g0, g1), an inner loop over the K / 256
// super-blocks carrying M accumulators (<8 x float>, lane = weight row), and a
// fully unrolled body (2 halves x 2 sub-block pairs x 8 chunks x M rows).

#include <cstddef>
#include <string>
#include <vector>

#include "dynacore/cpu/cpu_kernels.h"
#include "dynacore/dynakernel/llvm_jit.h"
#include "dynacore/quantization/repack.h"

namespace dynacore::dk {

namespace {

constexpr int64_t kABytes = sizeof(ActSuperQ8);
constexpr int64_t kOffQ = offsetof(ActSuperQ8, q);
constexpr int64_t kOffU = offsetof(ActSuperQ8, u);
constexpr int64_t kOffM = offsetof(ActSuperQ8, m);
constexpr int64_t kOffMs = offsetof(ActSuperQ8, ms);
static_assert(sizeof(BlockQ4_Kx8) == 1216 && kABytes == 320);

// "<i8 15, i8 15, ...>" (n lanes).
std::string splat_const(const char* elt, int n, long long v) {
  std::string s = "<";
  for (int i = 0; i < n; ++i) s += (i ? ", " : "") + std::string(elt) + " " + std::to_string(v);
  return s + ">";
}

// "<n x i32> <i32 a, i32 b, ...>" shuffle mask.
std::string mask(const std::vector<int>& idx) {
  std::string s = "<" + std::to_string(idx.size()) + " x i32> <";
  for (size_t i = 0; i < idx.size(); ++i) s += (i ? ", i32 " : "i32 ") + std::to_string(idx[i]);
  return s + ">";
}

std::vector<int> stride_mask(int n, int first, int step) {
  std::vector<int> m(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) m[static_cast<size_t>(i)] = first + step * i;
  return m;
}

// SSA value writer for one function body.
class Ir {
 public:
  explicit Ir(std::string& out) : t_(out) {}
  std::string v(const std::string& rhs) {
    std::string n = "%v" + std::to_string(next_++);
    t_ += "  " + n + " = " + rhs + "\n";
    return n;
  }
  void line(const std::string& s) { t_ += s + "\n"; }
  std::string gep(const std::string& base, const std::string& off) {
    return v("getelementptr inbounds i8, ptr " + base + ", i64 " + off);
  }
  std::string gep(const std::string& base, int64_t off) { return off == 0 ? base : gep(base, std::to_string(off)); }
  std::string load(const std::string& ty, const std::string& p, int align) {
    return v("load " + ty + ", ptr " + p + ", align " + std::to_string(align));
  }
  std::string splat(const std::string& elt, int n, const std::string& scalar) {
    const std::string vt = "<" + std::to_string(n) + " x " + elt + ">";
    const std::string a = v("insertelement " + vt + " poison, " + elt + " " + scalar + ", i64 0");
    return v("shufflevector " + vt + " " + a + ", " + vt + " poison, <" + std::to_string(n) + " x i32> zeroinitializer");
  }
  std::string fma(const std::string& a, const std::string& b, const std::string& c) {
    return v("call <8 x float> @llvm.fma.v8f32(<8 x float> " + a + ", <8 x float> " + b + ", <8 x float> " + c + ")");
  }

 private:
  std::string& t_;
  int next_ = 0;
};

const std::string kV8F = "<8 x float>", kV8I = "<8 x i32>", kV16S = "<16 x i16>", kV32B = "<32 x i8>",
                  kV32S = "<32 x i16>", kV16I = "<16 x i32>";

}  // namespace

std::string_view llvm_lowering_name(LlvmLowering l) {
  switch (l) {
    case LlvmLowering::kPortable: return "portable";
    case LlvmLowering::kAvx2: return "avx2";
    case LlvmLowering::kAvxVnni: return "avxvnni";
  }
  return "?";
}

std::string llvm_kernel_symbol(const Q4KMatmulSpec& s, LlvmLowering l, const LlvmKernelOptions& o) {
  return "dk_llvm_q4k_sx8_" + std::string(llvm_lowering_name(l)) + "_m" + std::to_string(s.m) + "_n" +
         std::to_string(s.n) + "_k" + std::to_string(s.k) + "_b" + (s.bias ? "1" : "0") +
         (o.prefetch_bytes > 0 ? "_pf" + std::to_string(o.prefetch_bytes) : "") +
         (o.layout != PackedLayout::kX8 ? "_l" + std::string(packed_layout_name(o.layout)) : "");
}

Result<std::string> emit_q4k_int8_llvm_ir(const Q4KMatmulSpec& s, LlvmLowering l, const LlvmKernelOptions& o) {
  ENGINE_RETURN_IF_ERROR(validate_int8_spec(s));
  if (l != LlvmLowering::kPortable && l != LlvmLowering::kAvx2 && l != LlvmLowering::kAvxVnni) {
    return InvalidArgument("llvm ir: unknown lowering");
  }
  if (o.prefetch_bytes < 0 || o.prefetch_bytes % 64 != 0 || o.prefetch_bytes > (1 << 20)) {
    return InvalidArgument("llvm ir: prefetch_bytes must be a multiple of 64 in [0, 1 MiB]");
  }
  if (!valid_packed_layout(o.layout)) return InvalidArgument("llvm ir: unknown packed layout");
  // Weight block layout (BlockQ4_Kx8 or a compact variant, compact_layout.h).
  const PackedLayoutInfo li = packed_layout_info(o.layout);
  const int64_t kWBytes = li.block_bytes;
  const std::string sym = llvm_kernel_symbol(s, l, o);
  const int M = s.m;
  const int64_t kSb = s.k / 256;
  const std::string c15 = splat_const("i8", 32, 15), c4 = splat_const("i8", 32, 4);

  std::string t;
  t += "; Generated by DynaKernel (llvm_ir.cpp). Do not edit.\n";
  t += "; y[M,N] = x[M,K] . w[N,K]^T (+ bias); x sub-scaled int8 (ActSuperQ8), w " +
       (o.layout == PackedLayout::kX8 ? std::string("BlockQ4_Kx8") : "Q4_K x 8 layout " + std::string(packed_layout_name(o.layout))) +
       ".\n";
  t += "; m=" + std::to_string(M) + " n=" + std::to_string(s.n) + " k=" + std::to_string(s.k) +
       " bias=" + (s.bias ? "1" : "0") + " lowering=" + std::string(llvm_lowering_name(l)) + "\n";
  t += "source_filename = \"dynakernel\"\n\n";
  t += "@" + sym + ".count = internal global i64 0, align 8\n\n";
  t += "declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)\n";
  if (l == LlvmLowering::kAvx2) {
    t += "declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)\n";
    t += "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n";
  }
  if (l == LlvmLowering::kAvxVnni) t += "declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>)\n";
  if (o.prefetch_bytes > 0) t += "declare void @llvm.prefetch.p0(ptr, i32, i32, i32)\n";
  t += "\ndefine i64 @" + sym + "_calls() nounwind {\n";
  t += "  %c = load atomic i64, ptr @" + sym + ".count monotonic, align 8\n";
  t += "  ret i64 %c\n}\n\n";

  // Loop body first (its final accumulator names feed the loop phis).
  std::string body;
  Ir ir(body);
  const std::string poff = ir.v("mul i64 %i, " + std::to_string(kWBytes));
  const std::string p = ir.gep("%wg", poff);
  if (o.prefetch_bytes > 0) {  // the weight block prefetch_bytes ahead in the packed stream
    for (int64_t line = 0; line < kWBytes; line += 64) {
      ir.line("  call void @llvm.prefetch.p0(ptr " + ir.gep(p, o.prefetch_bytes + line) + ", i32 0, i32 3, i32 1)");
    }
  }
  auto load_scale8 = [&](int64_t off) {  // 8 fp32 values, stored as fp32 or fp16 (exact widening)
    if (!li.fp16) return ir.load(kV8F, ir.gep(p, off), 4);
    return ir.v("fpext <8 x half> " + ir.load("<8 x half>", ir.gep(p, off), 2) + " to " + kV8F);
  };
  const std::string dv = load_scale8(li.off_d);
  const std::string dminv = load_scale8(li.off_dmin);
  // 6-bit scale (is_min = false) or minimum of sub-block j for the 8 rows, as <8 x i8>.
  auto scale_or_min8 = [&](int j, bool is_min) {
    if (!li.scales6) return ir.load("<8 x i8>", ir.gep(p, (is_min ? li.off_mn : li.off_sc) + 8 * j), 1);
    auto sbyte = [&](int b) { return ir.load("<8 x i8>", ir.gep(p, li.off_scales + 8 * b), 1); };
    const std::string c63 = splat_const("i8", 8, 63), c15b = splat_const("i8", 8, 15), c6 = splat_const("i8", 8, 6),
                      c4b = splat_const("i8", 8, 4);
    if (j < 4) return ir.v("and <8 x i8> " + sbyte(is_min ? j + 4 : j) + ", " + c63);
    const std::string lo = is_min ? ir.v("lshr <8 x i8> " + sbyte(j + 4) + ", " + c4b)
                                  : ir.v("and <8 x i8> " + sbyte(j + 4) + ", " + c15b);
    const std::string hi = ir.v("shl <8 x i8> " + ir.v("lshr <8 x i8> " + sbyte(is_min ? j : j - 4) + ", " + c6) + ", " + c4b);
    return ir.v("or <8 x i8> " + lo + ", " + hi);
  };
  std::vector<std::string> a(static_cast<size_t>(M)), acc(static_cast<size_t>(M));
  for (int r = 0; r < M; ++r) {
    const std::string idx = r == 0 ? "%i" : ir.v("add i64 %i, " + std::to_string(r * kSb));
    a[static_cast<size_t>(r)] = ir.gep("%x", ir.v("mul i64 " + idx + ", " + std::to_string(kABytes)));
    acc[static_cast<size_t>(r)] = "%acc" + std::to_string(r);
  }
  for (int half = 0; half < 2; ++half) {
    std::vector<std::string> sumi(static_cast<size_t>(M));
    for (int jj = 0; jj < 2; ++jj) {
      const int jp = 2 * half + jj;
      const int js[2] = {2 * jp, 2 * jp + 1};
      std::string wq[2][8];  // [low/high nibble][chunk]
      for (int c = 0; c < 8; ++c) {
        const std::string q = ir.load(kV32B, ir.gep(p, li.off_qs + 256 * jp + 32 * c), 1);
        wq[0][c] = ir.v("and " + kV32B + " " + q + ", " + c15);
        wq[1][c] = ir.v("lshr " + kV32B + " " + q + ", " + c4);
      }
      std::string scale[2], nmin[2];  // per sub-block: weight scales (lowering-specific), -(dmin * mn)
      for (int h = 0; h < 2; ++h) {
        const std::string sc8 = scale_or_min8(js[h], false);
        if (l == LlvmLowering::kAvxVnni) {
          scale[h] = ir.v("zext <8 x i8> " + sc8 + " to " + kV8I);
        } else {  // each row's scale twice: two int16 lanes per row
          const std::string dup = ir.v("shufflevector <8 x i8> " + sc8 + ", <8 x i8> poison, " +
                                       mask({0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7}));
          scale[h] = ir.v("zext <16 x i8> " + dup + " to " + kV16S);
        }
        const std::string mn8 = scale_or_min8(js[h], true);
        const std::string mnf = ir.v("uitofp <8 x i8> " + mn8 + " to " + kV8F);
        nmin[h] = ir.v("fneg " + kV8F + " " + ir.v("fmul " + kV8F + " " + mnf + ", " + dminv));
      }
      for (int r = 0; r < M; ++r) {
        const std::string& ar = a[static_cast<size_t>(r)];
        std::string part[2];
        for (int h = 0; h < 2; ++h) {
          const std::string xp = ir.gep(ar, kOffQ + 32 * js[h]);
          std::string dot;  // avx2: <16 x i16>; portable: <32 x i16>; vnni: <8 x i32>
          for (int c = 0; c < 8; ++c) {
            const std::string b = ir.load("i32", ir.gep(xp, 4 * c), 1);
            const std::string bv = ir.splat("i32", 8, b);  // 4 activation bytes in every lane
            const std::string& w = wq[h][c];
            if (l == LlvmLowering::kAvxVnni) {
              const std::string wv = ir.v("bitcast " + kV32B + " " + w + " to " + kV8I);
              dot = ir.v("call <8 x i32> @llvm.x86.avx512.vpdpbusd.256(<8 x i32> " + (c == 0 ? "zeroinitializer" : dot) +
                         ", <8 x i32> " + wv + ", <8 x i32> " + bv + ")");
            } else {
              const std::string bb = ir.v("bitcast " + kV8I + " " + bv + " to " + kV32B);
              std::string prod;
              if (l == LlvmLowering::kAvx2) {
                prod = ir.v("call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> " + w + ", <32 x i8> " + bb + ")");
              } else {
                const std::string wz = ir.v("zext " + kV32B + " " + w + " to " + kV32S);
                const std::string xs = ir.v("sext " + kV32B + " " + bb + " to " + kV32S);
                prod = ir.v("mul " + kV32S + " " + wz + ", " + xs);
              }
              dot = c == 0 ? prod : ir.v("add " + (l == LlvmLowering::kAvx2 ? kV16S : kV32S) + " " + dot + ", " + prod);
            }
          }
          const std::string m16 = ir.load("i16", ir.gep(ar, kOffM + 2 * js[h]), 2);
          if (l == LlvmLowering::kAvxVnni) {
            const std::string mv = ir.splat("i32", 8, ir.v("sext i16 " + m16 + " to i32"));
            const std::string sm = ir.v("mul " + kV8I + " " + scale[h] + ", " + mv);
            part[h] = ir.v("mul " + kV8I + " " + dot + ", " + sm);
          } else {
            const std::string sm = ir.v("mul " + kV16S + " " + scale[h] + ", " + ir.splat("i16", 16, m16));
            if (l == LlvmLowering::kAvx2) {
              part[h] = ir.v("call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> " + dot + ", <16 x i16> " + sm + ")");
            } else {
              const std::string ev = ir.v("shufflevector " + kV32S + " " + dot + ", " + kV32S + " poison, " + mask(stride_mask(16, 0, 2)));
              const std::string od = ir.v("shufflevector " + kV32S + " " + dot + ", " + kV32S + " poison, " + mask(stride_mask(16, 1, 2)));
              const std::string pair = ir.v("add " + kV16S + " " + ev + ", " + od);
              const std::string pe = ir.v("sext " + kV16S + " " + pair + " to " + kV16I);
              const std::string se = ir.v("sext " + kV16S + " " + sm + " to " + kV16I);
              const std::string pr = ir.v("mul " + kV16I + " " + pe + ", " + se);
              const std::string e2 = ir.v("shufflevector " + kV16I + " " + pr + ", " + kV16I + " poison, " + mask(stride_mask(8, 0, 2)));
              const std::string o2 = ir.v("shufflevector " + kV16I + " " + pr + ", " + kV16I + " poison, " + mask(stride_mask(8, 1, 2)));
              part[h] = ir.v("add " + kV8I + " " + e2 + ", " + o2);
            }
          }
        }
        const std::string pab = ir.v("add " + kV8I + " " + part[0] + ", " + part[1]);
        std::string& si = sumi[static_cast<size_t>(r)];
        si = si.empty() ? pab : ir.v("add " + kV8I + " " + si + ", " + pab);
        // Weight minimums: acc -= (dmin * mn_j) * ms_j, fused (the kernel's fnmadd).
        for (int h = 0; h < 2; ++h) {
          const std::string ms = ir.load("float", ir.gep(ar, kOffMs + 4 * js[h]), 4);
          acc[static_cast<size_t>(r)] = ir.fma(nmin[h], ir.splat("float", 8, ms), acc[static_cast<size_t>(r)]);
        }
      }
    }
    // End of a half super-block: exact int32 -> float once, times d * u.
    for (int r = 0; r < M; ++r) {
      const std::string u = ir.load("float", ir.gep(a[static_cast<size_t>(r)], kOffU), 4);
      const std::string du = ir.v("fmul " + kV8F + " " + dv + ", " + ir.splat("float", 8, u));
      const std::string sf = ir.v("sitofp " + kV8I + " " + sumi[static_cast<size_t>(r)] + " to " + kV8F);
      acc[static_cast<size_t>(r)] = ir.fma(sf, du, acc[static_cast<size_t>(r)]);
    }
  }
  ir.line("  %i.next = add nuw nsw i64 %i, 1");
  ir.line("  %i.done = icmp eq i64 %i.next, " + std::to_string(kSb));
  ir.line("  br i1 %i.done, label %store, label %sb");
  ir.line("store:");
  const std::string g8 = ir.v("mul i64 %g, 8");
  for (int r = 0; r < M; ++r) {
    const std::string off = r == 0 ? g8 : ir.v("add i64 " + ir.v("mul i64 %ys, " + std::to_string(r)) + ", " + g8);
    const std::string yp = ir.v("getelementptr inbounds float, ptr %y, i64 " + off);
    std::string val = acc[static_cast<size_t>(r)];
    if (s.bias) {
      const std::string bp = ir.v("getelementptr inbounds float, ptr %bias, i64 " + g8);
      val = ir.v("fadd " + kV8F + " " + val + ", " + ir.load(kV8F, bp, 4));
    }
    ir.line("  store " + kV8F + " " + val + ", ptr " + yp + ", align 4");
  }

  t += "define void @" + sym +
       "(ptr noalias readonly %w, ptr noalias readonly %x, ptr noalias readonly %bias, ptr noalias %y, i64 %ys, "
       "i64 %g0, i64 %g1) nounwind {\n";
  t += "entry:\n";
  t += "  %calls.old = atomicrmw add ptr @" + sym + ".count, i64 1 monotonic, align 8\n";
  t += "  %none = icmp sge i64 %g0, %g1\n";
  t += "  br i1 %none, label %exit, label %group\n";
  t += "group:\n";
  t += "  %g = phi i64 [ %g0, %entry ], [ %g.next, %store ]\n";
  t += "  %wg.off = mul i64 %g, " + std::to_string(kSb * kWBytes) + "\n";
  t += "  %wg = getelementptr inbounds i8, ptr %w, i64 %wg.off\n";
  t += "  br label %sb\n";
  t += "sb:\n";
  t += "  %i = phi i64 [ 0, %group ], [ %i.next, %sb ]\n";
  for (int r = 0; r < M; ++r) {
    t += "  %acc" + std::to_string(r) + " = phi <8 x float> [ zeroinitializer, %group ], [ " + acc[static_cast<size_t>(r)] +
         ", %sb ]\n";
  }
  t += body;
  t += "  %g.next = add i64 %g, 1\n";
  t += "  %g.done = icmp eq i64 %g.next, %g1\n";
  t += "  br i1 %g.done, label %exit, label %group\n";
  t += "exit:\n";
  t += "  ret void\n}\n";
  return t;
}

}  // namespace dynacore::dk
