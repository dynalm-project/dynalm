// GgmlDevice (GPU through pinned ggml, docs/gpu-backend.md): every op the
// device lowers, run as the engine runs it (RecordingDevice -> one graph per
// segment), against CpuDevice on the same data; graph reuse with changing
// dynamic inputs; the paged KV layout with several sequences and scattered
// block tables; and the failure paths (direct op calls, unsupported ops and
// dtypes, aliasing, allocation failure, buffer reuse after free). No model
// files. Skips (exit 0) when the build has no GPU backend or no GPU exists.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <tuple>
#include <random>
#include <string>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include "dynacore/attention/paged_kv.h"
#include "dynacore/cpu/cpu_device.h"
#include "dynacore/device/segment.h"
#include "dynacore/execution/thread_pool.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"
#include "dynacore/ir/recording_device.h"
#include "dynacore/memory/mapped_file.h"
#include "dynacore/quantization/quant_formats.h"
#include "dynacore/tensor/fp16.h"
#include "dynacore/tensor/tensor.h"

using namespace dynacore;

namespace {

int g_checks = 0;

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::_Exit(1); /* no static teardown with a live GPU device */                   \
    }                                                                                  \
  } while (0)

std::mt19937 g_rng(1234);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g_rng); }

TensorView view2d(void* data, int64_t rows, int64_t cols, int64_t stride_bytes, DType t = DType::kF32,
                  DeviceLoc loc = {}) {
  TensorLayout l;
  l.dtype = t;
  l.shape = TensorShape{rows, cols};
  l.strides = {stride_bytes, t == DType::kF32 ? 4 : 0, 0, 0};
  return TensorView(data, l, loc);
}

struct Env {
  ThreadPool pool{4};
  CpuDevice cpu{pool, select_best_isa(cpu_info().features)};
  std::unique_ptr<Device> gpu;
  std::unique_ptr<ir::RecordingDevice> rec;
  std::vector<Tensor> keep;  // CPU and GPU tensors alive for the test
  // ggml's CPU backend evaluates GELU from an fp16 lookup table (~6e-4).
  double act_tol = 1e-4;

  SegmentStats stats() { return gpu->segment_executor()->segment_stats(); }

  // A host tensor and its GPU copy.
  std::pair<TensorView, TensorView> weight(DType t, const std::vector<std::byte>& bytes, int64_t rows, int64_t cols) {
    TensorLayout l = *TensorLayout::contiguous(t, rows > 0 ? TensorShape{rows, cols} : TensorShape{cols});
    auto host = Tensor::empty(t, l.shape);
    CHECK(host.ok());
    std::memcpy(host->data(), bytes.data(), bytes.size());
    auto dev = rec->upload(*host);
    CHECK(dev.ok());
    keep.push_back(*host);
    keep.push_back(*dev);
    return {host->view(), dev->view()};
  }
  // fp32 [rows, cols] scratch on both devices; GPU initialized from `init`.
  std::pair<TensorView, TensorView> act(int64_t rows, int64_t cols, const std::vector<float>* init = nullptr) {
    auto host = Tensor::zeros(DType::kF32, {rows, cols});
    CHECK(host.ok());
    if (init) std::memcpy(host->data(), init->data(), init->size() * 4);
    auto dev = rec->upload(*host);
    CHECK(dev.ok());
    keep.push_back(*host);
    keep.push_back(*dev);
    return {host->view(), dev->view()};
  }
  std::vector<float> download(const TensorView& v) {
    const int64_t rows = v.rank() > 1 ? v.dim(0) : 1, cols = v.dim(v.rank() - 1);
    std::vector<float> out(static_cast<size_t>(rows * cols));
    rec->download(v, out);
    if (Status st = rec->take_error(); !st.ok()) std::fprintf(stderr, "device error: %s\n", st.to_string().c_str());
    CHECK(rec->take_error().ok());
    return out;
  }
};

std::vector<float> host_rows(const TensorView& v) {
  const int64_t rows = v.rank() > 1 ? v.dim(0) : 1, cols = v.dim(v.rank() - 1);
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    std::memcpy(out.data() + r * cols, static_cast<const std::byte*>(v.data()) + r * (v.rank() > 1 ? v.stride(0) : 0),
                static_cast<size_t>(cols) * 4);
  }
  return out;
}

std::vector<float> randv(size_t n, float lo = -1, float hi = 1) {
  std::vector<float> v(n);
  for (float& x : v) x = uni(lo, hi);
  return v;
}

// max |a - b| / max(1e-6, max |a|)
double rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
  CHECK(ref.size() == got.size());
  double mx = 1e-6, d = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(std::isfinite(got[i]));
    mx = std::max(mx, std::abs(static_cast<double>(ref[i])));
    d = std::max(d, std::abs(static_cast<double>(ref[i]) - got[i]));
  }
  return d / mx;
}

void expect_close(const char* what, const std::vector<float>& ref, const std::vector<float>& got, double tol) {
  const double e = rel_err(ref, got);
  if (e > tol) std::fprintf(stderr, "%s: relative error %.3g > %.3g\n", what, e, tol);
  CHECK(e <= tol);
}

// Weight bytes for [n, k] of type t (random, valid scales).
std::vector<std::byte> make_weight(DType t, int64_t n, int64_t k) {
  const int64_t bytes = dtype_row_bytes(t, k) * n;
  std::vector<std::byte> out(static_cast<size_t>(bytes));
  if (t == DType::kF32) {
    auto v = randv(static_cast<size_t>(n * k), -0.2f, 0.2f);
    std::memcpy(out.data(), v.data(), out.size());
  } else if (t == DType::kF16) {
    auto* h = reinterpret_cast<uint16_t*>(out.data());
    for (int64_t i = 0; i < n * k; ++i) h[i] = fp32_to_fp16(uni(-0.2f, 0.2f));
  } else if (t == DType::kQ8_0) {
    auto v = randv(static_cast<size_t>(n * k), -0.2f, 0.2f);
    quant::quantize_q8_0(v.data(), reinterpret_cast<quant::BlockQ8_0*>(out.data()), n * k / 32);
  } else {  // K-quants: random bytes, small fp16 scales at the block's known offsets
    for (auto& b : out) b = static_cast<std::byte>(g_rng() & 0xFF);
    const int32_t bb = dtype_block_bytes(t);
    for (int64_t off = 0; off < bytes; off += bb) {
      auto put = [&](int64_t at, float f) {
        const uint16_t h = fp32_to_fp16(f);
        std::memcpy(out.data() + off + at, &h, 2);
      };
      if (t == DType::kQ4_K) {
        put(0, uni(1e-3f, 4e-3f));  // d
        put(2, uni(1e-3f, 4e-3f));  // dmin
      } else if (t == DType::kQ6_K) {
        put(bb - 2, uni(1e-4f, 4e-4f));  // d (last field)
      } else if (t == DType::kIQ4_NL) {
        put(0, uni(1e-3f, 4e-3f));  // d
      } else if (t == DType::kIQ4_XS) {
        put(0, uni(1e-4f, 3e-4f));  // d (times a 6-bit sub-block scale - 32)
      }
    }
  }
  return out;
}

void test_matmul(Env& e) {
  const int64_t k = 512, n = 96;
  // IQ4_NL / IQ4_XS: DynaCore's dequantization against ggml's own kernels.
  for (DType t : {DType::kF32, DType::kF16, DType::kQ8_0, DType::kQ4_K, DType::kQ6_K, DType::kIQ4_NL, DType::kIQ4_XS}) {
    auto [wc, wg] = e.weight(t, make_weight(t, n, k), n, k);
    auto bias = randv(static_cast<size_t>(n));
    std::vector<std::byte> bb(bias.size() * 4);
    std::memcpy(bb.data(), bias.data(), bb.size());
    auto [bc, bgpu] = e.weight(DType::kF32, bb, 0, n);
    // Second pass with the weight prepacked: on ggml-cpu, ggml's repacked
    // (interleaved int8) kernels for Q4_K / Q6_K / Q8_0; elsewhere a no-op.
    for (int pass = 0; pass < 2; ++pass)
    for (int64_t m : {1, 5, 33}) {
      if (pass == 1 && m == 1) e.rec->prepack_weight(wg);
      // x is a column slice (strided rows) of a wider buffer: a strided leaf.
      auto xs = randv(static_cast<size_t>(m * (k + 64)));
      auto [xc, xg] = e.act(m, k + 64, &xs);
      const TensorView xcv = view2d(static_cast<float*>(xc.data()) + 64, m, k, (k + 64) * 4);
      const TensorView xgv = view2d(static_cast<float*>(xg.data()) + 64, m, k, (k + 64) * 4, DType::kF32, xg.device());
      auto [yc, yg] = e.act(m, n);
      e.cpu.matmul(xcv, wc, &bc, yc);
      e.rec->matmul(xgv, wg, &bgpu, yg);
      // GPU mat-mat kernels may round activations to half precision: 1e-2 of the output scale.
      expect_close("matmul", host_rows(yc), e.download(yg), 1e-2);
    }
  }
  // ggml-cpu repacks the K-quant weights it has interleaved kernels for.
  if (e.gpu->device().type == DeviceType::kCpu) CHECK(e.gpu->prepack_stats().tensors > 0);
  std::printf("  matmul: f32 f16 q8_0 q4_K q6_K iq4_nl iq4_xs, M 1/5/33, strided x, bias; again prepacked (%lld repacked)\n",
              static_cast<long long>(e.gpu->prepack_stats().tensors));
}

void test_norms_elementwise(Env& e) {
  const int64_t m = 7, d = 256;
  auto x = randv(static_cast<size_t>(m * d), -3, 3), x2 = randv(static_cast<size_t>(m * d));
  auto w = randv(static_cast<size_t>(d), 0.5f, 1.5f), b = randv(static_cast<size_t>(d));
  std::vector<std::byte> wb(w.size() * 4), bbytes(b.size() * 4);
  std::memcpy(wb.data(), w.data(), wb.size());
  std::memcpy(bbytes.data(), b.data(), bbytes.size());
  auto [wc, wg] = e.weight(DType::kF32, wb, 0, d);
  auto [bc, bgv] = e.weight(DType::kF32, bbytes, 0, d);
  auto [xc, xg] = e.act(m, d, &x);
  auto [x2c, x2g] = e.act(m, d, &x2);
  auto [yc, yg] = e.act(m, d);
  e.cpu.rms_norm(xc, wc, 1e-6f, yc);
  e.rec->rms_norm(xg, wg, 1e-6f, yg);
  expect_close("rms_norm", host_rows(yc), e.download(yg), 1e-5);
  e.cpu.layer_norm(xc, wc, &bc, 1e-5f, yc);
  e.rec->layer_norm(xg, wg, &bgv, 1e-5f, yg);
  expect_close("layer_norm", host_rows(yc), e.download(yg), 1e-5);
  for (Activation a : {Activation::kSilu, Activation::kGelu, Activation::kGeluTanh}) {
    e.cpu.act_mul(a, xc, x2c, yc);
    e.rec->act_mul(a, xg, x2g, yg);
    expect_close("act_mul", host_rows(yc), e.download(yg), e.act_tol);
    e.cpu.activation(a, xc, yc);
    e.rec->activation(a, xg, yg);
    expect_close("activation", host_rows(yc), e.download(yg), e.act_tol);
  }
  // add, then in-place scale / softcap / fill on the result, one segment.
  e.cpu.add(xc, x2c, yc);
  e.cpu.scale(yc, 0.5f);
  e.cpu.softcap(yc, 1.5f);
  e.rec->add(xg, x2g, yg);
  e.rec->scale(yg, 0.5f);
  e.rec->softcap(yg, 1.5f);
  expect_close("add+scale+softcap", host_rows(yc), e.download(yg), 1e-5);
  e.cpu.fill(yc, 3.25f);
  e.rec->fill(yg, 3.25f);
  expect_close("fill", host_rows(yc), e.download(yg), 0);
  std::printf("  rms_norm, layer_norm, act_mul/activation (silu gelu gelu_tanh), add scale softcap fill\n");
}

void test_rope(Env& e) {
  const int32_t heads = 4, hd = 64;
  const int64_t m = 6;
  std::vector<int32_t> pos = {0, 1, 7, 100, 513, 2047};
  std::vector<float> factors(hd / 2);  // dim / 2 are read: enough for every case
  for (float& f : factors) f = uni(1.0f, 4.0f);
  // attn: LongRoPE's attention factor (Phi-3/4) scales the rotated dims only.
  struct Case { RopeStyle style; int32_t dim; RopeScaling sc; bool freq; float attn; };
  for (const Case cs : {Case{RopeStyle::kHalfSplit, 64, RopeScaling::kNone, false, 1.0f},
                        Case{RopeStyle::kInterleaved, 64, RopeScaling::kNone, false, 1.0f},
                        Case{RopeStyle::kHalfSplit, 32, RopeScaling::kNone, true, 1.0f},
                        Case{RopeStyle::kHalfSplit, 64, RopeScaling::kLinear, false, 1.0f},
                        Case{RopeStyle::kHalfSplit, 48, RopeScaling::kNone, true, 1.0f},
                        Case{RopeStyle::kHalfSplit, 64, RopeScaling::kNone, false, 1.19f},
                        Case{RopeStyle::kHalfSplit, 48, RopeScaling::kNone, true, 1.19f}}) {
    RopeConfig rc;
    rc.style = cs.style;
    rc.dim = cs.dim;
    rc.freq_base = 1000000.0f;
    rc.scaling = cs.sc;
    rc.scaling_factor = 4.0f;
    rc.attn_factor = cs.attn;
    auto x = randv(static_cast<size_t>(m * heads * hd));
    auto [xc, xg] = e.act(m, heads * hd, &x);
    e.cpu.rope(xc, heads, hd, pos, rc, cs.freq ? factors.data() : nullptr);
    e.rec->rope(xg, heads, hd, pos, rc, cs.freq ? factors.data() : nullptr);
    const std::vector<float> yc = host_rows(xc);
    expect_close("rope", yc, e.download(xg), 2e-4);
    // Position 0 rotates by nothing: the rotated dims are x * attn, the rest x.
    for (int32_t h = 0; h < heads; ++h) {
      for (int32_t d = 0; d < hd; ++d) {
        const size_t i = static_cast<size_t>(h * hd + d);
        CHECK(std::fabs(yc[i] - x[i] * (d < cs.dim ? cs.attn : 1.0f)) <= 1e-6f * (1.0f + std::fabs(x[i])));
      }
    }
  }
  std::printf("  rope: half-split, interleaved, partial dims + frequency factors, linear scaling, attention factor\n");
}

void test_embedding_gather(Env& e) {
  const int64_t vocab = 50, d = 128;
  for (DType t : {DType::kF16, DType::kQ8_0, DType::kQ6_K}) {
    auto [tc, tg] = e.weight(t, make_weight(t, vocab, t == DType::kQ6_K ? 256 : d), vocab, t == DType::kQ6_K ? 256 : d);
    const int64_t cols = t == DType::kQ6_K ? 256 : d;
    std::vector<int32_t> ids = {3, 0, 49, 3, 17};
    auto [yc, yg] = e.act(static_cast<int64_t>(ids.size()), cols);
    e.cpu.embedding(tc, ids, yc);
    e.rec->embedding(tg, ids, yg);
    expect_close("embedding", host_rows(yc), e.download(yg), 1e-6);
  }
  auto src = randv(8 * 32);
  auto [sc, sg] = e.act(8, 32, &src);
  auto [dc, dg] = e.act(3, 32);
  std::vector<int32_t> rows = {7, 0, 3};
  e.cpu.gather_rows(sc, rows, dc);
  e.rec->gather_rows(sg, rows, dg);
  expect_close("gather_rows", host_rows(dc), e.download(dg), 0);
  std::printf("  embedding (f16 q8_0 q6_K tables), gather_rows\n");
}

// Paged KV: three sequences with scattered physical blocks; a prefill-like
// step then two decode steps, kv_store + attention, f16 and f32 KV, GQA,
// sliding window and softcap. Two decode steps share one prepared graph with
// different positions (dynamic inputs must not be stale).
void test_paged_attention(Env& e) {
  for (KvLayout gpu_layout : {KvLayout::kSlotMajor, KvLayout::kHeadMajor}) {
  for (DType kvt : {DType::kF16, DType::kF32, DType::kQ8_0}) {
    for (int variant = 0; variant < 2; ++variant) {
      KvGeometry g;
      g.num_layers = 1;
      g.num_kv_heads = 2;
      g.head_dim = g.head_dim_v = 64;
      g.block_size = 16;
      g.num_blocks = 12;
      g.dtype = kvt;
      const int32_t heads = 4;
      const auto bytes = static_cast<size_t>(g.bytes_per_layer() / 2);
      auto kc = Tensor::zeros(DType::kI8, {static_cast<int64_t>(bytes)}), vc = Tensor::zeros(DType::kI8, {static_cast<int64_t>(bytes)});
      auto kg = e.rec->allocate(bytes), vg = e.rec->allocate(bytes);
      CHECK(kc.ok() && vc.ok() && kg.ok() && vg.ok());
      e.keep.push_back(*kc);
      e.keep.push_back(*vc);
      std::vector<std::vector<int32_t>> tables = {{5, 2, 9}, {0}, {7, 1, 3, 11}};
      // The CPU reference is head-major; the GPU pool uses gpu_layout over the
      // same block tables (same blocks, different order inside a block).
      KvGeometry gg = g;
      gg.layout = gpu_layout;
      auto views = [&](void* k, void* v, const KvGeometry* geo) {
        std::vector<KvLayerView> out;
        for (auto& t : tables) out.push_back(KvLayerView{k, v, geo, t});
        return out;
      };
      const auto cv = views(kc->data(), vc->data(), &g);
      const auto gv = views((*kg)->data(), (*vg)->data(), &gg);
      AttentionParams ap;
      ap.num_heads = heads;
      ap.scale = 0.125f;
      ap.softcap = variant == 1 ? 20.0f : 0.0f;
      ap.sliding_window = variant == 1 ? 9 : 0;
      // Step buffers per row count, reused across steps as the engine's
      // scratch is (stable pointers: the prepared graph is reused); new data
      // is copied in on the device before each step.
      struct StepBufs { TensorView kc, kg, vc, vg, qc, qg, oc, og; };
      std::map<int64_t, StepBufs> bufs;
      auto refill = [&](const TensorView& c, const TensorView& gdst, const std::vector<float>& data) {
        std::memcpy(c.data(), data.data(), data.size() * 4);
        auto [tc, tg] = e.act(c.dim(0), c.dim(1), &data);
        e.rec->copy(gdst.data(), tg.data(), data.size() * 4);
      };
      auto run_step = [&](std::vector<int32_t> pos, std::vector<int32_t> seq) {
        const auto m = static_cast<int64_t>(pos.size());
        if (!bufs.count(m)) {
          StepBufs b;
          std::tie(b.kc, b.kg) = e.act(m, 128);
          std::tie(b.vc, b.vg) = e.act(m, 128);
          std::tie(b.qc, b.qg) = e.act(m, heads * 64);
          std::tie(b.oc, b.og) = e.act(m, heads * 64);
          bufs[m] = b;
        }
        const StepBufs& b = bufs[m];
        refill(b.kc, b.kg, randv(static_cast<size_t>(m * 2 * 64)));
        refill(b.vc, b.vg, randv(static_cast<size_t>(m * 2 * 64)));
        refill(b.qc, b.qg, randv(static_cast<size_t>(m * heads * 64)));
        const TensorView &kcv = b.kc, &kgv = b.kg, &vcv = b.vc, &vgv = b.vg, &qcv = b.qc, &qgv = b.qg, &oc = b.oc,
                         &og = b.og;
        e.cpu.kv_store(kcv, vcv, pos, seq, cv);
        e.rec->kv_store(kgv, vgv, pos, seq, gv);
        AttentionParams pc = ap, pg = ap;
        pc.q = qcv;
        pc.out = oc;
        pg.q = qgv;
        pg.out = og;
        pc.positions = pg.positions = pos;
        pc.row_seq = pg.row_seq = seq;
        pc.kv = cv;
        pg.kv = gv;
        e.cpu.attention(pc);
        e.rec->attention(pg);
        // q8_0 KV: CPU and GPU quantize K/V themselves (rounding may differ by one step).
        expect_close("attention", host_rows(oc), e.download(og), kvt == DType::kQ8_0 ? 2e-2 : 3e-3);
      };
      // Prefill-like: seq 0 rows 0..20, seq 1 rows 0..9, seq 2 rows 0..39.
      std::vector<int32_t> pos, seq;
      for (int32_t s = 0; s < 3; ++s) {
        const int32_t len = s == 0 ? 21 : s == 1 ? 10 : 40;
        for (int32_t p = 0; p < len; ++p) {
          pos.push_back(p);
          seq.push_back(s);
        }
      }
      run_step(pos, seq);
      const SegmentStats before = e.stats();
      run_step({21, 10, 40}, {0, 1, 2});  // decode
      run_step({22, 11, 41}, {0, 1, 2});  // decode: same graph, new positions
      CHECK(e.stats().graph_reuses > before.graph_reuses);
    }
  }
  }
  std::printf("  paged kv_store + attention: in-place (slot-major) and gather (head-major); 3 sequences,\n"
              "    scattered blocks, f16/f32/q8_0 KV, GQA, window + softcap, prefill then decode with graph reuse\n");
}

// Errors are reported, not hidden; the device stays usable afterwards.
void test_failures(Env& e) {
  // Direct op calls (reference execution) are refused.
  {
    auto [xc, xg] = e.act(1, 32);
    e.gpu->fill(xg, 1.0f);
    const Status st = e.gpu->take_error();
    CHECK(!st.ok() && st.code() == StatusCode::kUnsupported);
    CHECK(e.gpu->take_error().ok());  // cleared on read
  }
  // Unsupported op: scatter_add_rows (MoE).
  {
    auto [sc, sg] = e.act(2, 32);
    auto [dc, dg] = e.act(4, 32);
    std::vector<int32_t> rows = {1, 2};
    std::vector<float> w = {0.5f, 0.5f};
    e.rec->scatter_add_rows(sg, rows, w, dg);
    std::vector<float> out(4 * 32);
    e.rec->download(dg, out);
    const Status st = e.rec->take_error();
    CHECK(!st.ok() && st.code() == StatusCode::kUnsupported);
  }
  // Views of one buffer. Reading a sub-view of a value written in full is
  // derived from that value (supported, correct); reading an enclosing view
  // after writing only a part of it cannot be derived and is refused.
  {
    auto x = randv(4 * 32);
    auto [ac, ag] = e.act(4, 64);
    auto [bc, bg] = e.act(4, 32, &x);
    auto [cc, cg] = e.act(4, 32);
    const TensorView shifted = view2d(static_cast<float*>(ag.data()) + 16, 4, 32, 256, DType::kF32, ag.device());
    e.rec->fill(ag, 1.0f);
    e.rec->add(shifted, bg, cg);  // derived: columns 16..47 of the filled value
    const std::vector<float> got = e.download(cg);
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == 1.0f + x[i]);

    auto [dc, dg] = e.act(4, 64);
    e.rec->add(bg, bg, shifted);  // writes part of ag
    e.rec->add(ag, ag, dg);       // reads all of ag: not derivable
    std::vector<float> out(4 * 64);
    e.rec->download(dg, out);
    const Status st = e.rec->take_error();
    CHECK(!st.ok() && st.code() == StatusCode::kUnsupported);
    CHECK(st.message().find("aliasing") != std::string::npos);
  }
  // A block id outside the pool is refused before anything runs.
  {
    KvGeometry g;
    g.num_layers = 1;
    g.num_kv_heads = 1;
    g.head_dim = g.head_dim_v = 32;
    g.block_size = 16;
    g.num_blocks = 2;
    g.dtype = DType::kF16;
    g.layout = KvLayout::kSlotMajor;
    auto kg = e.rec->allocate(static_cast<size_t>(g.bytes_per_layer() / 2));
    auto vg = e.rec->allocate(static_cast<size_t>(g.bytes_per_layer() / 2));
    CHECK(kg.ok() && vg.ok());
    std::vector<int32_t> table = {7}, pos = {0}, seq = {0};  // block 7 of 2
    std::vector<KvLayerView> kv = {KvLayerView{(*kg)->data(), (*vg)->data(), &g, table}};
    auto [kc, kgv] = e.act(1, 32);
    e.rec->kv_store(kgv, kgv, pos, seq, kv);
    e.rec->synchronize();
    const Status st = e.rec->take_error();
    CHECK(!st.ok() && st.message().find("out of range") != std::string::npos);
  }
  // Allocation failure: an error, not a crash.
  {
    auto huge = e.rec->allocate(size_t{1} << 46);
    CHECK(!huge.ok() && huge.status().code() == StatusCode::kOutOfMemory);
  }
  // The device still works.
  {
    auto x = randv(64);
    auto [xc, xg] = e.act(2, 32, &x);
    auto [yc, yg] = e.act(2, 32);
    e.cpu.add(xc, xc, yc);
    e.rec->add(xg, xg, yg);
    expect_close("add after errors", host_rows(yc), e.download(yg), 0);
  }
  std::printf("  failures: direct calls, scatter_add, aliasing, bad block id, allocation failure; device usable after\n");
}

// Device-side MoE (Device::moe_ffn) against a host reference: router softmax,
// top-k (normalized or raw weights), SwiGLU experts from 3-D weights, and the
// optional shared expert with a per-row sigmoid gate. fp32 weights, so the
// only difference is summation order.
void test_moe_ffn(Env& e) {
  const int64_t m = 5, d = 32, ff = 16, ne = 4, sf = 8;
  const int32_t k = 2;
  auto upload3 = [&](int64_t a, int64_t b, int64_t c, const std::vector<float>& v) {
    auto host = Tensor::empty(DType::kF32, {a, b, c});
    CHECK(host.ok());
    std::memcpy(host->data(), v.data(), v.size() * 4);
    auto dev = e.rec->upload(*host);
    CHECK(dev.ok());
    e.keep.push_back(*host);
    e.keep.push_back(*dev);
    return dev->view();
  };
  auto upload2 = [&](int64_t rows, int64_t cols, const std::vector<float>& v) {
    std::vector<std::byte> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return e.weight(DType::kF32, b, rows, cols).second;
  };
  const auto x = randv(static_cast<size_t>(m * d)), wr = randv(static_cast<size_t>(ne * d), -2, 2);
  const auto wg = randv(static_cast<size_t>(ne * ff * d)), wu = randv(static_cast<size_t>(ne * ff * d));
  const auto wd = randv(static_cast<size_t>(ne * d * ff));
  const auto sg = randv(static_cast<size_t>(sf * d)), su = randv(static_cast<size_t>(sf * d));
  const auto sd = randv(static_cast<size_t>(d * sf)), sr = randv(static_cast<size_t>(d));
  auto silu = [](double v) { return v / (1 + std::exp(-v)); };
  auto dot = [](const float* a, const float* b, int64_t n) {
    double t = 0;
    for (int64_t i = 0; i < n; ++i) t += static_cast<double>(a[i]) * b[i];
    return t;
  };
  // out[r] for one configuration, on the host.
  auto reference = [&](bool normalize, bool shared, bool gated) {
    std::vector<float> out(static_cast<size_t>(m * d), 0.0f);
    for (int64_t r = 0; r < m; ++r) {
      const float* xr = x.data() + r * d;
      std::vector<double> p(static_cast<size_t>(ne));
      double mx = -1e300, sum = 0;
      for (int64_t q = 0; q < ne; ++q) mx = std::max(mx, p[static_cast<size_t>(q)] = dot(wr.data() + q * d, xr, d));
      for (double& v : p) sum += (v = std::exp(v - mx));
      std::vector<int64_t> idx(static_cast<size_t>(ne));
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) { return p[a] > p[b]; });
      double kept = 0;
      for (int32_t j = 0; j < k; ++j) kept += p[static_cast<size_t>(idx[j])];
      for (int32_t j = 0; j < k; ++j) {
        const int64_t q = idx[static_cast<size_t>(j)];
        const double w = p[static_cast<size_t>(q)] / (normalize ? kept : sum);
        std::vector<float> h(static_cast<size_t>(ff));
        for (int64_t i = 0; i < ff; ++i) {
          h[static_cast<size_t>(i)] = static_cast<float>(silu(dot(wg.data() + (q * ff + i) * d, xr, d)) *
                                                         dot(wu.data() + (q * ff + i) * d, xr, d));
        }
        for (int64_t o = 0; o < d; ++o) out[r * d + o] += static_cast<float>(w * dot(wd.data() + (q * d + o) * ff, h.data(), ff));
      }
      if (shared) {
        std::vector<float> h(static_cast<size_t>(sf));
        for (int64_t i = 0; i < sf; ++i) {
          h[static_cast<size_t>(i)] = static_cast<float>(silu(dot(sg.data() + i * d, xr, d)) * dot(su.data() + i * d, xr, d));
        }
        const double gate = gated ? 1 / (1 + std::exp(-dot(sr.data(), xr, d))) : 1.0;
        for (int64_t o = 0; o < d; ++o) out[r * d + o] += static_cast<float>(gate * dot(sd.data() + o * sf, h.data(), sf));
      }
    }
    return out;
  };
  MoeFfnParams p;
  auto [xc, xg] = e.act(m, d, &x);
  auto [oc, og] = e.act(m, d);
  (void)xc;
  (void)oc;
  p.x = xg;
  p.out = og;
  p.router = upload2(ne, d, wr);
  p.gate = upload3(ne, ff, d, wg);
  p.up = upload3(ne, ff, d, wu);
  p.down = upload3(ne, d, ff, wd);
  p.experts = static_cast<int32_t>(ne);
  p.top_k = k;
  p.act = Activation::kSilu;
  const TensorView vg = upload2(sf, d, sg), vu = upload2(sf, d, su), vd = upload2(d, sf, sd), vr = upload2(1, d, sr);
  CHECK(e.rec->supports_moe_ffn());
  for (const auto& [normalize, shared, gated] : {std::tuple{true, false, false}, std::tuple{false, true, true},
                                                 std::tuple{true, true, false}}) {
    p.normalize_topk = normalize;
    p.sh_gate = shared ? vg : TensorView();
    p.sh_up = shared ? vu : TensorView();
    p.sh_down = shared ? vd : TensorView();
    p.sh_router = gated ? vr : TensorView();
    e.rec->moe_ffn(p);
    expect_close("moe_ffn", reference(normalize, shared, gated), e.download(og), 1e-4);
  }
  std::printf("  moe_ffn: top-2 of 4 experts (normalized / raw weights), shared expert with and without sigmoid gate\n");
}

// A value written in a step and read back under another [rows, cols] split
// of the same bytes, then written and read again in the first one: per-head
// Q/K norm reads Q [rows, heads x head_dim] as [rows x heads, head_dim].
void test_reshaped_view(Env& e) {
  const int64_t m = 3, heads = 4, hd = 32;
  auto x = randv(static_cast<size_t>(m * heads * hd), -2, 2), w = randv(static_cast<size_t>(hd), 0.5f, 1.5f);
  std::vector<std::byte> wb(w.size() * 4);
  std::memcpy(wb.data(), w.data(), wb.size());
  auto [wc, wg] = e.weight(DType::kF32, wb, 0, hd);
  auto [xc, xg] = e.act(m, heads * hd, &x);
  auto [yc, yg] = e.act(m, heads * hd);
  auto [zc, zg] = e.act(m, heads * hd);
  const TensorView yhc = view2d(yc.data(), m * heads, hd, hd * 4);
  const TensorView yhg = view2d(yg.data(), m * heads, hd, hd * 4, DType::kF32, yg.device());
  for (Device* d : {static_cast<Device*>(&e.cpu), static_cast<Device*>(e.rec.get())}) {
    const bool gpu = d != &e.cpu;
    d->add(gpu ? xg : xc, gpu ? xg : xc, gpu ? yg : yc);                  // y = 2x  as [m, heads x hd]
    d->rms_norm(gpu ? yhg : yhc, gpu ? wg : wc, 1e-6f, gpu ? yhg : yhc);  // in place as [m x heads, hd]
    d->add(gpu ? yg : yc, gpu ? xg : xc, gpu ? zg : zc);                  // z = y + x as [m, heads x hd]
  }
  expect_close("reshaped view (y)", host_rows(yc), e.download(yg), 1e-5);
  expect_close("reshaped view (z)", host_rows(zc), e.download(zg), 1e-5);
  std::printf("  same bytes under another row split in one step (per-head norm), read back in the first\n");
}

// A prepared graph binds buffers. Freeing a buffer drops every prepared graph,
// so a new buffer at a reused address never runs through a stale graph.
void test_buffer_reuse(Env& e) {
  for (int round = 0; round < 3; ++round) {
    auto x = randv(128);
    auto host = Tensor::empty(DType::kF32, {4, 32});
    CHECK(host.ok());
    std::memcpy(host->data(), x.data(), 512);
    std::vector<float> got(128);
    {
      auto dx = e.rec->upload(*host);
      auto dy = e.rec->allocate(512);
      CHECK(dx.ok() && dy.ok());
      const TensorView yv = view2d((*dy)->data(), 4, 32, 128, DType::kF32, dx->view().device());
      e.rec->scale(dx->view(), 1.0f);  // in place
      e.rec->add(dx->view(), dx->view(), yv);
      e.rec->download(yv, got);
      CHECK(e.rec->take_error().ok());
    }  // both buffers freed here
    for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == 2 * x[i]);
  }
  CHECK(e.stats().graph_evictions > 0);
  std::printf("  buffer reuse after free: prepared graphs dropped, results correct\n");
}

// Zero-copy weights: a tensor borrowed from a read-only file mapping is used in
// place (no device copy is counted); the mapping stays alive through the
// device buffer after every caller reference is gone.
void test_mapped_weights(Env& e) {
  const int64_t n = 64, k = 256;
  const std::vector<std::byte> w = make_weight(DType::kQ8_0, n, k);
  const auto path = std::filesystem::temp_directory_path() / ("dynalm-mapped-" + std::to_string(g_rng()));
  {
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    CHECK(f != nullptr);
    const char pad[100] = {};
    CHECK(std::fwrite(pad, 1, sizeof(pad), f) == sizeof(pad));  // unaligned tensor start
    CHECK(std::fwrite(w.data(), 1, w.size(), f) == w.size());
    CHECK(std::fclose(f) == 0);
  }
  const int64_t mapped_before = e.gpu->memory_stats().weight_bytes;
  TensorView wg;
  Tensor dev;
  bool in_place = false;
  {
    auto file = MappedFile::open(path.string());
    CHECK(file.ok());
    void* p = const_cast<std::byte*>((*file)->data() + 100);
    auto storage = Storage::borrow(p, w.size(), DeviceLoc{}, *file);
    auto layout = TensorLayout::contiguous(DType::kQ8_0, {n, k});
    CHECK(layout.ok());
    auto host = Tensor::from_storage(storage, TensorView(p, *layout));
    CHECK(host.ok());
    auto up = e.rec->upload(*host);
    CHECK(up.ok());
    dev = *up;
    wg = dev.view();
    // In place where the provider wraps host memory (Metal); otherwise copied.
    in_place = wg.data() == p;
  }  // file, storage and host tensor released here; the device buffer keeps the mapping
  std::filesystem::remove(path);
  if (in_place) {
    CHECK(e.gpu->memory_stats().weight_bytes == mapped_before);  // no device copy
  } else {
    CHECK(e.gpu->memory_stats().weight_bytes >= mapped_before + static_cast<int64_t>(w.size()));  // a real copy
  }
  auto [wc, unused] = e.weight(DType::kQ8_0, w, n, k);
  (void)unused;
  auto x = randv(static_cast<size_t>(3 * k));
  auto [xc, xg] = e.act(3, k, &x);
  auto [yc, yg] = e.act(3, n);
  e.cpu.matmul(xc, wc, nullptr, yc);
  e.rec->matmul(xg, wg, nullptr, yg);
  expect_close("mapped matmul", host_rows(yc), e.download(yg), 1e-2);
  std::printf(in_place ? "  zero-copy weights: mapped in place, no device copy, mapping kept alive by the buffer\n"
                       : "  weights copied (this provider cannot wrap host memory); correct after the file is gone\n");
}

}  // namespace

// Every GPU provider built in (Metal, Vulkan, CUDA) runs the whole suite, and
// so does ggml's CPU backend (the experimental ggml-cpu device).
// DYNACORE_TEST_GPU=metal|vulkan|cuda|ggml-cpu restricts it to one.
int main() {
  if (!ggml_gpu_compiled(DeviceType::kCpu)) {  // ggml is in the build when any of its backends is
    std::printf("no ggml backend in this build (ENABLE_METAL/VULKAN/CUDA/GGML_CPU=OFF): nothing to test\n");
    return 0;
  }
  const char* only = std::getenv("DYNACORE_TEST_GPU");
  int ran = 0;
  std::string tested;  // provider labels, in the summary line (CI checks for "metal")
  for (auto [type, label] : {std::pair{DeviceType::kMetal, "metal"}, std::pair{DeviceType::kVulkan, "vulkan"},
                             std::pair{DeviceType::kCuda, "cuda"}, std::pair{DeviceType::kCpu, "ggml-cpu"}}) {
    if (!ggml_gpu_compiled(type) || (only != nullptr && *only != '\0' && std::string_view(only) != label)) continue;
    auto gpu = create_ggml_gpu_device(type);
    if (!gpu.ok()) {
      std::printf("%s: no usable device: %s; not tested\n", label, gpu.status().to_string().c_str());
      continue;
    }
    Env e;
    e.gpu = std::move(*gpu);
    if (type == DeviceType::kCpu) e.act_tol = 2e-3;
    e.rec = std::make_unique<ir::RecordingDevice>(*e.gpu, ir::RecordingDevice::Mode::kDeferred);
    CHECK(e.gpu->segment_executor() != nullptr);
    CHECK(!e.gpu->host_accessible());
    std::printf("device %s, ggml %s\n", std::string(e.gpu->name()).c_str(), ggml_revision().c_str());
    test_matmul(e);
    test_norms_elementwise(e);
    test_rope(e);
    test_embedding_gather(e);
    test_paged_attention(e);
    test_failures(e);
    test_buffer_reuse(e);
    test_reshaped_view(e);
    test_moe_ffn(e);
    test_mapped_weights(e);
    const SegmentStats st = e.stats();
    std::printf("segments %lld, graphs built %lld, reused %lld, evictions %lld\n", static_cast<long long>(st.segments),
                static_cast<long long>(st.graph_builds), static_cast<long long>(st.graph_reuses),
                static_cast<long long>(st.graph_evictions));
    e.keep.clear();
    ++ran;
    tested += std::string(" ") + label;
  }  // each GPU device is destroyed before exit (ggml-metal requirement)
  if (ran == 0) {
    std::printf("no GPU device: nothing tested\n");
    return 0;
  }
  std::printf("all ggml device tests passed on %d provider(s):%s (%d checks)\n", ran, tested.c_str(), g_checks);
  return 0;
}
