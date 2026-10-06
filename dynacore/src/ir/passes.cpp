#include "dynacore/ir/passes.h"

#include <algorithm>
#include <map>
#include <set>

#include "dynacore/ir/verifier.h"

namespace dynacore::ir {
namespace {

bool is_leaf(OpKind k) {
  return k == OpKind::kInput || k == OpKind::kWeight || k == OpKind::kKvCache || k == OpKind::kConstant;
}

bool is_matmul(OpKind k) { return k == OpKind::kMatMul || k == OpKind::kQuantizedMatMul; }

int64_t concrete_bytes(const Type& t, int64_t symbol_size = 1) {
  Type c = t;
  for (Dim& d : c.shape) {
    if (!d.known()) d = Dim::of(symbol_size);
  }
  const int64_t b = c.bytes();
  return b < 0 ? 0 : b;
}

// Rows (M) of a matmul's activation operand, 1 if unknown.
int64_t rows_of(const Graph& g, const Op& op) {
  const Type& x = g.value(op.inputs[0]).type;
  return x.rank() >= 1 && x.shape[0].known() ? x.shape[0].size : 1;
}

// Next op index > i that is not a leaf, or -1.
int32_t next_compute(const Graph& g, int32_t i) {
  for (int32_t j = i + 1; j < static_cast<int32_t>(g.ops().size()); ++j) {
    if (!is_leaf(g.op(j).kind)) return j;
  }
  return -1;
}

bool recorded_group_member(const Op& op) { return op.has("group"); }

}  // namespace

std::vector<std::vector<int32_t>> value_uses(const Graph& g) {
  std::vector<std::vector<int32_t>> uses(g.values().size());
  for (size_t i = 0; i < g.ops().size(); ++i) {
    for (ValueId v : g.op(static_cast<int32_t>(i)).inputs) uses[static_cast<size_t>(v)].push_back(static_cast<int32_t>(i));
  }
  return uses;
}

// --- canonicalization ----------------------------------------------------------

int canonicalize(Graph& g) {
  int rewrites = 0;
  for (Op& op : g.mutable_ops()) {
    if (is_matmul(op.kind) && op.inputs.size() >= 2) {
      const bool q = g.value(op.inputs[1]).type.quantized();
      const OpKind want = q ? OpKind::kQuantizedMatMul : OpKind::kMatMul;
      if (op.kind != want) {
        op.kind = want;
        ++rewrites;
      }
    } else if (op.kind == OpKind::kAttention || op.kind == OpKind::kGqa) {
      const OpKind want = op.int_attr("heads") != op.int_attr("kv_heads") ? OpKind::kGqa : OpKind::kAttention;
      if (op.kind != want) {
        op.kind = want;
        ++rewrites;
      }
    }
  }
  // No dead-code elimination: a recorded segment's results may be read by the
  // host after the sync point (logits), which the graph cannot see.
  return rewrites;
}

// --- memory analysis -----------------------------------------------------------

MemoryReport analyze_memory(const Graph& g, int64_t symbol_size) {
  MemoryReport r;
  const auto uses = value_uses(g);
  const auto nops = static_cast<int32_t>(g.ops().size());
  struct Interval {
    int32_t def, last;
    int64_t bytes;
  };
  std::vector<Interval> live;
  std::map<int32_t, int64_t> buffer_size;
  for (const Value& v : g.values()) {
    if (v.type.kind != ValueKind::kTensor || v.producer < 0 || is_leaf(g.op(v.producer).kind)) continue;
    const int64_t bytes = concrete_bytes(v.type, symbol_size);
    ++r.values;
    r.activation_bytes += bytes;
    buffer_size[v.buffer] = std::max(buffer_size[v.buffer], bytes);
    const auto& u = uses[static_cast<size_t>(v.id)];
    // Unused results may be read by the host after the segment: live to the end.
    live.push_back({v.producer, u.empty() ? nops : u.back(), bytes});
  }
  r.buffers = static_cast<int32_t>(buffer_size.size());
  for (const auto& [b, bytes] : buffer_size) r.buffer_bytes += bytes;
  // Peak of simultaneously live bytes.
  std::vector<int64_t> delta(static_cast<size_t>(nops) + 2, 0);
  for (const Interval& in : live) {
    delta[static_cast<size_t>(in.def)] += in.bytes;
    delta[static_cast<size_t>(in.last) + 1] -= in.bytes;
  }
  int64_t cur = 0;
  for (int64_t d : delta) r.peak_live_bytes = std::max(r.peak_live_bytes, cur += d);
  // Greedy reuse: a value takes the smallest free buffer that fits.
  std::sort(live.begin(), live.end(), [](const Interval& a, const Interval& b) { return a.def < b.def; });
  std::multimap<int32_t, int64_t> busy;  // last use -> buffer size
  std::multiset<int64_t> free_sizes;
  for (const Interval& in : live) {
    while (!busy.empty() && busy.begin()->first < in.def) {
      free_sizes.insert(busy.begin()->second);
      busy.erase(busy.begin());
    }
    auto it = free_sizes.lower_bound(in.bytes);
    int64_t size = in.bytes;
    if (it != free_sizes.end()) {
      size = *it;
      free_sizes.erase(it);
    } else {
      r.planned_bytes += in.bytes;
    }
    busy.emplace(in.last, size);
  }
  return r;
}

// --- cost model and kernel selection -------------------------------------------

CostModel CostModel::from(const KernelPlan& plan, int32_t threads) {
  CostModel cm;
  cm.threads = threads;
  cm.int8_max_rows = plan.int8_decode_max_rows;
  cm.expand_min_rows = plan.expand_min_rows;
  return cm;
}

double estimate_op_us(const Graph& g, const Op& op, const CostModel& cm) {
  if (is_leaf(op.kind)) return 0;
  double weight_bytes = 0, act_bytes = 0;
  for (ValueId in : op.inputs) {
    const Type& t = g.value(in).type;
    if (t.kind == ValueKind::kWeight) weight_bytes += static_cast<double>(concrete_bytes(t));
    else if (t.kind == ValueKind::kTensor) act_bytes += static_cast<double>(concrete_bytes(t));
  }
  act_bytes += static_cast<double>(concrete_bytes(g.value(op.result).type));
  double us = weight_bytes / (cm.dram_gbps * 1e3) + act_bytes / (cm.cache_gbps * 1e3);
  return us + cm.region_us;
}

void select_kernels(Graph& g, const CostModel& cm) {
  for (Op& op : g.mutable_ops()) {
    if (is_leaf(op.kind)) continue;
    if (is_matmul(op.kind)) {
      const Type& w = g.value(op.inputs[1]).type;
      const Type& x = g.value(op.inputs[0]).type;
      const int64_t m = rows_of(g, op);
      const int64_t k = x.shape[1].known() ? x.shape[1].size : 0;
      const std::string dt(dtype_name(w.dtype));
      const int64_t int8_rows = op.int_attr("int8_rows", cm.int8_max_rows);
      if (w.quantized() && m <= int8_rows && k % 32 == 0) op.kernel = "cpu.int8_dot." + dt;
      else if (m >= cm.expand_min_rows && w.dtype != DType::kF32) op.kernel = "cpu.panel_gemm." + dt;
      else op.kernel = "cpu.fused_dot." + dt;
    } else if (op.kind == OpKind::kGqa) {
      op.kernel = "cpu.gqa_grouped";
    } else {
      op.kernel = "cpu." + std::string(op_name(op.kind));
    }
  }
}

// --- fusion and execution planning ----------------------------------------------

ExecPlan plan_execution(Graph& g, const FusionOptions& opts, const CostModel& cm, FusionReport* report) {
  ExecPlan plan;
  FusionReport rep;
  const auto uses = value_uses(g);
  const auto nops = static_cast<int32_t>(g.ops().size());
  int32_t next_group = 0;
  int32_t i = 0;
  auto plain_matmul = [&](int32_t j) {
    const Op& o = g.op(j);
    return is_matmul(o.kind) && !recorded_group_member(o);
  };
  // Grouping and gating pay off where the kernels are latency-bound: the int8
  // decode path (few rows). For prefill-shaped matmuls the single-matmul
  // K-blocked GEMM beats matmul_many's 16-row panels (measured end to end,
  // DD-072), so those stay separate.
  auto decode_shaped = [&](int32_t j) {
    const Op& o = g.op(j);
    if (!o.kernel.empty()) return o.kernel.rfind("cpu.int8_dot.", 0) == 0;
    return rows_of(g, o) <= o.int_attr("int8_rows", cm.int8_max_rows);
  };
  while (i < nops) {
    const Op& op = g.op(i);
    if (is_leaf(op.kind)) {
      ++i;
      continue;
    }
    // Gated MLP: matmul(x, Wg) ; matmul(x, Wu) ; act_mul(gate, up), with the
    // gate/up results used by nothing else.
    if (opts.gated_mlp && plain_matmul(i) && decode_shaped(i) && op.inputs.size() == 2) {
      const int32_t j = next_compute(g, i);
      const int32_t a = j >= 0 ? next_compute(g, j) : -1;
      if (j >= 0 && a >= 0 && plain_matmul(j) && decode_shaped(j) && g.op(j).inputs.size() == 2 && g.op(j).inputs[0] == op.inputs[0] &&
          g.op(a).kind == OpKind::kActMul) {
        const Op& act = g.op(a);
        const ValueId gate = act.inputs[0], up = act.inputs[1];
        const bool shapes = g.value(op.inputs[1]).type.shape == g.value(g.op(j).inputs[1]).type.shape;
        const bool order_ok = (gate == op.result && up == g.op(j).result) || (gate == g.op(j).result && up == op.result);
        const bool private_uses = uses[static_cast<size_t>(op.result)].size() == 1 &&
                                  uses[static_cast<size_t>(g.op(j).result)].size() == 1;
        if (shapes && order_ok && private_uses) {
          // ops = {gate matmul, up matmul, act_mul}
          const int32_t gate_op = gate == op.result ? i : j, up_op = gate == op.result ? j : i;
          plan.steps.push_back({ExecStep::Kind::kGatedMatmul, {gate_op, up_op, a}});
          for (int32_t o : {i, j, a}) g.mutable_ops()[static_cast<size_t>(o)].fusion_group = next_group;
          ++next_group;
          ++rep.gated;
          rep.est_saved_us += 2 * cm.region_us +
                              2.0 * static_cast<double>(concrete_bytes(g.value(op.result).type)) / (cm.cache_gbps * 1e3);
          i = a + 1;
          continue;
        }
      }
    }
    // Matmuls sharing their input, back to back (Q/K/V).
    if (opts.group_shared_input && plain_matmul(i) && decode_shaped(i)) {
      std::vector<int32_t> run = {i};
      int32_t j = next_compute(g, i);
      while (j >= 0 && plain_matmul(j) && decode_shaped(j) && g.op(j).inputs[0] == op.inputs[0]) {
        bool independent = true;
        for (int32_t r : run) {
          for (ValueId in : g.op(j).inputs) independent = independent && in != g.op(r).result;
        }
        if (!independent) break;
        run.push_back(j);
        j = next_compute(g, j);
      }
      if (run.size() >= 2) {
        plan.steps.push_back({ExecStep::Kind::kMatmulGroup, run});
        for (int32_t o : run) g.mutable_ops()[static_cast<size_t>(o)].fusion_group = next_group;
        ++next_group;
        ++rep.groups;
        rep.grouped_ops += static_cast<int32_t>(run.size());
        rep.est_saved_us += static_cast<double>(run.size() - 1) * cm.region_us;
        i = run.back() + 1;
        continue;
      }
    }
    plan.steps.push_back({ExecStep::Kind::kOp, {i}});
    ++i;
  }
  if (report) *report = rep;
  return plan;
}

// --- pipeline ------------------------------------------------------------------

Result<ExecPlan> compile_segment(Graph& g, const CompileOptions& opts, CompileReport* report) {
  if (opts.verify) ENGINE_RETURN_IF_ERROR(verify(g));
  CompileReport rep;
  rep.canonicalized = canonicalize(g);
  if (report) rep.memory = analyze_memory(g);
  select_kernels(g, opts.cost);
  ExecPlan plan = plan_execution(g, opts.fusion, opts.cost, &rep.fusion);
  if (opts.verify) ENGINE_RETURN_IF_ERROR(verify(g));
  if (report) *report = rep;
  return plan;
}

ExecutionPlanner make_planner(CompileOptions opts) {
  return [opts](Graph& g) { return compile_segment(g, opts, nullptr); };
}

}  // namespace dynacore::ir
