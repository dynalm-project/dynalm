#include "bench/cpu_report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <tuple>

#include "api/json.h"

namespace dynalm::cpu_report {
namespace {

std::string fmt(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof(b), f, v);
  return b;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Medians over rounds of one (path, model, users).
struct Summary {
  bool found = false;
  double tok_s = 0, ttft = 0, itl50 = 0, itl99 = 0;
  int32_t errors = 0;
  int rounds = 0;
};

Summary summarize(const Report& r, const std::string& path, const std::string& model, int32_t users) {
  std::vector<double> t, f, i5, i9;
  Summary s;
  for (const Point& p : r.points) {
    if (p.path != path || p.model != model || p.users != users) continue;
    t.push_back(p.tok_s);
    f.push_back(p.ttft_p50_ms);
    i5.push_back(p.itl_p50_ms);
    i9.push_back(p.itl_p99_ms);
    s.errors += p.errors;
    ++s.rounds;
  }
  if (t.empty()) return s;
  s.found = true;
  s.tok_s = median(t);
  s.ttft = median(f);
  s.itl50 = median(i5);
  s.itl99 = median(i9);
  return s;
}

}  // namespace

std::string feature_list(const dynacore::CpuFeatures& f) {
  std::vector<std::string> v;
  auto add = [&](bool on, const char* n) {
    if (on) v.emplace_back(n);
  };
  add(f.sse42, "sse4.2");
  add(f.avx2, "avx2");
  add(f.fma, "fma");
  add(f.f16c, "f16c");
  add(f.avx_vnni, "avx_vnni");
  add(f.avx512f, "avx512f");
  add(f.avx512bw, "avx512bw");
  add(f.avx512vl, "avx512vl");
  add(f.avx512_vnni, "avx512_vnni");
  add(f.avx512_bf16, "avx512_bf16");
  add(f.amx_tile, "amx_tile");
  add(f.amx_int8, "amx_int8");
  add(f.amx_bf16, "amx_bf16");
  add(f.neon, "neon");
  add(f.arm_dotprod, "dotprod");
  add(f.arm_i8mm, "i8mm");
  add(f.arm_fp16, "fp16");
  add(f.arm_bf16, "bf16");
  add(f.arm_sve, "sve");
  add(f.arm_sme, "sme");
  std::string s;
  for (const std::string& x : v) s += (s.empty() ? "" : " ") + x;
  return s;
}

std::vector<Path> plan_paths(const dynacore::CpuFeatures& f, bool native_default, int threads) {
  const std::string t = std::to_string(threads);
  std::vector<Path> p;
  // DynaLM as users run it, and with each kernel family forced.
  p.push_back({"default", native_default ? "DynaLM default (DynaCore kernels)" : "DynaLM default (ggml kernels)", false,
               {"--cpu", "-t", t}, {}, native_default ? "native" : "ggml-cpu"});
  if (native_default) {
    p.push_back({"ggml-cpu", "DynaLM, ggml kernels", false, {"--backend", "ggml-cpu", "-t", t}, {}, "ggml-cpu"});
  } else {
    p.push_back({"dynacore", "DynaLM, DynaCore kernels (--mode exact)", false, {"--backend", "cpu", "--mode", "exact", "-t", t},
                 {}, "native"});
  }
  // DynaCore's own tiers, where the CPU has more than one (each forced; the
  // fastest is what the DynaCore path above already uses).
  const std::vector<std::string> native_args =
      native_default ? std::vector<std::string>{"--cpu", "-t", t}
                     : std::vector<std::string>{"--backend", "cpu", "--mode", "exact", "-t", t};
  if (f.arm_dotprod && f.arm_i8mm) {
    p.push_back({"qmm-neon_dot", "DynaCore, NEON dot-product tier (no i8mm)", false, native_args,
                 {{"DYNACORE_QMM_TIER", "neon_dot"}}, "native"});
  }
  if (f.arm_sme) {
    p.push_back({"sme-off", "DynaCore without SME (prefill on NEON)", false, native_args, {{"DYNACORE_SME", "0"}}, "native"});
  }
  if (f.avx_vnni) {
    p.push_back({"vnni", "DynaCore with the AVX-VNNI decode kernel (DYNACORE_VNNI=1)", false, native_args,
                 {{"DYNACORE_VNNI", "1"}}, "native"});
  }
  // The reference: llama.cpp's server, CPU only, the same threads.
  p.push_back({"llama.cpp", "llama.cpp (llama-server, CPU)", true, {"-ngl", "0", "-t", t}, {}, ""});
  return p;
}

const std::vector<ModelSpec>& all_models() {
  static const std::vector<ModelSpec> m = {
      {"1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf", 1117320736, int64_t{4} << 30},
      {"4b", "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", 2497280256, int64_t{8} << 30},
      {"8b", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", 5027783488, int64_t{12} << 30},
  };
  return m;
}

ModelChoice choose_models(const std::string& ids, bool quick, int64_t ram_bytes) {
  ModelChoice c;
  std::vector<std::string> want;
  if (!ids.empty()) {
    std::stringstream ss(ids);
    for (std::string id; std::getline(ss, id, ',');) {
      if (!id.empty()) want.push_back(id);
    }
  } else if (quick) {
    want = {"1.5b"};
  } else {
    for (const ModelSpec& m : all_models()) want.push_back(m.id);
  }
  for (const std::string& id : want) {
    const auto it = std::find_if(all_models().begin(), all_models().end(), [&](const ModelSpec& m) { return m.id == id; });
    if (it == all_models().end()) {
      c.error = "unknown model '" + id + "' (1.5b, 4b, 8b)";
      return c;
    }
    if (ram_bytes > 0 && ram_bytes < it->min_ram_bytes) {
      c.skipped.push_back(id + ": needs " + std::to_string(it->min_ram_bytes >> 30) + " GB of RAM, this machine has " +
                          fmt("%.1f", static_cast<double>(ram_bytes) / (1 << 30)) + " GB");
      continue;
    }
    c.run.push_back(*it);
  }
  return c;
}

LlamaAsset llama_asset(const std::string& os, const std::string& arch) {
  // llama.cpp build b11695 (commit 9a16f1dbe), the version of the published
  // M4 results; sha256 as GitHub publishes them for the release assets.
  static const struct {
    const char* os;
    const char* arch;
    const char* file;
    const char* sha;
  } kAssets[] = {
      {"macos", "arm64", "llama-b11695-bin-macos-arm64.tar.gz", "087505ef3ac701704f8807e1c31f848b23a7d00eb8c570a1d4d2e411aa7b304e"},
      {"macos", "x86_64", "llama-b11695-bin-macos-x64.tar.gz", "28e059dadefe3ee672f9f7e1b48bcb6125eb835a094fd602dd3c8d0404b62378"},
      {"linux", "x86_64", "llama-b11695-bin-ubuntu-x64.tar.gz", "7e77d0c9f498265803b2861f5ddf3ecdda5112dfd9ece6e4558e1d495b4ab1a3"},
      {"linux", "arm64", "llama-b11695-bin-ubuntu-arm64.tar.gz", "2a5d17078ea446474079ed81d1582dea8bcf2ccae940587dd032fe372adf8a87"},
      {"windows", "x86_64", "llama-b11695-bin-win-cpu-x64.zip", "cccd83322af7eb4f80291c174d073d09dceed3c0d0c30d3b76854163937f1777"},
      {"windows", "arm64", "llama-b11695-bin-win-cpu-arm64.zip", "868c758380b502ddecacab01a8def848afe7ed677410df5b822238b06969fafe"},
  };
  LlamaAsset a;
  a.build = "b11695";
  for (const auto& e : kAssets) {
    if (os == e.os && arch == e.arch) {
      a.url = std::string("https://github.com/ggml-org/llama.cpp/releases/download/b11695/") + e.file;
      a.sha256 = e.sha;
      a.zip = std::string(e.file).ends_with(".zip");
    }
  }
  return a;
}

bool accuracy_pass(double top1_pct, double ppl, double llama_ppl) {
  return top1_pct >= kMinTop1Pct && llama_ppl > 0 && ppl <= llama_ppl * (1.0 + kMaxPplIncreasePct / 100.0);
}

std::string to_json(const Report& r) {
  using json::Value;
  json::Object o;
  o["dynalm_version"] = r.dynalm_version;
  o["cpu"] = r.cpu_name;
  o["os"] = r.os;
  o["features"] = r.features;
  o["ram_bytes"] = static_cast<double>(r.ram_bytes);
  o["threads"] = r.threads;
  o["controlled"] = r.controlled;
  o["llama_cpp"] = r.llama_source;
  json::Array notes;
  for (const std::string& n : r.notes) notes.emplace_back(n);
  o["notes"] = std::move(notes);
  json::Object w;
  w["prompt_tokens"] = r.workload.prompt_tokens;
  w["output_tokens"] = r.workload.output_tokens;
  json::Array users;
  for (int32_t u : r.workload.users) users.emplace_back(u);
  w["users"] = std::move(users);
  w["rounds"] = r.workload.rounds;
  o["workload"] = std::move(w);
  json::Array paths;
  for (const Path& p : r.paths) {
    json::Object po;
    po["id"] = p.id;
    po["label"] = p.label;
    json::Array args, env;
    for (const std::string& a : p.serve_args) args.emplace_back(a);
    for (const auto& [k, v] : p.env) env.emplace_back(k + "=" + v);
    po["args"] = std::move(args);
    po["env"] = std::move(env);
    paths.emplace_back(std::move(po));
  }
  o["paths"] = std::move(paths);
  json::Array models;
  for (const std::string& m : r.models) models.emplace_back(m);
  o["models"] = std::move(models);
  json::Array points;
  for (const Point& p : r.points) {
    json::Object po;
    po["path"] = p.path;
    po["model"] = p.model;
    po["users"] = p.users;
    po["round"] = p.round;
    po["tok_s"] = p.tok_s;
    po["ttft_p50_ms"] = p.ttft_p50_ms;
    po["itl_p50_ms"] = p.itl_p50_ms;
    po["itl_p99_ms"] = p.itl_p99_ms;
    po["completed"] = p.completed;
    po["errors"] = p.errors;
    points.emplace_back(std::move(po));
  }
  o["points"] = std::move(points);
  json::Array acc;
  for (const Accuracy& a : r.accuracy) {
    json::Object ao;
    ao["path"] = a.path;
    ao["model"] = a.model;
    ao["ran"] = a.ran;
    if (!a.error.empty()) ao["error"] = a.error;
    ao["top1_pct"] = a.top1_pct;
    ao["kl_mean"] = a.kl_mean;
    ao["ppl"] = a.ppl;
    ao["llama_ppl"] = a.llama_ppl;
    ao["positions"] = static_cast<double>(a.positions);
    ao["pass"] = a.pass;
    acc.emplace_back(std::move(ao));
  }
  o["agreement_with_llama_cpp"] = std::move(acc);
  if (!r.machine_json.empty()) {
    if (auto m = json::parse(r.machine_json); m.ok()) o["machine"] = std::move(*m);
  }
  return json::dump(Value(std::move(o))) + "\n";
}

std::string to_markdown(const Report& r) {
  std::ostringstream md;
  md << "## DynaLM CPU report\n\n";
  md << "- **CPU:** " << r.cpu_name << " (" << r.features << "), " << r.threads << " threads\n";
  md << "- **OS:** " << r.os << ", RAM " << fmt("%.1f", static_cast<double>(r.ram_bytes) / (1 << 30)) << " GB\n";
  md << "- **DynaLM:** " << r.dynalm_version << "; **llama.cpp:** " << r.llama_source << "\n";
  md << "- **Workload:** prompt " << r.workload.prompt_tokens << ", " << r.workload.output_tokens
     << " output tokens, " << r.workload.rounds << " round(s), medians\n";
  md << "- **Run:** " << (r.controlled ? "controlled" : "**uncontrolled** (see notes): compare with care") << "\n";
  for (const std::string& n : r.notes) md << "- " << n << "\n";
  const Path* llama = nullptr;
  for (const Path& p : r.paths) {
    if (p.llama) llama = &p;
  }
  for (const std::string& model : r.models) {
    md << "\n### " << model << "\n\n";
    md << "| Path | Users | tok/s | vs llama.cpp | First token p50 | Token gap p50 / p99 |\n";
    md << "|---|---|---|---|---|---|\n";
    for (int32_t u : r.workload.users) {
      const Summary base = llama != nullptr ? summarize(r, llama->id, model, u) : Summary{};
      for (const Path& p : r.paths) {
        const Summary s = summarize(r, p.id, model, u);
        if (!s.found) continue;
        std::string ratio = "–";
        if (!p.llama && base.found && base.tok_s > 0) ratio = fmt("%.2f×", s.tok_s / base.tok_s);
        md << "| " << p.label << " | " << u << " | " << fmt("%.1f", s.tok_s) << " | " << ratio << " | "
           << fmt("%.0f", s.ttft) << " ms | " << fmt("%.1f", s.itl50) << " / " << fmt("%.1f", s.itl99) << " ms";
        if (s.errors > 0) md << " (" << s.errors << " errors)";
        md << " |\n";
      }
    }
  }
  if (!r.accuracy.empty()) {
    md << "\n### Agreement with llama.cpp (same tokens)\n\n";
    md << "| Path | Model | Top-1 | Mean KL | PPL (llama.cpp) | Verdict |\n|---|---|---|---|---|---|\n";
    for (const Accuracy& a : r.accuracy) {
      std::string label = a.path;
      for (const Path& p : r.paths) {
        if (p.id == a.path) label = p.label;
      }
      if (!a.ran) {
        md << "| " << label << " | " << a.model << " | – | – | – | not run: " << a.error << " |\n";
        continue;
      }
      md << "| " << label << " | " << a.model << " | " << fmt("%.2f %%", a.top1_pct) << " | " << fmt("%.5f", a.kl_mean)
         << " | " << fmt("%.4f", a.ppl) << " (" << fmt("%.4f", a.llama_ppl) << ") | " << (a.pass ? "pass" : "**FAIL**")
         << " |\n";
    }
    md << "\nPass: top-1 ≥ " << fmt("%.0f", kMinTop1Pct) << " % and perplexity ≤ llama.cpp's + "
       << fmt("%.0f", kMaxPplIncreasePct) << " %.\n";
  }
  return md.str();
}

}  // namespace dynalm::cpu_report
