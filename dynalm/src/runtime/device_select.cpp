#include "runtime/device_select.h"

namespace dynalm {

Status gpu_model_support(const ModelConfig& c, DType kv_dtype, DeviceKind kind) {
  // Routed MoE, with or without a (sigmoid-gated) shared expert, runs on the
  // GPU as one device op (Device::moe_ffn).
  // Verified (Granite-MoE 1B, bench_gpu_accuracy): Metal and Vulkan pass
  // DD-053 and the MoE batching checks; ggml-cpu is at llama.cpp-level
  // accuracy like all its models. CUDA is unchecked.
  if (c.moe.num_experts > 0 && kind == DeviceKind::kCuda) {
    return Unsupported("mixture-of-experts models on " + std::string(device_kind_name(kind)) +
                       " are not verified yet");
  }
  // Per-head Q/K norm runs on the GPU; verified against DD-053 for Qwen3 (4B)
  // and Gemma 3 (1B) on Metal and Vulkan. Others stay on the CPU until checked.
  if (c.attn_qk_norm && c.architecture != "qwen3" && c.architecture != "gemma3") {
    return Unsupported("per-head Q/K norm is verified on the GPU for Qwen3 and Gemma 3 only (" + c.architecture +
                       ": not yet)");
  }
  for (const RopeConfig* r : {&c.rope, c.has_rope_local ? &c.rope_local : nullptr}) {
    if (r != nullptr && r->scaling != RopeScaling::kNone && r->scaling != RopeScaling::kLinear) {
      return Unsupported("RoPE scaling other than none/linear (YaRN, Llama 3) is not supported on the GPU yet");
    }
  }
  if (kv_dtype != DType::kF16 && kv_dtype != DType::kF32 && kv_dtype != DType::kQ8_0) {
    return Unsupported("KV cache dtype " + std::string(dtype_name(kv_dtype)) + " is not supported on the GPU (f16, f32, q8_0)");
  }
  return Status::Ok();
}

DevicePlan plan_devices(const ModelConfig& c, const DeviceRequest& r, std::span<const DeviceKind> compiled,
                        bool ggml_cpu_built) {
  DevicePlan p;
  auto add_cpu_path = [&] {
    p.native_cpu_last = true;
    if (!r.cpu_ggml) return;
    if (!ggml_cpu_built) {
      p.note += (p.note.empty() ? "" : "; ") + std::string("ggml CPU kernels not built in (ENABLE_GGML_CPU=OFF)");
    } else if (Status s = gpu_model_support(c, r.kv_dtype, DeviceKind::kGgmlCpu); !s.ok()) {
      p.note += (p.note.empty() ? "" : "; ") + std::string("ggml CPU kernels: ") + s.message();
    } else {
      p.candidates.push_back(DeviceKind::kGgmlCpu);
    }
  };
  if (!r.backend_auto) {
    if (r.backend == DeviceKind::kCpu) {
      add_cpu_path();
    } else {
      p.candidates.push_back(r.backend);
    }
    return p;
  }
  bool any_gpu = false;
  Status first = Status::Ok();
  for (DeviceKind k : compiled) {
    if (k == DeviceKind::kCpu || k == DeviceKind::kGgmlCpu) continue;
    any_gpu = true;
    if (Status s = gpu_model_support(c, r.kv_dtype, k); s.ok()) {
      p.candidates.push_back(k);
    } else if (first.ok()) {
      first = s;
    }
  }
  if (p.candidates.empty()) {
    const std::string why = any_gpu ? first.message() : "this DynaLM has no GPU support built in (install a GPU build, see docs/gpu.md)";
    if (r.require_gpu) {
      p.error = Unsupported("--gpu: " + why + "; use --cpu");
      return p;
    }
    p.note = "auto selects the CPU: " + why;
  }
  if (!r.require_gpu) add_cpu_path();
  return p;
}

}  // namespace dynalm
