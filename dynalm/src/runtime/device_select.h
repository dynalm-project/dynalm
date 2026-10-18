#pragma once

// Device selection, as a pure decision: which devices to try, in which order,
// for a model and a device request. Engine::create opens them in that order
// and keeps the first that starts. No device is opened here, so the policy is
// unit-tested without a GPU (dynalm/tests/test_engine.cpp).

#include <span>
#include <string>
#include <vector>

#include "dynacore/base/status.h"
#include "dynacore/device/device_registry.h"
#include "model_ir/model_config.h"
#include "common/core.h"

namespace dynalm {

// OK when the ggml device `kind` (a GPU, or kGgmlCpu) runs this model with
// this KV dtype; otherwise Unsupported naming the first missing feature
// (MoE with a shared expert, MoE where not yet verified, per-head Q/K norm of
// an unverified architecture, RoPE scaling other than none/linear, KV dtype).
// Checked at startup, so neither `auto` nor `--backend metal` finds out on the
// first request.
Status gpu_model_support(const ModelConfig& c, DType kv_dtype, DeviceKind kind);

// What the caller asked for (EngineOptions' device fields).
struct DeviceRequest {
  DeviceKind backend = DeviceKind::kCpu;
  bool backend_auto = false;  // a GPU that runs the model, else the CPU
  bool require_gpu = false;   // with backend_auto: never the CPU (--gpu)
  bool cpu_ggml = false;      // the CPU: ggml's kernels first (EngineOptions::cpu_ggml)
  DType kv_dtype = DType::kF16;
};

struct DevicePlan {
  std::vector<DeviceKind> candidates;  // tried in order; the first that starts is used
  bool native_cpu_last = false;        // then DynaCore's native CPU kernels
  std::string note;                    // why devices were skipped (logged)
  Status error;                        // not OK: fail with it (nothing to try)
};

// `compiled`: the kinds built in (compiled_devices()); `ggml_cpu_built`: the
// ggml CPU kernels are (ENABLE_GGML_CPU or a GPU build).
//   auto:          the compiled GPUs that run the model (CUDA, Metal, Vulkan
//                  order), then the CPU path below; --gpu: GPUs only.
//   cpu + ggml:    kGgmlCpu when built and it runs the model, then native.
//   anything else: exactly that device (its own errors surface on start).
DevicePlan plan_devices(const ModelConfig& c, const DeviceRequest& r, std::span<const DeviceKind> compiled,
                        bool ggml_cpu_built);

}  // namespace dynalm
