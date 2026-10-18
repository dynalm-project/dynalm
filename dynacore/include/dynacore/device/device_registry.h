#pragma once

// Device selection (DD-045).
//
// The runtime asks for a backend by kind: the CPU always, and the GPUs this
// binary was built with (Metal, Vulkan, CUDA through ggml). Asking for one
// that is not built in yields a clear kUnsupported naming what is missing,
// never a silent CPU fallback. Adding a backend = implementing Device (see
// docs/gpu-backend.md) and one case here.

#include <memory>
#include <string_view>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/base/status.h"
#include "dynacore/execution/thread_pool.h"

namespace dynacore {

// kGgmlCpu: the CPU through ggml's CPU backend ("ggml-cpu"), the default CPU
// path when built in (ENABLE_GGML_CPU, and every GPU build).
enum class DeviceKind : uint8_t { kCpu, kCuda, kHip, kMetal, kVulkan, kGgmlCpu };

std::string_view device_kind_name(DeviceKind k);
Result<DeviceKind> parse_device_kind(std::string_view name);  // "cpu", "cuda", "hip", "metal", "vulkan", "ggml-cpu"
// The CPU and the GPU kinds compiled into this binary (`--backend auto` tries
// the GPUs). The experimental ggml-cpu kind is not listed.
std::vector<DeviceKind> compiled_devices();

// `pool` is used by host-side backends (CPU worker threads).
Result<std::unique_ptr<Device>> create_device(DeviceKind kind, ThreadPool& pool);

}  // namespace dynacore
