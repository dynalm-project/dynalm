#include "dynacore/device/device_registry.h"

#include <cctype>
#include <string>

#include "dynacore/cpu/cpu_device.h"
#include "dynacore/gpu/ggml_device.h"
#include "dynacore/hardware/cpu_info.h"
#include "dynacore/hardware/isa.h"

namespace dynacore {

std::string_view device_kind_name(DeviceKind k) {
  switch (k) {
    case DeviceKind::kCpu: return "cpu";
    case DeviceKind::kCuda: return "cuda";
    case DeviceKind::kHip: return "hip";
    case DeviceKind::kMetal: return "metal";
    case DeviceKind::kVulkan: return "vulkan";
    case DeviceKind::kGgmlCpu: return "ggml-cpu";
  }
  return "?";
}

Result<DeviceKind> parse_device_kind(std::string_view name) {
  for (DeviceKind k : {DeviceKind::kCpu, DeviceKind::kCuda, DeviceKind::kHip, DeviceKind::kMetal,
                        DeviceKind::kVulkan, DeviceKind::kGgmlCpu}) {
    if (device_kind_name(k) == name) return k;
  }
  return InvalidArgument("unknown backend '" + std::string(name) + "' (cpu, cuda, hip, metal, vulkan, ggml-cpu)");
}

namespace {

// The storage provider of a GPU device kind (ggml_device.h).
DeviceType gpu_type(DeviceKind k) {
  switch (k) {
    case DeviceKind::kCuda: return DeviceType::kCuda;
    case DeviceKind::kHip: return DeviceType::kHip;
    case DeviceKind::kMetal: return DeviceType::kMetal;
    case DeviceKind::kVulkan: return DeviceType::kVulkan;
    case DeviceKind::kCpu:
    case DeviceKind::kGgmlCpu: break;
  }
  return DeviceType::kCpu;
}

}  // namespace

std::vector<DeviceKind> compiled_devices() {
  std::vector<DeviceKind> out = {DeviceKind::kCpu};
  for (DeviceKind k : {DeviceKind::kCuda, DeviceKind::kMetal, DeviceKind::kVulkan}) {
    if (ggml_gpu_compiled(gpu_type(k))) out.push_back(k);
  }
  return out;
}

Result<std::unique_ptr<Device>> create_device(DeviceKind kind, ThreadPool& pool) {
  if (kind == DeviceKind::kCpu) {
    return std::unique_ptr<Device>(std::make_unique<CpuDevice>(pool, select_isa(cpu_info().features)));
  }
  // GPUs through pinned ggml (docs/gpu-backend.md). They run only compiled
  // steps: the engine wraps them in a RecordingDevice.
  if (kind == DeviceKind::kMetal || kind == DeviceKind::kVulkan || kind == DeviceKind::kCuda) {
    if (!ggml_gpu_compiled(gpu_type(kind))) {
      std::string flag(device_kind_name(kind));
      for (char& ch : flag) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
      return Unsupported("the " + std::string(device_kind_name(kind)) + " backend is not built in this version (configure with -DENABLE_" +
                         flag + "=ON); use --backend cpu");
    }
    return create_ggml_gpu_device(gpu_type(kind));
  }
  // ggml's CPU backend (experimental): the same segment device as the GPUs,
  // with this pool's thread count.
  if (kind == DeviceKind::kGgmlCpu) {
    if (!ggml_gpu_compiled(DeviceType::kCpu)) {
      return Unsupported("the ggml-cpu backend is not built in this version (configure with -DENABLE_GGML_CPU=ON, "
                         "or any GPU backend); use --backend cpu");
    }
    // ggml-cpu is compiled for the AVX2 tier on x86-64 (cmake/Ggml.cmake).
    if (!ggml_cpu_usable_here()) return Unsupported("the ggml-cpu backend needs a CPU with AVX2, FMA and F16C");
    GgmlDeviceOptions o;
    o.threads = static_cast<int32_t>(pool.size());
    return create_ggml_gpu_device(DeviceType::kCpu, o);
  }
  return Unsupported("the " + std::string(device_kind_name(kind)) +
                     " backend is not built in this version: GPU backends are designed (docs/gpu-backend.md, "
                     "DD-045) but not implemented; use --backend cpu");
}

}  // namespace dynacore
