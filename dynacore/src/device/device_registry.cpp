#include "dynacore/device/device_registry.h"

#include <string>

#include "dynacore/cpu/cpu_device.h"
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
  }
  return "?";
}

Result<DeviceKind> parse_device_kind(std::string_view name) {
  for (DeviceKind k : {DeviceKind::kCpu, DeviceKind::kCuda, DeviceKind::kHip, DeviceKind::kMetal,
                        DeviceKind::kVulkan}) {
    if (device_kind_name(k) == name) return k;
  }
  return InvalidArgument("unknown backend '" + std::string(name) + "' (cpu, cuda, hip, metal, vulkan)");
}

std::vector<DeviceKind> compiled_devices() { return {DeviceKind::kCpu}; }

Result<std::unique_ptr<Device>> create_device(DeviceKind kind, ThreadPool& pool) {
  if (kind == DeviceKind::kCpu) {
    return std::unique_ptr<Device>(std::make_unique<CpuDevice>(pool, select_isa(cpu_info().features)));
  }
  return Unsupported("the " + std::string(device_kind_name(kind)) +
                     " backend is not built in this version: GPU backends are designed (docs/gpu-backend.md, "
                     "DD-045) but not implemented; use --backend cpu");
}

}  // namespace dynacore
