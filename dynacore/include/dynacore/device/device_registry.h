#pragma once

// Device selection (DD-045).
//
// The runtime asks for a backend by kind; only CPU is built today. GPU kinds
// exist in the API so callers, configuration and error messages carry them
// already: asking for one yields a clear kUnsupported naming what is missing,
// never a silent CPU fallback. Adding a backend = implementing Device (see
// docs/gpu-backend.md) and one case here.

#include <memory>
#include <string_view>
#include <vector>

#include "dynacore/device/device.h"
#include "dynacore/base/status.h"
#include "dynacore/execution/thread_pool.h"

namespace dynacore {

enum class DeviceKind : uint8_t { kCpu, kCuda, kHip, kMetal, kVulkan };

std::string_view device_kind_name(DeviceKind k);
Result<DeviceKind> parse_device_kind(std::string_view name);  // "cpu", "cuda", "hip", "metal", "vulkan"
// Kinds compiled into this binary.
std::vector<DeviceKind> compiled_devices();

// `pool` is used by host-side backends (CPU worker threads).
Result<std::unique_ptr<Device>> create_device(DeviceKind kind, ThreadPool& pool);

}  // namespace dynacore
