#pragma once

// Backend selection (DD-045).
//
// The runtime asks for a backend by kind; only CPU is built today. GPU kinds
// exist in the API so callers, configuration and error messages carry them
// already: asking for one yields a clear kUnsupported naming what is missing,
// never a silent CPU fallback. Adding a backend = implementing Backend (see
// docs/gpu-backend.md) and one case here.

#include <memory>
#include <string_view>
#include <vector>

#include "dynacore/device/backend.h"
#include "dynacore/base/status.h"
#include "dynacore/execution/thread_pool.h"

namespace engine {

enum class BackendKind : uint8_t { kCpu, kCuda, kHip, kMetal, kVulkan };

std::string_view backend_kind_name(BackendKind k);
Result<BackendKind> parse_backend_kind(std::string_view name);  // "cpu", "cuda", "hip", "metal", "vulkan"
// Kinds compiled into this binary.
std::vector<BackendKind> available_backends();

// `pool` is used by host-side backends (CPU worker threads).
Result<std::unique_ptr<Backend>> create_backend(BackendKind kind, ThreadPool& pool);

}  // namespace engine
