#include "backends/backend_registry.h"

#include <string>

#include "backends/cpu/cpu_backend.h"
#include "platform/cpu_info.h"
#include "platform/isa.h"

namespace engine {

std::string_view backend_kind_name(BackendKind k) {
  switch (k) {
    case BackendKind::kCpu: return "cpu";
    case BackendKind::kCuda: return "cuda";
    case BackendKind::kHip: return "hip";
    case BackendKind::kMetal: return "metal";
    case BackendKind::kVulkan: return "vulkan";
  }
  return "?";
}

Result<BackendKind> parse_backend_kind(std::string_view name) {
  for (BackendKind k : {BackendKind::kCpu, BackendKind::kCuda, BackendKind::kHip, BackendKind::kMetal,
                        BackendKind::kVulkan}) {
    if (backend_kind_name(k) == name) return k;
  }
  return InvalidArgument("unknown backend '" + std::string(name) + "' (cpu, cuda, hip, metal, vulkan)");
}

std::vector<BackendKind> available_backends() { return {BackendKind::kCpu}; }

Result<std::unique_ptr<Backend>> create_backend(BackendKind kind, ThreadPool& pool) {
  if (kind == BackendKind::kCpu) {
    return std::unique_ptr<Backend>(std::make_unique<CpuBackend>(pool, select_best_isa(cpu_info().features)));
  }
  return Unsupported("the " + std::string(backend_kind_name(kind)) +
                     " backend is not built in this version: GPU backends are designed (docs/gpu-backend.md, "
                     "DD-045) but not implemented; use --backend cpu");
}

}  // namespace engine
