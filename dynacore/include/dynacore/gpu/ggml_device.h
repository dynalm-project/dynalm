#pragma once

// GgmlDevice: runs DynaLM forward steps on a GPU through the public graph API
// of a pinned ggml revision (docs/gpu-backend.md). Metal is the first
// provider; the device is written against ggml's backend-neutral interfaces
// (device registry, backend buffers, graph compute) so CUDA can follow.
//
// Execution model (docs/gpu-backend.md, sections 4-6):
//   - The device is a SegmentExecutor. It runs only behind a RecordingDevice
//     (compiled execution): each recorded step becomes one ggml graph,
//     prepared once per structure (op stream, shapes, per-sequence rows, KV
//     span bucket) and reused; per step only the dynamic inputs (token ids,
//     positions, KV row indices, attention masks) are uploaded.
//   - Memory handed out by allocate()/upload() is ggml backend buffers. The
//     pointers are never host-dereferenced by the runtime
//     (host_accessible() == false); data moves through upload/download.
//   - The paged KV pool keeps DynaLM's layout ([block][kv_head][slot][dim],
//     dynacore/attention/paged_kv.h): kv_store writes rows with set_rows at
//     their paged addresses. Attention gathers each sequence's K/V rows into a
//     contiguous span (padded to a bucket and masked) for ggml's flash
//     attention. The gather is the temporary stage-1 approach; it reads the
//     KV about three times instead of once (docs/gpu-backend.md, section 7).
//   - Unsupported ops, types or aliasing patterns make the step fail with a
//     named error. Nothing falls back to the CPU inside a step.
//
// Calling the Device ops directly (reference execution) is an error reported
// through take_error(): a GPU device must not be driven op by op.

#include <cstdint>
#include <memory>
#include <string>

#include "dynacore/base/status.h"
#include "dynacore/device/device.h"

namespace dynacore {

struct GgmlDeviceOptions {
  // Prepared graphs kept (each holds its workspace); the cache is cleared when
  // it is full (decode/prefill structures are few).
  int32_t max_graphs = 64;
  // Attention spans are padded to a multiple of this many positions so a
  // decode stream reuses one graph per bucket instead of one per token.
  int32_t kv_bucket = 256;
};

// True when DynaCore was built with any ggml GPU backend (ENABLE_METAL,
// ENABLE_VULKAN, ENABLE_CUDA), or with the one named by `type`.
bool ggml_gpu_compiled();
bool ggml_gpu_compiled(DeviceType type);

// A GPU of the given provider (kMetal, kVulkan or kCuda) as a DynaCore Device:
// its first device, or device DYNACORE_GPU_INDEX (0-based) when set.
// kUnsupported when the provider is not compiled in or has no usable device.
Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType type, const GgmlDeviceOptions& opts = {});

// The pinned ggml revision compiled in ("" when not compiled).
std::string ggml_revision();

}  // namespace dynacore
