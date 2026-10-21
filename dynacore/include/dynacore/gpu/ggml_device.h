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
#include <vector>

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
  // DeviceType::kCpu (ggml's CPU backend): compute threads; 0 = ggml's default.
  int32_t threads = 0;
};

// Slot-major (in-place) attention reads the pool cells [0, span) of a step, with
// a mask admitting each row's own cells; ggml's sparse path instead visits only a
// row's finite mask entries (at most `kv_max` per row: the longest attended
// context, bucketed). The span is set by the highest physical block a step
// uses, not by how much KV it uses: once the prefix cache holds most of the
// pool, new sequences get high block ids and dense attention runs over nearly
// the whole pool every step (Metal, Qwen2.5-1.5B, 16 users, 65,536-cell pool:
// ITL p50 21 ms fresh, 136 ms with a full cache, 23 ms with the sparse bound;
// test/results/beta-phase4-20261021). The sparse kernel costs ~10% when the
// span is close to the context (one user), and more than dense attention in
// prompt-heavy steps at moderate spans (4 users, spans ~10x the context: -6%
// tok/s; 16 users at ~27x: ITL p99 120 -> 200 ms), so it is chosen per step and
// only for spans far beyond the context (the cache-filled pool was ~85x).
// DYNACORE_GPU_SPARSE: "1" always (where kv_max < span), "0" never, unset or
// anything else: kAuto. Read once.
enum class GpuSparseMode : uint8_t { kAuto, kAlways, kNever };
GpuSparseMode gpu_sparse_mode();
// kAuto uses the sparse bound only on Metal, and only when the span is at least
// kSparseSpanRatio x kv_max and kv_max <= kSparseMaxKv (above it ggml-metal
// falls back to the dense kernel anyway). It applies to every step type: a
// decode-only variant (sparse only when no sequence has more than 8 rows) left
// the cache-filled slowdown in place (16 users, ITL p50 135 ms), because the
// prompt rows of mixed steps are where the dense span costs most
// (test/results/beta-final-20261021/kv).
inline constexpr int64_t kSparseSpanRatio = 32;
inline constexpr int64_t kSparseMaxKv = 4096;
// The n_kv_max to set on a slot-major attention op (0 = dense).
int32_t sparse_kv_bound(int64_t span, int64_t kv_max, GpuSparseMode mode, DeviceType type);

// True when DynaCore was built with any ggml GPU backend (ENABLE_METAL,
// ENABLE_VULKAN, ENABLE_CUDA), or with the one named by `type`.
bool ggml_gpu_compiled();
bool ggml_gpu_compiled(DeviceType type);
// ggml's CPU kernels are built in and this CPU can run them (x86-64: AVX2,
// FMA and F16C, the tier they are compiled for; any ARM64 CPU).
bool ggml_cpu_usable_here();

// A GPU of the given provider (kMetal, kVulkan or kCuda) as a DynaCore Device:
// its first device, or device DYNACORE_GPU_INDEX (0-based) when set.
// kUnsupported when the provider is not compiled in or has no usable device.
Result<std::unique_ptr<Device>> create_ggml_gpu_device(DeviceType type, const GgmlDeviceOptions& opts = {});

// The devices a provider reports (descriptions, e.g. "Apple M4 Pro"), without
// initializing any of them: for diagnostics (`dynalm doctor`). Empty when the
// provider is not compiled in or finds no device.
std::vector<std::string> ggml_gpu_device_names(DeviceType type);

// The pinned ggml revision compiled in ("" when not compiled).
std::string ggml_revision();

}  // namespace dynacore
