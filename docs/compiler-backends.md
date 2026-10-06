# Compiler backends

The DynaCore compiler lowers IR to calls on a `Device`. This page records which backend
exists, which is designed, and the measurements behind each decision.

## Stage 1: the C++ kernel library (implemented)

- An `ExecPlan` step becomes one call on the inner device. A single op maps to its `Device`
  method, a shared-input group to `matmul_many`, and a gated MLP to `matmul_gated`.
- The CPU device picks the instruction set once at start-up (`generic`, `avx2`, `neon`) and the
  kernel path per call from the kernel plan:
  - int8 decode dot (DD-053);
  - fused dequantize-dot;
  - fp32 panel GEMM (DD-036);
  - grouped GQA attention (DD-066).
- This backend runs on every CPU DynaLM supports, so compiled mode is never less portable than
  reference mode.

New DynaCore ops added for the compiler, with defaults so every device keeps working:

| Op | Default | CPU implementation |
|---|---|---|
| `Device::matmul_gated(act, x, Wg, Wu, out, gate_scratch, up_scratch)` | two matmuls + `act_mul` | Fused int8 path: both dot products per output column, activation in the epilogue. Other row counts use the default. |
| `MatmulJob::bias` | applied per job | Added per chunk in `matmul_many`. Jobs reading the same activations share one int8-quantized copy. |

## Stage 2: generated intrinsic kernels (not built)

The plan was for the compiler to emit AVX2/VNNI intrinsics specialized per shape and
quantization format, beating the hand-written kernels. The measurements say there is little
left to win inside a single kernel on this machine:

| Measurement (i7-1255U, 10 threads) | GB/s |
|---|---|
| DRAM read ceiling (`tools/proto/read_bw.cpp`) | 18.4 |
| Best M = 1 GEMVs (`bench_decode_matmul`, int8 path) | 14.0–17.6 |
| ffn_gate q4_K [8960 × 1536] | 16.2 (88% of ceiling) |
| ffn_down q6_K [1536 × 8960] | 17.6 (96%) |
| lm_head q6_K [151936 × 1536] | 17.1 (93%) |

- **Decode GEMVs.** A generated GEMV can gain at most 4–12% on the large decode matrices, and
  only on matrices that are not already within noise of the ceiling.
- **VNNI.** An AVX-VNNI int8 GEMM was measured **slower** than fp32 for batched shapes
  (DD-058). Per-block scales cost as many instructions as the FMAs that `vpdpbusd` replaces.
  Requantizing weights to per-row scales would change that. It needs a second weight copy
  and a new accuracy gate, and is not started.

Stage 2 will start only for a kernel whose `bench_op_trace` time is far from its roofline, for
example a future prefill kernel or a new quantization format. It needs an A/B harness first.

## Stage 3: native code (not built)

There is no case yet where generated machine code beats compiled C++ intrinsics. It is not
started.

## GPU (designed)

A CUDA device implements `Device` (contract: [gpu-backend.md](gpu-backend.md)).

- The IR and the compiler are device-neutral:
  - types carry a device;
  - plans are lists of `Device` calls;
  - `matmul_gated` and `matmul_many` map to a fused CUDA kernel and to a grouped GEMM launch.
- The deferred recording device is the natural place for CUDA-graph capture: one cached plan
  per decode shape.
- Nothing is implemented: this machine has no NVIDIA GPU (`dynalm doctor`: GPU not detected).

## ARM (NEON/SVE)

- The CPU device already has a NEON tier.
- Compiled mode works on it unchanged, because fusions fall back to the default sequences when
  the int8 path is not available (`matmul_gated` → two matmuls + act_mul).
- It is untested on hardware here; CI runs the arm64 build.
