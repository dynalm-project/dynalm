# CPU mission: experiment ledger

Baseline B0 = Phase 8 CPU run (`test/results/phase8-cpu-20261024`, controlled: AC power, Energy Mode
Automatic, 5 rounds). CPU suite v1 (`docs/perf/cpu-suite-v1.md`) on B0: **0.77× vs llama.cpp**
(C1 0.89, C2 0.80, C3 1.09, C4 0.93, C5 0.29, C6 0.88, C7 0.72, C8 0.96), 0.81× vs Ollama (3 of 8
comparable). Peak RSS, llama.cpp ÷ DynaLM: 1.36× (1.5b), 1.11× (4b).

Development A/B runs below are back to back on the same machine, mostly **on battery in Low Power
Mode** (absolute numbers are about 2× slower than on AC; only the ratios between the two arms are
used). Each entry: bottleneck, change, correctness, measurement, decision.

## E1 — drop the file pages of packed weights (kept)

- Bottleneck: peak RSS 3.0 GiB for a 1.04 GB model. After load 2.09 GiB = weights + the packed
  Q4_K/Q6_K copy (the qmm kernels read only the copy, but the file pages read while packing stay
  resident).
- Change: `release_file_pages()` (mapped_file.cpp): drops resident pages of a range inside a live
  read-only MappedFile (macOS `msync(MS_INVALIDATE)`; `MADV_DONTNEED` is ignored there for file
  pages; Linux `MADV_DONTNEED`); never touches memory outside a registered mapping. CpuDevice calls
  it after packing. `DYNACORE_QMM_RELEASE_SOURCE=0` keeps the pages.
- Correctness: the bytes stay valid (they fault back in from the file); test_qmm checks data
  intact, partial pages, out-of-mapping and heap ranges refused.
- Measured (1.5b, 8 users, 512-token prompts, back to back x2): RSS 2,402 → 1,471 MiB; output
  44.8–45.2 tok/s in both arms (no speed change). After load: 2,090 → 1,159 MiB.

## E2 — SME fp32 panel gemm for prefill matmuls (kept)

- Bottleneck: at 4–8 users and for long prompts the CPU is bound by prompt compute (512 prompt
  tokens per 128 generated); the int8 qmm gemm ran at ~1.1 TOPS (FFN down int16: ~0.5).
- Measured first: M4 Pro SME (Low Power Mode) fp32 FMOPA 1.13 TFLOPS on 1 thread, 2.70 on 8;
  int8 SMOPA 5.63 TOPS on 8. A prototype panel gemm reached 2.59 TFLOPS (m=512).
- Change: qmm SME tier (`qmm_sme.cpp`, `-march=armv8.6-a+sme`, no SVE outside streaming mode —
  `armv9-a` made the compiler emit SVE in normal code: SIGILL): 32x32 blocks in the four fp32 ZA
  tiles; activations fp32 (not quantized), weights dequantized exactly into k-major panels
  (NEON, from the packed copy, so released pages stay released). Registry id `sme_f32_gemm`
  (accuracy class: reference), `KernelPlan::sme_min_rows` = 64 (`DYNACORE_SME_ROWS`), runtime
  check FEAT_SME + 512-bit streaming vectors; `DYNACORE_SME=0` off. CPU solo-prefill chunk 512
  where SME runs (`kCpuSmeSoloPrefillChunk`), 32 rows before.
- Correctness: test_qmm: panels bit-equal to `dequantize_row` (all 11 formats, packed and not),
  SME blocks vs a double product on ragged shapes; wikitext-2 below.
- Microbenchmark (`BENCH_QMM_SME=1 bench_qmm 8`), qmm int8 vs SME, m = 512: 1.5B gate/up 1,189 →
  2,204 GOPS; FFN down Q6_K 441 → 1,660; Q8_0 8B gate 1,087 → 2,231. Crossover ~m = 48.
- First attempt failed: scalar panel dequantization (0.6 G values/s per thread) with 32-row
  prompt chunks made prefill 2.5× slower (TTFT 2,048 tokens 10.7 → 27.7 s); fixed by the NEON
  panel writers and the 512-row solo chunk.

## E3 — SME flash attention for prompt chunks (kept)

- Bottleneck (bench_op_trace, `BENCH_TRACE_PREFILL=512`, 1.5b): attention 42% of a prefill step
  (621 ms of 1,466): the per-row path re-reads and re-converts the fp16 K/V of the whole context
  for every query row (~143 GFLOPS).
- Change: `CpuDevice::attention_sme` for runs of >= 16 consecutive positions of one sequence
  (`KernelPlan::sme_attention_min_rows`, `DYNACORE_SME_ATTN_ROWS`): 32 query rows x the query
  group against 32-key blocks; S^T = K (x) Q into the k-major P panel, online softmax, O += P^T V
  accumulated in ZA (`block_acc`). Other rows (decode, short runs) keep the per-row path.
- Correctness: test_cpu_attention (new): SME vs per-row on f16 / f32 / q8_0 KV, head sizes
  64 / 128 / 256, groups 1–8, runs starting mid-block next to decode rows, sliding window, softcap:
  max relative difference <= 1e-4.
- Measured: attention 621 → 201 ms per 512-token chunk; step 1,466 → 991 ms.

## E2 + E3 together

- End to end, 1.5b, 1 user, 2,048-token prompt, back to back (battery, Low Power Mode):
  TTFT 10,840 → 4,071 ms (2.66×), prompt 155 → 316 tok/s; gap between tokens unchanged
  (18–19 ms); RSS +66 MiB (activation and weight panels).
- wikitext-2 (20 x 512) vs the fp32 reference: Qwen2.5-1.5B PPL 11.6357 (ref 11.6358), mean KL
  1.5e-7, top-1 99.96%; Qwen3-4B PPL 18.4396 (ref 18.4393), mean KL 5.6e-7, top-1 100%. The int8
  qmm prefill had mean KL 0.0010 / 0.0005.
- 30/30 test programs pass; strict native check passes.

## E4 — prefill bursts on the CPU (kept)

- Bottleneck: at 4–8 users DynaLM folded 32-row prompt chunks into almost every decode step
  (DD-079 protection): every such step stalled all streams (gap p90 160 ms at 4 users, 232 ms at
  8, battery) and the prompts ran at the small-chunk rate. Requests fell out of step, so the
  overlap never ended. llama.cpp's gaps stay low because its closed-loop requests stay in step.
- Change: `SchedulerConfig::prefill_burst_max_stall_ms` (Engine default 2000 where the SME
  prefill runs; `DYNALM_PREFILL_BURST_MS`): while prompts prefill, steps carry prompt rows only
  (solo budget 512 shared by the prompts) and decoding streams wait, at most the bound since
  their last token; then a regular mixed step.
- Correctness: test_engine `test_prefill_burst`: with a long bound B's prompt never shares a step
  with A's decode rows and A's greedy text equals A alone; with a 1 ms bound A keeps decoding in
  mixed steps. The DD-079 test runs with bursts off.
- Measured (1.5b, 512-token prompts, battery, back to back): 4 users 42.0 → 69.5 tok/s, gap
  p90 160 → 32 ms, p99 165 → 34 ms, whole request p99 17.7 → 7.4 s, first token p50 2.6 → 3.5 s
  (prompts now finish together); 8 users 53.6 → 74.6 tok/s, gap p90 232 → 52 ms, p99 265 → 54 ms,
  first token p50 4.1 → 6.5 s, p99 13.5 → 7.3 s.

## E5 — bounded prefix-cache retention on the CPU (kept)

- Bottleneck: 4b RSS grew with every request (2.57 GiB after load, 4.73 after 1/4/8-user rounds)
  until the whole lazily allocated KV pool (4.5 GiB at 32,768 tokens) was resident: the prefix
  cache kept every finished prompt.
- Change: CPU default `prefix_cache_max_blocks` = pool / 4 (`kCpuPrefixCacheShare`); blocks the
  cache gives up return to the free list, which is reused first.
- Measured: same sequence plus two more 8-user rounds: 3.90 GiB, flat (3,895 / 3,898 / 3,900
  MiB after the last three rounds).

## E6 — vectorized online softmax in the SME attention (kept)

- Bottleneck: after E3 the per-block softmax of `attention_sme` (scalar loops over 32 keys per
  query row, plus a scalar 32 x 32 K transpose) was most of the remaining attention time.
- Change: softmax over the 32 contiguous query lanes of a block with NEON (per-lane key bounds,
  running max / sum arrays, softcap applied first); the K block goes into its k-major panel with
  a NEON 4 x 4 transpose (`transpose_into_panel`).
- Correctness: test_cpu_attention (SME vs per-row, max relative difference <= 1e-4) passes.
- Measured (`BENCH_TRACE_PREFILL=512 bench_op_trace`, 1.5b, battery, back to back): attention
  253 → 178 ms per 512-token chunk; step 1,050 / 1,006 → 882 / 935 ms.

## E7 — decode attention: fp16 KV read directly by the grouped kernels (kept)

- Bottleneck (`BENCH_TRACE_CTX=2048 bench_op_trace`, 4b, decode after 2,048 tokens): attention
  13.5 ms of a 49 ms step (28 %), 8 MB of KV per layer at ~22 GB/s. The grouped path converted
  every fp16 K/V block run into an fp32 buffer first (about 30 % of the kernel time: fp32 KV
  808 vs fp16 1,178 µs per call on 1 thread).
- Change: `attn_scores_heads_f16` / `attn_accum_heads_f16` (NEON): the grouped GQA kernels read
  fp16 and convert in registers (exact), no fp32 copy; 4 keys per pass in the score kernel and
  16–32 head-dimension floats per pass in the accumulation (more independent FMA chains), each
  key's arithmetic unchanged. Tiers without them keep converting.
- Correctness: test_cpu_attention adds grouped vs one head at a time (<= 1e-5 relative); 30/30
  test programs pass.
- Measured: `BENCH_ATTN_HEADS=32,8 bench_attention`, 2,048 tokens, 8 threads: 193 → 116 µs
  (cold, 36 rotating layers: 155 µs); in the model 13.5 → 9.7 ms per step. End to end (4b, 1
  user, 2,048-token prompt, battery, back to back x2): gap p50 51.3 / 51.5 → 48.3 / 49.2 ms,
  first token 11,359 / 11,425 → 10,700 / 10,793 ms.

## E8 — elementwise ops of prompt chunks (kept)

- Bottleneck (`BENCH_TRACE_PREFILL=512`, 4b, after 1,024 tokens): elementwise ops 270 ms of a
  2,837 ms chunk: `add` ran on one thread (699 µs per call), `rms_norm` summed squares in scalar
  double and ran one task per row (per-head q/k norms: 16,384 rows of 128; 727 µs per call),
  `kv_store` ran on one thread, and `silu` called libm `expf` (no vectorization; act_mul 1.8 ms).
- Change: `add` over (row, column chunk) tasks from 64 K values (smaller stays on the calling
  thread); `rms_norm` at least 8,192 values per task and a NEON float64 sum of squares (still
  double); `kv_store` rows split across the pool from 64 rows (each row writes its own slot);
  `silu(x) = x * (x >= 0 ? 1 : e) / (1 + e)` with `e = exp_nonpos(-|x|)` (polynomial exp,
  relative error ~1e-7) and a NEON loop for act_mul.
- Correctness: wikitext-2 (20 x 512) vs the fp32 reference: 1.5b PPL 11.6359 (ref 11.6358), mean
  KL 1.6e-7, top-1 99.96 %; 4b PPL 18.4398 (ref 18.4393), mean KL 4.7e-7, top-1 100 %. 30/30
  test programs pass.
- Measured (same trace): act_mul 66 → 26 ms, rmsnorm 105 → 22 ms, add 50 → 13.5 ms, kv_write
  33 → 7 ms; chunk 2,837 → 2,653 ms (-6.5 %).

## E9 — SME attention: shared K/V panels, 128-key softmax steps (kept)

- Bottleneck (4b prefill trace after E8): attention 687 ms of a 2,653 ms chunk (0.56 TFLOPS).
  Every (32-row query block, KV head) unit converted and transposed the whole context's K/V
  again (16 times for a 512-row chunk), and every 32 keys the O accumulators made a round trip
  through ZA (4 `block_acc` calls per head, each loading and storing 32 x 32 floats).
- Change: pass 1 converts K/V once per call into fp32 panels (K by 32-key block, V by 32-dim
  chunk with keys contiguous), per slab of KV heads within 32 MiB (`sme_kv_panels_`; one head
  per slab can exceed it at very long contexts); pass 2 runs the units on the shared panels with
  128-key softmax steps (4 QK blocks, then one PV product of depth 128 per chunk). Steps whose
  keys all precede the block's first query (no window) skip the mask.
- Correctness: test_cpu_attention (SME vs per-row incl. windows and softcap) passes; wikitext-2:
  1.5b PPL 11.6358 (ref 11.6358), mean KL 1.7e-7; 4b PPL 18.4394 (ref 18.4393), mean KL 4.0e-7;
  30/30 test programs pass.
- Measured (traces): 4b attention 687 → 437 ms per 512-token chunk (chunk 2,653 → 2,436 ms);
  1.5b attention 178 → 122 ms (chunk 882 → 832 ms). End to end, E7 + E8 + E9 vs the E6 build
  (battery, back to back): 1.5b 1 user 2,048-token prompt: first token 4,052 / 4,063 → 3,480 /
  3,483 ms, whole request 6,487 / 6,477 → 5,817 / 5,827 ms; 4b: first token 11,403 → 9,504 ms,
  gap p50 51.8 → 47.8 ms. Peak RSS +6 MiB (1.5b), +21 MiB (4b).
- Measured, not used: SME fp16 → fp32 widening FMOPA gives the same FLOPS as fp32 on this chip
  (2.81 vs 2.57 TFLOPS on 8 threads, Low Power Mode); only int8 SMOPA doubles it (5.63 TOPS), and
  its integer accumulation cannot apply per-32 scales inside ZA.

## Rejected

- R1 qmm i8mm gemm from 2 rows (`DYNACORE_QMM_GEMM_ROWS=2`) for batched decode: 8 users gap p50
  50.6 → 55.9 ms, 75.3 → 71.2 tok/s. The gemv tiles stay up to 8 rows.
- R3 two independent SDOT chains in the packed gemv lane dots (`lane_dot32` / `lane_dot16`):
  decode-step matvecs (`BENCH_QMM_STEP=1 bench_qmm`, 8 threads) 5.64 / 5.58 ms before, 6.04 /
  6.05 ms after: no gain (the out-of-order core already overlaps the chains). Reverted.
- R2 reference vs compiled execution on the CPU: one run showed reference 12.3 vs compiled
  15.6 ms per decode step, the repeat 12.3 vs 11.9 ms, 4b identical (20.3 ms): noise; kept as is.

## Development comparison against llama.cpp (battery, Low Power Mode, back to back, 1 user, 1.5b)

| Prompt | Output tok/s (DynaLM / llama.cpp) | First token ms | Gap p50 ms | Whole request ms |
|---|---|---|---|---|
| 128 | 54.0 / 45.2, 53.4 / 44.4 | 311 / 484, 316 / 485 | 15.9 / 17.4, 16.3 / 18.3 | 2,379 / 2,839 |
| 2048 | 19.6 / 11.2, 19.5 / 11.1 | 4,074 / 8,658, 4,102 / 8,695 | 19.0 / 21.2, 19.4 / 21.3 | 6,578 / 11,516 |

Not a release result (uncontrolled conditions); the controlled suite decides.

## Development suite run (E1–E5 build; battery, Low Power Mode; 2 rounds; NOT controlled)

`test/results/cpu-dev1-20261025`, CPU suite v1 settings with `REQUIRE_AC=0`, DynaLM and llama.cpp
only (Ollama not run):

| Workload | vs llama.cpp |
|---|---|
| C1 1.5b 1u 128p output tok/s | 1.18× |
| C2 1.5b 1u 2048p whole request | 1.78× |
| C3 1.5b 4u 512p aggregate tok/s | 2.07× |
| C4 1.5b 8u 512p aggregate tok/s | 1.67× |
| C5 1.5b 8u 512p gap p95 | 1.16× |
| C6 4b 1u 512p output tok/s | 1.49× |
| C7 4b 1u 2048p first token | 2.13× |
| C8 4b 1u 2048p gap p50 | 1.17× |

Geometric mean 1.54× over 8 of 8. Peak RSS (`maximum resident set size`), llama.cpp ÷ DynaLM:
M1 2.75× (4,125 / 1,503 MiB), M2 3.97× (15,442 / 3,894 MiB); by `peak memory footprint` 2.3× and
3.4×. llama.cpp allocates its whole KV cache (8 slots x 4,096, 4.6 GiB on 4b) at start; its RSS
was 10.8–12.8 GiB on 4b in Phase 8 too.

**Why this is not the result:** Low Power Mode caps the CPU at about 2.6 GHz (measured: dependent
add chain) and slows the engines unequally. llama.cpp 1.5b single-user decode: 133 tok/s on AC
(Phase 8) vs 45.7 here (2.9× slower); DynaLM 118 vs 54 (2.2× slower). In Low Power Mode the
DynaLM gemv kernels are compute bound (8 threads read 222 GB/s in a plain read loop, the decode
matvecs reach 82–125 GB/s); on AC both engines are near memory bandwidth for single-user decode
(llama.cpp 6.0 ms per token, DynaLM 6.6 ms in Phase 8), so C1/C6/C8 cannot approach 2× there
without reading fewer bytes per token. Only the controlled AC run decides.

## Development suite run 2 (E1–E9 build; battery, Low Power Mode; 2 rounds; NOT controlled)

`test/results/cpu-dev2-20261025`, DynaLM only, compared with run 1 by `cpu_suite.py compare`:

| Workload | Metric | Baseline | Candidate | Candidate ÷ baseline | Noise | Verdict |
|---|---|---|---|---|---|---|
| C1 | output tok/s | 54.3 | 55.5 | 1.021× | ±1% | within noise |
| C2 | whole request p50 | 6465.1 | 5751.4 | 1.124× | ±1% | better |
| C3 | aggregate output tok/s | 69.3 | 74.0 | 1.069× | ±0% | better |
| C4 | aggregate output tok/s | 74.8 | 79.9 | 1.069× | ±0% | better |
| C5 | gap between tokens p95 | 52.8 | 50.2 | 1.052× | ±0% | better |
| C6 | output tok/s | 16.4 | 17.3 | 1.056× | ±0% | better |
| C7 | first token p50 | 11375.4 | 9417.1 | 1.208× | ±0% | better |
| C8 | gap between tokens p50 | 51.4 | 47.5 | 1.081× | ±0% | better |
| M1 | peak RSS MiB | 1,503 | 1,512 | 0.994× | – | same |
| M2 | peak RSS MiB | 3,894 | 3,945 | 0.987× | – | same |


Decision: KEEP (better: C2, C3, C4, C5, C6, C7, C8)

Not a measurement against llama.cpp (llama.cpp was not rerun): run-1 ratios times these give C1 1.20, C2 2.00, C3 2.21, C4 1.79, C5 1.22, C6 1.57, C7 2.57, C8 1.26, geometric mean ~1.66× — a derived development estimate only.

## Sanitizers (E1–E9 code)

`build/native-asan` and `build/native-tsan` (RelWithDebInfo, strict native): test_cpu_attention
(SME on), test_qmm and test_engine pass with no AddressSanitizer / ThreadSanitizer reports.

## Controlled CPU suite v1 run (E1–E9 build) — the result

`test/results/cpu-suite-v1-20261025`, run 2026-10-25 09:25–11:52 by `cpu_final.sh`: AC power,
Energy Mode Automatic (all 30 server starts logged `AC, powermode 0` in `conditions.log`, each
after the 1-minute load average fell below 3), 5 rounds, random engine order per round, 8 threads,
4,096 context per user, 8 slots, 128 output tokens, greedy, `ignore_eos`. Apple M4 Pro, 24 GiB,
macOS 27.0.1. The battery was charging (12 % to 95 %) during the run.

Traceability (`<run dir>/state/`): base commit `0570cf7` plus the uncommitted working tree
(`diff.patch`, `status.txt`), SHA-256 of `dynalm`, `bench_ppl` and `llama-server`. The diff, status
and binary hashes were checked again after the run and had not changed. Engines: DynaLM 0.2.0
(native kernels, `neon+qmm/neon_i8mm+sme_f32`), llama.cpp `9a16f1db` (Release, NATIVE, REPACK,
LLAMAFILE, BLAS Apple), Ollama 0.40.9. Models: `qwen2.5-1.5b-instruct-q4_k_m.gguf` and
`Qwen3-4B-Q4_K_M.gguf`, both Q4_K_M, same files for every engine.

Correctness gate (same run, same binaries; wikitext-2 test, 20 × 512 tokens, against the
llama.cpp reference logits): 1.5b PPL 11.6358, mean KL 1.7e-7, top-1 100 %; 4b PPL 18.4394,
mean KL 4.0e-7, top-1 100 %. Pass (limits: PPL ≤ +1 %, mean KL ≤ 0.0025). The 30 test programs,
and ASan / TSan (previous section), ran on this same source.

Validity: DynaLM and llama.cpp produced 128 of 128 tokens on all 600 requests each, with no
failures. Ollama on 1.5b stopped at end of text after 13–14 tokens on most requests in every
round (it does not honour `ignore_eos` for this model), so its 1.5b workloads (C1–C5) are not
comparable and are left out; Ollama on 4b produced 128 of 128 tokens on all requests.

### Score (`cpu_suite.py score`; ratio per round, median over 5 rounds, ± = round-to-round half-spread)

| Workload | Metric | vs llama.cpp | vs Ollama |
|---|---|---|---|
| C1 1.5b 1u 128p | output tok/s | 0.96× (±0%) | not comparable |
| C2 1.5b 1u 2048p | whole request p50 | 1.74× (±2%) | not comparable |
| C3 1.5b 4u 512p | aggregate output tok/s | 1.92× (±3%) | not comparable |
| C4 1.5b 8u 512p | aggregate output tok/s | 1.68× (±2%) | not comparable |
| C5 1.5b 8u 512p | gap between tokens p95 | 1.17× (±3%) | not comparable |
| C6 4b 1u 512p | output tok/s | 1.38× (±1%) | 1.31× (±1%) |
| C7 4b 1u 2048p | first token p50 | 2.23× (±1%) | 2.10× (±1%) |
| C8 4b 1u 2048p | gap between tokens p50 | 1.11× (±1%) | 1.07× (±1%) |

**Geometric mean vs llama.cpp: 1.47× over 8 of 8 workloads** (B0, Phase 8: 0.77×).
Vs Ollama: 1.43× over 3 of 8 workloads (incomplete; C1–C5 not comparable).

| Memory | DynaLM peak server RSS (MiB, median) | llama.cpp ÷ DynaLM | Ollama ÷ DynaLM |
|---|---|---|---|
| M1 1.5b | 1,510 | 2.73× (4,118 ÷ 1,510) | 2.32× (3,495 ÷ 1,510) |
| M2 4b | 3,945 | 3.39× (13,390 ÷ 3,945) | 2.80× (11,067 ÷ 3,945) |

llama.cpp allocates its whole KV cache (8 slots × 4,096) at start; DynaLM grows its pool on
demand. Both are the configuration the suite fixes; the ratio reflects that design difference.

### Per-metric medians (`<run dir>/per_metric.txt`, from the harness `single` / `multi` tables)

| Model, users, prompt | DynaLM | llama.cpp | Ollama |
|---|---|---|---|
| 1.5b 1u 128: out tok/s, TTFT, ITL p50 | 127.5, 160 ms, 6.5 ms | 132.4, 198 ms, 6.0 ms | short output |
| 1.5b 1u 2048: TTFT, ITL p50 | 2,079 ms, 7.5 ms | 4,325 ms, 7.7 ms | short output |
| 1.5b 4u 512: aggregate out tok/s, ITL p50 | 140.9, 13.4 ms | 73.0, 23.5 ms | short output |
| 1.5b 8u 512: aggregate out tok/s, ITL p50 / p99 | 146.9, 23.6 / 27.6 ms | 87.9, 28.4 / 31.2 ms | short output |
| 4b 1u 128: out tok/s, TTFT, ITL p50 | 54.5, 401 ms, 15.2 ms | 55.7, 524 ms, 13.9 ms | 55.9, 519 ms, 13.8 ms |
| 4b 1u 512: out tok/s, TTFT | 38.0, 1,254 ms | 27.7, 2,521 ms | 29.1, 2,378 ms |
| 4b 1u 2048: TTFT, ITL p50 | 5,599 ms, 20.4 ms | 12,443 ms, 22.6 ms | 11,848 ms, 21.6 ms |

Prompt processing (prompt ÷ TTFT, 2,048 tokens, 1 user): 1.5b 985 vs 474 tok/s (2.08×), 4b
366 vs 165 tok/s (2.22×) against llama.cpp. Short-context single-user decode is slower than
llama.cpp: ITL p50 6.5 vs 6.0 ms (1.5b) and 15.2 vs 13.9 ms (4b), about 8–9 % behind. With a
2,048-token KV cache DynaLM decodes faster (7.5 vs 7.7 ms, 20.4 vs 22.6 ms).

Outside the score (4b, 4 and 8 users; not suite workloads): aggregate 51.8 vs 27.0 and 54.3 vs
30.4 tok/s against llama.cpp; DynaLM's 8-user ITL p99 is 1,093 ms (decode rows waiting behind a
prompt chunk), against 88 ms for llama.cpp.

### Verdict

- **2× CPU speed: not met.** 1.47× geometric mean against llama.cpp, measured. Only C7 (4b
  long-prompt first token, 2.23×) reaches 2×; C3 (1.92×) is close. C1 is a loss (0.96×).
- **2× memory efficiency: met** for both models and both references (2.32×–3.39× by peak RSS).
- The battery development estimate (~1.66×) was too high: on AC llama.cpp gains more than DynaLM,
  as expected from single-user decode being bandwidth bound on AC.
- Remaining gaps: short-context single-user decode (C1, C6 decode part, C8), and the gap between
  tokens while prompts arrive (C5 1.17×; the 4b 8-user p99 above).

## Cycle 2 (decode and scheduling) — reference: the controlled E1–E9 run above

Baseline state saved in the session scratchpad (`cycle2-base/`: commit, status, diff, untracked
files, `dynalm` / `bench_op_trace` / `bench_ppl` binaries; `dynalm` SHA-256 `8b40ec49…`, the binary
of the controlled run).

### Diagnosis: short-context decode (AC, Energy Mode Automatic)

- Machine limits: 8-thread streaming read 238 GB/s, 4.37 GHz (scratchpad `bw`, `freq`).
- Qwen2.5-1.5B decode reads ~972 MB of weights per token (28 layers × ~27.9 MB + LM head 191 MB),
  so the bandwidth floor is ~4.1 ms; the controlled ITL p50 is 6.5 ms (~150 GB/s, 63 % of peak).
  llama.cpp: 6.0 ms. Bandwidth is not yet the limit.
- `bench_op_trace` (64–128 context): matvecs are 91 % of op time; attention 3.6 %, elementwise ops
  under 3 %.
- `bench_decode_matmul` (isolated, packed qmm gemv): LM head 216 GB/s and FFN down (Q4_K) 212 GB/s,
  but FFN gate/up 166 GB/s and attention Q/O 130 GB/s. In the real step (`bench_decode_context`)
  the per-phase bandwidth was lower still (gate+up 155, down 148, QKV 88, O 93 GB/s).
- Thread pool (scratchpad `poolbench`): an 8-thread region costs 1.2–1.3 µs to fork and join; the
  spin primitive (`yield` vs `isb`) makes little difference. Claiming chunks is what costs: 256
  chunks on the shared counter take ~10 µs. `matmul_chunks_per_thread = 32` (DD-067) makes decode
  chunks tiny: FFN gate chunks of 2 work items = 28 KB of weights, ~1 µs of work each.
- `DYNACORE_MATMUL_CHUNKS` sweep, isolated gemv: 32 → 1 or 2 chunks per thread raises FFN gate
  166 → 228 GB/s, FFN down (Q4_K) 212 → 237 GB/s (the measured DRAM peak), attention Q 130 → 200
  GB/s (only at 1 chunk per thread). Decode steps (`bench_decode_context`, 64 context): 7.3–7.7 ms
  at 32 chunks, 5.3–5.8 ms at 1. The decode kernels are bandwidth capable; chunking is the loss.

### E10 — decode matmul chunks: equal per thread, sized by weight bytes (kept)

- Hypothesis: decode-shaped matmuls lose 20–30 % of bandwidth to chunks too small to keep a
  prefetch stream going; chunks of at least 512 KB of weights (but never fewer chunks than
  threads) recover it, while large matmuls (LM head: ~370 chunks) keep dynamic balancing for
  uneven P/E cores.
- Change: `KernelPlan::gemv_min_chunk_bytes` (2 MiB, `DYNACORE_GEMV_CHUNK_KB`, 0 = old
  behaviour) and `gemv_grain()` in `cpu_device.cpp`, used for the qmm gemv path, the fused
  gate/up decode path and `matmul_many` when every job has at most `qmm::kGemvMaxRows` rows:
  every thread gets the same number of chunks, as many as hold at least 2 MiB of weights (1 to
  `matmul_chunks_per_thread`). Small matmuls get one chunk per thread; the LM head keeps ~19
  per thread for uneven cores. Prefill (gemm, SME) partitioning is unchanged.
- First form (rejected): "at least 512 KiB per chunk, never fewer chunks than threads". A chunk
  count just above the thread count (4B FFN down in 2 MiB chunks: 10 chunks, 8 threads) left two
  threads with twice the work; 4B gained only 7 %.
- Correctness: partitioning only (every output element is computed by the same code). 30/30
  test programs pass; gate bit-identical to E9 (1.5b PPL 11.635792, mean KL 1.7e-7; 4b PPL
  18.439435, mean KL 4.0e-7; top-1 100 %).
- A first A/B ran after the machine had dropped to battery / Low Power Mode (2.6 GHz, decode
  compute bound there); it is discarded.
- Measured (AC, Energy Mode Automatic; `bench_decode_context`, 8 threads, same binary,
  `DYNACORE_GEMV_CHUNK_KB=0` vs default vs one chunk per thread, 3 rounds, rotated order;
  scratchpad `e10_ab3.sh`). Decode step p50, median of 3 (round-to-round spread under 1 %):

  | Model, context | Old grain | E10 (2 MiB) | 1 chunk per thread |
  |---|---|---|---|
  | 1.5b, 64 | 6.70 ms | 5.26 ms (−21.5 %) | 5.24 ms |
  | 1.5b, 1024 | 7.03 | 5.87 (−16.5 %) | 5.83 |
  | 1.5b, 2048 | 7.45 | 6.31 (−15.3 %) | 6.31 |
  | 1.5b, 4096 | 8.46 | 7.25 (−14.3 %) | 7.27 |
  | 4b, 64 | 15.03 | 12.66 (−15.8 %) | 12.58 |
  | 4b, 1024 | 17.23 | 14.88 (−13.6 %) | 14.86 |
  | 4b, 2048 | 19.34 | 16.83 (−13.0 %) | 16.79 |
  | 4b, 4096 | 24.05 | 21.16 (−12.0 %) | 21.22 |

- Correctness (final form): 30/30 test programs pass; gate bit-identical to E9 (1.5b PPL
  11.635792, mean KL 1.7e-7; 4b PPL 18.439435, mean KL 4.0e-7; top-1 100 %).
- Decision: KEEP. Suite effect: controlled run at the end of this section (C1 +11 %, C6 +6 %, C8 +9 %).

### Diagnosis: 4B eight-user tail gap (controlled run raw data)

`4b-cpu-dynalm-r1.requests.jsonl`, 8 users: all 8 requests arrive at t = 0; the first prompt is
admitted alone and prefilled in one burst step (first token 1.33 s), the other seven prefill
together (first token 10.7 s). The first request finishes at 18.6 s; its successor's 512-token
prompt then arrives while the seven others decode, and runs as one ~1.3 s burst step (E4) during
which they wait. A burst step is never interrupted, so its length, not the 2,000 ms stall bound,
sets the gap: the ~1.1 s p99. It repeats every wave because that request stays out of step.
llama.cpp's requests stay in step (4 + 4 prompts, finishing together).

### E11 — capped prefill bursts while streams decode (rejected, reverted)

- Hypothesis: capping a burst step's prompt rows while streams wait, and giving the decoders a
  decode-only step between such capped steps, bounds the gap at about one capped step.
- Change: `SchedulerConfig::prefill_burst_decode_chunk` (default 0 = unchanged;
  `DYNALM_PREFILL_BURST_DECODE_CHUNK`), counter `steps_burst_decode_turn`. Bursts while no stream
  decodes keep the whole 512-row budget.
- Correctness: test_engine `test_prefill_burst` gains a capped case: at most `cap` prompt rows per
  burst step, never mixed with A's decode rows, A decodes between any two bursts, A's greedy text
  equals A alone. All engine tests pass.
- Development A/B (battery, Low Power Mode, NOT controlled; 2 rounds, rotated order, same prompts
  per round; 4b, 8 users, 512-token prompts, 32 requests; scratchpad `abenv.sh`):

  | Cap | Aggregate tok/s | Gap p50 / p99 ms | First token p50 ms | Request p99 ms |
  |---|---|---|---|---|
  | 0 (current) | 31.0, 30.7 | 125 / 2,143, 2,151 | 15,238, 13,091 | 35,238, 35,734 |
  | 128 | 27.4, 26.7 | 125 / 757, 803 | 19,658, 19,682 | 40,657, 41,804 |
  | 256 | 30.6, 30.2 | 125 / 1,188, 1,194 | 13,812, 16,188 | 36,896, 36,530 |

  128 cuts the tail 63 % but costs 12 % aggregate throughput and 30–50 % first-token time; 256
  cuts it 45 % for ~1.5 % throughput (battery; not used for the decision).
- Measured (AC, Energy Mode Automatic, with E10; 3 rounds, rotated order, same prompts per round;
  the benchmark reports gap p50 / p90 / p99, not p95):

  | Cap | Aggregate tok/s | Gap p50 / p90 / p99 ms | First token p50 ms | Request p99 ms |
  |---|---|---|---|---|
  | 0 | 56.3, 56.3, 57.4 | 60 / 62–64 / 1,082, 1,077, 112 | 9,378, 9,336, 10,199 | 18,907, 18,814, 18,291 |
  | 256 | 52.5, 53.3, 52.5 | 61 / 63 / 799, 796, 799 | 10,551, 9,070, 10,567 | 21,431, 20,807, 21,442 |
  | 128 | 47.8, 47.8, 46.4 | 61 / 64–66 / 511, 512, 518 | 11,027, 11,020, 12,779 | 22,588, 22,687, 24,567 |

- Decision: REJECT. The tail gain (−26 % at 256, −52 % at 128) costs 7–16 % aggregate
  throughput and 13–34 % whole-request p99. In round 3 without a cap the requests happened to
  stay in step and the p99 was 112 ms; caps make prompts finish at different times, so streams
  fall out of step by themselves (cap 256, round 3: 799 ms). The code and its test were reverted.
- Conclusion: the 4B tail is structural on this CPU. A 512-token 4B prompt is ~1.3 s of prefill
  compute; a stream either waits for it or the prompt is sliced and throughput drops. Options
  that do not trade throughput: faster prefill (the SME path is the lever), or keeping
  concurrent requests in step at admission. The second only helps synchronized arrivals such as
  the closed-loop benchmark, so it was not tried. The suite scores the 8-user gap on 1.5B only
  (C5), where a prompt is ~0.5 s.

### Controlled CPU suite v1 run with E10 (2026-10-25, 17:05–19:37)

`cpu_final.sh` with `D=20261025-e10`, results in `test/results/cpu-suite-v1-20261025-e10` (state
snapshot in its `state/`). Same workloads, models and settings as the E1–E9 run. All 30 server
starts on AC power, Energy Mode Automatic; all 1,200 DynaLM and llama.cpp requests returned 128 of
128 tokens. Gate: 1.5B PPL 11.6358, mean KL 1.7e-7; 4B PPL 18.4394, mean KL 4.0e-7 (identical to
the E1–E9 run).

| Workload | E1–E9 vs llama.cpp | E10 vs llama.cpp | E10 vs Ollama |
|---|---|---|---|
| C1 1.5b 1u 128p, output tok/s | 0.96× | 1.08× | not comparable |
| C2 1.5b 1u 2048p, whole request p50 | 1.74× | 1.78× | not comparable |
| C3 1.5b 4u 512p, aggregate tok/s | 1.92× | 1.95× | not comparable |
| C4 1.5b 8u 512p, aggregate tok/s | 1.68× | 1.68× | not comparable |
| C5 1.5b 8u 512p, gap p95 | 1.17× | 1.19× | not comparable |
| C6 4b 1u 512p, output tok/s | 1.38× | 1.46× | 1.40× |
| C7 4b 1u 2048p, first token p50 | 2.23× | 2.26× | 2.14× |
| C8 4b 1u 2048p, gap p50 | 1.11× | 1.23× | 1.19× |
| Geometric mean | 1.47× (8 of 8) | 1.53× (8 of 8) | 1.53× (3 of 8) |

Memory (peak server RSS, median): M1 1,508 MiB (llama.cpp ÷ DynaLM 2.73×, Ollama 2.32×), M2
3,944 MiB (2.89×, 2.27×). The llama.cpp and Ollama 4B peaks were lower this run (11,403 and
8,949 MiB against 13,390 and 10,087); DynaLM's did not change.

`cpu_suite.py compare` (E1–E9 run vs this run, DynaLM only): C1 1.112×, C6 1.055×, C8 1.092×
better; C2–C5 and C7 within noise; RSS the same. Decision: KEEP.

Verdict: E10 raises the score from 1.47× to 1.53× vs llama.cpp and closes the C1 gap (now ahead).
The mission target (2×) is not reached. The low workloads are C1 (1.08×), C5 (1.19×), C8 (1.23×)
and C6 (1.46×): decode on one stream and gaps under new prompts.

### Release validation (2026-10-25, before v0.3.0 review)

- AddressSanitizer + UBSan (`build/native-asan`) and ThreadSanitizer (`build/native-tsan`),
  macOS arm64, rebuilt from the current tree: `tools/ci/run_tests.sh`, 30 of 30 programs pass.
  The CPU tests (test_qmm, test_matmul_registry, test_cpu_attention, test_engine) also pass
  under both with `DYNACORE_GEMV_CHUNK_KB` = 0 (the old chunking), 1 and 64, so every branch of
  the E10 grain runs under the sanitizers.
- x86-64: Ubuntu 24.04 linux/amd64 container (Rosetta; AVX2, FMA, F16C; no AVX-512 or AVX-VNNI),
  strict native (`DYNALM_NATIVE_ONLY=ON`, `ENABLE_GGML_CPU=OFF`). GCC 13.3 and Clang 18.1: build,
  all tests, `tools/ci/check_native.sh` and `dynalm doctor` (kernel self-test avx2 vs generic:
  max rel. diff 2.0e-6) pass. Functional only: no x86 performance was measured.
- Found by the CPU-only release tree: `test_device_selection` decided "GPU" from the backend
  name, and DynaCore's CPU device under `--execution auto` is named "compiled/CPU". It failed on
  ARM builds without a GPU backend (hidden where native Metal is built). Fixed in the test (it now
  checks `Engine::device_kind()`); no engine change.
