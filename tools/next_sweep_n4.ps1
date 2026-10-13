# Phase 4 runs (results\dynalm-next\phase4\plan.md); dot-sourced by next_sweep.ps1.
$Ffn = Join-Path $Root "build\msvc-release\dynacore\bench_ffn_kernels.exe"
# llama.cpp binaries: DYNALM_LLAMACPP_DIR, else this workspace's .bench\llamacpp.
$Llama = if ($env:DYNALM_LLAMACPP_DIR) { $env:DYNALM_LLAMACPP_DIR } else { Join-Path $Root ".bench\llamacpp" }
$Q15Full = Join-Path $Root $Q15
foreach ($c in 0, 4) {
  Add-Run "N4-PROBE-cpu$c" $Ffn @("$c", "probe")
  Add-Run "N4-K-cpu$c" $Ffn @("$c", "8960", "1536", "16,32,48,64")
}
$C16 = @("--concurrency", "16", "--prompt", "128", "--output", "128", "--requests", "32", "--warmup-requests", "1", "-c", "16384")
Add-Bench "N4-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
$alt = @(@("ref", "split"), @("split", "ref"), @("ref", "split"))
for ($r = 0; $r -lt 3; $r++) { foreach ($a in $alt[$r]) { Add-Bench ("N4-C16-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 $C16 $Arms[$a] } }
foreach ($c in 1, 2, 4, 8, 16) {
  $n = [math]::Max(4, 2 * $c)
  Add-Bench "N4-SCALE-q15-c$c" $Q15 @("--concurrency", "$c", "--prompt", "128", "--output", "128", "--requests", "$n", "--warmup-requests", "1", "-c", "16384")
}
Add-Run "N4-LB-q15" (Join-Path $Llama "llama-bench.exe") @("-m", $Q15Full, "-p", "2048", "-n", "128", "-t", "$Threads", "-r", "3") @{} $Q15
Add-Run "N4-LBB-q15" (Join-Path $Llama "llama-batched-bench.exe") @("-m", $Q15Full, "-c", "8192", "-npp", "128", "-ntg", "128", "-npl", "1,2,4,8,16", "-t", "$Threads") @{} $Q15
# Addendum W16 (plan.md): wide decode, alone and with the split kernel.
$Arms["wide"] = @{ DYNACORE_DECODE_WIDE_ROWS = "16" }
$Arms["sw"] = @{ DYNACORE_INT16_PREFILL_ROWS = "64"; DYNACORE_INT16_PREFILL_SPLIT = "1"; DYNACORE_DECODE_WIDE_ROWS = "16" }
$rotW = @(@("ref", "wide", "sw"), @("wide", "sw", "ref"), @("sw", "ref", "wide"))
for ($r = 0; $r -lt 3; $r++) { foreach ($a in $rotW[$r]) { Add-Bench ("N4W-C16-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 $C16 $Arms[$a] } }
