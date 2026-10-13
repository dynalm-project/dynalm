# Release-candidate runs (results\dynalm-next\release-rc-*\benchmark_plan.md); dot-sourced by next_sweep.ps1.
# Executables: DYNALM_ORIG_EXE (reference) and DYNALM_RC_EXE (candidate) must be set.
if (-not $env:DYNALM_ORIG_EXE -or -not $env:DYNALM_RC_EXE) { throw "set DYNALM_ORIG_EXE and DYNALM_RC_EXE" }
$Orig = $env:DYNALM_ORIG_EXE
$Rc = $env:DYNALM_RC_EXE
foreach ($e in $Orig, $Rc) { if (-not (Test-Path $e -PathType Leaf)) { throw "missing executable $e" } }
function Add-BenchExe($id, $exe, $model, [string[]]$extra, $env = @{}) {
  Add-Run $id $exe (@("benchmark", $model, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl"),
      "--request-log", (Join-Path $Out "$id.requests.jsonl")) + $extra) $env $model
}
$RArms = [ordered]@{
  "orig"  = @{ exe = $Orig; env = @{} }
  "rc"    = @{ exe = $Rc; env = @{} }
  "split" = @{ exe = $Rc; env = @{ DYNACORE_INT16_PREFILL_ROWS = "64"; DYNACORE_INT16_PREFILL_SPLIT = "1" } }
  "wide"  = @{ exe = $Rc; env = @{ DYNACORE_DECODE_WIDE_ROWS = "16" } }
  "sw"    = @{ exe = $Rc; env = @{ DYNACORE_INT16_PREFILL_ROWS = "64"; DYNACORE_INT16_PREFILL_SPLIT = "1"; DYNACORE_DECODE_WIDE_ROWS = "16" } }
}
$names = @("orig", "rc", "split", "wide", "sw")
$wl = @(
  @{ id = "Q15"; model = $Q15; args = @("--concurrency", "1,4", "--prompt", "128,512", "--output", "128", "--requests", "8", "--warmup-requests", "1", "-c", "8192") },
  @{ id = "Q3"; model = $Q3; args = @("--concurrency", "1,4", "--prompt", "128", "--output", "64", "--requests", "8", "--warmup-requests", "1", "-c", "8192") }
)
foreach ($w in $wl) {
  for ($r = 0; $r -lt 3; $r++) {
    for ($i = 0; $i -lt $names.Count; $i++) {
      $a = $names[($i + $r) % $names.Count]
      Add-BenchExe ("R-{0}-r{1}-{2}" -f $w.id, ($r + 1), $a) $RArms[$a].exe $w.model $w.args $RArms[$a].env
    }
  }
}
$c16 = @("--concurrency", "16", "--prompt", "128", "--output", "128", "--requests", "32", "--warmup-requests", "1", "-c", "16384")
$cn = @("rc", "split", "sw")
for ($r = 0; $r -lt 3; $r++) {
  for ($i = 0; $i -lt $cn.Count; $i++) {
    $a = $cn[($i + $r) % $cn.Count]
    Add-BenchExe ("R-C16-r{0}-{1}" -f ($r + 1), $a) $RArms[$a].exe $Q15 $c16 $RArms[$a].env
  }
}
# Tail-latency traces (diagnostic): one point each, scheduler step trace.
foreach ($a in "orig", "rc") {
  Add-BenchExe "R-TR-q15-c4-$a" $RArms[$a].exe $Q15 @("--concurrency", "4", "--prompt", "128", "--output", "128", "--requests", "16",
    "--warmup-requests", "1", "-c", "8192", "--step-trace", (Join-Path $Out "R-TR-q15-c4-$a.steps.jsonl"))
  Add-BenchExe "R-TR-q3-c4-$a" $RArms[$a].exe $Q3 @("--concurrency", "4", "--prompt", "128", "--output", "64", "--requests", "16",
    "--warmup-requests", "1", "-c", "8192", "--step-trace", (Join-Path $Out "R-TR-q3-c4-$a.steps.jsonl"))
}
