# Release campaign runs (results\dynalm-release-*\campaign\plan.md); dot-sourced by next_sweep.ps1.
# One invocation = one measurement session (-Session N); arm order rotates with the session.
# Executables: DYNALM_BASE_EXE (reference build) and DYNALM_CAND_EXE (candidate build) must be set.
if (-not $env:DYNALM_BASE_EXE -or -not $env:DYNALM_CAND_EXE) { throw "set DYNALM_BASE_EXE and DYNALM_CAND_EXE" }
$BaseExe = $env:DYNALM_BASE_EXE
$CandExe = $env:DYNALM_CAND_EXE
foreach ($e in $BaseExe, $CandExe) { if (-not (Test-Path $e -PathType Leaf)) { throw "missing executable $e" } }
function Add-Arm($id, $arm, $model, [string[]]$wargs) {
  $spec = $Campaign[$arm]
  Add-Run $id $spec.exe (@("benchmark", $model, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl"),
      "--request-log", (Join-Path $Out "$id.requests.jsonl")) + $wargs + $spec.args) $spec.env $model
}
$split = @{ DYNACORE_INT16_PREFILL_ROWS = "64"; DYNACORE_INT16_PREFILL_SPLIT = "1" }
$Campaign = @{
  "orig"  = @{ exe = $BaseExe; env = @{}; args = @() }
  "rc"    = @{ exe = $CandExe; env = @{}; args = @() }
  "wide"  = @{ exe = $CandExe; env = @{ DYNACORE_DECODE_WIDE_ROWS = "16" }; args = @() }
  "wide4" = @{ exe = $CandExe; env = @{ DYNACORE_DECODE_WIDE_ROWS = "16"; DYNACORE_WIDE_ROWS8 = "0" }; args = @() }
  "split" = @{ exe = $CandExe; env = $split; args = @() }
  "sw"    = @{ exe = $CandExe; env = ($split + @{ DYNACORE_DECODE_WIDE_ROWS = "16" }); args = @() }
  "lat"   = @{ exe = $CandExe; env = @{}; args = @("--policy", "latency") }
  "pm16"  = @{ exe = $CandExe; env = @{ DYNALM_PROTECT_MIN = "16"; DYNALM_DECODE_TARGET = "3" }; args = @() }
}
function Rotate([string[]]$a, [int]$by) { $n = $a.Count; 0..($n - 1) | ForEach-Object { $a[($_ + $by) % $n] } }
$s = $Session
$wl = @(
  @{ id = "HC-q15"; model = $Q15; arms = @("orig", "rc", "wide", "wide4", "sw"); rounds = 1
     args = @("--concurrency", "8,12,16", "--prompt", "128", "--output", "128", "--requests", "32", "--warmup-requests", "1", "-c", "16384") },
  @{ id = "LC-q15"; model = $Q15; arms = @("orig", "rc", "split", "wide"); rounds = 1
     args = @("--concurrency", "1,4", "--prompt", "128,512", "--output", "128", "--requests", "8", "--warmup-requests", "1", "-c", "8192") },
  @{ id = "TL-q15"; model = $Q15; arms = @("rc", "lat", "pm16"); rounds = 2
     args = @("--concurrency", "4", "--prompt", "128", "--output", "128", "--requests", "16", "--warmup-requests", "1", "-c", "8192") },
  @{ id = "HC-q3"; model = $Q3; arms = @("orig", "rc", "wide", "wide4", "sw"); rounds = 1
     args = @("--concurrency", "8,12,16", "--prompt", "128", "--output", "64", "--requests", "32", "--warmup-requests", "1", "-c", "16384") },
  @{ id = "LC-q3"; model = $Q3; arms = @("orig", "rc", "split", "wide"); rounds = 1
     args = @("--concurrency", "1,4", "--prompt", "128", "--output", "64", "--requests", "8", "--warmup-requests", "1", "-c", "8192") },
  @{ id = "TL-q3"; model = $Q3; arms = @("rc", "lat", "pm16"); rounds = 2
     args = @("--concurrency", "4", "--prompt", "128", "--output", "64", "--requests", "16", "--warmup-requests", "1", "-c", "8192") }
)
foreach ($w in $wl) {
  for ($r = 1; $r -le $w.rounds; $r++) {
    $order = Rotate $w.arms ($s - 1 + ($r - 1))
    if ($r % 2 -eq 0) { [array]::Reverse($order) }
    foreach ($a in $order) { Add-Arm ("{0}-s{1}r{2}-{3}" -f $w.id, $s, $r, $a) $a $w.model $w.args }
  }
}
