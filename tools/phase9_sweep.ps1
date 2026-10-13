<#
.SYNOPSIS
  Phase 9 validation of the experimental solo-prefill chunk (SchedulerConfig::solo_prefill_chunk):
  DD-053 accuracy, correctness tests, in-process A/B, long-prompt / mixed-length and open-loop
  (staggered-arrival) workloads. The serving A/B (V5) uses tools\pk8_validation.ps1 -Feature soloprefill.

.DESCRIPTION
  Stages (-Stage, any subset):
    V0  DD-053: bench_int8_accuracy, BENCH_PREFILL_ONLY=1 BENCH_PREFILL_CHUNKS=32,64,128,
        Qwen2.5-1.5B and Qwen3-4B, INT16 prefill off and DYNACORE_INT16_PREFILL_ROWS=128.
    V1  test_solo_prefill (Qwen2.5-1.5B) and test_int16_prefill.
    V2  dynalm benchmark in-process, 128/128, c1/5/8; arms off / fifo (--solo-prefill-chunk 128) /
        split (+ --solo-prefill-split). 1.5B: 16 requests, 3 repeats; 4B: 8 requests, 2 repeats.
        Rotated order.
    V3  long prompts and mixed traffic, off vs fifo only (split is never set), 3 repeats, order
        off/fifo, fifo/off, off/fifo:
          L1  c1, --prompt 512,2048 --output 32: Qwen2.5-1.5B (3 requests) and Qwen3-4B (2 requests)
          M1  Qwen2.5-1.5B c5, --prompt-mix 128,128,128,128,2048 --output 64, 10 requests
          M2  Qwen2.5-1.5B c8, --prompt-mix 128 x 7 + 2048 --output 64, 16 requests
          A1  Qwen2.5-1.5B bench_long_prompt, BENCH_LONG_CONFIGS=default,solo-fifo (order alternates):
              8 streams decode, then a 2048-token prompt arrives
    V4  open-loop validation of the lone-prompt rule, Qwen2.5-1.5B, arms off / lone, output 64,
        at most 16 in flight, --warmup-requests 2, per-request logs; use -IdleSeconds 60:
          P    --arrival-trace v4\traces\probe.csv (2048-token prompt, then a 128-token prompt DELTA
               later; 8 events 45 s apart), 4 pairs
          Q1   Poisson 0.05 req/s seed 11, mix 128 x 7 + 2048, 20 requests, 4 pairs
          Q2   Poisson 0.10 req/s seed 12, 28 requests, 4 pairs
          Q3   Poisson 0.15 req/s seed 13, 32 requests, 4 pairs
          Q2b  Poisson 0.10 req/s seed 22, 28 requests, 2 pairs (robustness)
    V3R re-validation of the lone-prompt rule (FIFO solo chunk only when exactly one prompt prefills
        and none decodes), Qwen2.5-1.5B, arms off / lone (--solo-prefill-chunk 128; split never set),
        4 paired repeats in order off/lone, lone/off, off/lone, lone/off; use -IdleSeconds 60:
          R1  c1, --prompt 512,2048 --output 32, 4 requests, --warmup-requests 1
          R2  c5, --prompt-mix 128,128,128,128,2048 --output 64, 20 requests (4 long), --warmup-requests 2
          R3  c8, --prompt-mix 128 x 7 + 2048 --output 64, 24 requests (3 long), --warmup-requests 2

  Every run is one process, started and awaited here. Nothing else is started, stopped or killed.
  Refuses to start when a dynalm / llama / ollama process runs, when DYNACORE_* / DYNALM_* variables
  are set (the script sets a run's own variables and removes them afterwards), or when free RAM is
  below -MinFreeGb. A run's output file is never overwritten.

  -Only id,id   run just these ids (e.g. one Qwen3-4B run per call, in the foreground)
  -Resume       continue in an existing -Out; ids already finished with exit 0 are skipped

  Output (-Out): preflight.json, plan.json, <id>.jsonl (dynalm benchmark), <id>.log, runs.jsonl.

.EXAMPLE
  pwsh -NoProfile -File tools\phase9_sweep.ps1 -Stage V0,V1,V2,V3,V4 -DryRun
.EXAMPLE
  pwsh -NoProfile -File tools\phase9_sweep.ps1 -Stage V2 -Out results\phase9-v2 -Resume -Only V2-q3-r1-off
#>
[CmdletBinding()]
param(
  [ValidateSet("V0", "V1", "V2", "V3", "V4", "V3R", "A3", "B")]
  [string[]]$Stage = @("V0", "V1", "V2", "V3", "V4"),
  [string]$Out = "",
  [string[]]$Only = @(),
  [switch]$Resume,
  [int]$Threads = 10,
  [double]$MinFreeGb = 6,
  [int]$IdleSeconds = 0,  # idle wait before every run (V3R: 60), recorded in runs.jsonl
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Out) { $Out = "results\phase9-validation-" + (Get-Date -Format "yyyyMMdd-HHmmss") }

$Dynalm = "build\msvc-release\bin\dynalm.exe"
$Bench = "build\msvc-release\dynalm\benchmarks"
$Q15 = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$Q3 = "models\Qwen3-4B-Q4_K_M.gguf"

$runs = [System.Collections.Generic.List[object]]::new()
function Add-Run($id, $exe, [string[]]$a, [hashtable]$envs = @{}) {
  $runs.Add([pscustomobject]@{ id = $id; exe = $exe; args = $a; env = $envs })
}
function Add-Bench($id, $model, [string[]]$extra) {
  Add-Run $id $Dynalm (@("benchmark", $model, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl")) + $extra)
}
function Arm($arm) {
  switch ($arm) {
    "off" { @() }
    "fifo" { @("--solo-prefill-chunk", "128") }
    "split" { @("--solo-prefill-chunk", "128", "--solo-prefill-split") }
  }
}

if ($Stage -contains "V0") {
  foreach ($m in @(@("q15", $Q15), @("q3", $Q3))) {
    $e = @{ BENCH_PREFILL_ONLY = "1"; BENCH_PREFILL_CHUNKS = "32,64,128" }
    Add-Run "V0-$($m[0])-int16off" "$Bench\bench_int8_accuracy.exe" @($m[1], "$Threads") $e
    $e2 = $e.Clone(); $e2["DYNACORE_INT16_PREFILL_ROWS"] = "128"
    Add-Run "V0-$($m[0])-int16r128" "$Bench\bench_int8_accuracy.exe" @($m[1], "$Threads") $e2
  }
}
if ($Stage -contains "V1") {
  Add-Run "V1-test_int16_prefill" "build\msvc-release\dynacore\test_int16_prefill.exe" @()
  Add-Run "V1-test_solo_prefill" "$Bench\test_solo_prefill.exe" @($Q15, "$Threads")
}
if ($Stage -contains "V2") {
  $base = @("--concurrency", "1,5,8", "--prompt", "128", "--output", "128", "-c", "4096")
  Add-Bench "V2-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $orders = @(@("off", "fifo", "split"), @("split", "off", "fifo"), @("fifo", "split", "off"))
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($a in $orders[$r]) { Add-Bench ("V2-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 ($base + @("--requests", "16") + (Arm $a)) }
  }
  Add-Bench "V2-q3-w00" $Q3 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  for ($r = 0; $r -lt 2; $r++) {
    foreach ($a in $orders[$r]) { Add-Bench ("V2-q3-r{0}-{1}" -f ($r + 1), $a) $Q3 ($base + @("--requests", "8") + (Arm $a)) }
  }
}
if ($Stage -contains "V3") {
  # Approved V3 matrix (off vs fifo only; never --solo-prefill-split). Qwen3-4B M1 excluded.
  $order = @(@("off", "fifo"), @("fifo", "off"), @("off", "fifo"))
  $warm = @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  Add-Bench "V3-q15-w00" $Q15 $warm
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($a in $order[$r]) {
      Add-Bench ("V3-L1-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 (@("--concurrency", "1", "--prompt", "512,2048", "--output", "32", "--requests", "3", "-c", "8192") + (Arm $a))
    }
  }
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($a in $order[$r]) {
      Add-Bench ("V3-M1-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 (@("--concurrency", "5", "--prompt-mix", "128,128,128,128,2048", "--output", "64", "--requests", "10", "-c", "16384") + (Arm $a))
    }
  }
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($a in $order[$r]) {
      Add-Bench ("V3-M2-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 (@("--concurrency", "8", "--prompt-mix", "128,128,128,128,128,128,128,2048", "--output", "64", "--requests", "16", "-c", "16384") + (Arm $a))
    }
  }
  $a1 = @("default,solo-fifo", "solo-fifo,default", "default,solo-fifo")
  for ($r = 0; $r -lt 3; $r++) {
    Add-Run ("V3-A1-q15-r{0}" -f ($r + 1)) "$Bench\bench_long_prompt.exe" @($Q15, "$Threads") @{ BENCH_LONG_CONFIGS = $a1[$r] }
  }
  Add-Bench "V3-q3-w00" $Q3 $warm
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($a in $order[$r]) {
      Add-Bench ("V3-L1-q3-r{0}-{1}" -f ($r + 1), $a) $Q3 (@("--concurrency", "1", "--prompt", "512,2048", "--output", "32", "--requests", "2", "-c", "8192") + (Arm $a))
    }
  }
}
if ($Stage -contains "V3R") {
  # Lone-prompt rule re-validation (approved V3R plan): off vs lone; never --solo-prefill-split.
  $order = @(@("off", "lone"), @("lone", "off"), @("off", "lone"), @("lone", "off"))
  $armArgs = @{ off = @(); lone = @("--solo-prefill-chunk", "128") }
  Add-Bench "V3R-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $work = @(
    @{ id = "R1"; args = @("--concurrency", "1", "--prompt", "512,2048", "--output", "32", "--requests", "4", "--warmup-requests", "1", "-c", "8192") },
    @{ id = "R2"; args = @("--concurrency", "5", "--prompt-mix", "128,128,128,128,2048", "--output", "64", "--requests", "20", "--warmup-requests", "2", "-c", "16384") },
    @{ id = "R3"; args = @("--concurrency", "8", "--prompt-mix", "128,128,128,128,128,128,128,2048", "--output", "64", "--requests", "24", "--warmup-requests", "2", "-c", "16384") }
  )
  foreach ($w in $work) {
    for ($r = 0; $r -lt 4; $r++) {
      foreach ($a in $order[$r]) { Add-Bench ("V3R-{0}-q15-r{1}-{2}" -f $w.id, ($r + 1), $a) $Q15 ($w.args + $armArgs[$a]) }
    }
  }
}
if ($Stage -contains "A3") {
  # Phase 10 Track A3 (results\phase10-next-20261013-084626\trackA\plan.md): prefill order fifo vs srpt,
  # solo prefill off, per-request logs and scheduler step traces. Never --solo-prefill-chunk / split.
  $ta = "results\phase10-next-20261013-084626\trackA\traces"
  $order = @(@("fifo", "srpt"), @("srpt", "fifo"), @("fifo", "srpt"), @("srpt", "fifo"))
  $armArgs = @{ fifo = @("--prefill-order", "fifo"); srpt = @("--prefill-order", "srpt", "--prefill-aging-ms", "20000") }
  $common = @("--concurrency", "16", "--output", "64", "--warmup-requests", "2", "-c", "16384")
  $mix = @("--prompt-mix", "128,128,128,128,128,128,128,2048")
  Add-Bench "A3-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $work = @(
    @{ id = "H"; pairs = 4; args = @("--arrival-trace", "$ta\hol.csv") },
    @{ id = "Q3"; pairs = 4; args = $mix + @("--requests", "32", "--arrival-rate", "0.15", "--seed", "13") },
    @{ id = "P"; pairs = 3; args = @("--arrival-trace", "$ta\probe.csv") },
    @{ id = "S"; pairs = 3; args = $mix + @("--requests", "48", "--arrival-rate", "0.15", "--seed", "31") }
  )
  # Addendum A4 (plan.md): fifo vs srpt with aging 5000 ms on H and Q3 (ids A4-*).
  $armArgs["srpt5"] = @("--prefill-order", "srpt", "--prefill-aging-ms", "5000")
  $order4 = @(@("fifo", "srpt5"), @("srpt5", "fifo"), @("fifo", "srpt5"), @("srpt5", "fifo"))
  foreach ($w in @($work[0], $work[1])) {
    for ($r = 0; $r -lt 4; $r++) {
      foreach ($a in $order4[$r]) {
        $id = "A4-{0}-q15-r{1}-{2}" -f $w.id, ($r + 1), $a
        Add-Bench $id $Q15 ($common + $w.args + @("--request-log", (Join-Path $Out "$id.requests.jsonl"), "--step-trace", (Join-Path $Out "$id.steps.jsonl")) + $armArgs[$a])
      }
    }
  }
  foreach ($w in $work) {
    for ($r = 0; $r -lt $w.pairs; $r++) {
      foreach ($a in $order[$r]) {
        $id = "A3-{0}-q15-r{1}-{2}" -f $w.id, ($r + 1), $a
        Add-Bench $id $Q15 ($common + $w.args + @("--request-log", (Join-Path $Out "$id.requests.jsonl"), "--step-trace", (Join-Path $Out "$id.steps.jsonl")) + $armArgs[$a])
      }
    }
  }
}
if ($Stage -contains "B") {
  # Phase 10 Track B (results\phase10-next-20261013-084626\trackB\plan.md): INT16 prefill off vs
  # DYNACORE_INT16_PREFILL_ROWS=64 (the Phase 7 configuration; Q4_K packed weights only, Q6_K packing stays off).
  # Scheduler defaults (fifo, solo prefill off). The variable is set for its run only.
  $int16 = @{ DYNACORE_INT16_PREFILL_ROWS = "64" }
  $order = @(@("off", "int16"), @("int16", "off"), @("off", "int16"), @("int16", "off"))
  foreach ($m in @(@("q15", $Q15), @("q3", $Q3))) {
    Add-Run "B-DD053-$($m[0])-int16" "$Bench\bench_int8_accuracy.exe" @($m[1], "$Threads") (@{ BENCH_PREFILL_ONLY = "1"; BENCH_PREFILL_CHUNKS = "32,64" } + $int16)
  }
  Add-Bench "B-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $work = @(
    @{ id = "L"; pairs = 3; args = @("--concurrency", "1", "--prompt", "2048", "--output", "32", "--requests", "4", "--warmup-requests", "1", "-c", "8192") },
    @{ id = "Q3"; pairs = 3; args = @("--concurrency", "16", "--output", "64", "--warmup-requests", "2", "-c", "16384", "--prompt-mix", "128,128,128,128,128,128,128,2048", "--requests", "32", "--arrival-rate", "0.15", "--seed", "13") }
  )
  foreach ($w in $work) {
    for ($r = 0; $r -lt $w.pairs; $r++) {
      foreach ($a in $order[$r]) {
        $id = "B-{0}-q15-r{1}-{2}" -f $w.id, ($r + 1), $a
        $e = if ($a -eq "int16") { $int16 } else { @{} }
        Add-Run $id $Dynalm (@("benchmark", $Q15, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl")) + $w.args + @("--request-log", (Join-Path $Out "$id.requests.jsonl"))) $e
      }
    }
  }
  # bench_profile runs one 256-row forward (above the 64-row int16 limit): per-op reference only. The int16
  # per-op comparison comes from the B-L runs' op_ms (32-row prefill chunks, where int16 engages).
  Add-Run "B-PROF-q15-off" "$Bench\bench_profile.exe" @($Q15, "$Threads")
  Add-Run "B-MATMUL-q15" "$Bench\bench_decode_matmul.exe" @($Q15, "$Threads", "16,32,64,128") @{ BENCH_INT16_PREFILL = "1" }
}
if ($Stage -contains "V4") {
  # Open-loop validation of the lone-prompt rule (V4 plan): off vs lone; never --solo-prefill-split.
  # Every run writes a per-request log next to its point JSON. The trace / seed is fixed per comparison.
  $trace = "results\phase9-adaptive-20261012-135244\v4\traces\probe.csv"
  $order = @(@("off", "lone"), @("lone", "off"), @("off", "lone"), @("lone", "off"))
  $armArgs = @{ off = @(); lone = @("--solo-prefill-chunk", "128") }
  $common = @("--concurrency", "16", "--output", "64", "--warmup-requests", "2", "-c", "16384")
  $mix = @("--prompt-mix", "128,128,128,128,128,128,128,2048")
  Add-Bench "V4-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $work = @(
    @{ id = "P"; pairs = 4; args = @("--arrival-trace", $trace) },
    @{ id = "Q1"; pairs = 4; args = $mix + @("--requests", "20", "--arrival-rate", "0.05", "--seed", "11") },
    @{ id = "Q2"; pairs = 4; args = $mix + @("--requests", "28", "--arrival-rate", "0.10", "--seed", "12") },
    @{ id = "Q3"; pairs = 4; args = $mix + @("--requests", "32", "--arrival-rate", "0.15", "--seed", "13") },
    @{ id = "Q2b"; pairs = 2; args = $mix + @("--requests", "28", "--arrival-rate", "0.10", "--seed", "22") }
  )
  foreach ($w in $work) {
    for ($r = 0; $r -lt $w.pairs; $r++) {
      foreach ($a in $order[$r]) {
        $id = "V4-{0}-q15-r{1}-{2}" -f $w.id, ($r + 1), $a
        Add-Bench $id $Q15 ($common + $w.args + @("--request-log", (Join-Path $Out "$id.requests.jsonl")) + $armArgs[$a])
      }
    }
  }
}

$sel = if ($Only.Count) { @($runs | Where-Object { $Only -contains $_.id }) } else { @($runs) }
if ($Only.Count -and $sel.Count -ne $Only.Count) { throw ("unknown -Only id(s): " + (($Only | Where-Object { $_ -notin $runs.id }) -join ", ")) }
Write-Host ("{0} of {1} runs -> {2}" -f $sel.Count, $runs.Count, $Out)
foreach ($r in $sel) {
  $envText = ($r.env.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " "
  Write-Host ("  {0,-24} {1}{2} {3}" -f $r.id, $(if ($envText) { "[$envText] " } else { "" }), $r.exe, ($r.args -join " "))
}
if ($DryRun) { Write-Host "`n(dry run: nothing started)"; return }

# --- preflight -------------------------------------------------------------------------
function Fail($msg) { throw "preflight: $msg" }
if ((Test-Path $Out) -and -not $Resume) { Fail "output directory $Out exists (use -Resume to continue it; results are never overwritten)" }
foreach ($r in $sel) { if (-not (Test-Path $r.exe)) { Fail "missing $($r.exe)" } }
foreach ($m in @($Q15, $Q3)) { if (-not (Test-Path $m)) { Fail "missing $m" } }
$vars = Get-ChildItem Env: | Where-Object { $_.Name -match '^(DYNACORE_|DYNALM_|BENCH_)' }
if ($vars) { Fail ("environment must be clean, found: " + (($vars | ForEach-Object Name) -join ", ")) }
$busy = Get-Process | Where-Object { $_.Name -match '^(dynalm|llama-server|llama-cli|ollama|ollama app)$' }
if ($busy) { Fail ("other inference processes are running (not touched): " + (($busy | ForEach-Object { "$($_.Name)#$($_.Id)" }) -join ", ")) }
$done = @{}
if (Test-Path (Join-Path $Out "runs.jsonl")) {
  Get-Content (Join-Path $Out "runs.jsonl") | Where-Object { $_.Trim() } | ForEach-Object {
    $j = $_ | ConvertFrom-Json
    if ($j.exit_code -eq 0) { $done[$j.id] = $true }
  }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$sha = [System.Security.Cryptography.SHA256]::Create()
$fs = [System.IO.File]::OpenRead((Resolve-Path $Dynalm))
$hash = -join ($sha.ComputeHash($fs) | ForEach-Object { $_.ToString("X2") })
$fs.Close()
$pre = Join-Path $Out ("preflight-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json")
[ordered]@{
  time = (Get-Date).ToString("o"); branch = (git rev-parse --abbrev-ref HEAD); head = (git rev-parse --short HEAD)
  dirty_files = @(git status --short).Count; dynalm_sha256 = $hash; threads = $Threads; stages = $Stage; only = $Only
  free_ram_gb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
  cpu = (Get-CimInstance Win32_Processor).Name
} | ConvertTo-Json | Set-Content $pre -Encoding utf8
$runs | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Out "plan.json") -Encoding utf8

# --- execute ---------------------------------------------------------------------------
$i = 0
foreach ($r in $sel) {
  $i++
  if ($done[$r.id]) { Write-Host "skip $($r.id) (finished earlier)"; continue }
  $log = Join-Path $Out "$($r.id).log"
  $jsonl = Join-Path $Out "$($r.id).jsonl"
  $reqlog = Join-Path $Out "$($r.id).requests.jsonl"
  $steplog = Join-Path $Out "$($r.id).steps.jsonl"
  if ((Test-Path $log) -or (Test-Path $jsonl) -or (Test-Path $reqlog) -or (Test-Path $steplog)) { throw "$($r.id): output from an unfinished earlier attempt exists; move it aside first (never overwritten)" }
  $traceHash = ""
  $ti = [array]::IndexOf([string[]]$r.args, "--arrival-trace")
  if ($ti -ge 0) {
    $tf = $r.args[$ti + 1]
    if (-not (Test-Path $tf)) { throw "$($r.id): arrival trace $tf not found" }
    $traceHash = (-join ($sha.ComputeHash([System.IO.File]::ReadAllBytes((Resolve-Path $tf))) | ForEach-Object { $_.ToString("X2") }))
  }
  if ($IdleSeconds -gt 0) {
    Write-Host ("idle {0} s before {1}" -f $IdleSeconds, $r.id)
    Start-Sleep -Seconds $IdleSeconds
  }
  # Conflicting inference process: never touched; wait and recheck (30 s steps, up to 30 min), then stop as a blocker.
  for ($w = 0; ; $w++) {
    $busy = Get-Process | Where-Object { $_.Name -match '^(dynalm|llama-server|llama-cli|ollama|ollama app)$' }
    if (-not $busy) { break }
    $names = ($busy | ForEach-Object { "$($_.Name)#$($_.Id)" }) -join ", "
    if ($w -ge 60) { throw "BLOCKER: inference process still running before $($r.id) after 30 min (not touched): $names" }
    Write-Host "waiting: inference process running ($names); recheck in 30 s"
    Start-Sleep -Seconds 30
  }
  # Free RAM: wait and recheck (30 s steps, up to 5 min), then stop.
  for ($w = 0; ; $w++) {
    $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
    if ($free -ge $MinFreeGb) { break }
    if ($w -ge 10) { throw "BLOCKER: only $free GB free RAM before $($r.id) after 5 min (need $MinFreeGb); stopping" }
    Write-Host "waiting: only $free GB free RAM (need $MinFreeGb); recheck in 30 s"
    Start-Sleep -Seconds 30
  }
  $t0 = Get-Date
  Write-Host ("`n== [{0}/{1}] {2}  {3:HH:mm:ss}  ({4} GB free)" -f $i, $sel.Count, $r.id, $t0, $free) -ForegroundColor Cyan
  foreach ($kv in $r.env.GetEnumerator()) { Set-Item -Path ("Env:" + $kv.Name) -Value $kv.Value }
  try {
    & $r.exe @($r.args) 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
  } finally {
    foreach ($kv in $r.env.GetEnumerator()) { Remove-Item ("Env:" + $kv.Name) -ErrorAction SilentlyContinue }
  }
  [ordered]@{ id = $r.id; exe = $r.exe; args = $r.args; env = $r.env; start = $t0.ToString("o"); end = (Get-Date).ToString("o");
              exit_code = $code; free_ram_gb_at_start = $free; idle_s_before = $IdleSeconds;
              no_inference_process = $true; dynalm_sha256 = $hash; trace_sha256 = $traceHash } |
    ConvertTo-Json -Compress -Depth 4 | Add-Content (Join-Path $Out "runs.jsonl") -Encoding utf8
  if ($code -ne 0) { Write-Host "run $($r.id) exited $code; stopping (no further runs)" -ForegroundColor Red; exit $code }
}
Write-Host "`nall selected runs finished"
