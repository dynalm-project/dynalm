<#
.SYNOPSIS
  End-to-end validation of the experimental 8-row packed INT8 kernel
  (DYNACORE_PACKED_8ROWS) with the existing comparison engine.

.DESCRIPTION
  Runs tools/compare_engines.py (DynaLM only; llama.cpp and Ollama skipped)
  with --dynalm-scrape-metrics, once as a discarded warm-up and then as
  -Pairs balanced baseline/experimental pairs:

    pair 1: base, exp    pair 2: exp, base    pair 3: base, exp   ... (ABBA order)

  Each run is one engine invocation with two model specs (concurrency 5 and
  8, prompt/output 128/128), so every point gets its own fresh server and its
  own /metrics scrape. The only difference between runs is
  DYNACORE_PACKED_8ROWS (0 = baseline, 1 = experimental), set in this
  process before the engine starts the server (the server inherits it; the
  flag is read at kernel initialization).

  Output (-Out, default results\pk8-validation-<timestamp>):
    preflight.json               machine, binary, git and environment record
    plan.json                    the run order
    run00-warmup\                discarded warm-up (engine output)
    runNN-<base|exp>\            engine output: results.jsonl, report.md, logs\,
                                 dynalm-metrics-*-c5-*.prom, -c8-*.prom
    runNN-<base|exp>.log         engine console output
    runNN-<base|exp>.json        run record: env, command, pair, start/end, exit code
  Analyze with:  python tools\pk8_analyze.py <Out>

  Nothing here changes kernels, defaults or the engine.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\pk8_validation.ps1 -DryRun
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\pk8_validation.ps1
#>
[CmdletBinding()]
param(
  [string]$Dynalm = "build\msvc-release\bin\dynalm.exe",
  [string]$Model = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
  [int]$Pairs = 8,
  [int]$Requests = 64,
  [int]$Threads = 10,
  [string]$Concurrency = "5,8",
  [int]$Prompt = 128,
  [int]$Output = 128,
  [int]$WarmupRequests = 16,
  [string]$Out = "",
  # Which opt-in feature the pairs compare (default: the 8-row kernel, as before).
  #   packed8rows:  DYNACORE_PACKED_8ROWS 0 / 1
  #   int16prefill: DYNACORE_INT16_PREFILL_ROWS 0 / 64 (-ExpValue overrides 64)
  #   soloprefill:  DYNALM_SOLO_PREFILL_CHUNK 0 / 128 (= dynalm serve --solo-prefill-chunk;
  #                 -ExpValue overrides 128). Scheduler, not kernels; split mode is not set.
  [ValidateSet("packed8rows", "int16prefill", "soloprefill")]
  [string]$Feature = "packed8rows",
  [string]$ExpValue = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Out) { $Out = "results\pk8-validation-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
if (Test-Path $Out) { throw "Output directory $Out already exists; choose a new -Out (existing results are never overwritten)." }

$feat = switch ($Feature) {
  "packed8rows" { @{ env = "DYNACORE_PACKED_8ROWS"; base = "0"; exp = "1"; gauge = "dynacore_packed_8rows_enabled"; counter = "dynacore_packed_rows8_dispatches_total" } }
  "int16prefill" { @{ env = "DYNACORE_INT16_PREFILL_ROWS"; base = "0"; exp = "64"; gauge = "dynacore_int16_prefill_max_rows"; counter = "dynacore_int16_prefill_matmuls_total" } }
  # serve reads DYNALM_<OPTION> for every option, so the variable name is built at run
  # time and is not a string in the binary: the option name is the needle instead.
  "soloprefill" { @{ env = "DYNALM_SOLO_PREFILL_CHUNK"; base = "0"; exp = "128"; gauge = "dynalm_scheduler_solo_prefill_chunk"; counter = "dynalm_scheduler_solo_prefill_steps_total"; needle = "solo-prefill-chunk" } }
}
if ($ExpValue) { $feat.exp = $ExpValue }

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Fail($msg) { throw "preflight: $msg" }

# --- preflight -------------------------------------------------------------------------
Step "Preflight"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Fail "python not found" }
if (-not (Test-Path $Dynalm)) { Fail "DynaLM binary not found: $Dynalm (build: tools\dev.cmd cmake --build --preset msvc-release)" }
if (-not (Test-Path $Model)) { Fail "model not found: $Model" }
$exe = Get-Item $Dynalm
# The binary must contain the experimental flag and its /metrics lines.
$bytes = [System.IO.File]::ReadAllBytes($exe.FullName)
$text = [System.Text.Encoding]::ASCII.GetString($bytes)
foreach ($needle in $feat.counter, $feat.gauge, $(if ($feat.needle) { $feat.needle } else { $feat.env })) {
  if (-not $text.Contains($needle)) { Fail "$Dynalm does not contain '$needle' (rebuild from the current tree)" }
}
# The engine must have the opt-in scrape.
if (-not (Select-String -Path tools\compare_engines.py -Pattern "dynalm-scrape-metrics" -Quiet)) {
  Fail "tools\compare_engines.py has no --dynalm-scrape-metrics"
}
# No variable that changes kernels or scheduling may leak into either configuration.
$leak = Get-ChildItem env: | Where-Object { $_.Name -like "DYNACORE_*" -or $_.Name -like "DYNALM_*" }
if ($leak) { Fail ("unset these first: " + (($leak | ForEach-Object { $_.Name }) -join ", ")) }
# No server already holding the engine's DynaLM port, no leftover dynalm serve.
if (Get-NetTCPConnection -LocalPort 8181 -State Listen -ErrorAction SilentlyContinue) { Fail "port 8181 is in use" }
$running = Get-CimInstance Win32_Process -Filter "Name='dynalm.exe'" | Where-Object { $_.CommandLine -match " serve " }
if ($running) { Fail "a 'dynalm serve' process is running (pid $($running.ProcessId -join ', '))" }
$freeGb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
if ($freeGb -lt 6) { Write-Warning "Only $freeGb GB free: below 3 GB the weight packing budget (DD-078) can skip packing and change both runs." }
$exeHash = (Get-FileHash $exe.FullName -Algorithm SHA256).Hash
$modelItem = Get-Item $Model
$git = [ordered]@{
  branch = (git rev-parse --abbrev-ref HEAD) -join ""
  head = (git rev-parse HEAD) -join ""
  status_lines = @(git status --short).Count
  diff_sha256 = $diff_sha256
}
$preflight = [ordered]@{
  created = (Get-Date -Format "o")
  machine = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
  free_ram_gb = $freeGb
  python = (& $py --version) -join ""
  dynalm = [ordered]@{ path = $exe.FullName; bytes = $exe.Length; mtime = $exe.LastWriteTime.ToString("o"); sha256 = $exeHash; version = ((& $exe.FullName --version) -join " | ") }
  model = [ordered]@{ path = $modelItem.FullName; bytes = $modelItem.Length }
  git = $git
  workload = [ordered]@{ concurrency = $Concurrency; prompt = $Prompt; output = $Output; requests = $Requests; threads = $Threads; pairs = $Pairs; warmup_requests = $WarmupRequests }
  feature = [ordered]@{ name = $Feature; env = $feat.env; base_value = $feat.base; exp_value = $feat.exp; gauge = $feat.gauge; counter = $feat.counter }
  dry_run = [bool]$DryRun
}
Write-Host ("binary  {0} ({1}, sha256 {2})" -f $exe.FullName, $exe.LastWriteTime, $exeHash.Substring(0, 16))
Write-Host ("model   {0} ({1:N0} bytes)" -f $modelItem.FullName, $modelItem.Length)
Write-Host ("git     {0} {1}, {2} changed/untracked paths" -f $git.branch, $git.head.Substring(0, 8), $git.status_lines)
Write-Host ("RAM     {0} GB free; no DYNACORE_*/DYNALM_* variables; port 8181 free" -f $freeGb)

# --- plan ----------------------------------------------------------------------------------
$plan = @()
for ($p = 1; $p -le $Pairs; $p++) {
  $order = if ($p % 2 -eq 1) { @("base", "exp") } else { @("exp", "base") }
  foreach ($cfg in $order) { $plan += [ordered]@{ pair = $p; cfg = $cfg } }
}
New-Item -ItemType Directory -Force $Out | Out-Null
$preflight | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Out "preflight.json") -Encoding UTF8
$plan | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $Out "plan.json") -Encoding UTF8
Write-Host ("plan    warm-up + {0} runs: {1}" -f $plan.Count, (($plan | ForEach-Object { "$($_.pair)$($_.cfg[0])" }) -join " "))

$specs = @($Concurrency.Split(",") | ForEach-Object { "{0}:{1}:{2}:{3}" -f $modelItem.FullName, $_, $Prompt, $Output })

function Invoke-Run([string]$name, [string]$cfg, [int]$pair, [int]$requests) {
  $runDir = Join-Path $Out $name
  $val = if ($cfg -eq "exp") { $feat.exp } else { $feat.base }
  $cmdArgs = @("tools\compare_engines.py", "--dynalm", $exe.FullName, "--skip", "llama.cpp", "--skip", "ollama",
            "--threads", "$Threads", "--requests", "$requests", "--dynalm-scrape-metrics", "--out", $runDir)
  foreach ($s in $specs) { $cmdArgs += @("--model", $s) }
  $envRec = [ordered]@{}
  $envRec[$feat.env] = $val
  $envRec["PYTHONUTF8"] = "1"
  $record = [ordered]@{ name = $name; cfg = $cfg; pair = $pair; env = $envRec;
                        command = "$py " + ($cmdArgs -join " "); start = (Get-Date -Format "o") }
  Write-Host ("[{0}] {1}={2}  {3}" -f $name, $feat.env, $val, $record.command)
  if ($DryRun) {
    $record.dry_run = $true
  } else {
    Set-Item -Path ("Env:" + $feat.env) -Value $val  # set before the engine starts the server
    $env:PYTHONUTF8 = "1"
    try {
      & $py @cmdArgs *> (Join-Path $Out "$name.log")
      $record.exit_code = $LASTEXITCODE
    } finally {
      Remove-Item ("Env:" + $feat.env) -ErrorAction SilentlyContinue
    }
    # Quick check of what the server reported (full checks: tools\pk8_analyze.py).
    $proms = Get-ChildItem $runDir -Filter "dynalm-metrics-*.prom" -ErrorAction SilentlyContinue
    $record.scrapes = @()
    foreach ($pf in $proms) {
      $m = @{}
      foreach ($line in Get-Content $pf.FullName) {
        if ($line -match "^(\S+)\s+(\S+)$" -and $Matches[1] -in @($feat.gauge, $feat.counter, "dynalm_requests_failed_total")) { $m[$Matches[1]] = [double]$Matches[2] }
      }
      $record.scrapes += [ordered]@{ file = $pf.Name; enabled = $m[$feat.gauge];
                                     dispatches = $m[$feat.counter]; failed = $m["dynalm_requests_failed_total"] }
      $ok = ($m[$feat.gauge] -eq [double]$val) -and
            (($cfg -eq "exp" -and $m[$feat.counter] -gt 0) -or ($cfg -ne "exp" -and $m[$feat.counter] -eq 0))
      $color = if ($ok) { "Green" } else { "Red" }
      Write-Host ("    {0}: gauge {1}, dispatches {2}, failed {3}" -f $pf.Name, $m[$feat.gauge],
                  $m[$feat.counter], $m["dynalm_requests_failed_total"]) -ForegroundColor $color
    }
    if ($record.exit_code -ne 0) { Write-Warning "$name exited with $($record.exit_code); see $name.log" }
  }
  $record.end = (Get-Date -Format "o")
  $record | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Out "$name.json") -Encoding UTF8
}

# --- runs ----------------------------------------------------------------------------------
Step "Warm-up (discarded)"
Invoke-Run "run00-warmup" "base" 0 $WarmupRequests
Step "Measured runs"
$i = 0
foreach ($r in $plan) {
  $i++
  Invoke-Run ("run{0:D2}-{1}" -f $i, $r.cfg) $r.cfg $r.pair $Requests
}

Step "Done"
Write-Host "Results: $Out"
Write-Host "Analyze: python tools\pk8_analyze.py $Out"
