# DynaLM Next benchmark runner (results/dynalm-next/*/plan.md).
#
# One run at a time. Before every run: free RAM >= 6 GB and no conflicting
# inference process (waits and rechecks, never kills), an idle gap, and a
# quiet check (total CPU busy < -QuietPct over 5 s, retried every 30 s for up
# to 15 min; a run that starts without passing it is recorded quiet=false).
# Every run appends one JSON record to <Out>\runs.jsonl.
#
#   pwsh -NoProfile -Command "& 'tools\next_sweep.ps1' -Stage N3 -Out results\dynalm-next\phase3\raw [-Only id,id] [-Resume] [-List]"
param(
  [Parameter(Mandatory = $true)][ValidateSet("N3", "N4", "R1", "R2")][string[]]$Stage,
  [Parameter(Mandatory = $true)][string]$Out,
  [string[]]$Only = @(),
  [switch]$Resume,
  [switch]$List,
  [int]$IdleSeconds = 60,
  [double]$QuietPct = 10,
  [int]$Threads = 10,
  # Measurement session number (stage R2): arm order rotates with it.
  [int]$Session = 1
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Dynalm = Join-Path $Root "build\msvc-release\bin\dynalm.exe"
$Bench = Join-Path $Root "build\msvc-release\dynalm\benchmarks"
$Q15 = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$Q3 = "models\Qwen3-4B-Q4_K_M.gguf"
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

$runs = New-Object System.Collections.Generic.List[object]
function Add-Run($id, $exe, [string[]]$arguments, $env = @{}, $model = "") {
  $runs.Add([pscustomobject]@{ id = $id; exe = $exe; args = $arguments; env = $env; model = $model })
}
function Add-Bench($id, $model, [string[]]$extra, $env = @{}) {
  Add-Run $id $Dynalm (@("benchmark", $model, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl"),
      "--request-log", (Join-Path $Out "$id.requests.jsonl")) + $extra) $env $model
}
$Arms = @{
  "ref"   = @{}
  "split" = @{ DYNACORE_INT16_PREFILL_ROWS = "64"; DYNACORE_INT16_PREFILL_SPLIT = "1" }
  "i16"   = @{ DYNACORE_INT16_PREFILL_ROWS = "64" }
}

if ($Stage -contains "N3") {
  $L = @("--concurrency", "1", "--prompt", "2048", "--output", "32", "--requests", "4", "--warmup-requests", "1", "-c", "8192")
  $Q = @("--concurrency", "16", "--output", "64", "--warmup-requests", "2", "-c", "16384", "--prompt-mix",
    "128,128,128,128,128,128,128,2048", "--requests", "32", "--arrival-rate", "0.15", "--seed", "13")
  $L4 = @("--concurrency", "1", "--prompt", "2048", "--output", "16", "--requests", "3", "--warmup-requests", "1", "-c", "8192")
  Add-Bench "N3-q15-w00" $Q15 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag")
  $rot = @(@("ref", "split", "i16"), @("split", "i16", "ref"), @("i16", "ref", "split"))
  for ($r = 0; $r -lt 3; $r++) { foreach ($a in $rot[$r]) { Add-Bench ("N3-L-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 $L $Arms[$a] } }
  $alt = @(@("ref", "split"), @("split", "ref"), @("ref", "split"))
  for ($r = 0; $r -lt 3; $r++) { foreach ($a in $alt[$r]) { Add-Bench ("N3-Q3-q15-r{0}-{1}" -f ($r + 1), $a) $Q15 $Q $Arms[$a] } }
  Add-Bench "N3-q3-w00" $Q3 @("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "2", "-c", "4096", "--no-diag")
  for ($r = 0; $r -lt 3; $r++) { foreach ($a in $alt[$r]) { Add-Bench ("N3-L4-q3-r{0}-{1}" -f ($r + 1), $a) $Q3 $L4 $Arms[$a] } }
}
if ($Stage -contains "N4") {
  . (Join-Path $PSScriptRoot "next_sweep_n4.ps1")  # Phase 4 runs, declared in results\dynalm-next\phase4\plan.md
}
if ($Stage -contains "R1") {
  . (Join-Path $PSScriptRoot "next_sweep_r1.ps1")  # release-candidate runs, declared in results\dynalm-next\release-rc-*\benchmark_plan.md
}
if ($Stage -contains "R2") {
  . (Join-Path $PSScriptRoot "next_sweep_r2.ps1")  # release campaign sessions, declared in results\dynalm-release-*\campaign\plan.md
}

$sel = if ($Only.Count -gt 0) { $runs | Where-Object { $Only -contains $_.id } } else { $runs }
Write-Host ("{0} of {1} runs -> {2}" -f @($sel).Count, $runs.Count, $Out)
if ($List) {
  foreach ($r in $sel) {
    $e = ($r.env.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join " "
    Write-Host ("  {0,-22} [{1}] {2}" -f $r.id, $e, ($r.args -join " "))
  }
  return
}

$hashCache = @{}
function Sha($path) {
  if (-not $path -or -not (Test-Path $path)) { return "" }
  $full = (Resolve-Path $path).Path
  if (-not $hashCache.ContainsKey($full)) { $hashCache[$full] = (Get-FileHash $full -Algorithm SHA256).Hash }
  return $hashCache[$full]
}
function Free-GB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2) }
function Conflicts { @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(dynalm|llama.*|ollama.*|bench_.*|test_.*)$' }) }
function Wait-Resources {
  $t0 = Get-Date
  while ($true) {
    $free = Free-GB
    $c = Conflicts
    if ($free -ge 6 -and $c.Count -eq 0) { return $free }
    if (((Get-Date) - $t0).TotalMinutes -gt 30) { throw "resources not available for 30 min ($free GB free, conflicts: $($c.Name -join ','))" }
    Write-Host ("  waiting: {0} GB free, conflicts: {1} (recheck in 30 s; never kills)" -f $free, ($c.Name -join ","))
    Start-Sleep 30
  }
}
# CPU busy % over 5 s and the processes that used the most CPU in that window.
function Sample-Quiet {
  $before = @{}
  foreach ($p in Get-Process -ErrorAction SilentlyContinue) { try { $before[$p.Id] = $p.TotalProcessorTime.TotalMilliseconds } catch {} }
  $busy = $null
  for ($try = 0; $try -lt 3 -and $null -eq $busy; $try++) {
    try {
      $busy = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 5 -ErrorAction Stop).CounterSamples |
        Measure-Object -Property CookedValue -Average
    } catch { Start-Sleep 2 }
  }
  if ($null -eq $busy) {  # counter API unavailable: Win32_Processor load, 5 samples
    $busy = 1..5 | ForEach-Object { Start-Sleep 1; (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average } |
      Measure-Object -Average
  }
  $top = foreach ($p in Get-Process -ErrorAction SilentlyContinue) {
    try {
      if ($before.ContainsKey($p.Id)) {
        $d = $p.TotalProcessorTime.TotalMilliseconds - $before[$p.Id]
        if ($d -gt 50) { [pscustomobject]@{ name = $p.Name; cpu_ms = [math]::Round($d) } }
      }
    } catch {}
  }
  return [pscustomobject]@{ busy = [math]::Round($busy.Average, 2); top = @($top | Sort-Object cpu_ms -Descending | Select-Object -First 4) }
}
function Wait-Quiet {
  $t0 = Get-Date
  $tries = 0
  while ($true) {
    $q = Sample-Quiet
    $tries++
    if ($q.busy -lt $QuietPct) { return [pscustomobject]@{ quiet = $true; busy = $q.busy; top = $q.top; tries = $tries } }
    if (((Get-Date) - $t0).TotalMinutes -gt 15) { return [pscustomobject]@{ quiet = $false; busy = $q.busy; top = $q.top; tries = $tries } }
    Write-Host ("  not quiet: CPU {0}% (top: {1}); recheck in 30 s" -f $q.busy, (($q.top | ForEach-Object { "$($_.name) $($_.cpu_ms)ms" }) -join ", "))
    Start-Sleep 30
  }
}
function Power-State {
  $plan = (powercfg /getactivescheme) -join " "
  $ac = "unknown"
  try {
    $b = Get-CimInstance Win32_Battery -ErrorAction Stop | Select-Object -First 1
    if ($b) { $ac = if ($b.BatteryStatus -eq 2) { "ac" } else { "battery($($b.BatteryStatus))" } } else { $ac = "no-battery" }
  } catch {}
  return [pscustomobject]@{ plan = $plan; source = $ac }
}

$log = Join-Path $Out "runs.jsonl"
# Source state of the workspace the runner lives in (recorded per run).
$GitHead = (git -C $Root rev-parse HEAD 2>$null)
$GitDirty = @(git -C $Root status --porcelain 2>$null).Count
$done = @{}
if ($Resume -and (Test-Path $log)) {
  foreach ($l in Get-Content $log) { if ($l.Trim()) { $done[($l | ConvertFrom-Json).id] = $true } }
}
$i = 0
foreach ($r in $sel) {
  $i++
  if ($done.ContainsKey($r.id)) { Write-Host "skip $($r.id) (done)"; continue }
  $outFile = Join-Path $Out "$($r.id).jsonl"
  if (Test-Path $outFile) { throw "refusing to overwrite $outFile (move it away or use -Resume)" }
  $free0 = Wait-Resources
  Write-Host ("idle {0} s before {1}" -f $IdleSeconds, $r.id)
  Start-Sleep $IdleSeconds
  $free = Wait-Resources
  $q = Wait-Quiet
  $pw = Power-State
  $t0 = Get-Date
  Write-Host ("`n== [{0}/{1}] {2}  {3:HH:mm:ss}  ({4} GB free, CPU {5}% quiet={6})" -f $i, @($sel).Count, $r.id, $t0, $free, $q.busy, $q.quiet) -ForegroundColor Cyan
  $saved = @{}
  foreach ($k in $r.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $r.env[$k]) }
  $logFile = Join-Path $Out "$($r.id).log"
  & $r.exe @($r.args) 2>&1 | Tee-Object -FilePath $logFile | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" }
  $code = $LASTEXITCODE
  foreach ($k in $r.env.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  $t1 = Get-Date
  $rec = [ordered]@{
    id = $r.id; start = $t0.ToString("s"); end = $t1.ToString("s"); seconds = [math]::Round(($t1 - $t0).TotalSeconds, 1)
    exit_code = $code; env = $r.env; args = ($r.args -join " ")
    free_ram_gb_at_start = $free; cpu_busy_pct_pre = $q.busy; quiet = $q.quiet; quiet_tries = $q.tries; top_pre = $q.top
    power = $pw; exe_sha256 = (Sha $r.exe); exe = $r.exe; model_sha256 = (Sha $r.model)
    git_head = $GitHead; git_dirty_paths = $GitDirty; free_ram_gb_at_end = (Free-GB)
  }
  ($rec | ConvertTo-Json -Compress -Depth 5) | Add-Content -Path $log -Encoding utf8
}
Write-Host "all selected runs finished"
