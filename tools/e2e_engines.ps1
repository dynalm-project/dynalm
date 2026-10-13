<#
.SYNOPSIS
  Repeated, order-rotated engine comparison (DynaLM baseline, llama.cpp,
  Ollama) on one model with tools/compare_engines.py.

.DESCRIPTION
  For each repetition r = 1..Reps the engine order is rotated by r-1
  (rep 1: dynalm, llama.cpp, ollama; rep 2: llama.cpp, ollama, dynalm; ...).
  Every (repetition, engine) runs two compare_engines invocations with the
  other engines skipped, each server started fresh by the engine:
    c1     concurrency 1,      -C1Requests requests
    cmulti -MultiConcurrency,  -MultiRequests requests per point
  Prompt/output lengths, threads and the load generator (dynalm benchmark
  --url: streaming, ignore_eos, temperature 0, one uncounted warm-up request
  per point) are the same for every engine. DynaLM runs with no DYNACORE_* or
  DYNALM_* variables (the default dispatch) and its /metrics is scraped.

  compare_engines restarts Ollama (taskkill /F "ollama app.exe" and
  "ollama.exe") so OLLAMA_NUM_PARALLEL applies. The preflight refuses to run
  while Ollama is running unless -AllowOllamaRestart is given.

  Output (-Out, default results\e2e-engines-<timestamp>):
    preflight.json, plan.json, runNN-<engine>-<c1|cmulti>\ (engine output),
    runNN-...log (console), runNN-....json (command, start/end, exit code).
  Analyze: python tools\e2e_engines_analyze.py <Out>

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\e2e_engines.ps1 -DryRun
#>
[CmdletBinding()]
param(
  [string]$Dynalm = "build\msvc-release\bin\dynalm.exe",
  [string]$LlamaServer = ".bench\llamacpp\llama-server.exe",
  [string]$Ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
  [string]$Model = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
  [string]$Engines = "dynalm,llama.cpp,ollama",
  [int]$Reps = 3,
  [int]$Threads = 10,
  [int]$Prompt = 128,
  [int]$Output = 128,
  [int]$C1Requests = 16,
  [string]$MultiConcurrency = "5,8",
  [int]$MultiRequests = 40,
  [string]$Out = "",
  [switch]$AllowOllamaRestart,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $Out) { $Out = "results\e2e-engines-" + (Get-Date -Format "yyyyMMdd-HHmmss") }
if (Test-Path $Out) { throw "Output directory $Out already exists; choose a new -Out." }
function Fail($m) { throw "preflight: $m" }

$engineList = @($Engines.Split(",") | ForEach-Object { $_.Trim() })
foreach ($e in $engineList) { if ($e -notin "dynalm", "llama.cpp", "ollama") { Fail "unknown engine $e" } }
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Fail "python not found" }
if (-not (Test-Path $Model)) { Fail "model not found: $Model" }
if (-not (Test-Path $Dynalm)) { Fail "DynaLM not found: $Dynalm" }
if ("llama.cpp" -in $engineList -and -not (Test-Path $LlamaServer)) { Fail "llama-server not found: $LlamaServer" }
if ("ollama" -in $engineList -and -not (Test-Path $Ollama)) { Fail "ollama not found: $Ollama" }
$leak = Get-ChildItem env: | Where-Object { $_.Name -like "DYNACORE_*" -or $_.Name -like "DYNALM_*" -or $_.Name -like "OLLAMA_*" }
if ($leak) { Fail ("unset these first (they change an engine's configuration): " + (($leak | ForEach-Object { $_.Name }) -join ", ")) }
foreach ($port in 8181, 8182, 11434) {
  $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  if ($busy -and -not ($port -eq 11434 -and $AllowOllamaRestart)) { Fail "port $port is in use" }
}
$ollamaRunning = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
if ("ollama" -in $engineList -and $ollamaRunning -and -not $AllowOllamaRestart) {
  Fail "Ollama is running; compare_engines will kill it (taskkill /F). Re-run with -AllowOllamaRestart to accept."
}
$freeGb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
$ver = { param($exe, $a) try { ((& $exe $a 2>&1) | Out-String).Trim() -replace "\s+", " " } catch { "?" } }
function Sha256($path) {  # .NET, so it works in Windows PowerShell and pwsh alike
  $fs = [System.IO.File]::OpenRead((Resolve-Path $path).Path)
  try { ([System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($fs))).Replace("-", "") } finally { $fs.Dispose() }
}
$preflight = [ordered]@{
  created = (Get-Date -Format "o")
  machine = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
  free_ram_gb = $freeGb
  git = [ordered]@{ branch = (git rev-parse --abbrev-ref HEAD) -join ""; head = (git rev-parse HEAD) -join ""; status_lines = @(git status --short).Count }
  dynalm = [ordered]@{ path = (Resolve-Path $Dynalm).Path; sha256 = (Sha256 $Dynalm); version = (& $ver $Dynalm "--version") }
  llama_server = if ("llama.cpp" -in $engineList) { [ordered]@{ path = (Resolve-Path $LlamaServer).Path; sha256 = (Sha256 $LlamaServer); version = (& $ver $LlamaServer "--version") } } else { $null }
  ollama = if ("ollama" -in $engineList) { [ordered]@{ path = $Ollama; version = (& $ver $Ollama "--version") } } else { $null }
  model = [ordered]@{ path = (Resolve-Path $Model).Path; bytes = (Get-Item $Model).Length }
  workload = [ordered]@{ engines = $engineList; reps = $Reps; threads = $Threads; prompt = $Prompt; output = $Output;
                         c1_requests = $C1Requests; multi_concurrency = $MultiConcurrency; multi_requests = $MultiRequests }
  dry_run = [bool]$DryRun
}
Write-Host ("DynaLM  {0} sha256 {1}" -f $preflight.dynalm.path, $preflight.dynalm.sha256.Substring(0, 16))
if ($preflight.llama_server) { Write-Host ("llama   {0} ({1})" -f $preflight.llama_server.path, $preflight.llama_server.version) }
if ($preflight.ollama) { Write-Host ("ollama  {0} ({1})" -f $preflight.ollama.path, $preflight.ollama.version) }
Write-Host ("RAM {0} GB free; no DYNACORE_*/DYNALM_*/OLLAMA_* variables" -f $freeGb)
if ($freeGb -lt 6) { Write-Warning "below 6 GB free: results may be affected by paging or skipped weight packing" }

$plan = @()
for ($r = 1; $r -le $Reps; $r++) {
  $rot = @(for ($i = 0; $i -lt $engineList.Count; $i++) { $engineList[($i + $r - 1) % $engineList.Count] })
  foreach ($e in $rot) {
    $plan += [ordered]@{ rep = $r; engine = $e; part = "c1"; conc = "1"; requests = $C1Requests }
    $plan += [ordered]@{ rep = $r; engine = $e; part = "cmulti"; conc = $MultiConcurrency; requests = $MultiRequests }
  }
}
New-Item -ItemType Directory -Force $Out | Out-Null
$preflight | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Out "preflight.json") -Encoding UTF8
$plan | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $Out "plan.json") -Encoding UTF8
Write-Host ("plan: {0} invocations: {1}" -f $plan.Count, (($plan | ForEach-Object { "$($_.rep):$($_.engine)/$($_.part)" }) -join " "))

$modelFull = (Resolve-Path $Model).Path
$i = 0
foreach ($p in $plan) {
  $i++
  $name = "run{0:D2}-{1}-{2}" -f $i, ($p.engine -replace "\.", ""), $p.part
  $skip = @("dynalm", "llama.cpp", "ollama") | Where-Object { $_ -ne $p.engine }
  $cmdArgs = @("tools\compare_engines.py", "--dynalm", (Resolve-Path $Dynalm).Path, "--threads", "$Threads", "--requests", "$($p.requests)",
               "--model", ("{0}:{1}:{2}:{3}" -f $modelFull, $p.conc, $Prompt, $Output), "--out", (Join-Path $Out $name))
  if ($p.engine -eq "llama.cpp") { $cmdArgs += @("--llama-server", (Resolve-Path $LlamaServer).Path) }
  if ($p.engine -eq "ollama") { $cmdArgs += @("--ollama", $Ollama) }
  if ($p.engine -eq "dynalm") { $cmdArgs += "--dynalm-scrape-metrics" }
  foreach ($s in $skip) { $cmdArgs += @("--skip", $s) }
  $rec = [ordered]@{ name = $name; rep = $p.rep; engine = $p.engine; part = $p.part; command = "$py " + ($cmdArgs -join " "); start = (Get-Date -Format "o") }
  Write-Host "[$name] $($rec.command)"
  if (-not $DryRun) {
    $env:PYTHONUTF8 = "1"
    & $py @cmdArgs *> (Join-Path $Out "$name.log")
    $rec.exit_code = $LASTEXITCODE
    if ($rec.exit_code -ne 0) { Write-Warning "$name exited with $($rec.exit_code)" }
  } else { $rec.dry_run = $true }
  $rec.end = (Get-Date -Format "o")
  $rec | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Out "$name.json") -Encoding UTF8
}
Write-Host "Results: $Out"
Write-Host "Analyze: python tools\e2e_engines_analyze.py $Out"
