<#
.SYNOPSIS
  Phase 8 prefill-pipeline investigation: scheduler chunk-size sweep, long-prompt
  workloads and per-operation profiles, with the existing in-process tools only.

.DESCRIPTION
  Every run is one process, started and awaited here, then gone. The runs are:
    dynalm benchmark <model> ...   in-process engine; no server, no port
    bench_long_prompt <model> 10   existing long-prompt scheduler benchmark
    bench_decode_matmul <model> 10 <M list>   matmul cost per path, by row count
    bench_profile <model> 10       per-op split of a 256-token prefill

  Parts:
    A  Qwen2.5-1.5B, 128/128, c1/5/8: chunk {8,16,32,64}/budget 64 and {128}/budget 128,
       3 repeats in rotated order.
    B  Qwen3-4B, 128/128, c1/5/8: chunk {16,32,128}, 2 repeats in rotated order.
    C  Qwen2.5-1.5B long prompt.
       C1: 2048/32 at c1, chunk {32,64,128,512}.
       C2: prompt mix 4 x 128 + 1 x 2048, 64 out, c5, chunk {16,32,64,128}.
       Each has 2 repeats. bench_long_prompt runs once.
    D  bench_decode_matmul at M = 8,16,32,64,128, plus bench_profile, for both models.

  The scheduler chunk/budget is the only variable inside a part. INT16 prefill stays
  at its production default (off): the script refuses to start if any DYNACORE_* or
  DYNALM_* variable is set.

  It never kills, restarts or touches another process. If a dynalm, llama-server or
  ollama process is running, it refuses to start. The output directory must be new.

  Output (-Out):
    preflight.json                       git, binary hash, machine, free RAM
    plan.json                            every run, in execution order
    <id>.jsonl                           dynalm benchmark points (one JSON line per point)
    <id>.log                             console output of the run
    runs.jsonl                           per-run record: command, start, end, exit code

.EXAMPLE
  pwsh -NoProfile -File tools\phase8_sweep.ps1 -DryRun
.EXAMPLE
  pwsh -NoProfile -File tools\phase8_sweep.ps1 -Parts A,B,C,D -Out results\phase8-prefill-<ts>\raw
#>
[CmdletBinding()]
param(
  [string]$Out = "",
  [ValidateSet("A", "B", "C", "D")]
  [string[]]$Parts = @("A", "B", "C", "D"),
  [int]$Threads = 10,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Out) { $Out = "results\phase8-prefill-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "\raw" }

$Dynalm = "build\msvc-release\bin\dynalm.exe"
$Bench = "build\msvc-release\dynalm\benchmarks"
$Q15 = "models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$Q3 = "models\Qwen3-4B-Q4_K_M.gguf"

# --- plan ------------------------------------------------------------------------------
$runs = [System.Collections.Generic.List[object]]::new()
function Add-Bench($id, $model, [string[]]$extra) {
  $a = @("benchmark", $model, "-t", "$Threads", "--out", (Join-Path $Out "$id.jsonl")) + $extra
  $runs.Add([pscustomobject]@{ id = $id; exe = $Dynalm; args = $a })
}
function Chunk($c) { if ($c -eq 128) { @("--chunk", "128", "--prefill-budget", "128") } elseif ($c -eq 512) { @("--chunk", "512", "--prefill-budget", "512") } else { @("--chunk", "$c", "--prefill-budget", "64") } }

# Discarded warm-up: pages the 1.5B weights in, settles clocks.
if ($Parts -contains "A" -or $Parts -contains "C") {
  Add-Bench "w00-warmup" $Q15 (@("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag") + (Chunk 32))
}
if ($Parts -contains "A") {
  $orders = @(@(32, 8, 64, 16, 128), @(128, 16, 64, 8, 32), @(16, 32, 8, 128, 64))
  for ($r = 0; $r -lt 3; $r++) {
    foreach ($c in $orders[$r]) {
      Add-Bench ("A-r{0}-ch{1}" -f ($r + 1), $c) $Q15 (@("--concurrency", "1,5,8", "--prompt", "128", "--output", "128", "--requests", "16", "-c", "4096") + (Chunk $c))
    }
  }
}
if ($Parts -contains "C") {
  $o1 = @(@(32, 128, 64, 512), @(512, 64, 128, 32))
  $o2 = @(@(32, 16, 128, 64), @(64, 128, 16, 32))
  for ($r = 0; $r -lt 2; $r++) {
    foreach ($c in $o1[$r]) {
      Add-Bench ("C1-r{0}-ch{1}" -f ($r + 1), $c) $Q15 (@("--concurrency", "1", "--prompt", "2048", "--output", "32", "--requests", "3", "-c", "8192") + (Chunk $c))
    }
    foreach ($c in $o2[$r]) {
      Add-Bench ("C2-r{0}-ch{1}" -f ($r + 1), $c) $Q15 (@("--concurrency", "5", "--prompt-mix", "128,128,128,128,2048", "--output", "64", "--requests", "10", "-c", "16384") + (Chunk $c))
    }
  }
  $runs.Add([pscustomobject]@{ id = "C3-bench_long_prompt"; exe = "$Bench\bench_long_prompt.exe"; args = @($Q15, "$Threads") })
}
if ($Parts -contains "B") {
  Add-Bench "w01-warmup-4b" $Q3 (@("--concurrency", "1", "--prompt", "128", "--output", "16", "--requests", "4", "-c", "4096", "--no-diag") + (Chunk 32))
  $orders = @(@(32, 16, 128), @(16, 128, 32))
  for ($r = 0; $r -lt 2; $r++) {
    foreach ($c in $orders[$r]) {
      Add-Bench ("B-r{0}-ch{1}" -f ($r + 1), $c) $Q3 (@("--concurrency", "1,5,8", "--prompt", "128", "--output", "128", "--requests", "8", "-c", "4096") + (Chunk $c))
    }
  }
}
if ($Parts -contains "D") {
  foreach ($m in @($Q15, $Q3)) {
    $tag = if ($m -eq $Q15) { "q15" } else { "q3" }
    $runs.Add([pscustomobject]@{ id = "D-matmul-$tag"; exe = "$Bench\bench_decode_matmul.exe"; args = @($m, "$Threads", "8,16,32,64,128") })
    $runs.Add([pscustomobject]@{ id = "D-profile-$tag"; exe = "$Bench\bench_profile.exe"; args = @($m, "$Threads") })
  }
}

Write-Host ("{0} runs -> {1}" -f $runs.Count, $Out)
foreach ($r in $runs) { Write-Host ("  {0,-22} {1} {2}" -f $r.id, $r.exe, ($r.args -join " ")) }
if ($DryRun) { Write-Host "`n(dry run: nothing started)"; return }

# --- preflight -------------------------------------------------------------------------
function Fail($msg) { throw "preflight: $msg" }
if (Test-Path $Out) { Fail "output directory $Out already exists (results are never overwritten)" }
foreach ($r in $runs) { if (-not (Test-Path $r.exe)) { Fail "missing $($r.exe)" } }
foreach ($m in @($Q15, $Q3)) { if (-not (Test-Path $m)) { Fail "missing $m" } }
$vars = Get-ChildItem Env: | Where-Object { $_.Name -match '^(DYNACORE_|DYNALM_)' }
if ($vars) { Fail ("environment must be clean, found: " + (($vars | ForEach-Object Name) -join ", ")) }
$busy = Get-Process | Where-Object { $_.Name -match '^(dynalm|llama-server|llama-cli|ollama|ollama app)$' }
if ($busy) { Fail ("other inference processes are running (not touched): " + (($busy | ForEach-Object { "$($_.Name)#$($_.Id)" }) -join ", ")) }
$os = Get-CimInstance Win32_OperatingSystem
$freeGb = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
if ($freeGb -lt 6) { Fail "only $freeGb GB free RAM; Qwen3-4B runs need about 6 GB" }
$sha = [System.Security.Cryptography.SHA256]::Create()
$fs = [System.IO.File]::OpenRead((Resolve-Path $Dynalm))
$hash = -join ($sha.ComputeHash($fs) | ForEach-Object { $_.ToString("X2") })
$fs.Close()
New-Item -ItemType Directory -Path $Out | Out-Null
[ordered]@{
  time = (Get-Date).ToString("o"); branch = (git rev-parse --abbrev-ref HEAD); head = (git rev-parse --short HEAD)
  dirty_files = @(git status --short).Count; dynalm_sha256 = $hash; free_ram_gb = $freeGb; threads = $Threads
  cpu = (Get-CimInstance Win32_Processor).Name; parts = $Parts
} | ConvertTo-Json | Set-Content (Join-Path $Out "preflight.json") -Encoding utf8
$runs | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Out "plan.json") -Encoding utf8

# --- execute ---------------------------------------------------------------------------
$i = 0
foreach ($r in $runs) {
  $i++
  $log = Join-Path $Out "$($r.id).log"
  $t0 = Get-Date
  Write-Host ("`n== [{0}/{1}] {2}  {3:HH:mm:ss}" -f $i, $runs.Count, $r.id, $t0) -ForegroundColor Cyan
  & $r.exe @($r.args) 2>&1 | Tee-Object -FilePath $log
  $code = $LASTEXITCODE
  [ordered]@{ id = $r.id; exe = $r.exe; args = $r.args; start = $t0.ToString("o"); end = (Get-Date).ToString("o"); exit_code = $code } |
    ConvertTo-Json -Compress -Depth 4 | Add-Content (Join-Path $Out "runs.jsonl") -Encoding utf8
  if ($code -ne 0) { Write-Host "run $($r.id) exited $code; stopping (no further runs)" -ForegroundColor Red; exit $code }
}
Write-Host "`nall $($runs.Count) runs finished"
