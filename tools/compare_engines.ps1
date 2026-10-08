<#
.SYNOPSIS
  One-shot CPU benchmark on Windows: DynaLM vs llama.cpp vs Ollama on three models.

.DESCRIPTION
  Sets everything up and runs tools/compare_engines.py:
    1. Finds dynalm.exe (builds it with the msvc-release preset if missing).
    2. Finds llama-server.exe in .bench\llamacpp, or downloads the latest official
       llama.cpp Windows CPU build from GitHub (ggml-org/llama.cpp).
    3. Finds Ollama, or installs it with winget.
    4. Checks the model files and free RAM, stops leftover servers.
    5. Runs every engine on every model, one server at a time, with the same load
       generator (dynalm benchmark --url), and writes a Markdown report.

  Close other heavy apps first: results are only fair with free RAM.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\compare_engines.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\compare_engines.ps1 -Threads 8 -Skip ollama
#>
[CmdletBinding()]
param(
  # Compute threads for every engine.
  [int]$Threads = 10,
  # FILE[:CONCURRENCY:PROMPTS:OUTPUTS]; lists are comma-separated, models
  # separated by ';' when passed with -File.
  [string[]]$Models = @(
    "models\qwen2.5-1.5b-instruct-q4_k_m.gguf:1,4:128,512:128",
    "models\Qwen3-4B-Q4_K_M.gguf:1,4:128:64",
    "$env:USERPROFILE\.dynalm\models\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf:1:128:32"
  ),
  # Engines to leave out: dynalm, llama.cpp, ollama.
  [string[]]$Skip = @(),
  # Output directory (default: results\compare-<timestamp>).
  [string]$Out = ""
)

# Continue: native tools write progress to stderr, which "Stop" turns into errors
# in Windows PowerShell 5.1. Failures are checked explicitly below.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
# With -File, "-Skip a,b" arrives as one string.
$Skip = @($Skip | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Models = @($Models | ForEach-Object { $_ -split ";" } | Where-Object { $_ })
if (-not $Out) { $Out = "results\compare-" + (Get-Date -Format "yyyyMMdd-HHmmss") }

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# --- Python -----------------------------------------------------------------
Step "Python"
$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { throw "Python 3 is required (https://www.python.org/downloads/)." }
& $Python --version

# --- DynaLM -----------------------------------------------------------------
Step "DynaLM"
$Dynalm = Join-Path $Root "build\msvc-release\bin\dynalm.exe"
if (-not (Test-Path $Dynalm)) {
  Write-Host "Building DynaLM (msvc-release)..."
  cmd /c "tools\dev.cmd cmake --preset msvc-release && tools\dev.cmd cmake --build --preset msvc-release"
  if ($LASTEXITCODE -ne 0) { throw "DynaLM build failed." }
}
& $Dynalm --version

# --- llama.cpp --------------------------------------------------------------
Step "llama.cpp"
$LlamaDir = Join-Path $Root ".bench\llamacpp"
$LlamaServer = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path $LlamaServer)) {
  New-Item -ItemType Directory -Force $LlamaDir | Out-Null
  Write-Host "Looking up the latest llama.cpp Windows CPU build..."
  $releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15" `
    -ErrorAction Stop -Headers @{ "User-Agent" = "dynalm-bench" }
  $asset = $null
  foreach ($r in $releases) {
    $asset = $r.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cpu-x64\.zip$" } | Select-Object -First 1
    if ($asset) { break }
  }
  if (-not $asset) { throw "No llama.cpp Windows CPU build found in the latest releases." }
  $zip = Join-Path $LlamaDir $asset.name
  Write-Host "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)..."
  Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing -ErrorAction Stop
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $LlamaDir)
  Remove-Item $zip
}
cmd /c "`"$LlamaServer`" --version 2>&1" | Select-String "version"
if (-not $?) { throw "llama-server.exe does not run." }

# --- Ollama -----------------------------------------------------------------
Step "Ollama"
$Ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $Ollama) {
  $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
  if (Test-Path $candidate) { $Ollama = $candidate }
}
if (-not $Ollama -and $Skip -notcontains "ollama") {
  Write-Host "Installing Ollama with winget..."
  winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
  $Ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
  if (-not (Test-Path $Ollama)) { throw "Ollama install failed; rerun with -Skip ollama to leave it out." }
}
if ($Ollama) { & $Ollama --version }

# --- Models -----------------------------------------------------------------
Step "Models"
$specs = @()
foreach ($m in $Models) {
  # Split off the options after the file path (a drive letter has a colon too).
  if ($m -match '^(.+?\.(gguf|safetensors))(:.*)?$') { $file = $Matches[1]; $rest = $Matches[3] } else { $file = $m; $rest = "" }
  if (-not [System.IO.Path]::IsPathRooted($file)) { $file = Join-Path $Root $file }
  if (-not (Test-Path $file)) {
    Write-Warning "Skipping missing model: $file"
    continue
  }
  $gb = (Get-Item $file).Length / 1GB
  Write-Host ("{0,-55} {1,6:N1} GB" -f (Split-Path -Leaf $file), $gb)
  $specs += "$file$rest"
}
if ($specs.Count -eq 0) { throw "No model files found. Pull them with: dynalm pull qwen2.5:1.5b / qwen3:4b" }

# --- Memory and leftover servers ---------------------------------------------
Step "System"
$freeGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
Write-Host ("Free RAM: {0:N1} GB" -f $freeGb)
if ($freeGb -lt 6) { Write-Warning "Less than 6 GB free: close other apps or results will be skewed by paging." }
foreach ($p in "llama-server", "ollama", "ollama app") {
  Get-Process -Name $p -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
Get-CimInstance Win32_Process -Filter "Name='dynalm.exe'" |
  Where-Object { $_.CommandLine -match " serve " } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# --- Run --------------------------------------------------------------------
Step "Benchmark (this takes a while; one server at a time)"
$pyArgs = @("tools\compare_engines.py", "--dynalm", $Dynalm, "--llama-server", $LlamaServer,
            "--threads", "$Threads", "--out", $Out)
if ($Ollama) { $pyArgs += @("--ollama", $Ollama) } else { $pyArgs += @("--skip", "ollama") }
foreach ($s in $Skip) { $pyArgs += @("--skip", $s) }
foreach ($s in $specs) { $pyArgs += @("--model", $s) }

New-Item -ItemType Directory -Force $Out | Out-Null
$env:PYTHONUTF8 = "1"
& $Python @pyArgs | Tee-Object -FilePath (Join-Path $Out "run.log")

# --- Clean up ---------------------------------------------------------------
foreach ($p in "llama-server", "ollama") {
  Get-Process -Name $p -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

Step "Done"
$OutFull = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $Root $Out }
Write-Host "Report:  $(Join-Path $OutFull 'report.md')"
Write-Host "Raw:     $(Join-Path $OutFull 'results.jsonl')"
Write-Host "Log:     $(Join-Path $OutFull 'run.log')"
Write-Host "Note: Ollama ignores ignore_eos; compare its rows by the 'Mean output tokens' column."
