<#
.SYNOPSIS
  DynaLM installer for Windows (per-user, no administrator rights).

.DESCRIPTION
  Downloads dynalm-windows-x86_64.zip and SHA256SUMS from the GitHub release,
  verifies the checksum, installs bin\dynalm.exe and bin\dynacorec.exe into
  %LOCALAPPDATA%\Programs\DynaLM, adds its bin folder to the user PATH and
  runs `dynalm --version`. -FromSource builds this checkout instead (needs
  Visual Studio 2022 Build Tools with the C++ workload).

.EXAMPLE
  irm https://raw.githubusercontent.com/dynalm-project/dynalm/main/scripts/install.ps1 | iex
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Version v0.1.0
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Archive .\dynalm-windows-x86_64.zip -Sha256Sums .\SHA256SUMS
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -FromSource
#>
param(
  [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\DynaLM'),
  [string]$Version = 'latest',
  [string]$Archive = '',
  [string]$Sha256Sums = '',
  [string]$Repo = $(if ($env:DYNALM_REPO) { $env:DYNALM_REPO } else { 'dynalm-project/dynalm' }),
  [switch]$NoPath,
  [switch]$FromSource
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is far faster without the progress bar
function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

$cpu = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
switch ($cpu) {
  'AMD64' { $platform = 'windows-x86_64' }
  'ARM64' { if (-not $FromSource) { throw 'No Windows ARM64 release is published yet. Use -FromSource.' }; $platform = 'windows-arm64' }
  default { throw "Unsupported CPU architecture '$cpu' (supported: x86-64)." }
}
$bin = Join-Path $Prefix 'bin'

if ($FromSource) {
  $Root = Split-Path -Parent $PSScriptRoot
  if (-not (Test-Path (Join-Path $Root 'CMakeLists.txt'))) { throw "-FromSource needs a DynaLM checkout ($Root)." }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { throw 'Visual Studio 2022 (Build Tools) with the C++ workload is required for -FromSource.' }
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vs) { throw 'Visual Studio is installed without the C++ workload ("Desktop development with C++").' }
  $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
  $vsArch = if ($cpu -eq 'ARM64') { 'arm64' } else { 'x64' }
  $build = Join-Path $Root 'build\install-release'
  $steps = @(
    "call `"$vcvars`" $vsArch >nul",
    "cmake -S `"$Root`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DENABLE_TESTS=OFF -DENABLE_BENCHMARKS=OFF -DDYNALM_STATIC_RUNTIME=ON",
    "cmake --build `"$build`"",
    "cmake --install `"$build`" --prefix `"$Prefix`""
  ) -join ' && '
  Say "Building DynaLM from source ($platform)"
  cmd /c $steps
  if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }
} else {
  $name = "dynalm-$platform"
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("dynalm-install-" + [Guid]::NewGuid())
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  try {
    if (-not $Archive) {
      $base = if ($Version -eq 'latest') { "https://github.com/$Repo/releases/latest/download" } else { "https://github.com/$Repo/releases/download/$Version" }
      Say "Downloading $name.zip ($Version)"
      $Archive = Join-Path $tmp "$name.zip"
      $Sha256Sums = Join-Path $tmp 'SHA256SUMS'
      try { Invoke-WebRequest -UseBasicParsing -Uri "$base/$name.zip" -OutFile $Archive }
      catch { throw "No $name.zip in release '$Version' of $Repo. Try -FromSource. ($_)" }
      try { Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $Sha256Sums }
      catch { throw "Release '$Version' has no SHA256SUMS; refusing to install an unverified download." }
    }
    if (-not (Test-Path $Archive)) { throw "Archive not found: $Archive" }
    if ((Split-Path -Leaf $Archive) -ne "$name.zip") { throw "$(Split-Path -Leaf $Archive) is not the archive for this machine ($name.zip)." }
    if ($Sha256Sums) {
      $line = Get-Content $Sha256Sums | Where-Object { $_ -match "^\s*([0-9a-fA-F]{64})\s+\*?$([regex]::Escape("$name.zip"))\s*$" } | Select-Object -First 1
      if (-not $line) { throw "$name.zip is not listed in $Sha256Sums" }
      $want = ($line -split '\s+')[0].ToLower()
      # .NET directly: Get-FileHash is missing when Windows PowerShell inherits PowerShell 7's PSModulePath.
      $sha = [Security.Cryptography.SHA256]::Create()
      $fs = [IO.File]::OpenRead((Resolve-Path $Archive))
      try { $have = -join ($sha.ComputeHash($fs) | ForEach-Object { $_.ToString('x2') }) } finally { $fs.Dispose(); $sha.Dispose() }
      if ($want -ne $have) { throw "SHA-256 mismatch for $name.zip (expected $want, got $have)" }
      Say 'Checksum verified'
    } else {
      Say 'No SHA256SUMS given: skipping checksum verification of a local archive'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory((Resolve-Path $Archive).Path, $tmp)
    $src = Join-Path $tmp $name
    if (-not (Test-Path (Join-Path $src 'bin\dynalm.exe'))) { throw "Archive does not contain $name\bin\dynalm.exe" }
    Say "Installing to $Prefix"
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    Copy-Item -Force (Join-Path $src 'bin\*.exe') $bin
    $doc = Join-Path $src 'share\doc\dynalm'
    if (Test-Path $doc) {
      $docDest = Join-Path $Prefix 'share\doc\dynalm'
      if (Test-Path $docDest) { Remove-Item -Recurse -Force $docDest }
      New-Item -ItemType Directory -Force -Path (Split-Path $docDest) | Out-Null
      Copy-Item -Recurse $doc $docDest
    }
  } finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  }
}

$exe = Join-Path $bin 'dynalm.exe'
& $exe --version
if ($LASTEXITCODE -ne 0) {
  throw "The installed dynalm.exe did not run (exit $LASTEXITCODE). If Windows reports an Application Control policy, Smart App Control blocked it."
}
Say "Installed: $exe"

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($userPath -split ';') -notcontains $bin) {
  if ($NoPath) {
    Write-Host "Not added to PATH (-NoPath). Run it as: $exe"
  } else {
    [Environment]::SetEnvironmentVariable('Path', (($userPath, $bin) -join ';').TrimStart(';'), 'User')
    Say "Added $bin to your user PATH (open a new terminal)."
  }
}
Write-Host 'Next:  dynalm doctor   ->   dynalm pull qwen3:4b   ->   dynalm run qwen3:4b'
