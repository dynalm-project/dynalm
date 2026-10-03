<#
.SYNOPSIS
  DynaLM installer for Windows: builds from source with Visual Studio's C++
  toolchain and installs dynalm.exe.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -AddToPath
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Prefix C:\Tools\DynaLM

.NOTES
  Needs Visual Studio 2022 (or its Build Tools) with the "Desktop development
  with C++" workload, which also provides CMake and Ninja:
    winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
  The binary links the C/C++ runtime statically (no redistributable needed).
#>
param(
  [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\DynaLM'),
  [switch]$AddToPath,
  [switch]$NoServer
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# Locate Visual Studio's C++ toolchain.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "Visual Studio (Build Tools) not found. See: Get-Help $PSCommandPath -Full" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
}
if (-not $vs) { throw 'Visual Studio is installed without the C++ workload ("Desktop development with C++").' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }

$build = Join-Path $Root 'build\install-release'
$server = if ($NoServer) { 'OFF' } else { 'ON' }
$configure = "cmake -S `"$Root`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl " +
             "-DENABLE_TESTS=OFF -DENABLE_BENCHMARKS=OFF -DENABLE_SERVER=$server -DDYNALM_STATIC_RUNTIME=ON"
$steps = @(
  "call `"$vcvars`" $arch >nul",
  $configure,
  "cmake --build `"$build`"",
  "cmake --install `"$build`" --prefix `"$Prefix`""
) -join ' && '

Say "Building DynaLM ($arch) with $vs"
cmd /c $steps
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }

$exe = Join-Path $Prefix 'bin\dynalm.exe'
& $exe version
Say "Installed: $exe"

$bin = Join-Path $Prefix 'bin'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (($userPath -split ';') -notcontains $bin) {
  if ($AddToPath) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $bin), 'User')
    Say "Added $bin to your user PATH (open a new terminal)."
  } else {
    Write-Host "To run 'dynalm' from anywhere, re-run with -AddToPath or add $bin to PATH."
  }
}
Write-Host 'Try:  dynalm info   |   dynalm run <model.gguf> -p "Hello"   |   dynalm serve <model.gguf>'
Write-Host 'Note: Windows Smart App Control may block freshly built programs; use an official signed release or the Docker image if it does.'
