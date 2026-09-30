@echo off
rem Runs a command inside the MSVC x64 developer environment.
rem   tools\dev.cmd cmake --preset msvc-release
rem   tools\dev.cmd cmake --build --preset msvc-release
setlocal
if defined VCINSTALLDIR goto run
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VSROOT=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
if not defined VSROOT (
  echo dev.cmd: MSVC not found. Install VS Build Tools 2022 with the C++ workload. 1>&2
  exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
:run
rem Fall back to pip-installed cmake/ninja when VS doesn't provide them.
where cmake >nul 2>nul || set "PATH=%APPDATA%\Python\Python312\Scripts;%PATH%"
where ninja >nul 2>nul || set "PATH=%APPDATA%\Python\Python312\Scripts;%PATH%"
%*
