#Requires -Version 5.1
<#
  RPGBarEngineCUDA -- Windows build script (MSVC + CUDA), PowerShell flavour.

  The Windows counterpart of build_wsl.sh: same sources, same C ABI, only the
  host toolchain differs (MSVC instead of g++).

  Why this does NOT call vcvars64.bat:
    vcvars discovers the Windows SDK through the registry (reg.exe). In locked
    down environments that is unavailable and the script dies before it ever
    reaches the compiler. Everything vcvars would set up is derived here from
    the filesystem instead: the MSVC toolset is the newest directory under
    VC\Tools\MSVC, the SDK the newest under Windows Kits\10, and the VS install
    itself comes from vswhere.exe (which needs no registry). No reg.exe.

  Why the default targets four architectures instead of "native":
    native emits SASS for this machine's GPU only. Ship that to someone else
    and you get
      "no kernel image is available for execution on the device"
    -- usable locally, not distributable. Distribution means listing the
    architectures explicitly.
    (CUDA 13 dropped offline compilation for sm<7.5, so 75 -- Turing, i.e.
     GTX 16 / RTX 20 -- is the lowest target still available. Older cards have
     to fall back to the CPU backend.)

  This file is intentionally ASCII-only: Windows PowerShell 5.1 reads .ps1 as
  ANSI unless a BOM is present, so non-ASCII comments would be mangled.

  Usage:
    .\build_win.ps1
    .\build_win.ps1 -Arch 120
    .\build_win.ps1 -Arch "86;89;120" -BuildDir D:\cuda-build
    .\build_win.ps1 -Nvcc "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"
#>
[CmdletBinding()]
param(
  [string]$Arch     = "75;86;89;120",
  [string]$BuildDir = "",
  [string]$Nvcc     = ""
)

# Deliberately NOT "Stop": PowerShell turns anything a native tool writes to
# stderr into an ErrorRecord, and with ErrorActionPreference=Stop that becomes
# terminating. cmake/ninja/nvcc all write progress and warnings to stderr, so
# "Stop" would abort the build on a mere warning. Failures are detected
# explicitly instead, through $LASTEXITCODE after each native call.
$ErrorActionPreference = "Continue"

function Fail([string]$msg) { Write-Host "[x] $msg"; exit 1 }
function Info([string]$msg) { Write-Host "[*] $msg" }

$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if (-not $BuildDir) { $BuildDir = Join-Path $root "build-win" }

# ---------- 1. Visual Studio install (vswhere -> no registry) ---------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
  Fail "vswhere.exe not found -- no Visual Studio / Build Tools installed on this machine."
}
$vsRoot = @(& $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath) | Select-Object -First 1
if (-not $vsRoot) {
  Fail "No VS install with the C++ toolset -- select the 'Desktop development with C++' workload."
}
if (-not (Test-Path $vsRoot)) { Fail "vswhere reported '$vsRoot' but that path does not exist." }
Info "Visual Studio : $vsRoot"

# ---------- 2. newest MSVC toolset ------------------------------------------
$msvcParent = Join-Path $vsRoot "VC\Tools\MSVC"
$toolset = @(Get-ChildItem $msvcParent -Directory -ErrorAction SilentlyContinue |
             Where-Object { Test-Path (Join-Path $_.FullName "bin\Hostx64\x64\cl.exe") } |
             Sort-Object Name -Descending) | Select-Object -First 1
if (-not $toolset) { Fail "No MSVC toolset containing bin\Hostx64\x64\cl.exe under $msvcParent" }
$msvc = $toolset.FullName
Info "MSVC toolset  : $($toolset.Name)"

# ---------- 3. newest Windows SDK -------------------------------------------
# The SDK does not have to live on the system drive -- on this machine VS is on
# E: and so is the SDK, so scan every fixed drive before giving up.
$sdkCands = @("${env:ProgramFiles(x86)}\Windows Kits\10", "$env:ProgramFiles\Windows Kits\10")
foreach ($d in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
  if ($d.Root) { $sdkCands += (Join-Path $d.Root "Windows Kits\10") }
}
$sdkRoot = @($sdkCands | Where-Object { $_ -and (Test-Path (Join-Path $_ "Include")) } |
             Select-Object -Unique) | Select-Object -First 1
if (-not $sdkRoot) { Fail "Windows SDK not found (looked for <drive>\Windows Kits\10\Include)." }
$sdkVer = @(Get-ChildItem (Join-Path $sdkRoot "Include") -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "um\windows.h") } |
            Sort-Object Name -Descending) | Select-Object -First 1
if (-not $sdkVer) { Fail "No Windows SDK version with Include\<ver>\um\windows.h under $sdkRoot" }
$sdk = $sdkVer.Name
Info "Windows SDK   : $sdk   ($sdkRoot)"

# ---------- 4. environment (the part vcvars would have set) -----------------
$env:PATH = (@(
  (Join-Path $msvc "bin\Hostx64\x64"),
  (Join-Path $sdkRoot "bin\$sdk\x64")
) + @($env:PATH -split ";" | Where-Object { $_ })) -join ";"

$env:INCLUDE = (@(
  (Join-Path $msvc "include"),
  (Join-Path $sdkRoot "Include\$sdk\ucrt"),
  (Join-Path $sdkRoot "Include\$sdk\shared"),
  (Join-Path $sdkRoot "Include\$sdk\um"),
  (Join-Path $sdkRoot "Include\$sdk\winrt"),
  (Join-Path $sdkRoot "Include\$sdk\cppwinrt")
) + @($env:INCLUDE -split ";" | Where-Object { $_ })) -join ";"

$env:LIB = (@(
  (Join-Path $msvc "lib\x64"),
  (Join-Path $sdkRoot "Lib\$sdk\ucrt\x64"),
  (Join-Path $sdkRoot "Lib\$sdk\um\x64")
) + @($env:LIB -split ";" | Where-Object { $_ })) -join ";"

# Some CMake modules and MSBuild targets read these. WindowsSDKVersion must keep
# its trailing backslash -- that is what the convention is, not a typo.
$env:VCToolsInstallDir = $msvc + "\"
$env:WindowsSdkDir     = $sdkRoot + "\"
$env:WindowsSDKVersion = $sdk + "\"

# ---------- 5. nvcc ---------------------------------------------------------
if (-not $Nvcc) {
  $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
  if ($cmd) { $Nvcc = $cmd.Source }
}
if (-not $Nvcc) {
  $cudaBase = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
  $v = @(Get-ChildItem $cudaBase -Directory -ErrorAction SilentlyContinue |
         Sort-Object Name -Descending |
         Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") }) | Select-Object -First 1
  if ($v) { $Nvcc = Join-Path $v.FullName "bin\nvcc.exe" }
}
if (-not $Nvcc -or -not (Test-Path $Nvcc)) {
  Fail "nvcc.exe not found -- install the CUDA Toolkit or pass -Nvcc <path>."
}
Info "nvcc          : $Nvcc"

# The toolkit root, derived from nvcc's own location (.../CUDA/v13.4/bin/nvcc.exe).
# This gets handed to find_package(CUDAToolkit) explicitly: left to itself it
# locates the toolkit by querying the registry on Windows, and a registry query
# shells out to reg.exe, which locked-down environments tend to have disabled.
# Pointing it straight at the root keeps the build off that path entirely.
$cudaRoot = Split-Path (Split-Path $Nvcc -Parent) -Parent
$env:CUDA_PATH = $cudaRoot
Info "CUDA toolkit  : $cudaRoot"

# ---------- 6. cmake / ninja (both ship with VS) ----------------------------
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
$cmake = if ($cmakeCmd) { $cmakeCmd.Source } else { Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" }
$ninja = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path $cmake)) { Fail "cmake not found (looked on PATH and under $vsRoot)." }
if (-not (Test-Path $ninja)) { Fail "ninja not found under $vsRoot." }
Info "cmake         : $cmake"
Info "ninja         : $ninja"

# ---------- 7. sanity: both host and device compilers must answer ------------
Info "arch          : $Arch"
Info "build dir     : $BuildDir"
# Not invoking cl.exe just to print its banner: with no input files it exits
# non-zero and prints to stderr, which is noise. That it exists is all we need
# here -- the real proof it works is CMake's compiler-ID test below.
$cl = Join-Path $msvc "bin\Hostx64\x64\cl.exe"
if (-not (Test-Path $cl)) { Fail "cl.exe missing at $cl" }
& $Nvcc --version | Select-String "release"
Info "cl.exe        : $cl"

if (Test-Path $BuildDir) {
  try { Remove-Item -Recurse -Force $BuildDir -Confirm:$false -ErrorAction Stop }
  catch { Write-Host "[!] could not clean $BuildDir -- continuing anyway ($($_.Exception.Message))" }
}

# ---------- 8. configure ----------------------------------------------------
# CMAKE_CUDA_STANDARD is raised to 20 by CMakeLists (the synced dtypes.h uses
# std::bit_cast) -- do not override it here. CUDA is optional in CMakeLists and
# silently degrades to a host-only build when nvcc is missing; we already proved
# nvcc exists above, so whatever comes out of this script carries the backend.
& $cmake `
  -S $root -B $BuildDir -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CXX_COMPILER=$cl" `
  "-DCMAKE_CUDA_COMPILER=$Nvcc" `
  "-DCUDAToolkit_ROOT=$cudaRoot" `
  "-DCMAKE_CUDA_ARCHITECTURES=$Arch" `
  "-DCMAKE_MAKE_PROGRAM=$ninja" `
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if ($LASTEXITCODE -ne 0) { Fail "configure failed (exit $LASTEXITCODE)." }

# ---------- 9. build --------------------------------------------------------
& $cmake --build $BuildDir -j
if ($LASTEXITCODE -ne 0) { Fail "build failed (exit $LASTEXITCODE)." }

$dll = Join-Path $BuildDir "rpg_engine.dll"
if (-not (Test-Path $dll)) { Fail "build reported success but $dll is missing." }

Write-Host ""
Write-Host ("[ok] artifact: {0}  ({1} bytes)" -f $dll, (Get-Item $dll).Length)
Write-Host ("[i]  install into the game:  python scripts\sync_engine.py --backend gpu --lib `"$dll`"")
