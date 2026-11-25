<#
.SYNOPSIS
  Build the portable Windows SAM 3 CLI bundle into dist\sam3-portable\.
.DESCRIPTION
  1. Creates an isolated venv from the hash-pinned lock file.
  2. Replaces the CPU torch wheel with the pinned CUDA 12.8 build (GPU required at run time).
  3. Runs PyInstaller with packaging\windows\sam3_portable.spec.
  Nothing is uploaded or published. No checkpoints are bundled.
#>
param(
  [string]$Python = "py -3.12",
  [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
  [switch]$SkipGpuCheck
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $Root

Invoke-Expression "$Python -m venv .venv-build"
$Py = ".\.venv-build\Scripts\python.exe"
& $Py -m pip install --upgrade "pip<26" uv
& $Py -m uv pip sync --python $Py --require-hashes requirements\lock\py312-runtime.txt
# GPU torch: same versions as the lock, CUDA build. Not hash-locked yet (docs/TASKS.md T-008).
& $Py -m uv pip install --python $Py --reinstall "torch==2.10.0" "torchvision==0.25.0" --index-url $TorchIndex
& $Py -m uv pip install --python $Py -e . --no-deps
& $Py -m uv pip install --python $Py "pyinstaller==6.16.0"

if (-not $SkipGpuCheck) {
  & $Py -c "import torch,sys; print('CUDA', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
}

& $Py -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\windows\sam3_portable.spec
Copy-Item LICENSE dist\sam3-portable\LICENSE.txt
Copy-Item packaging\windows\README_PORTABLE.txt dist\sam3-portable\README.txt
Write-Host "Built dist\sam3-portable\sam3-cli.exe"
