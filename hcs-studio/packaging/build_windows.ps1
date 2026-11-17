# Build the Windows installer of HCS Studio.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Variant cpu|cuda]
#
# Needs: Python 3.12 (x64), uv (https://docs.astral.sh/uv/), Inno Setup 6 (iscc on PATH or
# in its default folder). Run from the hcs-studio folder. Output: dist\HCS-Studio-*-setup.exe
param([ValidateSet("cpu", "cuda")][string]$Variant = "cpu")
$ErrorActionPreference = "Stop"

uv venv --python 3.12 .venv-build
$py = ".venv-build\Scripts\python.exe"
if ($Variant -eq "cuda") {
    uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu124
} else {
    uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cpu
}
uv pip install --python $py -e ".[gui,segmentation,packaging]"

# The user guide is shipped as HTML next to the program.
& $py scripts\build_docs.py dist\docs

& $py -m PyInstaller packaging\hcs_studio.spec --noconfirm --distpath dist --workpath build
Copy-Item -Recurse -Force dist\docs "dist\HCS Studio\docs"

# Check the bundle before packing it.
& "dist\HCS Studio\hcs-studio.exe" info
if ($LASTEXITCODE -ne 0) { throw "hcs-studio info failed" }
& "dist\HCS Studio\hcs-studio.exe" selftest
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }

$version = & $py -c "import hcs_studio; print(hcs_studio.__version__)"
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DAppVersion=$version" "/DVariant=$Variant" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-ChildItem dist\HCS-Studio-*-setup*
