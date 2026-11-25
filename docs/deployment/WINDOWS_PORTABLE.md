# Portable Windows executable

A one-folder PyInstaller bundle (`dist\sam3-portable\`) containing
`sam3-cli.exe`, the Python runtime, PyTorch (CUDA 12.8) and the `sam3`
package. Copy/zip the folder to any Windows x64 machine with an NVIDIA GPU; no
installation or admin rights needed.

**Status:** build scripts prepared and the CLI tested on Linux with a fake
model; not yet built/run on Windows (T-009).

## Contents

| File | Role |
|---|---|
| `packaging/windows/sam3_cli.py` | CLI: `sam3-cli segment --image … --prompt … [--checkpoint …] --out …` |
| `packaging/windows/sam3_portable.spec` | PyInstaller spec (includes BPE vocab, excludes notebooks/plots) |
| `packaging/windows/build.ps1` | builds from the hash-pinned runtime lock, swaps in CUDA torch, runs PyInstaller |
| `packaging/windows/README_PORTABLE.txt` | end-user readme copied into the bundle |
| `.github/workflows/windows-portable.yml` | **manual** CI build → private workflow artifact (7 days) |

## Build (Windows 10/11 x64, NVIDIA GPU, Python 3.12, Git)

```powershell
git clone <repo> sam3; cd sam3
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
dist\sam3-portable\sam3-cli.exe --help
dist\sam3-portable\sam3-cli.exe segment --image assets\images\truck.jpg --prompt truck `
    --checkpoint C:\models\sam3.pt --out out
```

Expected bundle size: several GB (CUDA libraries). Use `-SkipGpuCheck` only on
build machines without a GPU (the result still needs a GPU to run).

## Outputs

`out\results.json` (count, per-instance score, XYXY box, pixel area, and
provenance: versions, prompt, threshold, checkpoint SHA-256), plus
`<image>_masks.png` (16-bit label image) and `<image>_overlay.png`.

## Constraints and decisions

* **GPU required** (KI-009). A CPU build becomes possible after T-004.
* **No weights in the bundle** (licence + size + secrecy). Users pass
  `--checkpoint` or download with their own Hugging Face licence acceptance.
* Windows CUDA torch is not hash-locked yet (KI-015, T-008).
* Code signing: sign `sam3-cli.exe` with the organisation's certificate before
  distribution (SmartScreen); add to the build once a certificate is available.
* Distribution (GitHub Release, file share, Intune) requires explicit approval;
  the workflow intentionally only creates a temporary artifact.
* Alternative if PyInstaller proves fragile with CUDA/Triton: a relocatable
  "embedded Python" folder (python-3.12 embeddable zip + `pip install --target`)
  launched by a `sam3-cli.cmd` wrapper — same lock file, no freezing.
