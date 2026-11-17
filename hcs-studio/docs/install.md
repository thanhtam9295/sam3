# Installing HCS Studio

## Windows installer

Run `HCS-Studio-<version>-cpu-setup.exe` (or the `-cuda-` variant for NVIDIA GPUs; it is
larger and comes as a `setup.exe` plus `.bin` files that must stay in the same folder).
No administrator rights are needed: by default it installs for the current user. It adds
**HCS Studio** to the Start menu (and optionally the desktop), the user guide, and, if
you tick it, the `hcs-studio` command to `PATH`.

After installing, check it: open a command prompt and run

```bat
"%LOCALAPPDATA%\Programs\HCS Studio\hcs-studio.exe" selftest
```

(or `hcs-studio selftest` if you added it to `PATH`). It writes a small synthetic plate to
a temporary folder, processes it with two worker processes, resumes it, writes the
report, computes PCA/UMAP profiles and runs a small deconvolution; every line should read
PASS. Uninstall from *Settings → Apps*; your settings (`%USERPROFILE%\.hcs_studio`) and
run folders are kept.

| Variant | Contains | Use when |
|---|---|---|
| `cpu` | everything except CUDA | no NVIDIA GPU, or Cellpose on the CPU is fast enough |
| `cuda` | PyTorch with CUDA 12.4 | an NVIDIA GPU with a recent driver (535 or newer) |

Cellpose model weights are downloaded the first time a model is used (internet access to
huggingface.co is needed once). **SAM 3 is not included**: it needs its own Python
environment with `numpy<2` and the gated weights; see *Setting up SAM 3* in the user guide
and point **Tools → SAM 3 settings** at that environment.

## From Python (Windows, macOS, Linux)

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) (or pip):

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[gui,segmentation]"
```

On Windows the interpreter is `.venv\Scripts\python.exe`. For an NVIDIA GPU, install the
CUDA build of PyTorch first (`uv pip install torch --index-url
https://download.pytorch.org/whl/cu124`). Then `python -m hcs_studio` opens the app,
`hcs-studio selftest` checks the installation.

## Building the installer

The installer is built on Windows by `packaging/build_windows.ps1` (PyInstaller makes a
folder with two programs, `HCS Studio.exe` for the window and `hcs-studio.exe` for the
command line; Inno Setup packs it). It is also built by the GitHub Actions workflow
`hcs-studio-windows` (run it from the Actions tab, or push a tag `hcs-studio-v<version>`),
which first runs the test suite on Windows, then builds, installs and self-tests the
installer and keeps it as a downloadable artifact.

## Hardware

| | Minimum | Recommended for 384-well screens |
|---|---|---|
| Memory | 8 GB | 32 GB or more (about 1–2 GB per worker for 2-channel 1080 × 1080 fields; deconvolution needs more) |
| Processor | 4 cores | 8–16 cores (one worker per core, minus one) |
| Disk | | an SSD for run folders; plan about 2–3 × the raw data size with corrected images, about 0.5 × without (`--no-images`) |
| GPU | none | NVIDIA with 8 GB or more, for Cellpose and deconvolution |
