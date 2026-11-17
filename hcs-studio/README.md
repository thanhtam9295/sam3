# HCS Studio

Desktop image analysis for high-content and high-throughput fluorescence microscopy
(Nikon `.nd2`, Operetta CLS, ImageXpress). The full specification is in `PROMPT.md`;
the project rules are in `CLAUDE.md`.

**Status: Phase 1 (readers and browser).** Plates from Nikon, Operetta CLS, ImageXpress,
OME-TIFF, OME-Zarr and plain TIFF folders can be browsed on a plate map and opened in the
embedded napari viewer. The readers are tested on synthetic plates only; no processing
modules exist yet. See `docs/index.md`.

## Install

Python 3.11 or 3.12 is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[gui,dev]"
```

On Windows the interpreter path is `.venv\Scripts\python.exe`. The GPU is optional:
install PyTorch with CUDA (and optionally CuPy) into the same environment and the status
bar will report the device; without them the app runs on the CPU.

## Run

```bash
python -m hcs_studio
```

`hcs-studio info` prints the version and GPU status without opening a window.
`hcs-studio inspect <path>` describes how a plate folder is read.

## Develop

```bash
./scripts/ci.sh
```

runs `ruff`, `mypy --strict` and `pytest`. On headless Linux the GUI tests need
`xvfb` and Mesa software OpenGL (`apt-get install xvfb libegl1 libgl1-mesa-dri libxcb-cursor0`);
the script uses `xvfb-run` automatically, and GUI tests are skipped when no display is available.

## Layout

The core package (`hcs_studio` except `hcs_studio.gui`) never imports Qt or napari, so the
CLI and worker processes run headless. A test enforces this.
