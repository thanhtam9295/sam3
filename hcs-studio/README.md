# HCS Studio

Desktop image analysis for high-content and high-throughput fluorescence microscopy
(Nikon `.nd2`, Operetta CLS, ImageXpress). The full specification is in `PROMPT.md`;
the project rules are in `CLAUDE.md`.

**Status: Phase 5 (measurements).** Plates from Nikon, Operetta CLS, ImageXpress,
OME-TIFF, OME-Zarr and plain TIFF folders can be browsed and viewed. Correction,
enhancement, deconvolution, segmentation (Cellpose 4, SAM 3, threshold + watershed) and
measurement steps (object features, colocalization with Costes thresholds, QC and counts)
can be combined into a pipeline, previewed on a field and run on a plate from the
command line, producing Parquet/CSV tables per object, field and well. Algorithms are
tested on synthetic data; real-data validation is pending. See `docs/index.md`,
`docs/processing.md`, `docs/measurements.md` and `docs/validation.md`.

## Install

Python 3.11 or 3.12 is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[gui,dev,segmentation]"
```

On Windows the interpreter path is `.venv\Scripts\python.exe`. The GPU is optional:
install PyTorch with CUDA into the same environment (`pip install -e ".[gpu]"`, or the
CUDA build for your driver from pytorch.org) and the status bar will report the device.
Deconvolution then runs on the GPU; without it everything runs on the CPU.

## Run

```bash
python -m hcs_studio
```

`hcs-studio info` prints the version and GPU status without opening a window.
`hcs-studio inspect <path>` describes how a plate folder is read.
`hcs-studio run --pipeline pipe.yaml --input <plate>` runs a pipeline unattended.
`hcs-studio benchmark` times deconvolution on this machine.
`hcs-studio train-cellpose --data <folder>` fine-tunes Cellpose on edited masks.

SAM 3 needs a separate environment; see *Setting up SAM 3* in `docs/index.md`.

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
