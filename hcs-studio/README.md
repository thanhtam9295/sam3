# HCS Studio

Desktop image analysis for high-content and high-throughput fluorescence microscopy
(Nikon `.nd2`, Operetta CLS, ImageXpress). The full specification is in `PROMPT.md`;
the project rules are in `CLAUDE.md`.

**Status: Phases 11–14 (plate annotation; projection and background removal; pharmacology;
comparison and figure export, PROMPT_ADDON.md) done; Phases 1–10 of PROMPT.md
done; validation on real instrument data pending.** Plates from Nikon, Operetta CLS,
ImageXpress, OME-TIFF, OME-Zarr and plain TIFF folders can be browsed and viewed.
Correction, enhancement, deconvolution, segmentation (Cellpose 4, SAM 3, threshold +
watershed) and measurement steps (object features, colocalization, QC and counts) are
combined into pipelines and run on wells or plates as resumable background jobs. Results
appear on a clickable plate (any readout, % positive, robust Z, Z′) in the app and in a
self-contained `index.html` that works offline. Wells can be stitched; objects can be
gated, classified (trained on clicked examples), clustered, and analysed for spatial
organisation, also after a run. PCA/UMAP profiles normalised to negative controls, a
distance-to-control score and dose-response fits are linked to the plate. Plates are
annotated in a Layout tab (cell lines, compounds and combinations with units, roles;
dilution-series and combination-matrix helpers, validation, versions, CSV/XLSX import and
export). Dose–response curves (4PL/5PL/GR), 6 × 6 combination matrices and synergy
scores (Bliss, HSA, Loewe, ZIP) come with plate QC and an HTML/PDF report. Fields can be
compared side by side with linked contrast and exported as reproducible figures. Algorithms are tested on synthetic data;
real-data validation is pending. See `docs/index.md`, `docs/processing.md`,
`docs/measurements.md` and `docs/validation.md`.

## Install

On Windows, use the installer (`HCS-Studio-<version>-cpu-setup.exe`, or `-cuda-` for
NVIDIA GPUs), built by the `hcs-studio-windows` GitHub workflow; see `docs/install.md`.
Then run `hcs-studio selftest`.

From Python 3.11 or 3.12, with [uv](https://docs.astral.sh/uv/):

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

`hcs-studio info` prints the version and GPU status without opening a window;
`hcs-studio selftest` checks the installation end to end.
`hcs-studio acceptance …` runs the acceptance tests on real data (see `docs/acceptance.md`).
`hcs-studio inspect <path>` describes how a plate folder is read.
`hcs-studio run --pipeline pipe.yaml --input <plate> --workers auto` runs a pipeline unattended;
`--resume <run folder>` continues an interrupted run.
`hcs-studio report <run folder>` regenerates the shareable `index.html`.
`hcs-studio analyze <run folder> --pipeline steps.yaml` applies gates, classifiers and
clustering to a finished run; `hcs-studio train-classifier` trains a classifier from a
labels CSV.
`hcs-studio profile <run folder> [...]` computes PCA/UMAP profiles, distance to control and
dose-response curves.
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
