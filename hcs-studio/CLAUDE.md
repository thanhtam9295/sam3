# HCS Studio — project rules

Desktop image analysis for high-content cellular imaging. Full specification: `PROMPT.md`.
Work proceeds phase by phase (PROMPT.md §9); stop for review after each phase.

## Rules

1. **Never modify raw data.** Raw folders are read-only. All outputs go to a new run folder (PROMPT.md §7).
2. **Every output carries provenance**: software version, git commit, model name + checkpoint hash, all parameters, input file paths + checksums, timestamp. Saved as `run_manifest.json`.
3. **Scientific correctness over speed.** Each algorithm module ships with unit tests against synthetic data with a known answer (e.g. Pearson of two identical images = 1.0; RL deconvolution of a blurred point source re-concentrates intensity).
4. **Lazy, chunked I/O.** Plates can be hundreds of GB. Never load a whole plate into RAM. Use dask + OME-Zarr caching.
5. **GPU optional, never required.** Every GPU path (CUDA/PyTorch/CuPy) has a CPU fallback and auto-detection.
6. **UI never blocks.** All processing runs in a worker process/queue with progress, cancel and resume.
7. **Pipelines are data.** A pipeline is a YAML/JSON file the UI can save, load and re-run on another plate.
8. **Stop and ask** before adding a heavy dependency not listed below, or when a vendor file format behaves differently than expected — show the sample file structure first.
9. Type hints everywhere, `ruff` + `mypy` clean, `pytest` green before reporting a phase as done.
10. Keep a `CHANGELOG.md` and update `docs/` for every user-facing feature.

Also: enhanced images (CLAHE, gamma, sharpening) are for viewing and segmentation input only.
Intensity measurements and colocalization use linear data (AF-corrected, optionally deconvolved).

## Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 or 3.12 | Single conda/uv environment; pin versions in `pyproject.toml` |
| GUI shell | PySide6 (Qt 6) | Main window, docks, plate map, dialogs |
| nD viewer | napari (embedded as a Qt widget) | Multichannel, Z, T, labels layers, 3D rendering |
| Plate map & charts | Qt Graphics View for plate; pyqtgraph / plotly (QtWebEngine) for PCA/UMAP and histograms | Plots must be interactive (lasso → highlight wells/cells) |
| File readers | `nd2`, `tifffile`, `bioio` + `bioio-nd2` / `bioio-tifffile` / `bioio-ome-tiff`, custom Operetta & ImageXpress parsers | PROMPT.md §4 |
| Arrays | numpy, dask, zarr / `ome-zarr` | Cache converted data as OME-Zarr |
| Image processing | scikit-image, scipy.ndimage, OpenCV | |
| GPU | PyTorch, CuPy (optional) | |
| Deconvolution | Own Richardson–Lucy (CuPy/PyTorch FFT) + `pycudadecon` optional; `psfmodels` for Gibson–Lanni / vectorial PSFs | |
| Denoising (optional) | `careamics` (Noise2Void / N2V2) | |
| Segmentation | cellpose ≥ 4.2 (`cpsam_v2`, `cpdino`, `cpdino-vitb`); SAM 3.1 (`facebookresearch/sam3`, gated HF weights); optional `micro-sam` | PROMPT.md §5.4 |
| Features | scikit-image `regionprops_table`, `mahotas` | |
| ML | scikit-learn, `umap-learn`, `hdbscan` | |
| Tables | pandas + pyarrow/Parquet | Per-cell tables can be 10⁷ rows |
| Reports | Jinja2 → HTML; matplotlib for static figures | |
| Packaging | PyInstaller or Briefcase → Windows installer; also `python -m hcs_studio` | |
| Tests | pytest, pytest-qt, hypothesis | |

## Architecture decisions (approved)

1. Core is headless: only `hcs_studio.gui` may import PySide6 or napari (enforced by a test). GUI is the `gui` extra.
2. Modules declare a scope (field / well / plate) and typed inputs/outputs (image, labels, table), with an optional plate-level `fit` before per-field `apply`.
3. Images carry a `linear` flag; CLAHE/gamma/sharpening clear it; analyzer and colocalization refuse or warn on non-linear input.
4. Pipeline schema, executor, run-folder writer and manifest are built in Phase 2; Phase 6 keeps builder UI, job queue, resume, CLI.
5. Cellpose, SAM 3.1, micro-sam, careamics, CuPy are optional extras, imported lazily. SAM 3.1 pins `numpy<2`; run it in a separate process/environment if that conflicts.
6. No `bioio` (use `nd2`, `tifffile`, `ome-zarr`); pyqtgraph for in-app plots and plain embedded JS in `index.html` (no plotly/QtWebEngine); Leiden optional; no BM3D.
7. Packaging tool is chosen in Phase 10.

All napari access goes through `gui/viewer.py` (`ViewerPanel`).

## Commands

Run from this folder (`hcs-studio/`).

- Environment: `uv venv --python 3.12 .venv` then `uv pip install --python .venv/bin/python -e ".[gui,dev]"`
- Launch: `python -m hcs_studio` (or `hcs-studio`); `hcs-studio info` prints version and GPU status
- All checks: `./scripts/ci.sh` (ruff check, ruff format --check, mypy --strict, pytest)
- Headless Linux needs `xvfb libegl1 libgl1-mesa-dri libxcb-cursor0`; `ci.sh` wraps pytest in `xvfb-run`. `QT_QPA_PLATFORM=offscreen` does not work (napari needs a real OpenGL context).
