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
7. Packaging (Phase 10): PyInstaller one-folder bundle with two executables (`HCS Studio.exe` windowed GUI, `hcs-studio.exe` console CLI) + Inno Setup installer; built on Windows by `packaging/build_windows.ps1` and the `hcs-studio-windows` GitHub workflow. SAM 3 is never bundled.

All napari access goes through `gui/viewer.py` (`ViewerPanel`).

## Reader conventions

- Readers live in `io/`, subclass `PlateReader`, register in `READERS`, and return a `Plate` whose fields load lazy dask `(T, C, Z, Y, X)` arrays. Anything a reader assumes goes into `Plate.notes`.
- Reader tests use `tests/builders.py`: synthetic vendor layouts whose pixel values encode (well, field, channel, z).
- No public `.nd2` sample is reachable from the cloud container; the Nikon reader is tested against `FakeND2` and still needs validation on real files.
- User settings (overrides, favourites) go to `~/.hcs_studio` or `HCS_STUDIO_HOME`, never into raw folders.

## Processing conventions

- Modules live in `processing/`, subclass `Module`, register in `MODULES`, and take a `ModuleParams` pydantic model (`extra="forbid"`). User-facing channel numbers start at 1.
- `ImageData` is one timepoint, float32 `(C, Z, Y, X)`, with a `linear` flag. Set `preserves_linearity = False` on any module whose output must not be measured; linearity is never regained.
- Plate-level estimates go in `fit(context)`; `context.sample()` yields fields already processed by earlier steps. Summarise the state in `describe_state` for the manifest.
- `PipelineRunner.process` returns `final` and `corrected` (last linear image). Measurement code must use `corrected`.
- Every module needs a known-answer test (`tests/test_processing.py`, `tests/test_deconvolution.py`). `docs/processing.md` is generated by `scripts/gen_processing_docs.py`; a test fails when it is stale. Record what is and is not validated in `docs/validation.md`.
- Deconvolution: backends in `processing/deconvolution.py` (numpy CPU, PyTorch CUDA); memory per padded voxel is measured (~80 B, budget constant 96 B) and drives automatic tiling. PSF helpers in `processing/psf.py` keep PSFs as float32 (Z, Y, X), sum 1, centre at `n // 2`.
- `QDialog` subclasses must not define `self.result` (shadows `QDialog.result()`). Never read widgets from inside a background task; capture values first.
- Run outputs go through `pipeline/executor.py` (`run_pipeline`), which refuses output folders inside raw data.

## Segmentation conventions

- Segmentation modules live in `segmentation/` and register in the same `MODULES` registry. They add a `LabelSet` (int32 (Z, Y, X), `parent` + `parent_ids`) with `ImageData.with_labels`; pixel data and linearity are unchanged.
- Cells grown from nuclei share the nucleus ID. Other links use `overlap_parents` / `point_parents`.
- SAM 3 runs only in a separate environment (`.venv-sam3`, numpy < 2) via `segmentation/sam3_worker.py`, which must import only stdlib, numpy, torch, PIL and sam3. Its CPU shim and bfloat16 autocast are deliberate; do not edit the upstream `sam3/` package.
- SAM 3.1 is the video release; still images use `build_sam3_image_model` (`facebook/sam3`, `sam3.pt`). Access to the gated weights is granted; download with `HF_HUB_DISABLE_XET=1` (the Xet hosts are not allowed in the cloud environment). SAM 3 concept prompts are unreliable on dense small nuclei and wording-sensitive (see docs/validation.md); keep Cellpose as the batch default.
- The `fake` SAM worker model exists only for protocol and UI tests.
- Cellpose and SAM weights come from huggingface.co; the cloud environment `sam3-hcs-studio` allows it.

## Measurement conventions

- Measurement steps subclass `MeasurementModule` (in `analysis/measure.py`): `apply` passes the image through; `measure(MeasureContext)` returns DataFrames by table name. The runner gives them `raw`, `corrected` (last linear image) and the current labels; never measure `final`.
- Table names: `objects` → `objects.parquet`, `coloc` → `coloc.csv`, `fields` → `fields.csv` per well; `output/tables.py` builds `tables/` after the run (streamed Parquet merge, QC flags, per-well summary).
- Colocalization definitions follow Fiji Coloc 2 except the Costes search (first crossing from the top) and p (= 1 − Coloc 2's value); see docs/validation.md. Column reference: docs/measurements.md (update it when adding features).
- pandas-stubs types `df.loc[i, col]` as a scalar union; in tests read rows with `to_dict("index")`.

## Batch conventions

- `run_pipeline` (pipeline/executor.py) orchestrates; `process_well` writes one well, `well_summary.json` last (with `pipeline_hash`) = completion marker. Worker processes rebuild the plate with `reopen_plate(plate.source)`; anything sent to workers (`RunPlan`, `Job`) must be picklable. Use the `spawn` context everywhere.
- Resume skips wells that `well_is_complete`; fitted states live in `state/fitted.pkl`. `well_summary.json` is rewritten after every run with plate-level summaries, so test "not redone" on per-field outputs (previews), not on the summary.
- Modules that use the GPU override `uses_gpu()`; modules that must not run in several processes set `exclusive = True`.
- Jobs (pipeline/jobs.py) run in their own process and report dict events (`run_started`, `well_started`, `field_done`, `well_finished`, `status`, `log`, `finished`); the GUI Jobs dock only listens.

## Results conventions

- `output/results.py` (`RunResults`) reads a run folder for display: readouts, % positive, robust Z, Z′, previews. The app's Results tab and `index.html` use it; keep their maths identical (the browser test compares them).
- `index.html` (template `output/templates/report.html.j2`) must stay self-contained: inline CSS/JS, data as embedded JSON (escape `</`), images by relative path only. `test_report_works_offline_in_a_browser` runs it in Chromium via Node Playwright with the network blocked.
- Per field, previews are `fNNN.png` (enhanced), `fNNN_corrected.png`, `fNNN_raw.png` and `fNNN_outline_<labels>.png` (RGBA); never burn outlines into the composites.
- The focus score measures edge sharpness relative to contrast; it must not depend on cell number (sparse wells are often the interesting ones).

## Analysis conventions

- Table steps subclass `TableModule` (`analysis/table_steps.py`) and register in `MODULES`. They run after all wells (`run_table_steps`, called by `aggregate_run`), never per field: `fit_table(sample)` for plate-level state, `transform` returns new columns aligned with the well's objects, `summarize` returns per-well values and JSON details. Column prefixes: `gate_`, `class_`, `cluster_`, `spatial_` (excluded from feature statistics and from features).
- Annotations go to `wells/<W>/objects_annotations.parquet`; never rewrite `objects.parquet`. `pipeline_hash` leaves table steps out, so they can change without invalidating a resume; `analyze_run` applies them to a finished run (the run's own table steps first unless `include_run_steps=False`).
- `stitch_well` is an image-scope step that `process_well` recognises; it adds `well_x_um`, `well_y_um`, `in_overlap_duplicate` to objects and `stitching` + `field_size_um` to `well_summary.json`. Duplicates are left out of counts, statistics, sampling and spatial analysis.
- Classifiers are joblib dicts (`FORMAT = "hcs-studio-classifier/1"`); record the file's SHA-256 in the analysis summary.
- `docs/processing.md` covers table steps too (regenerate after changing their parameters).

## Profiling conventions

- `analysis/profiling.py` (`run_profiling`, `load_profile`) is headless; it writes `advanced/` of the first run (or `output`). Normalisation is robust Z per plate relative to negative-control **objects**; well profiles = median of object Z-scores (do not scale by control-well spread: too few wells). Clip ±10 before PCA/UMAP/distances.
- `RunResults` adds `profile_*` readouts from `advanced/well_profiles.parquet` (rows of that run only) and `profile_embedding()` for plots; the report embeds it (well points + ≤ 3000 sampled objects).
- Figures for files go through matplotlib (`save_scatter`, `_figures`); pyqtgraph's SVG exporter fails with this Qt version.
- UMAP is seeded (single-threaded); record every option in `profiling.json`.

## Hardening conventions

- Never load a whole plate's objects or colocalization rows: `aggregate_run` summarises well by well (`object_statistics`, `coloc_statistics`) and streams the plate files. Sample (`sample_objects`, `_sample`) for anything plate-wide.
- `run_pipeline` checks free disk space before starting (`estimate_output_bytes`, `check_disk_space`); a full disk (ENOSPC) marks the outcome `fatal` and stops the run (resumable). `MemoryError` gives an actionable message. Keep these messages written for users.
- Per-step timings: `PipelineRunner.timings`; per well in `well_summary.json` → `timing`; per run in `run_manifest.json` → `performance` (`run_performance`). Use them before optimising.
- Run options `save_images` and `max_memory_gb` travel through `RunPlan`, `Job`, the run dialog and the CLI; `deconvolution.MEMORY_LIMIT` caps the deconvolution budget in workers.
- The GUI logs to `<settings>/logs/hcs-studio.log` and shows unexpected errors without closing (`gui/launch.py`). `hcs-studio selftest` (`selftest.py`) and `hcs-studio gui --smoke-test` check installations; keep them fast and dependency-light.
- `multiprocessing.freeze_support()` must run first in the frozen entry (`packaging/entry.py`); worker processes of the bundle restart the executable.
- `scripts/make_test_plate.py` writes realistic synthetic plates (384 wells); `scripts/pipelines/perf_384.yaml` is the reference pipeline for performance numbers in docs/validation.md.

## Add-on phases 11–15 (PROMPT_ADDON.md) — approved; Phases 11–12 built

Order: 11 annotation → 12 projection + background → 13 pharmacology → 14 comparison/export → 15 integration (all prerequisites, Phases 1–10, are built). Stop for review after each phase.

### Plate annotation data model (Phase 11; replaces `model/layout.py`)

```
PlateLayout (model/annotation.py)
 ├─ plate_id / barcode, plate_type (96 | 384 | 1536), assay, date, operator, version, parent_version
 └─ wells: dict[(row, col)] → WellAnnotation
     ├─ cell_line, passage, seeding_density (cells/well)
     ├─ treatments: tuple[Treatment, ...]   (0 = untreated/vehicle only, 1 = single agent, 2+ = combination)
     │    └─ Treatment(compound, concentration (base units), unit_kind, display_unit, vehicle)
     ├─ role: sample | negative_control | positive_control | untreated | blank | empty | excluded
     │        (+ unstained, needed by the AF-reference step)
     ├─ replicate_group: derived from (cell_line, sorted treatments) unless overridden
     ├─ time_point_h, notes, custom: dict[str, str]
```

- Concentrations are stored in base units: molar → M; mass → g/L (ng/mL etc.); percent → fraction (v/v). Molar and mass convert only with a molecular weight. One tested utility: `model/units.py`.
- Treatments are compared order-free: (A, B) == (B, A). Concentration 0 of a drug means "this drug absent" (single-agent edge of a matrix).
- Layouts are immutable values; an edit makes a new version. Versions live in `<settings>/layouts/<plate key>/vNNNN.json` (plate key = barcode, else a hash of the plate path). Each run copies the version it used to `plate_layout.json` and records `{version, sha256}` in `run_manifest.json`. Templates live in `<settings>/layout_templates/`.
- Only the processing-relevant part of a layout (unstained/blank wells) enters `pipeline_hash`; changing names, doses or roles of samples does not block resuming. Annotation can be re-applied to a finished run (tables and report refreshed, no image reprocessing).
- Per-well tables get the annotation columns joined at aggregation; per-object tables are joined by well at read time (no duplication across 10⁷ rows).

### Annotation conventions (Phase 11, built)

- Model: `model/annotation.py` (PlateLayout, WellAnnotation, Treatment, AnnotationFilter);
  helpers `model/layout_tools.py` (pure functions returning new layouts: `dilution_series`,
  `combination_matrix`, `set_role`, `exclude_edges`, `apply_fields`, copy/paste, templates,
  `validate` → `Issue`s); files `io/layout_io.py` (`load_layout` → `LayoutImport`,
  `save_layout`, plate-map XLSX, figures); versions/templates `io/layout_store.py`.
  `model/layout.py` is a compatibility shim (`load_plate_layout`, `run_layout_file`).
- Values are canonicalised to 12 significant digits so CSV/XLSX/JSON round trips are exact;
  keep that when adding fields. Custom keys starting with `_` are internal (not exported
  as columns, e.g. the edge-exclusion marker).
- `apply_fields` on an unannotated well starts from a sample, not from `EMPTY`.
- GUI: `gui/layout_editor.py` (`LayoutEditor`; its document is `self.doc`, never
  `self.layout`, which is `QWidget.layout()`), `gui/annotation_filter.py`. The main window
  keeps `layout_document` (shown), `_run_layout` + `layout_path` (what runs use);
  `layout_for_run()` saves unsaved edits as a new version first.
- Echo/D300 dispense-file import is deferred until a sample file is provided.

### Projection and background conventions (Phase 12, built)

- `processing/projection.py` (`projection`) and `processing/background.py` (`background`)
  report per field through `ImageData.reports` (`{"module": …}`): planes used (from 1),
  focus scores, per-channel object/background mean changes, `warnings`. `process_well`
  collects report `warnings` into `well_summary.json` → `warnings` and log events.
- `Background.needs_fit` is a property (only `basic` and `constant` + blank wells fit).
- Measurements never use saved files: `image_dtype="original"` only changes what is
  written (`to_storage_dtype`).
- `stitch_well.save_mosaic` writes `mosaic.ome.tif` with `float_mosaic` (overlaps averaged).
- Large synthetic Z-stack plates: `utils/synthetic.write_ome_zarr_plate` (compact; a
  384 × 3 × 20-plane plate writes in ~10 s).

### New dependencies (approved list for 11–15)

| Library | Phase | Use |
|---|---|---|
| openpyxl | 11 | XLSX import/export of layouts and results |
| reportlab | 13/15 | PDF reports (pure-Python wheels; bundles with PyInstaller) |
| imageio-ffmpeg | 14 | MP4 export (GIF needs only Pillow) |
| synergy | 13 | MuSyC, optional; ask first |

Not added: lmfit (`scipy.optimize.least_squares` covers bounds, robust loss, covariance CIs), weasyprint (needs system Pango/Cairo; hard on Windows).

## Commands

Run from this folder (`hcs-studio/`).

- Environment: `uv venv --python 3.12 .venv` then `uv pip install --python .venv/bin/python -e ".[gui,dev,segmentation]"`
- Launch: `python -m hcs_studio` (or `hcs-studio`); `hcs-studio info` prints version and GPU status
- Describe how a path is read: `hcs-studio inspect <path>`
- Run a pipeline: `hcs-studio run --pipeline p.yaml --input <plate> --wells A01-H12 --workers auto`; `--resume <folder>`, `--retry-failed <folder>`; list modules: `hcs-studio modules`
- Deconvolution timing: `hcs-studio benchmark --shape 30 2048 2048 --iterations 40 [--device gpu]`
- Fine-tune Cellpose: `hcs-studio train-cellpose --data <folder>`; real-SAM 3 tests: `HCS_SAM3_PYTHON=.venv-sam3/bin/python HCS_SAM3_WEIGHTS=1 pytest -k real_sam3`
- Regenerate a run's report: `hcs-studio report <run folder>`
- Profiles: `hcs-studio profile <run> [<run> ...] --object-type nuclei [--level well] [--no-umap]`
- Apply table steps to a finished run: `hcs-studio analyze <run folder> --pipeline steps.yaml [--only]`; train a classifier from a labels CSV: `hcs-studio train-classifier --run <folder> --labels labels.csv --output model.joblib`
- Acceptance tests on real data (docs/acceptance.md): `hcs-studio acceptance counts|coloc|run|kill-resume|beads …` (PASS/FAIL + JSON report)
- Check an installation: `hcs-studio selftest`; GUI: `hcs-studio gui --smoke-test`
- Layouts: `hcs-studio layout <file> --validate [--export out.xlsx [--plate-map]]`; re-annotate a run: `hcs-studio annotate <run> --layout <file>`
- Synthetic 384-well plate: `python scripts/make_test_plate.py <folder> --wells 384 --sites 2`
- Windows installer (on Windows): `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Variant cuda]`; user guide as HTML: `python scripts/build_docs.py <folder>`
- All checks: `./scripts/ci.sh` (ruff check, ruff format --check, mypy --strict, pytest)
- Headless Linux needs `xvfb libegl1 libgl1-mesa-dri libxcb-cursor0`; `ci.sh` wraps pytest in `xvfb-run`. `QT_QPA_PLATFORM=offscreen` does not work (napari needs a real OpenGL context).
