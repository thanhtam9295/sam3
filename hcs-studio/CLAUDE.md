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

## Add-on phases 11–15 (PROMPT_ADDON.md) — approved; Phases 11–15 built

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

### Pharmacology conventions (Phase 13, built)

- Headless package `pharmacology/`: `normalize.py` (well table from runs, normalization,
  outliers, plate QC), `curves.py` (Prism-form 4PL/5PL/3PL, CIs, AUC, GR, F test),
  `combination.py` (matrix detection, Bliss/HSA/Loewe/ZIP, summaries), `analysis.py`
  (`PharmOptions`, `analyse`, `run_pharmacology`), `report.py` (all files, figures,
  HTML/PDF), `synthetic.py` (known-answer 384-well plate + minimal run folder).
- Concentrations are fitted in base units (M); display units only for labels and CSVs.
  `effect` (% inhibition scale) is what synergy models use.
- Vehicles in % and treatments of control wells are not drugs (`drug_treatments`).
- Keep synergy definitions in step with docs/pharmacology.md; the reference-data tests in
  `tests/data/pharmacology/` (Prism, SynergyFinder) run when files are present.
- GUI: `gui/pharmacology_panel.py`; analysis runs in a `BackgroundTask` (thread), not in
  the job queue; re-analysis after an exclusion does not write files (Write report does).

### Comparison conventions (Phase 14, built)

- Headless package `compare/`: `spec.py` (`Comparison`, `PanelSpec`, `ChannelDisplay`,
  `ExportSettings`: pydantic, JSON round trip), `sources.py` (`load_panel`, cached;
  corrected / enhanced / raw / `step:<n>` recomputed with `state/fitted.pkl`;
  representative field), `render.py` (`compose`, overlays, crop, matplotlib figure with
  geometry in mm), `export.py` (figure + sidecar, `regenerate`, data, movies, presets,
  batch), `arrange.py` (grid, by annotation, combination matrix, one compound).
- Determinism: figures are drawn on an Agg canvas with fixed metadata; the sidecar freezes
  automatic choices (fields, channel display) so regeneration is pixel-identical.
  `svg.fonttype = none`, `pdf.fonttype = 42` keep text editable.
- GUI: `gui/compare_panel.py` (pyqtgraph views; ImageItem gets RGB transposed to (x, y));
  the main window keeps `last_selection` (run, wells) from Results, Pharmacology and
  Profiles. Do not name widgets `width`/`layout` (they shadow QWidget methods).
- Test plate with images: `tests/compare_builders.make_run` (2 × 6 wells, 2 channels,
  5 planes, 2 fields, layout).
- Memory: `sources` caches panels up to `CACHE_LIMIT_BYTES` (LRU) and reuses one reopened
  plate per run (`_context`, with a lock); `export.MAX_MEGAPIXELS` guards PNG/TIFF.
  Sidecar checksums use `fingerprint()` (files: SHA-256; folders: a tree digest).
- Time-lapse movies need `raw`/`step:<n>` sources (run outputs keep timepoint 1 only).
- Tests: pytest's `faulthandler_timeout = 180` dumps stacks of a hung test; keep it.
- GUI tests: never `qtbot.waitSignal` on a signal that worker threads emit **several** times
  (e.g. `ThumbnailLoader.ready`, once per field): the later queued emissions reach the destroyed
  wait loop and segfault in pytest-qt's event processing after the test. Wait with
  `qtbot.waitUntil` on state instead (`ThumbnailLoader.idle()`).
- GUI tests: heavy pure-Python work (numba compiles, matplotlib figures) in a `BackgroundTask`
  thread while `qtbot.waitUntil` spins the event loop can be starved of the GIL for minutes late
  in a long GUI session (not reproducible in isolation); such tests take the `inline_tasks`
  fixture (tests/conftest.py), which runs the work synchronously.

### Integration conventions (Phase 15, built)

- Run steps subclass `RunStep` (`analysis/run_steps.py`), register in `MODULES`, pass images
  through, and implement `run(folder, log)` returning a record. `Pipeline.run_steps()`;
  `image_steps()` excludes them; `pipeline_hash` ignores them. `run_pipeline` runs them after
  the final manifest write (`finish_run_steps`: manifest `run_steps`, then `write_report`).
- `output/experiment.py` (`write_experiment_report`) and `output/methods.py`
  (`methods_paragraphs`: add a `DESCRIPTIONS` entry for every new module).
- `index.html` data: `report_data()` adds `pharmacology` (from `pharmacology/report_data.json`,
  this run's wells only), normalized readouts and `links`. Keep the template self-contained.
- Regression: `tests/test_regression.py` + `pharmacology/synthetic.write_image_plate`; the
  baseline `tests/data/regression/baseline.json` changes only deliberately
  (`HCS_UPDATE_BASELINE=1`).

### New dependencies (approved list for 11–15)

| Library | Phase | Use |
|---|---|---|
| openpyxl | 11 | XLSX import/export of layouts and results |
| reportlab | 13/15 | PDF reports (pure-Python wheels; bundles with PyInstaller) |
| imageio-ffmpeg | 14 | MP4 export (GIF needs only Pillow) |
| synergy | 13 | MuSyC, optional; ask first |
| h5py | 16 | ilastik HDF5 files and `.ilp` reading (approved 2026-11-19) |

Not added: lmfit (`scipy.optimize.least_squares` covers bounds, robust loss, covariance CIs), weasyprint (needs system Pango/Cairo; hard on Windows).

## Phase 16 — ilastik backend (PROMPT_ILASTIK.md) — approved 2026-11-19; steps (a)–(h) built

Paused 2026-11-19 after step (a); resumed 2026-11-24 (user: "CONTINUE"): step (b) built against a
fake ilastik (no `.ilp` samples yet; installing ilastik in the container was not permitted this
time). Real-ilastik tests stay opt-in (`HCS_ILASTIK`). Phases 21, 22 and 25 use ilastik only as one
option (see below).

Build order (stop for review after each): (a) settings, detection, HDF5/axis utilities — built;
(b) HeadlessRunner + Pixel Classification/Autocontext + Probabilities → Objects; (c) Object
Classification (incl. on Cellpose masks); (d) Multicut + Tracking; (e) FijiRunner, macro export,
parity test. Ask for sample `.ilp` projects + images before (b). Facts checked 2026-11-19 against ilastik's source and docs on GitHub
(www.ilastik.org is not reachable from the cloud container) and the ilastik4ij README/examples.
Re-check every flag with `--help` / `--version` on the installed ilastik before relying on it.

### Architecture

```
segmentation/ilastik_backend/
 ├─ settings.py    detection (ilastik + Fiji/ilastik4ij), version check, threads/RAM, max processes
 ├─ axes.py        TCZYX ⇄ ilastik axis strings (yxc, zyxc, tyxc, tzyxc); one tested utility
 ├─ h5io.py        ilastik-ready HDF5 (/data, gzip, chunked, axistags JSON) read/write
 ├─ project.py     .ilp reader (h5py, read-only): workflow, dims, channels, classes, version, sha256
 ├─ job.py         IlastikJob / IlastikResult (pydantic, picklable)
 ├─ runners/       HeadlessRunner (default), FijiRunner (ilastik4ij macro), PythonApiRunner
 └─ steps.py       pipeline modules: ilastik_pixel, ilastik_autocontext, ilastik_objects,
                   ilastik_multicut, ilastik_tracking, probabilities_to_objects, export_ilastik_h5
```

- Runners never import ilastik: HeadlessRunner and FijiRunner are subprocesses; ilastik's pinned
  dependencies stay out of the app environment (same rule as SAM 3).
- All runners take the same `IlastikJob` and return the same `IlastikResult`; steps do not know the runner.
- ilastik steps are `exclusive` (one ilastik process at a time; ilastik is multithreaded) and call
  ilastik once per **well** with all its fields (`--raw_data` takes several files; other roles pair
  by position). Needs a small executor hook so a step can process a well's fields as one batch.
- Outputs per PROMPT_ILASTIK.md §8 (`<run>/ilastik/…`, `wells/<W>/ilastik/…`); manifest records
  `.ilp` SHA-256, ilastik version, runner and the exact command line.

### Verified ilastik facts (ilastik 1.4.2 installed in the cloud container, 2026-11-19)

- `python -m ilastik --version` → 1.4.2 (Python 3.11). Top-level flags as in `--help`: `--headless
  --project --readonly --new_project --workflow --configfile --logfile --redirect_output --debug
  --exit_on_failure --nn_device` (+ GUI-only ones). `--new_project` needs an absolute path.
- Workflow input flags come from the role names (lower case, `_`); confirmed per workflow:
  pixel / autocontext `--raw_data [--prediction_mask]`; object classification from label image
  `--raw_data --label_image` (keeps instance IDs: use it for Cellpose/SAM masks), from binary
  `--raw_data --segmentation_image`, from prediction `--raw_data --prediction_maps`; multicut
  `--raw_data --probabilities [--superpixels]`; tracking from binary `--raw_data
  --segmentation_image`, from prediction `--raw_data --prediction_maps`.
- `--export_source` names: pixel "Probabilities", "Simple Segmentation", "Uncertainty", "Features",
  "Labels"; autocontext "Probabilities Stage 2", "Simple Segmentation Stage 2", … "Probabilities
  All Stages"; objects "Object Predictions", "Object Probabilities", "Blockwise Object …",
  "Object Identities" (+ `--table_filename`); multicut "Multicut Segmentation"; tracking
  "Tracking-Result", "Merger-Result", "Object-Identities" (+ "Plugin" / `--export_plugin` from the
  tracking export applet, e.g. "CSV-Table").
- `--output_format`: … "hdf5", "compressed hdf5", "n5", "compressed n5", "single-scale OME-Zarr",
  "multi-scale OME-Zarr", "numpy", "tif", "multipage tiff", "blockwise hdf5" (+ image sequences).
- Role flags take several files (`nargs="+"`), paired by position across roles.
- Env: `LAZYFLOW_THREADS` (-1 = auto), `LAZYFLOW_TOTAL_RAM_MB` (≥ 500). Headless defaults to
  read-only projects (to be confirmed by hashing the .ilp before/after a run).
- `.ilp` (HDF5): root datasets `workflowName`, `ilastikVersion`, `time`; groups `Input Data`,
  `FeatureSelections`, `PixelClassification` (`LabelNames`, `LabelColors`, `PmapColors`), export
  settings. The classifier is a **pickle**: never unpickle .ilp content in HCS Studio; read only
  plain datasets/attributes. Training axes/channels: `Input Data/Role Names`[0] of the last lane in
  `Input Data/infos` (`axistags` JSON + `shape`), as `ilastik.experimental.parser` reads them.
- Avoid `--cutout_subregion` (source and docs disagree on its axis order); tile in HCS Studio.
- ilastik4ij macro keys (examples/): pixel `projectfilename= inputimage= pixelclassificationtype=
  saveonly=false`; objects `inputproborsegimage= secondinputtype=`; multicut
  `boundarypredictionimage=`; tracking `input= inputproborsegimage= secondinputtype=`; HDF5 import
  `select= datasetname= axisorder=`, export `input= exportpath= datasetname= compressionlevel=`.
  The Configure command's keys are unconfirmed. Two published example macros have variable bugs.
- Incompatible with ilastik4ij: 1.3.3b2, 1.3.3rc2, 1.3.3, 1.3.3post1, 1.3.3post2, 1.4.0b27.
  Latest stable ilastik: 1.4.2 (May 2026); 1.5.0b1 (Nov 2026) is a beta.

### Installing ilastik for tests (Linux, conda packages)

ilastik is not on conda-forge; it is on the `ilastik-forge` channel (deps from conda-forge, flexible
priority; strict priority does not solve). 1.4.2's package misses `yapsy` and pulls a `hytra` without
`pluginsystem` (tracking then fails to load): pin both.

```bash
micromamba create -y -p <env> --channel-priority flexible -c ilastik-forge -c conda-forge "ilastik=1.4.2" yapsy "hytra=1.1.5"
```

The environment is ~3.4 GB; run as `<env>/bin/python -m ilastik --headless …`. Tracking uses the
DPCT solver (no CPLEX/Gurobi: the learning-tracking variant is unavailable, as in most installs).

### Conventions (step a)

- `axes.py` is the only TCZYX ⇄ ilastik conversion (`ilastik_axes`, `to_ilastik`, `from_ilastik`);
  axes missing from a string must have length 1; the channel axis is kept by default.
- `h5io.write_h5` writes `/data` (gzip, chunks 1 t × ≤32 z × ≤256² × all c) with vigra `axistags`
  JSON exactly as ilastik writes it, via a `.partial` file + rename; `read_h5` prefers
  `exported_data`. ilastik input spec: `<file>/data` (`input_spec`).
- `settings.py`: `IlastikSettings` in `<settings>/ilastik.json`; `detect()` order = settings,
  `HCS_ILASTIK`, install folders (newest first); `check_version` blocks `INCOMPATIBLE` and
  < 1.3.3post3, warns on pre-releases. `environment()` sets `LAZYFLOW_*` only when not default.
  Starting ilastik for `--version` takes seconds: the GUI runs detection in a `BackgroundTask`.
- GUI: `gui/ilastik_settings.py` (Tools → ilastik settings…). CLI: `hcs-studio ilastik`.

### Conventions (step b, built)

- Executor hook: `Module.batch_scope = "well"` + `apply_batch(images, state, keys, workdir)`
  (`keys` = (field, timepoint); `workdir` = `BatchFolders(run, well, well_name)` or None for
  previews). `process_well` runs every field up to the first well-batch step
  (`PipelineRunner.batch_index`), calls it once (`PipelineRunner.apply_batch`, under the
  exclusive lock), then continues each field with `process(begin=…, raw=…, corrected=…)`.
  Measurement steps before a well-batch step are refused (their tables would be lost).
- `project.py` (`read_project` → `ProjectInfo`; `workflow_key`; `check_compatible`; never
  unpickles), `job.py` (`IlastikJob`, `IlastikResult`, `EXPORT_SOURCES`, `IlastikError`),
  `runners/headless.py` (`HeadlessRunner.command/run`: `--headless --project= --export_source=
  --output_format=hdf5 --output_filename_format=<dir>/{nickname}_<suffix>.h5 --raw_data …`;
  no `--readonly`; SHA-256 before/after; outputs matched by prefix), `steps.py`
  (`ilastik_pixel`, `ilastik_autocontext`: needs_fit (detect ilastik + project info →
  manifest), exclusive, batch_scope well; class channels `ilastik_<class>` appended, linear
  unchanged; `probabilities_to_objects` (`probabilities_to_labels`); `ilastik_class_areas`
  (MeasurementModule → fields `ilastik_area_pct_<class>`, prefix in
  `FIELD_MEASUREMENT_PREFIXES`)), `presets.py` (`pixel_preset`). Probabilities are channels,
  not label sets (measure_objects measures every label set by default).
- Tests: `tests/fake_ilastik_headless.py` (real flag set; ramp of channel 1 between the
  project's `fake/lo`/`fake/hi`; `FAKE_ILASTIK_LOG`, `FAKE_ILASTIK_FAIL_ON`,
  `FAKE_ILASTIK_TOUCH_PROJECT`), `tests/test_ilastik_steps.py` (`make_project` writes a minimal
  `.ilp`). CLI `hcs-studio ilastik --info <ilp> | --preset <ilp> --output p.yaml`.

### Conventions (step c, built)

- `objects.py`: image step `ilastik_objects` (needs_fit, exclusive, batch_scope well; `input`
  labels → `--label_image` with an existing label set (keeps IDs), prediction_maps /
  segmentation_image → ilastik's own objects: calls "Object Identities" then "Object
  Predictions", relabelled into label set `<name>_objects`); `majority_classes` (class 0 wins
  only when nothing else is predicted); writes `wells/<W>/ilastik/object_classes_<name>.parquet`
  (+ `.json` with the project's class list). Table step `ilastik_object_classes`
  (`class_<name>`, `class_<name>_agreement`; `count_fractions` over **all** project classes so
  wells without a class report 0, not missing). Same column names as `classify_objects`, so
  readouts/pharmacology need nothing new.
- Executor: several well-batch steps run in order (`batch_indices`); `batch_index` = the last;
  measurement steps must come after the last one.
- `presets.objects_preset`: `segment_cellpose` → `ilastik_objects` → `measure_objects` → `qc`
  → `ilastik_object_classes`. Fake ilastik: workflows containing "ObjectClassification" accept
  `--label_image/--prediction_maps/--segmentation_image`; class 2 for objects ≥ `fake/min_area`.

### Conventions (step d, built)

- `multicut_tracking.py`: `_IlastikWellStep` (project, fit, folders `_setup`, `_run`);
  `ilastik_multicut` (`--probabilities` = class channels, export "Multicut Segmentation" →
  relabelled label set `name`); `ilastik_tracking` (per field a (T, …) stack; inputs
  `segmentation_image` (mask from a label set, else first class channel ≥ 0.5) or
  `prediction_maps`; exports "Object-Identities" (per-frame label set `tracked`, IDs kept to
  match the CSV) and `--export_source=Plugin --export_plugin=CSV-Table` (`output_extension`
  csv) → `wells/<W>/ilastik/tracks_<name>.csv` (field, timepoint, object_id, track_id,
  parent_track_id, lineage_id)); `read_tracking_csv` (column spellings in `CSV_COLUMNS`,
  unconfirmed); table step `ilastik_tracks` subclasses Phase 22d `Tracking` (transform maps
  ilastik's links, then `track_table`; summarize/finish_table inherited → `trk_` readouts,
  `tracking/tracks.csv`). `IlastikJob.export_plugin/output_extension`. Presets
  `multicut_preset`, `tracking_preset`; CLI `--preset <multicut.ilp> --boundary <pixel.ilp>`.

### Conventions (step e, built)

- Runner choice: steps call `runners.runner_state()` in `fit` (settings `runner`; Fiji launcher,
  `ilastik4ij` jar name, `fiji_configure` → manifest) and `runners.run_job(state, job, log)` per
  call; `make_runner` falls back to headless (note in the log + `IlastikResult.note`) for jobs
  `fiji.unsupported` refuses. `_projects_used(folder, state, runners)` merges one entry per
  project SHA-256 with the runners that actually ran it.
- `runners/fiji.py`: `FijiRunner` writes `wells/<W>/ilastik/fiji_<suffix>.ijm` (literal
  statements only: Configure, per input Import HDF5 → `rename` → command → `rename` → Export
  HDF5 → `close("*")`, then `File.saveString` marker) and runs `<launcher> --headless --console
  -macro <file>`; success = marker + outputs (Fiji exits 0 on macro errors). Macro keys come from
  the ilastik4ij source (`COMMANDS`, `SECOND_INPUTS`): `projectfilename inputimage
  pixelclassificationtype|autocontextpredictiontype|objectexportsource inputproborsegimage
  secondinputtype boundarypredictionimage`; Import `select datasetname axisorder`; Export `input
  exportpath datasetname compressionlevel` (file axes `tzyxc`); Configure `executablefile
  numthreads maxrammb` (ram 0 → 4096). Paths go in as POSIX; `[ ] "` refused. Plugin limits:
  no `--label_image`, tracking only "Tracking-Result", no export plugins, one ilastik start per
  image, needs the ilastik program (not a Python).
- `folder_macro` / `project_macro` (`EXPORT_CHOICES`, `SECOND_CHOICES`) write macros for people
  (`#@` parameters, loop over a folder's `.h5`); CLI `--macro`. `h5io.export_plate` (CLI
  `--export-h5`) writes raw fields as ilastik-ready HDF5 outside raw data.
- Tests: `tests/fake_fiji.py` interprets the runner's macros like the plugin (writes `tzcyx`
  temp inputs, calls ilastik with `--readonly=1 --input_axes --output_axis_order
  --output_internal_path`, `FAKE_FIJI_LOG`); the fake ilastik accepts those flags and bare
  `<file>.h5` inputs. `tests/test_ilastik_fiji.py`: parity headless vs Fiji per workflow and for
  a whole pipeline; real parity with `HCS_ILASTIK` + `HCS_FIJI` + `HCS_ILASTIK_PIXEL_PROJECT`.

### Conventions (step f, built: the spec's remaining §5/§6/§9 items)

- Project library = the model registry (kind `ilastik`, `registry:NAME@V` refs work in every
  ilastik step); `training_set.project_metadata` → `hyperparameters` (workflow, classes, axes,
  channels, training set). Register through `services.training.register_model`.
- `ilastik_backend/training_set.py`: `representative_fields` (layout groups: cell line × control
  role / top dose per compound; else spread), `export_training_set` (plate → raw; run folder →
  `raw` / `step:<n>` / `ilastik` = `ilastik_stage(run)`, recomputed via
  `compare.sources._recompute` with the run's fitted states; `training_set.json`
  `FORMAT`; refuses raw-data folders), `ilastik_gui_command` (`--project=` or `--new_project=
  --workflow=` from `GUI_WORKFLOWS`; GUI workflow names unconfirmed).
- `runners.runner_state` honours `HCS_ILASTIK_RUNNER` (one run with another runner; CLI
  `--run … --runner`). CLI `--training-set`, `--stage`, `--layout-file`, `--run`, `--input`,
  `--runs`.
- GUI `gui/ilastik_panel.py` (`IlastikPanel`, dock "ilastik" tabified with Segmentation;
  injectable `choose_project/choose_folder/choose_save/confirm/launch`; ilastik's window is
  watched with a QTimer (`_check_ilastik`) → `ilastik_closed` offers registration;
  `preset_ready` → `PipelinePanel.set_pipeline`; `check_fit` on every field change).
- Done-when test `tests/test_ilastik_done_when.py` (subprocess run killed with SIGKILL on its
  process group, CLI resume; `HCS_SLOW=1` for 384 wells).

### Conventions (step g, built: ilastik as an option of Phases 21, 22, 25)

- `Module.fit_without_plate` (True for the ilastik steps): `training.validate.predict` fits such
  steps once per validation (`states`) instead of refusing them.
- `detect_spheroids` method `probability` (`probability_channel`, default first non-background
  `ilastik_` channel; `assays.spheroids.probability_spheroids`: ≥ threshold, holes filled, size
  filter, numbered by size; 2D, Z max-projected) and `confluence` method `probability`
  (`class_channel`, threshold default 0.5; needs no fit; reads `corrected`, so the class channel
  must be in the linear image). Presets `presets.spheroid_preset` (from
  `assays.spheroids.preset` with the detection swapped) and `confluence_preset`; CLI `--for`
  (`_ilastik_preset` in app.py, shared by `--preset` and `--run`).

### Conventions (step h, built)

- `api_worker.py` runs in ilastik's Python (stdlib, numpy, h5py, xarray, ilastik only; exit 3 =
  no API): job JSON (project, workflow, [input spec, output]) → project loaded once,
  `get_probabilities` / `get_probabilities_stage_2`, outputs as `exported_data` + axistags.
  `runners/python_api.py` (`PythonApiRunner`, `is_python`, cached `api_available`); settings
  runner `python_api`; `runner_state` falls back to headless with `runner_note` (not a Python, or
  no API); `make_runner` falls back per job (only probabilities, no roles).
- `selftest.check_installation(project, runners, settings)` → `InstallationCheck` (per runner
  PASS/FAIL, seconds, max difference vs headless ≤ 1e-4); GUI "Test installation…" (separate
  `test_task`, injectable `choose_project`), CLI `--test [project]`.
- `project._training_layout`: as ilastik's parser; `confirmed=True` when found that way.
- Tests: `tests/test_ilastik_api.py` generates a fake ilastik environment (`bin/python3` wrapper,
  `ilastik/experimental/api`, `python -m ilastik` = fake headless, minimal `xarray`).

### Testing plan

- CI uses a fake ilastik executable (a Python script that parses the real flag set and writes
  known-answer outputs), like the fake SAM worker. Real-ilastik and Fiji parity tests run only when
  `HCS_ILASTIK=<executable>` / `HCS_FIJI=<launcher>` are set (skipped with a note otherwise).
- The tiny bundled `.ilp` must be created in ilastik (it cannot be made without ilastik).

## Platform phases 17–28 (PROMPT_PLATFORM.md) — architecture approved and built 2026-11-19; Phases 17–28 built

Assumptions until the user says otherwise: the scheduler uses the `lab_events` JSON Schemas (a
Python package sits beside them); one PC uses the local registry before server mode.

### Status and build order

Phases 0–15 are built (verified on synthetic data only, see docs/qc_review.md); Phase 16 is
built (steps a–e, against stand-in ilastik/Fiji programs). Proposed order: **17 → 18 → 19 → 20 → 22 → 21 → 23 → 24 → 25 → 26 → 27 → 28**.
Changes against PROMPT_PLATFORM.md §1: 22 before 21 (spheroid growth curves use kinetics); 21's
ilastik detection mode, 22's ilastik confluence/tracking and 25's ilastik registration wait for
Phase 16 (b–e), while their other modes do not. 20 needs only Phases 5–13 and can be pulled
forward whenever 19 is blocked on scheduler or instrument details.

### Architecture (§2), concrete changes

1. **Database** — package `hcs_studio/db/`: SQLAlchemy 2.x typed models (`models.py`), engine and
   sessions (`engine.py`; `sqlite:///<settings>/registry.sqlite` local, `postgresql+psycopg://…`
   server), Alembic migrations shipped *inside* the package (`db/migrations/`, run with Alembic's
   API at start-up, bundled as PyInstaller data). SQLite: WAL, `foreign_keys=ON`, and a hard refusal
   to open a database on a network filesystem (UNC/mapped network drive on Windows; nfs/cifs/smb/
   fuse mounts on Linux/macOS) with a message pointing to server mode. First migration: `data_root`,
   `event` (service events) and `job`; Phase 17 adds the catalog. Postgres tests opt-in
   (`HCS_TEST_POSTGRES_URL`).
2. **Large data in files** — per-object Parquet stays in run folders; `db/parquet.py` (Phase 17)
   queries it across runs with DuckDB, run folders resolved through data roots.
3. **Data roots** — `hcs_studio/paths.py`: `DataRoot(name, windows, linux, macos)` from settings
   (mirrored in the `data_root` table); `RootedPath(root, rel)` with POSIX `rel`;
   `to_rooted(path)` = longest matching root (case-insensitive on Windows), `resolve(rooted)` for
   this OS; paths outside every root become root `"ABSOLUTE"` with a warning. Manifests gain
   rooted copies of every path (absolute paths kept for old readers).
4. **Service layer** — `hcs_studio/services/`: `Actor(id, name, kind: os|account|scheduler|system)`,
   `ServiceContext(actor, reason, client, app_version)`; `@service("layout.save",
   reason_required=False, object_type=…)` validates the reason, runs the change, and emits
   `Event(type, utc, actor, reason, object_type, object_id, before, after, client, app_version)`
   to an `EventBus` (sinks: DB `event` table, log). Retrofit: layout version save, layout applied to
   a run (`annotate`), pharmacology exclusions/inclusions and report writes, run start / resume /
   retry / cancel, table steps on a finished run, classifier / Cellpose training outputs, settings
   changes. Core functions stay pure; GUI and CLI call services (an AST test forbids calling the
   wrapped functions from `gui/` and `app.py`, like the headless-core test). CLI: `--reason`.
   Phase 26 adds a hash-chain sink without touching callers.
5. **Job queue** — `pipeline/queue.py`: `JobQueue` protocol (`submit(JobSpec) → id`, `status`,
   `events(id, since)`, `cancel`, `retry(failed_only)`, `list`); `JobSpec(kind, payload,
   priority: interactive > auto > batch, owner: Actor, resources: Resources(cpu_workers, gpu,
   ram_gb))`. `LocalJobQueue` moves the scheduling now in `gui/jobs_panel.py` into headless core,
   persists jobs in the `job` table (survives restarts; interrupted jobs detected as today), runs
   each job in its own process as now, orders by priority (no pre-emption of running jobs),
   serialises GPU jobs. The Jobs dock becomes a view of the queue. Phase 27 adds `RemoteJobQueue`.
6. **Readout registry** — `hcs_studio/readouts.py`: `ReadoutSpec(pattern, display, unit, level:
   object|field|well|fit, direction: up|down|either|none, default_normalization, description,
   module)`. Patterns cover today's generated names (`{labels}_count`, `{labels}_per_mm2`,
   `{labels}_{feature}_{stat}`, `{channel}_…`, coloc pairs, `gate_{name}_fraction`,
   `class_{name}_{cls}_fraction`, `cluster_…`, `spatial_…`, `profile_…`, pharmacology normalized
   readouts and fit parameters); `resolve(column) → Readout` fills unit/display from the matched
   feature. Retrofit Phases 5–15 (+16 placeholders). A regression-run test asserts **every numeric
   column of `per_well_summary.csv` resolves**. Consumers moved to the registry: `RunResults`
   (ordering, names, units), Results tab, `index.html`, pharmacology `default_readouts`.
   `level: fit` is an addition to §2 for EC50/AUC/synergy values.
7. **`lab_events`** — its own distribution at the repo root, `sam3/lab_events/` (`pyproject.toml`,
   `src/lab_events/schemas/v1/*.schema.json`, `validate()`, `make_event()`), depending only on
   `jsonschema`. Envelope: `schema_version "1.0"`, `event_id` (UUID), `type`, `emitted_at` (UTC
   RFC 3339), `source {system, host, version}`, `correlation_id?`, `payload`. Types: `plate.imaged`,
   `plate.dispensed`, `analysis.started|completed|failed`, `reimage.request`, `cherrypick.request`.
   Unknown major version → `UnsupportedVersion`; minors are additive. The JSON Schema files are the
   language-neutral contract if the scheduler is not Python. HCS Studio depends on it by path
   (bundled in the installer).
8. **Settings and secrets** — `hcs_studio/config.py`: one versioned `settings.json`
   (`config_version`, pydantic `AppSettings`: data_roots, database, sam3, ilastik, later
   notifications/assistant) with step-wise migrations; existing `sam3.json` / `ilastik.json` are
   imported once. `hcs_studio/secrets.py` wraps `keyring` (service `hcs-studio`); settings hold only
   references (`secret:smtp/password`); a logging filter redacts known secret values; headless
   servers without a keyring backend read `HCS_SECRET_<NAME>`; tests use an in-memory backend.
9. **Synthetic data** — package `hcs_studio/synthetic/`: one `SyntheticPlate` builder (vendor
   layout, wells, fields, channels, Z, T) with composable features that each write their truth to
   `truth.json`; the existing generators (`utils/synthetic.py`, `pharmacology/synthetic.py`,
   `scripts/make_test_plate.py`) become thin wrappers. Features arrive with their phases
   (artifacts 18, time courses 22, spheroids 21, screens with seeded hits 23, profiling phenotypes
   24, cell cycle/foci/N:C 20); plus `synthetic.registry()` for Phase 17's multi-run tests.

### Dependencies (approved 2026-11-19)

`sqlalchemy>=2.1,<2.2`, `alembic>=1.20,<2`, `jsonschema>=4.26,<5`, `keyring>=25.7,<26`,
`lab-events` (path dependency `../lab_events`, `[tool.uv.sources]`). Ask before the later ones: with their phases: `duckdb` (17; 2.0.0 is a new major — pin after testing), `watchdog`,
`httpx` (19), `statsmodels` only if a phase needs LOESS/mixed models (17's uniformity maps can use
the polynomial surfaces already in `processing/background.py`), `psycopg` (27).

### Platform conventions (built)

- **Database**: models in `db/models.py` (foundation) and `db/catalog.py` (Phase 17); import new
  model modules in `db/schema.py`. Every schema change needs an Alembic revision in
  `db/migrations/versions/` (autogenerate with `alembic_config(engine)` + `command.revision`, then
  review); `tests/test_db.py` fails when models and migrations differ. `get_engine()` migrates on
  first use; `session_scope()` commits/rolls back. `HCS_DATABASE_URL` overrides the URL (tests
  run against a registry in their `HCS_STUDIO_HOME`; `tests/conftest.py` gives every test session a
  temporary home). SQLite returns naive datetimes; compare via timestamps.
- **Services**: a state change is a function `@service("domain.action", object_type=…,
  reason_required=…)` taking `ServiceContext` first and calling `ctx.note(object_id=…,
  before=…, after=…)`; existing ones in `services/operations.py` and `services/registry.py`.
  GUI/CLI build the context with `services.context(reason)` *before* starting a background task.
  `tests/test_services.py` lists core functions the GUI/CLI must not import (`WRAPPED`): add new
  state-changing core functions there. Read-only work (previews, write=False) stays direct.
- **Job queue**: `pipeline/queue.LocalJobQueue` (headless; `poll()` drives it); submit/cancel/
  retry/resume through `services.operations.*_job`. The Jobs dock is a view (`panel.queue`).
- **Readouts**: register new readouts with `READOUTS.register(ReadoutSpec(...))` next to the
  module that produces them; placeholders `{name}` (lazy tokens), `{x:a|b}` (choices),
  `{n:int}`, `{x:any}` (free text); resolution prefers the most specific pattern, ties → well
  level. `tests/test_readouts.py` runs a full pipeline and requires every numeric per-well and
  per-object column to resolve. Store values even when unregistered (the scanner reports them).
- **Data roots**: store paths with `output.manifest.rooted_text()` / `paths.to_rooted()`; resolve
  with `paths.resolve()`. Root names ≥ 2 characters (never drive letters); `ABSOLUTE` reserved.
- **Settings/secrets**: `config.AppSettings` (`settings.json`, `config_version`, `MIGRATIONS`);
  `config.check_no_secrets` refuses plain secrets (keys containing password/token/api_key/secret/
  webhook must be `secret:<name>`); `secrets.get_secret` (env `HCS_SECRET_<NAME>` first, then
  keyring), values redacted in logs.
- **lab_events**: `sam3/lab_events` (schemas in `src/lab_events/schemas/v1/`); `validate()` /
  `make_event()`; new optional fields = minor version, breaking = new major folder.
- **Synthetic**: `hcs_studio.synthetic` (`SyntheticPlate` + features writing `truth.json`;
  `registry_runs`). Do not change the Phase 13/15 generators (regression baseline).
- **Phase 17a**: scanner `db/scanner.py` (keys: plate `barcode:` / `name:…|raw`, acquisition
  `<rooted raw>|t`, run = rooted folder; `fingerprint()` = manifest, per-well table, layout,
  pharmacology files); queries `db/queries.py` (DataFrames/dicts, shared by GUI/CLI/assistant);
  GUI `gui/experiments_panel.py` (Experiments tab; all queries in `BackgroundTask`); CLI
  `hcs-studio db …`.
- **Phase 17b (built)**: `normalization/` is headless and works on `{well: value}` +
  `WellInfo(well, role, compounds)`: `plate.normalize_plate` (methods, `positional_guard` →
  `PositionalCorrectionBlocked` for B-score), `uniformity.build_map/apply_map` (degree +
  `edge_rings`; robust refit never drops > half the wells), `batch.batch_effects`,
  `concordance.replicate_concordance`. Registry glue in `db/normalization.py` (`find_run` by
  id / folder name / barcode, `plate_values`, `normalize_run`, `concordance`, `day_effects`);
  maps stored via `services.registry.save_uniformity_map` (`uniformity_map` table). Synthetic:
  `synthetic.screening_plate` (seeded hits + positional artifact), `synthetic.write_values_run`
  (a run folder from values).
- **Phase 17c (built)**: `monitoring/` is headless: `charts.control_chart(points, rules,
  baseline_n)` (points = DataFrame `when`, `value`, `label`; `Rule.parse("2of3_2s")`),
  `charts.events_for` (empty field = applies to all), `msr` (MSR formulas from Eastwood 2006;
  **not yet verified against the NCATS AGM** — www.ncbi.nlm.nih.gov answers automated clients with
  a reCAPTCHA; ask the user to confirm before changing them), `bridging.bridge`, `report.
  write_bridging_report`. Registry glue `db/monitoring.py` (`run_metrics`, `chart`,
  `reference_potency`, `msr_history`, `bridging` pairs plates by barcode). Charts mix plate types
  unless filtered by assay or experiment.
- **Phase 18 plan**: (a) field metrics, rules, decisions, `qc/` files, CLI — built; (b) — built: QC
  before segmentation, object-level QC, compound autofluorescence/precipitation,
  exclusions propagated to measurements, IC50 fits (hollow points), plate maps (hatched) and
  reports, QC in the registry, 384-well timing; (c) QC dashboard (heat maps, gallery by
  severity, accept/override with reason via services), learned QC, `qc_report.html`.
- **Phase 18a (built)**: `qc/metrics.py` (`field_metrics(stack (C,Z,Y,X), MetricSettings)` →
  `c<n>_<metric>` per channel + content-channel metrics; PLLS = CellProfiler/centrosome
  definition; focus metrics on the image clipped at p99; `_properties` = vectorised regionprops
  area/mean/axes — keep it vectorised, regionprops loops were the bottleneck), `qc/rules.py`
  (pydantic `QcRules`, schema `hcs-studio-qc-rules/1`, `sha256()`, `default_rules()`; add new
  metrics to `CHANNEL_METRICS`/`FIELD_METRICS`), `qc/decide.py` (`apply_rules` → `QcResult`;
  `best_z_offset` from a robust plane over the plate), `qc/run.py` (`run_qc` streams fields with
  threads, refuses raw-data outputs, writes `qc/`; field numbers are the reader's `Field.index`).
  Service `services.operations.run_image_qc` (`qc.run`); `run_qc` is in `WRAPPED`. Synthetic:
  `synthetic.artifacts.Artifacts` / `qc_plate` (labels in `truth["labels"]`). Thresholds are
  synthetic-tuned: ask the user for 20–50 bad + ~50 good real fields per instrument.
- **Phase 18b (built)**: `Pipeline.qc: QcRules | None` (dropped from `to_dict` when None and
  from `pipeline_hash`); `executor.image_qc_stage` runs before `_fit_or_load` (events
  `qc_started`/`qc_finished`, manifest `image_qc`, `performance.wall_s.image_qc`, `QcStop` →
  status `qc_failed`). Propagation goes through files, not new plumbing: `output/tables.py`
  `merge_image_qc` (field exclusions into `qc_flags.excluded`) and `image_qc_wells`
  (`qc/well_qc.csv` → `qc_failed/qc_flags/qc_reasons`); pharmacology `with_image_qc` in
  `load_wells`, `_qc_failed` in `normalize` (controls) and `exclude_outliers` (reason "image QC:
  …"). Object QC = table step `qc/objects.py` (`qc_` prefix in ANNOTATION_PREFIXES; objects with
  `qc_excluded` are dropped by `object_statistics` and `TableModule.selected`, counts unchanged).
  Compound artifacts `qc/compound.py` (series = single-agent wells per compound × cell line).
  Registry `Run.image_qc` (migration 0004), `WellResult.qc_flag = "qc_failed"`. QC metric
  speed: never use large structuring elements (distance transforms instead).
- **Phase 18c (built)**: reviews in `qc/review.py` (`Override`, `qc/overrides.json` append-only,
  latest per (level, well, field) wins; `redecide` rewrites qc files from `rule_excluded` +
  reviews; `apply_override` (in `WRAPPED`) also calls `executor.refresh_run_tables` for run
  folders); service `operations.review_qc` (`qc.review`, reason required). Thumbnails +
  `emb_*` embedding in `qc/thumbs.py` (written during measurement); learned QC `qc/learned.py`
  (features = QC metrics + emb as plate robust Z; trained with normalisation "none" on
  pre-normalised features, then marked `plate_robust_z`; `LearnedQc` in rules with SHA-256).
  `qc/report.py` writes `qc_report.html` from `write_qc` (every QC write). GUI `gui/qc_panel.py`
  (QC tab; widget is `metric_combo` — `metric` shadows `QWidget.metric()`). CLI `qc-review`,
  `qc-train`.
- **Phase 19 decisions** (2026-11-24): exports from ImageXpress (MetaXpress), Operetta
  (Harmony) and Nikon NIS-Elements `.nd2` are watched; notifications by email (SMTP);
  scheduler events over both HTTP and a drop folder. Stdlib only (polling watcher, no
  watchdog — polling also works on network shares; urllib/smtplib).
- **Phase 19 (built)**: `automation/completeness.py` (`CHECKERS` per reader; ImageXpress counts
  TIFFs from the `.HTD` and checks pixel data is fully written), `automation/watcher.py`
  (`FolderWatcher.scan(now)`: candidates = sub-folders and `.nd2` files; signature = files,
  bytes, newest mtime; quiet period then completeness; `plate_imaged_event` — naive instrument
  times become local-aware; already-registered raw data is skipped after a restart),
  `automation/intake.py` (`receive`: schema validation, event log `lab_event`, key = barcode +
  acquisition ID/start + time point, plus same barcode + raw + time point within
  `duplicate_window_hours`; `send` writes `<outbox>/<µs time>_<type>_<id>.json` and/or POSTs),
  `automation/routing.py` (`Rule.matches`: every given criterion; `default`; `check_routing`),
  `automation/runner.py` (`AutoRunner.step()`: inbox, watchers, `queue.poll()`, then each
  pending/waiting/running `auto_run`: completeness → route → layout by barcode → `submit_job`
  with system actor `automation`, priority `auto` → on finish pharmacology/screen/kinetics,
  `scan_folders`, `write_report`, `summarise` (verdict + key numbers), `analysis.*` events;
  `qc_failed` → `reimage.request` + `analysis.failed`; `retry` (in `WRAPPED`), `is_night`),
  `automation/notify.py` (`Notifier`, `Notice`; secrets via `secret:` refs), `automation/status.py`
  (read-only lists). Service `services/automation.retry_auto_run` (`automation.retry`, reason).
  CLI `cli_automation.py` (`auto serve|status|events|check|retry|test-notify|send`; server mode
  = PostgreSQL → `DbJobQueue`). GUI `gui/automation_panel.py` (Automation tab, event-log viewer;
  `reason` injectable). Tests use synthetic ImageXpress exports with real job processes and an
  in-test SMTP server (`tests/test_automation_flow.py`).
- **Phase 20 (oncology assays)**: one module per step, stop after each (20a cell cycle — built;
  20b foci, 20c apoptosis, 20d translocation, 20e proliferation). Ask the user for one real control
  plate per assay plus their current results.
- **Phase 20a (built)**: `assays/base.py` `AssayModule(TableModule)` (`roles`, `prefix`,
  `preset()`, `well_roles()` from the run layout — untreated wells without drugs count as vehicle,
  `control_objects()`, `run_warnings()`, `output_folder()`); table steps get `run_folder` before
  `fit_table` and `finish_table(state, folder, per_well)` after the wells (`run_table_steps`).
  `assays/cell_cycle.py`: pure functions `g1_peak`, `fit_dna` (DJF, binned ML, 3 starts),
  `phase_of`, `gate_edu`, `otsu_log`, `robust_upper_log`; module `cell_cycle` (prefix `cc_`, in
  ANNOTATION_PREFIXES and tables `_LABEL_PREFIXES`); report; `preset()` uses a **triangle**
  threshold (Otsu biases DNA content on arrested fields). Synthetic: `synthetic/cell_cycle.py`
  (`sample_nuclei`, `write_cycle_run`, `CycleNuclei`). GUI `gui/assay_wizard.py` (Tools → Assay
  wizard…); CLI `hcs-studio assay list|preset|run`. New assays: register in
  `processing/base.load_builtin_modules`, readouts in `readouts.py`, methods `DESCRIPTIONS`,
  `gen_processing_docs.NOTES`, docs/assays.md.
- **Phase 20b (built)**: `assays/foci.py`: image step `detect_foci` (fit on vehicle fields via
  `context.sample(n, wells)`; `spot_response` 2D/3D, `find_foci` = threshold ∩ nuclei, watershed at
  local maxima, size filter; max area defaults to π·diameter²; label set `foci`, parent nuclei →
  `measure_objects` gives `foci_count` per nucleus, missing = 0) and table step `foci` (prefix
  `dd_`; pan-nuclear = nuclear mean ≥ pan_factor × vehicle median; `cc_phase` breakdown when
  cell_cycle ran first). `AssayModule.describe_state` returns the state; `preview_values(state)`
  feeds the wizard's plot (assay-generic). Synthetic `synthetic/foci.py` (`FociNuclei`, per-nucleus
  truth).
- **Phase 20c (built)**: `assays/apoptosis.py`, table step `apoptosis` (prefix `ap_`): thresholds
  `vehicle_robust` (default; a vehicle percentile falls inside the positives) / `vehicle_percentile`
  / `controls` (Otsu); `marker_object_type: cells` looks up the cell with the nucleus's
  well+field+object_id; condensed = area and `dna_feature` (p95) vs vehicle medians, fragments ≤
  0.25 × area left out of counts; layout time point 0 h → role `time_zero` (`well_roles`
  override; never vehicle); `finish_table` adds GR/`ap_response`/`ap_viable_vs_vehicle` and
  `z_prime_apoptotic`. `AssayModule.layout()`. Synthetic `synthetic/apoptosis.py`
  (`write_apoptosis_run`, `ApoptosisNuclei`). `run_table_steps` does not write tables — tests
  read `result.per_well`.
- **Phase 20d (built)**: `assays/translocation.py`: image step `cytoplasm` (`cytoplasm_labels`:
  `expand_labels` territories → ring = outer minus gap; `cells` restricts; mode `cells`) and table
  step `translocation` (prefix `tl_`; ratio = nucleus row vs cytoplasm row by well+field+
  timepoint+object_id, or cells' `<ch>_cytoplasm_mean`; `mode` agonist: vehicle unstimulated /
  inhibitor: positive controls unstimulated — matches pharmacology % inhibition, so EC50/IC50
  need no new code). Preset uses **Otsu** nuclei (triangle inflates masks into the cytoplasm)
  and a **constant** background (polynomial over-subtracts under dense cells). Synthetic
  `synthetic/translocation.py` needs plate rows/cols in the manifest for `RunResults`.
- **Phase 20e (built)**: `assays/proliferation.py`: image step `mitotic_errors` (2D,
  `find_micronuclei`: small segmented nuclei moved out + threshold components not touching the
  1-px-dilated nuclei, nearest nucleus via EDT indices; `cells` sets the nuclei's parent to the
  cells → `nuclei_count` per cell; refuses cells made from the nuclei and runs only before other
  children of the nuclei) and table step `proliferation` (prefix `pf_`; markers ki67/edu/ph3;
  secondary-only = role `unstained`; `mixture_threshold` = 1D EM, equal-density point — Otsu is
  biased by unequal class sizes). Synthetic `synthetic/proliferation.py`.
- **Phase 21 plan** (user asked for 21 before 22, 2026-11-21): (a) one spheroid per well,
  detection, size/shape/volume, QC flags, gallery, 384 batch + volume dose–response — built;
  (b) live/dead viability, necrotic core (radial shells), invasion (skeleton), organoid mode;
  (c) confocal 3D (Cellpose 3D nuclei, per-plane attenuation correction, Z-coverage warning).
  Deferred: ilastik detection (Phase 16 b), growth curves (Phase 22). Ask for one brightfield and
  one confocal spheroid plate with treatment, and an invasion example.
- **Phase 21a (built)**: `assays/spheroids.py`: `detect_spheroids` (`texture_map` = local SD of a
  DoG with odd-reflection padding; Otsu with a 4 × p10 noise floor; half-height refinement per
  component, every refined part kept, untextured parts dropped), `measure_spheroids`
  (MeasurementModule returning the **objects** table for the spheroids — never also
  `measure_objects` on them; writes a 0-row table for empty wells so they are reported),
  `spheroid` (AssayModule, prefix `sp_`, output `spheroids/` not `assays/`, crops cut from
  `previews/fNNN.png` + outline overlay). `SPHEROID_FEATURES` in readouts.py. `channel_colormap`:
  transmitted-light names → gray. Synthetic `synthetic/spheroids.py` (`BrightfieldSpheroid`,
  radial smooth meniscus — a min-of-distances meniscus has diagonal ridges that read as texture).
- **Phase 21b (built)**: in `assays/spheroids.py`: `ExtraOptions` + `spheroid_extras` (called from
  `spheroid_table` when planes/invasion are given): `radial_shells` (depth / max depth),
  `necrotic_core` (half height rim 10–30 % vs centre > 85 % depth; component at the deepest
  point), `stain_threshold` (outside-objects median + k robust SD), `invasion_measures` (opening
  via EDTs, kept protrusions ≥ min reach, skeleton end points). `detect`: interior level = median
  of the Otsu dark class of the component (not the eroded component: wrong for small objects);
  `reach_px` widens the ROI. `Spheroid.mode == "organoids"` → `organoids()` (imaged area from
  `well_summary.json` fields × field px area; OFE from `seeding_density`). Readouts:
  `SPHEROID_EXTRAS` (object + `sp_` well), `sp_org_*`. Synthetic: `core_ratio`, `strands`,
  `organoids` spec keys; `LiveDeadStains` feature.
- **Phase 21c (built)**: `processing/attenuation.py` (`attenuation`: `plane_signal` = foreground
  p90 minus plane background; `correction_factors` Theil–Sen exponential or per-plane; scales only
  signal above background; capped). `detect_3d` (method bright on stacks; brightfield stacks
  refused → project). `measure_spheroids` on stacks: 2D table from the projected labels +
  `add_3d` (volume/surface via `features._surface_area`/sphericity/Z coverage; `spheroid_extras`
  with `spacing` → `*_volume_um3`) + `add_child_counts` (label sets with `parent == spheroids`).
  `stain_threshold` = max(noise floor, half height to the in-object p99). `SpheroidParams.volume`
  `auto`; flag `incomplete_z` (excluded by default). `SyntheticPlate(z_step_um=…)`,
  `FieldContext.z_step_um`; synthetic `ConfocalSpheroid`.
- **Phase 22 plan** (2026-11-23): (a) time-course data, (b) kinetic analysis (growth fits,
  doubling time, IC50(t), GR over intervals — check the formula against Hafner et al.), (c) views
  and QC (sparklines, heat maps, focus drift, missing time points, evaporation), (d) optional
  tracking (ask before starting; scipy only). Done when: doubling time within 5 % on synthetic
  data; IC50(t) and GR on a real kinetic plate (asked the user for one per instrument with the
  treatment time).
- **Phase 22a (built)**: times: `Plate.time_points_s` / `acquired_at`, `ImageMetadata
  .time_interval_s`, `plate_times(plate)` → (times, source); readers (ImageXpress
  `acquisition-time-local`, ND2 `events()`, OME-Zarr t scale + `hcs_studio` plate attrs);
  manifest `_times`. `kinetics/drift.py` (`register_timepoints`, recognised in `process_well`
  like `stitch_well`: `shifts()` once per field, `correct(image, shift, crop)`; consecutive
  pairs, Hann window; jumps > max_shift_um not applied; `common_area` crop; refused with
  stitching in `PipelineRunner`). `kinetics/confluence.py` (`texture` = local SD of image −
  median filter; `texture_level` → level/noise/texture/`halo_sigma`; erosion by halo).
  `_Context.sample` samples (field, time point) pairs. `MeasureContext.state`; `_concat` merges
  1-row `fields` tables of several steps. `output/tables.field_means` (`FIELD_MEASUREMENT_PREFIXES`)
  for per-well means of per-field measurements. `kinetics/timecourse.py`
  (`build_time_course`, `treatment_time` (given > layout `treated_at` > `plate.dispensed`
  event), `assemble` (identity = barcode else plate name), `readout_matrix`); written by
  `aggregate_run` as `tables/time_course.*`. The run layout is written when it has wells or a
  treatment time. CLI `hcs-studio kinetics table`. Synthetic: `SyntheticPlate(timepoints,
  time_points_s, acquired_at)`, `FieldContext.timepoint/time_s`, `synthetic/kinetics.py`
  (`GrowingCells`, `exponential`); `write_imagexpress_plate(timepoints=…)`.
- **Phase 22b (built)**: `kinetics/growth.py` (`fit_growth` on log y: exponential, lag
  (exact profile over a λ grid), logistic (least_squares); AICc, AIC when n ≤ k + 1; `constant`
  when a flat line is as good; `curve_metrics`). `kinetics/response.py` (`conditions(layout)`,
  `ic50_over_time` (% of same-time vehicle median, `fit_curve`), `gr_over_time` (x₀ = each well's
  value at the last time point ≤ 0 h since treatment), `gr_intervals`; `responds()` skips flat
  series — the pharmacology 4PL takes seconds on flat data). `kinetics/analysis.py` (table step
  `kinetics`, `finish_table` builds the time course itself; `analyse`, `write_kinetics`,
  `analyse_runs`; `kn_` per-well values, readouts in readouts.py). CLI `hcs-studio kinetics
  analyse`. Synthetic `response_course` (+ `onset_h`), `drug_rate`. The time-dependent GR
  definition of Hafner et al. was not readable (NCBI/EBI blocked): interval convention unchecked.
- **Phase 22c (built)**: `kinetics/qc.py` (`kinetic_qc` → `KineticQc` with checks pass/warn/fail/
  not assessed + `well_flags` + `timeline`; `check_time_points`, `check_focus` (needs the `qc`
  step's `focus_score`), `check_edges` (condition key role|cell line|compound|conc on edge and
  inside), `check_controls`, `control_history` (registry: `kn_doubling_h` of `PlateWell` vehicle
  wells per run, given rooted folders excluded)); `analyse(course, layout, params, plate_shape,
  history)`; `kn_qc_flags`. `kinetics/report.py` (`report_data`, `write_report` →
  self-contained `kinetics/report.html`, `heatmap_figure`); linked in `output/report.report_links`.
  Time course `fields_imaged`; `_objects_measured` = 0 where imaged without objects.
- **Phase 22d (built; user said "Proceed with 22d", scipy only)**: `kinetics/tracking.py`
  (`track_objects` per field: augmented LAP (`_assign`, Jaqaman 2008) on predicted positions,
  gate = max speed × interval; divisions near the mother's last position with area guard,
  `_reappears` (track lost in the last max_gap frames → not a division) and `min_cycle_h`;
  gap closing by LAP; `track_table` (cycle from interval midpoints), `well_summary`; table step
  `tracking` (prefix `trk_` in ANNOTATION_PREFIXES and `_LABEL_PREFIXES`; times from the
  manifest; ids field × 100 000 + track; `tracking/tracks.csv`). Synthetic `simulate_lineages`,
  `MovingCells` (per-field cache; truth `cells` with id/parent/y/x).
- **Phase 23 plan** (2026-11-24): (a) scoring + hit calling — built; (b) review views (screen
  report: plate heat maps, score vs plate order, distribution, replicate scatter; hit gallery
  via Phase 14 compare; GUI tab); (c) Echo pick list (configurable columns), confirmation
  layouts pre-registered, `cherrypick.request`, screen summary. Ask for: a library plate map, a
  sample Echo pick list, current hit-calling rules; RDKit (PAINS) only if approved.
- **Phase 23a (built)**: package `screening/`: `scores.py` (`plate_scores` → pct/z/b/ssmd_single
  + Z′ via `normalization.plate.normalize_plate`; `ssmd`), `rules.py` (AST whitelist evaluator),
  `screen.py` (`ScreenConfig` pydantic, `score_screen` → wells/compounds/plates, `_artifacts` from
  `qc_failed`/`low_count`/`qc_flags`, `finalize` with `Override`s from `overrides.json`,
  `run_screen`, `write_screen`), `library.py` (`read_library_map`, `read_transfers`,
  `assay_layouts` stamp/quadrants/echo onto a control template). `db/library.py`
  (`import_library`, `library_map`). Services `operations.run_screen`/`review_hit`,
  `registry.import_library_map` (core functions in `WRAPPED`). CLI `hcs-studio screen
  score|review|library|layouts`. Synthetic `synthetic/screening.write_screen`. The positional
  guard skips single-concentration compounds (speed).
- **Phase 23b (built)**: `screening/report.py` (`report_data` column-oriented JSON, `write_report`
  → self-contained `report.html` with canvas plots, `dataset` counts for tests; `write_gallery` →
  `gallery/` via `compare.sources.load_panel` + `render.auto_contrast` (linked per strip) +
  `compose`); `rules.thresholds(rule)`; `ScreenResult.run_folders`; `run_screen(..., gallery=N)`
  writes report (always) and gallery. GUI `gui/screen_panel.py` (Screen tab: `open`, `review`
  via `services.operations.review_hit`). CLI `screen report <dir> [--gallery N]`, `screen score
  --gallery N`. Keep backslashes out of f-strings (Python 3.11 target).
- **Phase 23c (built)**: `screening/cherrypick.py` (`ConfirmationDesign` replicates/dose_response,
  `cherry_pick` → `CherryPick` picks + layouts by barcode; replicate r of hit i at free position
  i + r·n; series fill the first slot of every row first; volumes c·V/stock rounded to droplets,
  skipped when < 1 droplet or > `max_rounding_pct` off; `Backfill`; `PickListFormat` columns =
  field or fixed text; `write_pick_list`; `event_payload`; `run_cherry_pick` (in `WRAPPED`);
  `volume_check`). Service `operations.cherry_pick_hits` (`screen.cherrypick`: registers layouts
  via `LayoutStore.save`, sends events via `automation.intake.send`). `report.write_summary` →
  `summary.html` (after scoring and cherry-pick). CLI `screen cherrypick`. Echo format unverified:
  ask for the user's sample pick list.
- **Phase 24 plan** (2026-11-24; pycytominer 1.7.1 + copairs 0.6.0 approved as extra `profiling`,
  they pull duckdb 2.0.1 and statsmodels 0.15): (a) features, CellProfiler naming, CP import —
  built; (b) pycytominer pipeline aggregate → annotate → normalize (mad_robustize vs vehicle per
  plate, spherize) → feature_select → consensus + single-cell QC + reproduction test — built; (c) — built: copairs
  mAP activity/consistency, similarity, clustered heat maps, UMAP, find-similar, kNN MoA, batch
  checks, report. Ask for a Cell Painting plate with reference compounds and MoA annotations.
- **Phase 24a (built)**: `features.correlations` (+ `measure_objects.correlation`, readout
  `{channel}_{partner}_correlation`); package `cellpainting/`: `naming.py` (`rename_objects`,
  `cp_name`, `compartment`; table in the docstring; E501-exempt), `single_cell.py`
  (`well_single_cells` joins compartments on (field, timepoint, object_id), drops other
  compartments' child summaries, `Metadata_` columns from layout; `write_parquet_stream`;
  `write_single_cells`; `feature_columns`), `preset.py` (`preset`, `channel_map`, ROLES),
  `cellprofiler.py` (`read_cellprofiler` SQLite/CSV folder/Parquet, pycytominer linking).
  Test data `tests/data/cellpainting/` (pycytominer SQ00014613, BSD-3, licence copied). CLI
  `hcs-studio cellpainting preset|single-cells|import`. Synthetic `synthetic/cellpainting.py`.
- **Phase 24b (built)**: `cellpainting/profiles.py` (`ProfilingOptions`, `build_profiles`:
  `single_cell_qc` → `aggregate_wells` → optional `annotate` → `normalize_profiles` (vehicle =
  `Metadata_Role == 'negative_control'`, per plate; spherize across plates) → `select_features`
  (pycytominer operations one by one, as `feature_select` does, logging removals) →
  `consensus_profiles` (falls back to plate+well when `consensus_by` is absent); `run_profiling`
  (in `WRAPPED`) writes stages + JSON). Service `operations.cellpainting_profiles`; CLI
  `cellpainting profile`. Every stage must stay equal to pycytominer's direct call
  (`tests/test_cellpainting_profiles.py`, data `tests/data/cellpainting/BR00121431/`).
  Spherize needs full-rank vehicles (more wells than features).
- **Phase 24c (built)**: `cellpainting/evaluation.py` (`EvaluationOptions`, `evaluate`:
  `activity` (copairs `average_precision` + `mean_average_precision`; reference vehicles get a
  unique `Metadata_reference_index`, held-out vehicles = treatment `vehicle (held out)`, ≤ ⅓ of
  a plate's vehicles), `consistency` (active consensus profiles by MoA, pos_diffby compound),
  `similarity`, `similar`, `moa_suggestions` (kNN, leave one compound out), `batch_effects`
  (plate/date/batch, before = globally standardised aggregated, after = selected), `_umap`;
  `write_evaluation` → CSV/JSON/figures/`evaluation.html`; `run_evaluation` (in `WRAPPED`)).
  copairs needs NaN-free numpy features. Service `operations.cellpainting_evaluation`; CLI
  `cellpainting evaluate|similar`, `profile --evaluate`. Synthetic `profile_cells` (no images).
- **Phase 25 plan** (2026-11-24; user: build without stopping, review all phases at the end):
  (a) ground truth + metrics — built; (b) registry, pinned models, manifest models, training
  jobs — built; (c) comparison report, downstream impact vs MSR, spot checks, drift, Models tab
  — built.
  Ask for 20–40 annotated fields of the hardest cell line.
- **Phase 25a/b (built)**: package `training/`: `metrics.py` (AP = TP/(TP+FP+FN) per IoU,
  one-to-one matching — Hungarian at ≤ 0.5 because IoU exactly 0.5 can tie; merges/splits by
  majority overlap; `classifier_metrics`, `calibration`), `groundtruth.py` (versions in
  `<settings>/annotations/<name>/vNNNN/`, immutable; `content_sha256` excludes splits;
  `assign_splits` by plate/well/cell_line + `check_leakage`; `import_pairs`, `export_pairs`),
  `validate.py` (`segmentation_steps`, `predict`, `validate_segmentation`), `registry.py`
  (`model_version` + `spot_check` tables, migration 0006; store `<settings>/models/<name>/vNNNN/`
  read-only; `resolve` checks status and SHA-256; `registry:NAME@VERSION` refs in Cellpose
  `model`, classifier `model_path`, learned QC; `pipeline_models` → manifest `models`;
  `pin_pipeline`), `train.py` (`TrainingSpec`; trainers steps (grid search), cellpose,
  classifier; `run_training_job` = queue entry via `Job.extra["task"] == "train"`).
  Module `segment_model` (`segmentation/registered.py`). Services `services/training.py`
  (`groundtruth.*`, `models.*`); CLI in `cli_models.py` (`gt`, `validate-seg`, `models`), also
  scanned by the services AST test. Synthetic `synthetic/groundtruth.py` (easy/hard).
- **Phase 25c (built)**: `training/compare.py` (`compare_candidate`: baseline = production of the
  same name else base model; test split of the training set; report in
  `<model store>/vNNNN/reports/`; `update_metrics` "test"), `training/impact.py` (`swap_model`,
  `rerun` from the manifest's plate source + run layout, counts Bland–Altman, IC50 ratio vs MSR
  (registry `msr_history` else default 3); counts fall back to objects when the pipeline has no
  `qc` step), `training/qa.py` (spot checks in `spot_check`, once per run; drift = median shift
  in training IQR + spread ratio; the training summary is of the **model's own masks** on train
  fields, not ground truth — else model error reads as drift), run step `model_qa`
  (`qa/model_qa.json`). GUI `gui/models_panel.py` (Models tab; `confirm` injectable for tests).
- **Phase 26 decisions** (2026-11-24): research use (strict audit always; password and re-auth
  off by default, `security.regulated_mode` turns them on); server mode uses local accounts
  only (no SSO); argon2-cffi approved. Plan: (a) accounts, roles, audit chain — built; (b) run
  states, signatures, locking, versions, report DRAFT mark + signatures + audit section — built.
- **Phase 26a (built)**: `security/roles.py` (`PERMISSIONS` prefix → lowest role; checked in the
  `service` wrapper, denials emitted with status `denied`), `security/accounts.py` (no accounts
  → OS user is admin; else account role, unknown → viewer; argon2id; lockout; in-process
  `Session` with timeout), `security/audit.py` (`chain_new_rows` in `database_sink`, the
  `audit_head` row UPDATE serialises writers; canonical JSON incl. id; `verify_chain`;
  checkpoints `<settings>/audit/chain_head.jsonl` + `audit_head_copies`). `Actor.role`,
  `services.context()` uses the signed-in session else the OS user. The database sink is
  strict (`AuditError`). Migration 0007 (account, audit_head, run_state, signature, event
  columns with server defaults). Services `services/security.py`; CLI `cli_security.py`
  (`accounts`, `audit`); GUI `gui/security_dialogs.py` (AuditDialog, SignInDialog,
  `require_sign_in` at launch). Settings `AppSettings.security`.
- **Phase 26b (built)**: `security/signoff.py` (`status` from `run_state` else the manifest:
  completed / completed_with_failures → completed, else draft; `sign` (reviewed needs completed,
  approved needs reviewed; regulated: approver ≠ reviewer; `reauth_required` needs the password,
  checked by the service), approval locks (`run_state.locked`, files chmod read-only, report
  rewritten first), `unlock`, `new_version` (hard links for `IMMUTABLE_DIRS/FILES`, copies the
  rest; manifest `run_version`, `parent_run`)). `services.operations.guard_runs` in every
  run-changing service; `analyze_run(new_version=True)` (reason required). Services
  `run.sign|unlock|new_version`; CLI `run-state`, `sign`, `unlock`, `analyze --new-version`,
  `db backup` (`db/backup.py`, SQLite online backup). Report: `signoff_data` (from
  `signatures.json` + manifest; no registry needed), `audit_rows`, DRAFT mark. The executor
  rewrites the report after the final status when there are no run steps. GUI `SignRunDialog`
  (Tools → Sign the run in Results…).
- **Phase 27 decisions** (2026-11-24): PostgreSQL-backed job queue (`SELECT … FOR UPDATE SKIP
  LOCKED`; no Redis); Docker Compose first, Windows service documented (IT not decided); own
  canvas OME-Zarr viewer in the web portal (no vizarr/OpenSeadragon); extra `server` =
  fastapi, uvicorn, psycopg[binary]; httpx in the core (desktop connected mode). PostgreSQL 16
  is installed in the cloud container (`/usr/lib/postgresql/16/bin`) for tests.
  Start a throwaway cluster as the `postgres` user in `/var/tmp/hcs-pg` (the scratchpad path is
  not readable by it): `initdb -A trust`, `pg_ctl -o '-p 55432 -c listen_addresses=127.0.0.1'`;
  then `HCS_TEST_POSTGRES_URL=postgresql+psycopg://postgres@127.0.0.1:55432/postgres pytest …`
  (tests create and drop one database each).
- **Phase 27a (built)**: tables `queue_job`/`queue_event`/`queue_worker` (`db/server.py`,
  migration 0008); `server/queue.py` (`DbJobQueue`: JobQueue protocol + `claim` (atomic
  conditional UPDATE; Postgres `FOR UPDATE SKIP LOCKED`; order priority → fair share (owner's
  running jobs) → age; GPU jobs only to GPU workers), `heartbeat` (returns cancel requests),
  `record` (events numbered under the job row lock), `release`, `reap` (stale heartbeat →
  queued again with `resume` = run folder; `max_attempts` → interrupted); payload paths are
  data-root paths), `server/worker.py` (`Worker` runs each job in a `JobProcess`; `run_pool`
  restarts dead workers and reaps). Job processes exit when their parent dies
  (`jobs._exit_with_parent`). CLI `cli_server.py`: `worker`, `worker-pool`, `queue`.
- **Phase 27b (built)**: `server/api.py` (`create_app`; **no `from __future__ import
  annotations`** — FastAPI must resolve endpoint annotations imported inside `create_app`; DB
  session tokens `api_token` (SHA-256 only, migration 0009; expiry `server.session_hours`, idle
  timeout); every change via services with the token's account and client `api:<host>`; run
  files path-guarded; image tiles = raw `<u2`/`<f4` chunks (`X-Shape`, `X-Dtype`), 512 px,
  power-of-two levels; WebSocket `/api/jobs/{id}/ws?token=`; `job_spec_from_request` opens the
  plate on the server from a data-root path; results go to `server.results_root`).
  `server/client.py` (`ServerClient` — any httpx-like client, tests pass FastAPI `TestClient`;
  `RemoteJobQueue` = JobQueue protocol + the attributes the Jobs dock reads). Workers index
  finished runs (`register_run`). Desktop connected mode: `gui/security_dialogs.connect_server`
  (server sign-in at launch when `server.url` is set; `accounts.adopt_session` so local
  services act as the server account). `AppSettings.server`; CLI `serve`. Ruff bugbear
  `extend-immutable-calls` for fastapi markers.
- **Phase 27c (built)**: portal `server/static/portal.html` (served at `/`; vanilla JS; cookie
  session `hcs_token` (HttpOnly, SameSite=Strict, Secure behind HTTPS) so reports' relative
  links work; canvas viewer composes raw tiles per channel; tested in Chromium by
  `test_portal_in_a_browser`). `deploy/` (Dockerfile + `Dockerfile.dockerignore`, build
  context = repo root; compose: db, api, worker-gpu, worker-cpu, proxy — only the proxy
  publishes a port; nginx.conf with WebSocket upgrade; settings example). Docs `server.md`,
  `it_security.md`. Done-when test `tests/test_server_e2e.py` (uvicorn + pool).
- **Phase 28 decisions** (2026-11-24): `anthropic` SDK approved as extra `assistant`
  (`anthropic>=2.13,<3`, API version 2026-10-21); live evaluation later (the user runs it with
  their key). Messages API with HCS Studio's own tool loop (not Managed Agents, not the beta
  tool runner): tools run locally on lab data, every call is audited/confirmable. Defaults
  `claude-opus-5-5` effort medium (routine), `claude-fable-5-2` effort high (stronger; a
  Covered Model — 30-day retention, not under ZDR: docs say to switch to Opus with ZDR).
  No `thinking` parameter (adaptive by default), no sampling parameters, no forced
  `tool_choice` (400 on these models); server-side refusal fallback (`fallbacks: "default"`,
  beta `server-side-fallback-2026-07-01`) for the models in `FALLBACK_MODELS`.
- **Phase 28 (built)**: package `assistant/`: `views.py` (`VIEWS` = the only data the assistant
  sees, documented; `build_snapshot` writes a separate SQLite file per conversation; `Privacy`
  (compound codes, notes), `Pseudonyms.reveal` on screen only; `scrub` for manifests/QC JSON —
  path-like values and keys naming people/hosts/files/notes removed; add a view before a tool
  may use new data), `sql.py` (`ReadOnlyDatabase`: `mode=ro`, `query_only`, authorizer —
  CTE reads have db None, stored tables need to be views — progress-handler timeout, row cap),
  `tools.py` (`TOOL_SPECS` strict schemas, all properties required + nullable; inputs
  re-validated by `_check_types`; `ToolOutcome` content/summary/sql/ui; `NeedsConfirmation`
  for thumbnails), `conversation.py` (`Conversation.ask/resume/preview`, `PayloadLog`,
  `unsupported_numbers` grounding check, structured-output methods draft; system prompt =
  fixed instructions + `schema_text` with a cache breakpoint, top-level `cache_control`;
  append `response.content` unchanged), `usage.py` (weighted tokens, `assistant_usage`,
  migration 0010), `session.py` (`LocalAssistant`, `RemoteAssistant`), `evaluation.py`
  (`ITEMS` with ground-truth SQL by barcode, `grade`, `run_evaluation`). Services
  `services/assistant.py` (`assistant.question|tool|thumbnail|answer` = viewer;
  `settings.assistant`, `settings.assistant_key` = admin). Server `server/assistant_api.py`
  (no future import; conversations per account; `app.state.assistant.client_factory` for
  tests). GUI `gui/assistant_panel.py` (Assistant tab, `confirm` injectable;
  `AssistantSettingsDialog`); CLI `cli_assistant.py`. Tests use dict responses from fake
  clients (`client.messages.create` / `client.beta.messages.create`).

## Windows conventions (from the first Windows CI runs, 2026-11-25)

- Every text read/write names its encoding (`encoding="utf-8"`); `tests/test_encoding.py` scans
  src, tests and scripts. Windows defaults to cp1252 (µ, ′, ×, –, ⚠ break).
- Platform branches use `sys.platform == "win32"` (mypy narrows on it, not on `os.name` or
  try/except ImportError); `scripts/ci.sh` also runs `mypy --platform win32`. ctypes: set
  `restype`/`argtypes` to `wintypes.HANDLE` etc. (the default C int truncates 64-bit handles).
- Process trees: `taskkill /F /T` on Windows, `killpg` elsewhere; job processes watch a handle to
  their parent on Windows (`jobs._exit_with_parent`). Test subprocesses inherit `os.environ`
  (Windows needs `SYSTEMROOT`). Shebang-script fakes skip on Windows.
- Relative paths written to JSON, HTML or digests use `.as_posix()` (never `str(rel)`); CLI
  output is UTF-8 (`app.utf8_output`, redirected Windows output is cp1252 otherwise).
- Paths: compare with `str(Path(...))` (native separators); a path for another OS is text
  (`paths.resolve_text`); test data roots get a location for every platform.
- GUI start: an OpenGL failure of the viewer writes `<settings>/software_opengl` and restarts with
  software OpenGL (`gui/opengl.prepare_opengl`, called by `app.main` before Qt/napari/PyOpenGL
  are imported: Windows loads Qt's `opengl32sw.dll` copied as `<settings>/opengl/opengl32.dll`
  and sets `QT_OPENGL_DLL`, so Qt and PyOpenGL share one GL module — Qt's
  `AA_UseSoftwareOpenGL` alone fails, PyOpenGL keeps the system opengl32; Linux
  `LIBGL_ALWAYS_SOFTWARE=1`; `HCS_SOFTWARE_OPENGL=1|0`); the windowed bundle shows a
  modal dialog on unhandled errors, so CI smoke-tests the console executable (exit 5 = OpenGL
  failed, then a second start must pass).
- Windows CI: commit message `[build installer]` on a `claude/**` branch runs
  `hcs-studio-windows` (tests + installer; the installer is uploaded before the smoke test).

## Commands

Run from this folder (`hcs-studio/`).

- Environment: `uv venv --python 3.12 .venv` then `uv pip install --python .venv/bin/python -e ".[gui,dev,segmentation,profiling,server,assistant]"`
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
- Pharmacology: `hcs-studio pharm <run> --readout nuclei_count --normalize percent_inhibition --models bliss,hsa,loewe,zip`
- Experiment report of a run: Tools menu, the `experiment_report` run step, or `hcs-studio analyze <run> --pipeline steps.yaml --only`
- ilastik: `hcs-studio ilastik --detect [--set-ilastik <program>] [--runner headless|fiji|python_api] [--save] | --test [<ilp>] | --info <ilp> | --preset <ilp> [--class C --channels 1] --output p.yaml | --macro <ilp> [--export E --second S --axes A] --output m.ijm | --export-h5 <plate> --wells W --output <folder> | --training-set <plate|run> [--stage ilastik] --output <folder> | --run <ilp> --input <plate> --runs <folder> [--runner fiji]`; real-ilastik tests: `HCS_ILASTIK=<ilastik program or its env's python> pytest tests/test_ilastik.py`; real parity: `+ HCS_FIJI=<Fiji> HCS_ILASTIK_PIXEL_PROJECT=<ilp> pytest tests/test_ilastik_fiji.py -k real`
- Registry: `hcs-studio db info | scan <folders> [--raw] | search | potency | ic50s | plate <barcode>`; normalization: `hcs-studio db normalize <run> --readout R --method robust_z|b_score|… | uniformity <run> --name N | maps | batch | concordance <a> <b>`; assays: `hcs-studio assay list | preset cell_cycle --map dna=DAPI [--map edu=EdU] --channels … --output p.yaml | run <run> cell_cycle --map dna=DAPI`; kinetics: `hcs-studio kinetics table <run> [<run>… --output c.csv] [--treated-at ISO]` | `analyse <run> [<run>…] [--readout R --interval-h H --output dir]`; image QC: `hcs-studio qc <plate> --output <folder> [--rules r.yaml] | --write-rules r.yaml`, `hcs-studio qc-review <run> --well W [--field N] --action confirm|include|exclude --reason R`, `hcs-studio qc-train <runs…> --output m.joblib`; monitoring: `hcs-studio db chart --metric z_prime|vehicle_cv_pct|log_ec50|… --readout R [--assay A --instrument I --compound C --cell-line L --rules 1_3s,2of3_2s,7x] | msr --compound C --cell-line L | bridge --instrument-a A --instrument-b B --readout R --output <folder> | event --kind reagent_lot --on YYYY-MM-DD`
- Automatic runs: `hcs-studio auto serve | status | events [--status rejected] | check | retry <key> --reason R | test-notify | send <event.json>` (docs/automation.md)
- Lab assistant: `hcs-studio assistant ask "…" [--strong --ids-only] | preview | status | configure --enable | set-key | eval --workdir <new>` (live eval costs tokens; also `HCS_ASSISTANT_EVAL=1 pytest tests/test_assistant_eval.py`)
- Figures: `hcs-studio figure --regenerate <file>.json` / `--run <run> --per-compound`
- Synthetic 384-well plate: `python scripts/make_test_plate.py <folder> --wells 384 --sites 2`
- Windows installer (on Windows): `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Variant cuda]`; user guide as HTML: `python scripts/build_docs.py <folder>`
- All checks: `./scripts/ci.sh` (ruff check, ruff format --check, mypy --strict, pytest)
- Headless Linux needs `xvfb libegl1 libgl1-mesa-dri libxcb-cursor0`; `ci.sh` wraps pytest in `xvfb-run`. `QT_QPA_PLATFORM=offscreen` does not work (napari needs a real OpenGL context).
