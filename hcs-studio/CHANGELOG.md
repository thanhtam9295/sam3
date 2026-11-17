# Changelog

## Unreleased

### Phase 10 — hardening and packaging
- Performance record: time per step (summed over fields), reading, writing, fields per
  minute, peak memory per worker and output size in `run_manifest.json` → `performance`;
  per well in `well_summary.json` → `timing`. Profiled on a synthetic 384-well plate
  (`scripts/make_test_plate.py`, `scripts/pipelines/perf_384.yaml`); see validation.md.
- Faster granularity features (OpenCV morphology with cached footprints; identical
  openings): object measurement about 40 % faster.
- Plate summaries are computed well by well; the plate object table and per-object
  colocalization rows are never loaded at once (10⁷-object plates). Training a classifier
  from the command line reads only the labelled wells and a sample.
- Run options: do not save corrected images (`--no-images`; masks, previews and tables are
  still written) and a RAM limit for the whole run (`--max-memory-gb`, also in the run
  dialog), which limits the worker count and the deconvolution tile size.
- Error handling: free disk space is checked before a run starts; a full disk stops the run
  at once (resumable) instead of failing every well; clear out-of-memory messages; Ctrl+C
  in the command line stops after the current field and prints the resume command (exit
  code 130); the application logs to `~/.hcs_studio/logs/hcs-studio.log` and shows
  unexpected errors without closing.
- `hcs-studio selftest` checks an installation end to end; `hcs-studio gui --smoke-test`.
- Windows installer: PyInstaller bundle (`HCS Studio.exe` + `hcs-studio.exe`) and Inno
  Setup (`packaging/`), CPU and CUDA variants, built and self-tested by the
  `hcs-studio-windows` GitHub workflow, which also runs the test suite on Windows.
- User guide shipped as HTML (`scripts/build_docs.py`); new `docs/install.md` and
  `docs/troubleshooting.md`.
- Flat-field: the profile is now a fitted 2D polynomial by default (accurate up to the
  border: corner error 3.8 % → 0.2 %); Gaussian smoothing remains an option. Changes the
  pipeline hash of runs with flat-field (they cannot be resumed with this version).
- Plate summary of a 384-well plate: 132 → 46 s, 1.4 → 0.36 GB peak, identical results.
- Fixed: profiling failed on integer features (counts); synthetic test plates now shade
  the background like real optics; the frozen bundle misses no matplotlib backend or
  napari resources (found by the self-test).

### Phase 9 — phenotypic profiles
- Profiling of one or more runs: feature selection (missing, near-constant, |r| > 0.95),
  robust Z per plate relative to negative-control objects, optional plate centring, PCA
  (scree, loadings) and seeded UMAP of objects or wells.
- Per-well profiles and distance to the negative controls (object SD units), shown as
  readouts on the results plate and in the report.
- Dose-response: four-parameter logistic fits (EC50, Hill, bottom, top, R²) of any per-well
  readout per compound.
- Profiles tab: PCA/UMAP coloured by controls, compound, concentration, class or cluster;
  lasso → wells marked on the results plate; click → object crop and viewer; plate click →
  points highlighted; scree and loadings; dose-response; SVG/PNG/CSV export.
- `advanced/` folder with coordinates, loadings, well profiles, `dose_response.csv`,
  figures (SVG, PNG) and `profiling.json`; `hcs-studio profile <runs…>`.
- `index.html`: PCA/UMAP panel linked to the plate (click a point → well; drag → mark wells;
  selected well highlighted).
- umap-learn and matplotlib are now dependencies.
- Fixed: rebuilding the results plate while wells were selected raised an error.

### Phase 8 — classification, clustering, whole-well analysis
- **Stitch well** step: field positions from stage coordinates, stage orientation found
  from the overlaps, sub-pixel refinement by phase correlation and global least squares;
  whole-well object coordinates, overlap duplicates counted once, well area from the union
  of fields, `previews/mosaic.png` (Results → Whole-well mosaic).
- Table steps, which run on the object tables after all wells: **Gate objects**,
  **Classify objects**, **Phenotypic clusters**, **Spatial clusters**. Annotations go to
  `objects_annotations.parquet` and the plate table; per-well values to the summary and
  the results plate; `tables/analysis_summary.json` records parameters, classifier hash
  and accuracy.
- Object classifier: random forest or histogram gradient boosting, plate robust-Z
  normalisation, stratified cross-validation with confusion matrix, active learning
  (least certain objects), saved with its feature list.
- Phenotypic clusters: k-means, HDBSCAN, Leiden (optional), with per-cluster feature
  profiles.
- Spatial statistics: DBSCAN/HDBSCAN colonies, Ripley K/L/H and pair correlation with
  border correction and CSR envelopes, nearest neighbours, Clark–Evans, Voronoi and
  Delaunay local density.
- Classify dock: gates drawn on histograms and scatter plots, examples labelled by clicking
  objects in the viewer, train / cross-validate / suggest, apply to a finished run, colour
  objects by class, gate, cluster or colony.
- CLI: `hcs-studio analyze <run> --pipeline steps.yaml`, `hcs-studio train-classifier`.
- The report lists the analysis steps and links each well's mosaic.
- scikit-learn is now a direct dependency.
- Fixed: the main window could open taller than the screen (about 2000 px); it now opens
  at 1500 × 950 or the screen size, whichever is smaller.

### Phase 7 — results plate and report
- Results tab: plate coloured by any per-well readout or by % positive above a threshold;
  values or robust Z-scores (plate, or negative controls); colour bar; Z′ from the layout's
  controls. Opens automatically when a job finishes; File → Open results….
- Well panel: annotation, QC flags, key statistics, field previews switchable between
  enhanced, corrected and raw, with toggleable outlines; Open in viewer (image and masks);
  raw / corrected / enhanced side by side.
- `index.html` in every run folder: the same results plate in a single self-contained file
  (works offline, zipped and shared); `hcs-studio report <folder>` regenerates it.
- Run folders now hold raw, corrected and enhanced previews per field and transparent
  outline overlays per segmentation (outlines are no longer burned into the preview).
- Fixed: the focus score depended on how many cells a field held, so sparse fields could
  be flagged out of focus and excluded from counts; it now measures edge sharpness.

### Phase 6 — pipelines and batch runs
- Pipeline builder: drag-and-drop step order, preview up to a step or all steps,
  **Run on plate…**.
- Plate map selection: click, Ctrl+click, Shift+click range and rubber-band drag.
- Run dialog: current field, current well, selected wells, whole plate, or several plates;
  output folder, worker processes (auto), checksums.
- Jobs dock: a queue of jobs, each in its own process; overall progress with ETA, status of
  every well, pause, cancel, retry failed wells, resume a run folder, open folder;
  interrupted jobs detected.
- Batch engine: wells processed by a pool of worker processes; GPU steps serialised by a
  shared lock; memory budget shared between workers; SAM 3 runs with one worker.
- Resume: a well is complete when its summary is written with the run's pipeline hash;
  interrupted wells are redone; plate-level estimates are saved and reused; resuming a
  folder made with a different pipeline, plate, metadata or layout is refused.
- An unreadable image fails only its own well; plate-level estimates skip it and log a
  warning.
- CLI: `--workers N|auto`, several `--input`, `--resume <folder>`, `--retry-failed <folder>`;
  exit codes 0 / 1 (some wells failed) / 2 (could not start).
- Run manifest records the plate source, pipeline hash and every attempt (history).
- Verified on a synthetic 384-well plate: killed part-way, resumed, identical to a clean run.

### Phase 5 — measurements
- Measurement steps: produce tables, always from the linear image and the current
  segmentations; the pipeline runner collects them per field.
- Object analyzer: morphology (2D and 3D, in µm), intensity statistics and percentiles,
  radial distribution, nuclear/cytoplasmic ratio, Haralick texture, granularity, optional
  Zernike moments, child-object counts (spots per cell), neighbours.
- Colocalization: Pearson, Spearman, Li's ICQ, Manders M1/M2, tM1/tM2 with Costes
  thresholds (or manual), overlap coefficients, Costes randomization test, object-based
  colocalization, saturation and chromatic-shift warnings; per object or per field.
- Register channels: chromatic-shift correction from given shifts or a bead image.
- Field QC: focus score, saturation, area and counts; plate-relative out-of-focus,
  saturated, edge and low-count flags; counts and densities per well without excluded fields.
- Run folders: `objects.parquet` and `coloc.csv` per well; `tables/per_object_all.parquet`
  (streamed, unified schema), `per_well_summary.csv`, `per_field.csv`, `qc_flags.csv`,
  `coloc_all.csv`; per-well summaries in `well_summary.json`.
- Measurements dock: tables of the preview, cytofluorogram with Costes line and thresholds,
  colocalization map (nMDP), and export of the exact inputs plus a macro for Fiji Coloc 2.
- `docs/measurements.md`: reference of every measurement column.

### Phase 4 — segmentation
- Label sets on images: named segmentations (nuclei, cells, spots, …) with parent links.
- Cellpose 4.2 step (`cpsam_v2` default, `cpsam`, `cpdino`, `cpdino-vitb`, or a trained model
  file): one or two channels, optional diameter in µm, flow and cell-probability thresholds,
  2D, 3D or stitched planes, GPU when available, model cached per process.
- SAM 3 step (text concept) and interactive SAM 3 in the viewer (concept, positive/negative
  points, boxes), through a worker process in a separate SAM 3 environment, with a CPU
  compatibility layer and a first-run settings dialog.
- Classical fallback: threshold (Otsu, Li, triangle, or local with background removal) and
  watershed for touching objects.
- Hierarchy: cells from nuclei by expansion (IDs shared), spot detection (LoG/DoG) linked to
  cells, linking any segmentation to a parent by overlap.
- Object filters: edge, size, solidity and intensity gates; children removed with parents.
- Editing in the viewer: napari paint/erase/fill, plus merge, split by seed points and delete.
- Training pairs in Cellpose format from edited labels; Cellpose fine-tuning in the app and
  with `hcs-studio train-cellpose`.
- Run folders now hold label images (`masks/`), parent links (CSV), object counts per field,
  and previews with outlines.
- Optional `segmentation` extra (`cellpose`).
- SAM 3.1 is a video-tracking release; still images use the SAM 3 image model.
- Validated SAM 3 with its trained weights on real fluorescence images (see
  `docs/validation.md`): excellent for clicks and larger nuclei, unreliable for dense small
  nuclei; Cellpose stays the batch default. The SAM worker downloads over plain HTTPS.

### Phase 3 — deconvolution
- Theoretical PSFs from the metadata via `psfmodels` (vectorial or Gibson-Lanni; confocal
  model with pinhole in Airy units, derived from the pinhole size when known).
- Measured PSFs averaged from isolated beads (edge, saturated, crowded and off-centre beads
  rejected; sub-pixel centring), saved as OME-TIFF with the voxel size and resampled to the
  image on use.
- Deconvolution step: Richardson-Lucy with total-variation regularisation and Biggs-Andrews
  acceleration, stop on iteration count or relative change; Wiener for quick previews; 2D and
  3D; mirror padding, background estimation, non-negativity.
- CUDA through PyTorch when available, otherwise numpy on the CPU (`device`: auto/cpu/gpu).
- Automatic Y/X tiling with blended overlaps whenever an image would exceed the memory budget
  (`max_memory_gb`, default half the free RAM or 80 % of free GPU memory).
- Per-channel report (iterations, residual, runtime, tiles, PSF and its FWHM) in the preview
  and in each well's `well_summary.json`.
- Tools → PSF tool: compute or measure a PSF, view XY/XZ projections and FWHM, save it and
  link it to the Deconvolution step.
- `hcs-studio benchmark` reports deconvolution runtime and peak memory on this machine.
- Fixed: background estimation was biased low on noisy data (now smoothed before taking the
  low percentile).
- `docs/validation.md` lists what is verified and what still needs real data.
- Deferred: `pycudadecon`, CuPy backend, depth-variant PSFs.

### Phase 2 — correction and enhancement
- Processing-module interface: pydantic parameters, optional plate-level `fit`, per-field `apply`,
  and a `linear` flag that marks display-only results.
- Flat-field correction estimated from many fields (median-based), with dark offset.
- Autofluorescence: reference wells (scalar or image), AF-channel subtraction with robust
  bleed-through estimation, background subtraction (top-hat, rolling ball), linear unmixing.
- Enhancement: denoise (Gaussian, median, non-local means), contrast/gamma, CLAHE, sharpening
  (unsharp, DoG), clarity preset with a noise gate, Z projection (max, mean, extended depth of field).
- Pipelines as YAML/JSON files; executor writing a run folder with `run_manifest.json`
  (versions, parameters, fitted estimates, input checksums), `pipeline.yaml`, corrected and
  enhanced OME-TIFFs, PNG previews, per-well summaries and `logs/run.log`.
- `hcs-studio run` (unattended runs) and `hcs-studio modules` (parameter reference).
- Pipeline dock: step list, generated parameter forms, preview on the current field with
  before / after / side-by-side display and histograms, optional live preview, load and save.
- Deferred: Noise2Void and BM3D denoising, Operetta FFC profiles, 3D CLAHE, the OME-Zarr
  cache, and starting plate runs from the GUI (Phase 6; use `hcs-studio run` until then).

### Phase 1 — readers and browser
- Unified data model (plate → well → field → lazy T,C,Z,Y,X image) with optical metadata.
- Readers with auto-detection: Nikon `.nd2` (single file or folder), Operetta CLS Harmony index
  (V4 and V5+ layouts), ImageXpress `.HTD` exports (sites, wavelengths, Z-step folders, thumbnails
  ignored), OME-TIFF, OME-Zarr plates, and TIFF folders with a filename pattern.
- Data browser with favourites and vendor badges; clickable plate map (6 to 1536 wells) with hover
  preview; field gallery with thumbnails; fields open in the viewer with channel names, colours,
  micrometre scale and Z.
- Metadata panel with overrides saved to the user settings folder, never into raw data.
- Plate layout CSV import; filename-pattern dialog for unrecognised TIFF folders.
- `hcs-studio inspect <path>` prints how a folder is read, for checking against vendor software.
- Not yet validated on real instrument data. The Nikon reader is tested only against a stand-in
  for the `nd2` file object. FFC profiles and the OME-Zarr conversion cache are deferred to Phase 2.

### Phase 0 — scaffold
- Repository layout, `pyproject.toml` (core, `gui` and `dev` extras), `scripts/ci.sh`.
- Main window with data-browser, metadata and log docks around an embedded napari viewer.
- Synthetic two-channel test image shown at start-up (Help → Show test image).
- GPU auto-detection (PyTorch CUDA / Apple MPS / CuPy) with CPU fallback, shown in the status bar.
- `hcs-studio info` command; plugin registry; headless-import guarantee for the core package.
