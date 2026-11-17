# Build Prompt — "HCS Studio": Desktop Image Analysis for High-Content Cellular Imaging

> **How to use this file**
> 1. Create an empty project folder (e.g. `D:\dev\hcs-studio` or `~/dev/hcs-studio`).
> 2. Save this file there as `PROMPT.md`.
> 3. Start Claude Code in that folder, select **Claude Fable 5.2** as the model (there is no Fable 5.1; Fable 5.2 is the current Mythos-tier release).
> 4. Send: *"Read PROMPT.md. Create CLAUDE.md from the 'Project rules' section, then execute Phase 0 and stop for my review."*
> 5. After each phase, review, run the app on one real plate, then say *"Proceed to Phase N."*
>
> Building phase by phase (instead of "build everything") is the single biggest factor in getting a working app — each phase ends with tests on your own data before the next one starts.

---

## 1. Mission

Build a **local desktop application** for analysis of high-content and high-throughput fluorescence microscopy data, comparable in scope to Arivis Pro, HALO, Aivia and Huygens. It must:

- Browse plate/experiment data on **local drives** from **Nikon (.nd2)**, **Revvity Operetta CLS (Harmony export)** and **Molecular Devices ImageXpress (MetaXpress export)**.
- Let the user run **one task or a full pipeline** on a single image, a well, a selection of wells, or a whole plate:
  - Autofluorescence subtraction
  - Deconvolution (confocal and widefield)
  - Image enhancement and post-deconvolution clarity enhancement
  - Segmentation (Cellpose 4.2 / Cellpose-SAM and SAM 3.1)
  - Object analyzer, object classifier, object counting
  - Colocalization (Pearson, Manders M1/M2 with Costes threshold)
  - Whole-well cluster analysis (spatial and phenotypic)
  - Advanced multivariate analysis (PCA, UMAP)
- Write every result to a **new, self-describing output folder** and show an **interactive clickable plate map**; clicking a well opens the enhanced image(s), segmentation overlays and per-well results.

Target users: biologists in an oncology screening group. The UI must be usable without coding; every algorithm parameter must also be scriptable for batch use.

---

## 2. Project rules (copy into CLAUDE.md)

1. **Never modify raw data.** Raw folders are read-only. All outputs go to a new run folder (Section 7).
2. **Every output carries provenance**: software version, git commit, model name + checkpoint hash, all parameters, input file paths + checksums, timestamp. Saved as `run_manifest.json`.
3. **Scientific correctness over speed.** Each algorithm module ships with unit tests against synthetic data with a known answer (e.g. Pearson of two identical images = 1.0; RL deconvolution of a blurred point source re-concentrates intensity).
4. **Lazy, chunked I/O.** Plates can be hundreds of GB. Never load a whole plate into RAM. Use dask + OME-Zarr caching.
5. **GPU optional, never required.** Every GPU path (CUDA/PyTorch/CuPy) has a CPU fallback and auto-detection.
6. **UI never blocks.** All processing runs in a worker process/queue with progress, cancel and resume.
7. **Pipelines are data.** A pipeline is a YAML/JSON file the UI can save, load and re-run on another plate.
8. **Stop and ask** before adding a heavy dependency not listed here, or when a vendor file format behaves differently than expected — show me the sample file structure first.
9. Type hints everywhere, `ruff` + `mypy` clean, `pytest` green before reporting a phase as done.
10. Keep a `CHANGELOG.md` and update `docs/` for every user-facing feature.

---

## 3. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 or 3.12 | Single conda/uv environment; pin versions in `pyproject.toml` |
| GUI shell | **PySide6 (Qt 6)** | Main window, docks, plate map, dialogs |
| nD viewer | **napari** (embedded as a Qt widget) | Multichannel, Z, T, labels layers, 3D rendering |
| Plate map & charts | Qt Graphics View for plate; **pyqtgraph** / **plotly (QtWebEngine)** for PCA/UMAP and histograms | Plots must be interactive (lasso → highlight wells/cells) |
| File readers | `nd2` (Nikon), `tifffile`, `bioio` + `bioio-nd2` / `bioio-tifffile` / `bioio-ome-tiff`, custom Operetta & ImageXpress parsers | See Section 4 |
| Arrays | numpy, dask, zarr / `ome-zarr` | Cache converted data as OME-Zarr |
| Image processing | scikit-image, scipy.ndimage, OpenCV | |
| GPU | PyTorch, CuPy (optional) | |
| Deconvolution | Own Richardson–Lucy (CuPy/PyTorch FFT) + `pycudadecon` optional; `psfmodels` for Gibson–Lanni / vectorial PSFs | |
| Denoising (optional) | `careamics` (Noise2Void / N2V2) | Self-supervised, no clean ground truth needed |
| Segmentation | **cellpose ≥ 4.2** (models `cpsam_v2`, `cpdino`, `cpdino-vitb`); **SAM 3.1** (`facebookresearch/sam3`, gated weights on Hugging Face); optional `micro-sam` | Section 5.4 |
| Features | scikit-image `regionprops_table`, `mahotas` (Haralick/Zernike) | |
| ML | scikit-learn (RandomForest, HistGradientBoosting), `umap-learn`, `hdbscan` | |
| Tables | pandas + **pyarrow/Parquet** | Per-cell tables can be 10⁷ rows |
| Reports | Jinja2 → HTML; matplotlib for static figures | |
| Packaging | PyInstaller or Briefcase → Windows installer; also runnable via `python -m hcs_studio` | |
| Tests | pytest, pytest-qt, hypothesis | |

---

## 4. Data ingestion & browser

### 4.1 Unified data model
Implement a reader abstraction that maps every vendor format to one model:

```
Experiment
 └─ Plate (barcode, plate type 96/384/1536, rows×cols)
     └─ Well (row, col, e.g. "C07")
         └─ Field / site (index, stage x/y µm)
             └─ Image5D  (T, C, Z, Y, X) lazy dask array
                 + metadata: channel names, excitation/emission, pixel size XY/Z (µm),
                   objective NA, immersion RI, magnification, confocal vs widefield,
                   pinhole size, exposure, bit depth
```

### 4.2 Vendor readers (one module each, with a sample-data test)
- **Nikon `.nd2`** — use `nd2` package; read the experiment loops (XY positions/wells, Z, T, λ), well names from the position names / JOBS metadata if present, and full optical metadata (NA, RI, pinhole, modality).
- **Operetta CLS (Harmony export)** — parse `Images/Index.idx.xml` (or `Index.xml`/`Index.ref.xml` depending on Harmony version): wells, fields, planes, channels, TIFF file names, `ImageResolutionX/Y`, objective NA/magnification, `AcquisitionType` (confocal vs non-confocal). Handle flat-field profiles in `FFC_Profile` if exported.
- **ImageXpress (MetaXpress)** — parse the `.HTD` plate descriptor + TIFF naming pattern `<Plate>_<Well>_s<site>_w<wavelength>[_z<n>][_thumb...].TIF`, and the per-TIFF MetaMorph/MDS XML tags for pixel size and wavelength. Ignore `_thumb` files.
- **Generic** — OME-TIFF, OME-Zarr, folders of TIFFs with a user-defined filename regex.

Auto-detect the format from folder contents. If detection fails, show a dialog that lets the user map the filename pattern to well/field/channel/z.

### 4.3 Data browser UI
- Left dock: **file-system tree** of local/network drives with favorites; folders recognized as plates show a plate icon + vendor badge.
- Center: **plate map** of the selected plate (correct geometry for 96/384/1536) with thumbnails on hover; click a well → field gallery → open in viewer.
- Right dock: **metadata panel** (optics, channels, dimensions) — editable overrides saved to a sidecar, never into raw files.
- Optional "plate layout" import (CSV: well, compound, concentration, treatment, control type) used later for normalization, grouping and plot coloring.

---

## 5. Processing modules

Each module is a class with: `params` (pydantic model), `run(image, metadata) -> result`, `validate()`, a Qt parameter panel auto-generated from the pydantic model, and a **live preview** on the current field in the viewer.

### 5.1 Autofluorescence (AF) subtraction
Offer three methods, selectable per channel:
1. **Reference-well subtraction** — user marks unstained/vehicle-only wells; compute per-channel AF profile (median image after flat-field, or scalar per field) and subtract, clipping at 0.
2. **Dedicated AF channel / linear unmixing** — if an AF/empty channel exists, estimate the bleed-through coefficient per channel by robust regression on background pixels and subtract `k·AF`. Support full linear spectral unmixing when a reference spectra matrix is provided.
3. **Background estimation** — rolling-ball / top-hat or morphological background with user-set radius (in µm, converted via pixel size).

Also include **flat-field and dark-frame correction** (BaSiC-style estimation from many fields of the plate, or vendor FFC profiles). Show before/after and an intensity histogram.

### 5.2 Deconvolution
- **PSF**: (a) theoretical via `psfmodels` from metadata (NA, RI immersion/sample, emission λ, pixel size, z-step, modality; for confocal use the pinhole-aware model), or (b) **measured PSF** from a bead stack (crop, average, center, normalize). PSF viewer with XY/XZ projections.
- **Algorithms**: Richardson–Lucy with **total-variation regularization** and acceleration (Biggs–Andrews); stopping by iteration count or relative-change threshold; optional **Wiener** for quick preview. 2D and 3D. GPU via CuPy/PyTorch FFT, tiled with overlap (blend seams) for large images.
- Handle edge artifacts (padding/mirroring), background offset estimation before deconvolution, and non-negativity.
- Report: iterations, final residual, runtime. Save result as float32 OME-Zarr / OME-TIFF.
- Note in docs: this is classic RL/TV deconvolution — validate against Huygens on a few bead and cell images before using it for quantitative claims.

### 5.3 Enhancement & post-deconvolution clarity
Pipeline-able steps, each with preview:
- **Denoise**: non-local means, BM3D-like (if available), or **Noise2Void** (careamics) trained per plate/channel.
- **Contrast**: percentile rescaling, gamma, **CLAHE** (2D/3D, tile size in µm).
- **Clarity / sharpening**: unsharp mask, difference-of-Gaussians local-contrast enhancement, and a "clarity" preset = mild denoise → RL-residual-aware unsharp → CLAHE (tuned to avoid ringing after deconvolution).
- **Projection**: max/mean/extended-depth-of-field for Z stacks.
- Strict separation: **enhanced images are for viewing and segmentation input only**; intensity measurements default to the AF-corrected (and optionally deconvolved) data, never to CLAHE/gamma output. The UI must make this explicit.

### 5.4 Segmentation
- **Cellpose 4.2+**: models `cpsam_v2` (default), `cpdino`, `cpdino-vitb`, plus user-trained custom models. Nucleus and cytoplasm/whole-cell modes; 2D and 3D; parameters: diameter (optional), flow threshold, cellprob threshold, min size. Batch on GPU.
- **SAM 3.1** (`facebookresearch/sam3`): interactive mode in the viewer — point/box prompts, and **text/exemplar concept prompts** ("nucleus", "mitotic cell", "spheroid") to segment all instances. Note weights are gated on Hugging Face — add a first-run dialog for the HF token and local checkpoint path. Use SAM 3.1 primarily for interactive correction, organoids/spheroids, and unusual objects; use Cellpose for high-throughput cell/nucleus batch runs.
- **Optional micro-SAM** backend (SAM fine-tuned for microscopy) if installed.
- **Classical fallback**: Otsu/local threshold + watershed for nuclei/spots.
- **Hierarchy**: nuclei → cells (cytoplasm by expansion or Cellpose cyto) → sub-cellular spots (LoG/DoG spot detector) linked by parent IDs.
- **Editing tools**: paint/erase/merge/split labels in napari; edits saved and usable as training data to fine-tune Cellpose.
- Filters: remove edge-touching objects, size/shape/intensity gates.

### 5.5 Object analyzer
Per object (and per parent cell) measure, in physical units:
- Morphology: area/volume, perimeter/surface area, equivalent diameter, eccentricity, solidity, extent, major/minor axis, orientation, form factor, Zernike moments.
- Intensity per channel: mean, median, integrated, min/max, SD, percentiles, radial distribution (center vs periphery), nuclear/cytoplasmic ratio.
- Texture per channel: Haralick (at configurable scales), granularity.
- Spots per cell: count, total/mean spot intensity, spot area.
- Neighborhood: number of neighbors within r µm, distance to nearest neighbor.
- Output: per-object Parquet table + per-well aggregates (count, mean, median, SD, % positive) as CSV/Parquet.

### 5.6 Object classifier
- **Rule-based gates**: user draws thresholds on histograms/scatter plots (e.g. "Ki67 positive if mean intensity > X").
- **Machine-learning classifier**: user clicks example objects in the viewer to label classes (e.g. mitotic / apoptotic / normal); train Random Forest or HistGradientBoosting on the feature table; show cross-validated accuracy and confusion matrix; apply to the plate; save the classifier with its feature list and normalization.
- Active-learning helper: suggest the most uncertain objects to label next.
- Results: per-object class + probability; per-well class counts and fractions on the plate map.

### 5.7 Object count
- Counts per field, per well, per class; density per mm²; optional exclusion of out-of-focus fields (focus-quality metric per field) and edge fields.

### 5.8 Colocalization
Per cell, per field, or per user ROI, for any channel pair:
- **Pearson's correlation coefficient** (with background-subtracted data and inside-mask only).
- **Manders M1/M2** using the **Costes automatic threshold** (also allow manual thresholds), and thresholded overlap coefficients.
- **Costes randomization test** (pixel-block scrambling, block size ≈ PSF size) → p-value.
- Optional: Spearman, Li's ICQ, object-based colocalization (fraction of spots in channel A overlapping/within d nm of channel B objects).
- Visuals: 2D cytofluorogram (scatter/density) with the Costes regression line and thresholds; colocalization map image.
- Warn the user if inputs are saturated, enhanced (CLAHE/gamma) instead of raw-linear, or if channels show chromatic shift (offer registration with a bead image).

### 5.9 Whole-well cluster analysis
Two complementary analyses:
1. **Spatial clustering** of objects across the stitched well: DBSCAN / HDBSCAN on centroids (eps in µm), Ripley's K/L and pair-correlation functions vs complete spatial randomness, nearest-neighbor distributions, colony/cluster size and count, Voronoi/Delaunay density maps.
2. **Phenotypic clustering**: HDBSCAN / k-means / Leiden on standardized per-cell features → cluster labels painted back onto cells in the viewer and summarized per well.
- **Well stitching**: assemble fields into a well mosaic using stage coordinates from metadata (optional fine registration by phase correlation) for whole-well visualization and spatial stats.

### 5.10 Advanced analysis — PCA / UMAP
- Inputs: per-cell or per-well feature tables from any run; optional plate layout for annotation.
- Preprocessing: feature selection (drop near-constant, highly correlated > 0.95), robust z-score per plate **relative to negative-control wells**, optional plate-effect correction.
- **PCA** (explained-variance scree, loadings table/biplot) and **UMAP** (n_neighbors, min_dist, metric configurable; seeded for reproducibility).
- Interactive plots: color by treatment/concentration/class/cluster/well; **lasso a region → highlight those cells in the viewer and those wells on the plate map**; click a point → open that cell's image crop.
- Per-well phenotypic profiles: well-level embeddings, distance-to-control score, dose-response curves of features and cluster fractions.
- Export figures (SVG/PNG) and embedding coordinates.

---

## 6. Pipelines, batch execution and performance

- **Pipeline builder**: drag-and-drop ordered list of modules (e.g. Flat-field → AF subtraction → Deconvolution → Clarity → Cellpose → Object analyzer → Classifier → Coloc → Cluster → PCA/UMAP). Each step can be toggled; "Run single step" or "Run all".
- Scope selector: current field / current well / selected wells (click/shift-click/drag on plate map) / whole plate / multiple plates.
- **Job queue** in a separate worker process with progress per well, ETA, pause/cancel, and **resume** (skip wells whose outputs and parameter hash already exist).
- Parallelism: process pool for CPU steps; single GPU worker serializing GPU steps; configurable RAM/VRAM limits; tiled processing for large fields.
- Logging to `run.log` and a log dock in the UI; failed wells listed with errors and a "retry failed" button.
- CLI mirror: `hcs-studio run --pipeline pipe.yaml --input <plate> --wells A01-H12` for unattended runs.

---

## 7. Output folder & the clickable results plate

Every run creates a **new folder** next to (never inside) the raw data, or in a user-chosen output root:

```
<output_root>/<PlateName>__<PipelineName>__<YYYYMMDD-HHMMSS>/
├── run_manifest.json          # provenance (Section 2, rule 2)
├── pipeline.yaml              # exact pipeline + parameters used
├── plate_layout.csv           # copy, if provided
├── index.html                 # standalone clickable plate report (opens in any browser)
├── wells/
│   └── C07/
│       ├── enhanced/          # AF-corrected, deconvolved, enhanced images (OME-TIFF / OME-Zarr)
│       ├── previews/          # 8-bit RGB composites + overlays (PNG/JPEG) for fast viewing
│       ├── masks/             # label images (nuclei, cells, spots)
│       ├── objects.parquet    # per-object features + class + cluster
│       ├── coloc.csv
│       └── well_summary.json
├── tables/
│   ├── per_well_summary.csv
│   ├── per_object_all.parquet
│   └── qc_flags.csv           # focus, saturation, low-count wells
├── advanced/
│   ├── pca_coordinates.parquet, pca_loadings.csv, umap_coordinates.parquet
│   └── figures/ (SVG/PNG)
└── logs/run.log
```

**Results plate view (in-app and in `index.html`)**:
- Plate grid where each well is clickable; well color = any chosen readout (cell count, % positive, mean intensity, Pearson, cluster fraction, UMAP-distance-to-control, QC flag) with a color bar and per-plate Z′/robust Z-score option.
- Click a well → panel with field thumbnails (enhanced composite), toggleable mask/class/cluster overlays, per-well statistics, and buttons "Open in viewer" / "Open folder".
- Side-by-side comparison: raw vs enhanced vs deconvolved for the same field.
- Tab for PCA/UMAP linked to the plate (selection syncs both ways).

`index.html` must be self-contained (embedded JS/CSS, relative paths to previews) so it can be zipped and shared without the app.

---

## 8. Architecture & repo layout

```
hcs-studio/
├── pyproject.toml, README.md, CLAUDE.md, CHANGELOG.md
├── src/hcs_studio/
│   ├── app.py                  # entry point
│   ├── io/                     # readers: nd2, operetta, imagexpress, ome, generic; zarr cache
│   ├── model/                  # Experiment/Plate/Well/Field dataclasses, plate layout
│   ├── processing/             # flatfield, autofluorescence, deconvolution, psf, enhance, denoise
│   ├── segmentation/           # cellpose_backend, sam3_backend, microsam_backend, classical, editing
│   ├── analysis/               # features, classifier, counting, coloc, spatial_cluster, phenotypic, pca_umap, qc
│   ├── pipeline/               # pipeline schema (pydantic), executor, job queue, cache/resume, cli
│   ├── output/                 # run folder writer, manifest, html report (jinja2 templates)
│   ├── gui/                    # main_window, browser, plate_map, viewer (napari), panels, plots, pipeline_builder
│   └── utils/                  # gpu detection, units, logging, config
├── tests/                      # unit tests per module + synthetic data generators + small real samples
├── sample_data/                # tiny anonymized samples per vendor (git-lfs or download script)
└── docs/                       # user guide per module, validation notes
```

Plugin-style registry so new modules/segmentation backends can be added without editing the core.

---

## 9. Build phases (stop after each for review)

**Phase 0 – Scaffold**: repo layout, `pyproject.toml`, environment (CUDA detection), CLAUDE.md, CI script (ruff, mypy, pytest), empty Qt window with docks, napari embedded and showing a test image.
*Done when*: app launches; `pytest` passes; GPU status shown in status bar.

**Phase 1 – Readers & browser**: data model, Nikon/Operetta/ImageXpress/OME readers, auto-detection, file tree, plate map, metadata panel, open field in napari with correct channels, scale (µm) and Z.
*Done when*: one real plate from each instrument opens correctly; metadata matches the vendor software.

**Phase 2 – Correction & enhancement**: flat-field, autofluorescence (3 methods), enhancement + clarity, live preview, before/after.

**Phase 3 – Deconvolution**: PSF generation + measured PSF import, RL-TV on GPU/CPU, tiling, bead-image validation test.
*Done when*: a bead stack shows reduced FWHM vs raw; runtime per 2048×2048×30 stack is reported.

**Phase 4 – Segmentation**: Cellpose 4.2 backends, SAM 3.1 interactive + concept prompts, classical fallback, hierarchy, editing tools, fine-tuning workflow.

**Phase 5 – Measurements**: object analyzer, counting, colocalization (Pearson, Manders + Costes, randomization), QC metrics.
*Done when*: coloc tests on synthetic images give expected values; results match a Fiji/JACoP (or Coloc 2) comparison on 3 real fields within tolerance.

**Phase 6 – Pipelines & batch**: pipeline builder, YAML save/load, job queue, resume, CLI, run-folder writer + manifest.

**Phase 7 – Results plate & report**: clickable results plate in-app, `index.html` report, readout heatmaps, side-by-side comparison.

**Phase 8 – Classifier & clustering**: rule gates, ML classifier with active learning, spatial + phenotypic whole-well clustering, well stitching.

**Phase 9 – PCA/UMAP**: normalization to controls, PCA/UMAP, linked interactive plots, dose-response, figure export.

**Phase 10 – Hardening & packaging**: performance profiling on a full 384-well plate, memory limits, error handling, user docs, Windows installer.

---

## 10. Acceptance tests on real data (I will provide)

- 1 Nikon confocal `.nd2` Z-stack (with bead PSF stack), 1 Operetta CLS plate export (confocal mode), 1 ImageXpress plate export.
- Full pipeline on a 384-well plate completes unattended, resumes after a forced kill, and produces the output folder in Section 7.
- Cell counts per well within ±5% of manual counts on 10 annotated fields.
- Pearson/Manders within ±0.02 of Fiji Coloc 2 on the same masked, background-corrected inputs.
- Clicking any well in `index.html` opens its enhanced previews offline.

---

## 11. First message to send Claude Code

> Read PROMPT.md in full. Create CLAUDE.md from Section 2 plus the stack in Section 3. Then propose any changes to the architecture you think are necessary (briefly, with reasons) and wait for my answer. After I approve, execute **Phase 0 only**, run the tests, and report what you built, how to launch it, and what you need from me (sample data, paths, GPU info) for Phase 1.
