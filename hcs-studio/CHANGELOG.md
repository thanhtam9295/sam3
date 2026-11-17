# Changelog

## Unreleased

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
