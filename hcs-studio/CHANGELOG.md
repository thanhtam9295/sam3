# Changelog

## Unreleased

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
