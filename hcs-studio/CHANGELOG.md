# Changelog

## Unreleased

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
