# HCS Studio — Add-on Prompt: Phase 16
## ilastik integration (Pixel Classification, Autocontext, Object Classification, Multicut, Tracking) + Fiji ilastik4ij plugin bridge

> **How to use this file**
> 1. Save it in the project folder as `PROMPT_ILASTIK.md`, next to `PROMPT.md` and `PROMPT_ADDON.md`.
> 2. Install the **latest stable ilastik** on the analysis PC. Optionally install **Fiji** with the **ilastik update site** enabled (Help ▸ Update ▸ Manage update sites ▸ tick "ilastik" ▸ restart Fiji).
> 3. In Claude Code, select **Claude Opus 5.5** and send the message in **Section 10**.
> 4. Phase 16 needs Phase 4 (segmentation backends) and Phase 6 (pipelines and batch). It can be built any time after Phase 6.

All project rules in `PROMPT.md` Section 2 still apply: raw data is read-only, every run has provenance, tests have known answers, the UI never blocks, and pipelines are data.

---

## 1. Goal

Make ilastik a **first-class segmentation and classification backend** in HCS Studio, alongside Cellpose and SAM:

- Users **train** in the ilastik GUI, which is interactive and needs very few labels and is good on difficult stains, organoids and textured phenotypes.
- HCS Studio **applies** the trained `.ilp` project to whole plates in batch, with the same queue, resume, provenance, plate map and analysis as every other backend.
- ilastik results (probability maps, segmentations, object classes, multicut labels, tracks) flow into the existing object analyzer, classifier, counting, colocalization, clustering, PCA/UMAP and pharmacology modules.
- The same workflows can also run **through Fiji's ilastik4ij plugin**, for labs that already use Fiji macros, with identical results.

---

## 2. Background on ilastik4ij (for Claude Code)

The Fiji plugin (`ilastik4ij.jar`, from the "ilastik" update site) adds these commands under **Plugins ▸ ilastik**:

| Command | Inputs | Output |
|---|---|---|
| Import HDF5 | file, `datasetname`, `axisorder` | image |
| Export HDF5 | image, output file, dataset name, gzip level (0 for raw, 9 for segmentations) | `.h5` |
| Configure ilastik executable location | ilastik path, threads (−1 means unlimited), RAM limit (MB) | settings |
| Run Pixel Classification Prediction | project `.ilp`, raw image, type = `Probabilities` or `Segmentation` | float multichannel probabilities, or a label image of class values 1, 2, … |
| Run Autocontext Prediction | same as Pixel Classification | same |
| Run Object Classification Prediction | project, raw image, probability or segmentation image, input type | image where each object's value is its predicted class |
| Run Multicut | project, raw image, boundary probability image | label image with a unique ID per object |
| Run Tracking | project, raw image with a time axis, probability or segmentation image with matching dimensions | stack where each object's value is its lineage ID |

Key constraints:
- **Axis order** uses the letters `x y z c t` (e.g. `tzyxc`), and the number of letters must equal the number of dimensions.
- A project only works on data with the **same spatial dimensionality and the same channel count and order** as its training data.
- **HDF5 input gives the best performance** because ilastik reads it block by block. Large multipage TIFFs should be converted to HDF5 first.
- Some old ilastik versions are incompatible with the plugin: 1.3.3b2, 1.3.3rc2, 1.3.3, 1.3.3post1, 1.3.3post2 and 1.4.0b27. Use the latest stable release, or 1.3.3post3 if you must use 1.3.3x.
- Every command can be called from a Fiji macro, for example:
  `run("Run Pixel Classification Prediction", "projectfilename=[<ilp>] input=[<image>] pixelclassificationtype=[Probabilities]");`
  Example macros for each workflow are in the ilastik4ij GitHub repo under `examples/`.

Sources: https://www.ilastik.org/documentation/fiji_export/plugin and https://github.com/ilastik/ilastik4ij. **Read both pages, and the ilastik headless documentation, before writing code. Verify every command-line flag against `--help` for the installed ilastik version, and don't rely on memory.**

---

## 3. Architecture: three execution routes behind one backend

Create `segmentation/ilastik_backend/` with a single `IlastikBackend` interface and three interchangeable *runners*:

1. **`HeadlessRunner` (default and recommended).** Calls the ilastik executable directly in headless mode (`run_ilastik.bat` / `run_ilastik.sh` / `ilastik.exe --headless --project=… --export_source=… --raw_data=… --output_format=… --output_filename_format=…`, plus `--prediction_maps` / `--segmentation_image` / `--input_axes` as the workflow needs). It runs as a subprocess from the job queue, with no Java or Fiji needed. Parse stdout/stderr into the log dock and map exit codes to clear errors.
2. **`PythonApiRunner` (optional).** If ilastik is installed as a Python package in a *separate* conda environment, use its Python API (e.g. `ilastik.experimental.api`) for Pixel Classification only, run in that environment through a small worker script. Never import ilastik into the main app environment, because its pinned dependencies conflict. Check that the API exists in the installed version first, and fall back to `HeadlessRunner` if it doesn't.
3. **`FijiRunner` (ilastik4ij bridge).** Runs Fiji headless (`ImageJ-win64.exe --headless --console -macro <generated.ijm> <args>`) with a generated macro that calls the ilastik4ij commands in Section 2, or uses `pyimagej` if installed. Use this for parity with existing Fiji macros, and so users can paste their own `.ijm`.

All three runners take the same `IlastikJob` (project, workflow, inputs, axis order, output type, threads, RAM) and return the same `IlastikResult`, so the pipeline doesn't care which runner was used. A **parity test** runs the same project with `HeadlessRunner` and `FijiRunner` on a sample image and checks that the outputs are identical, or within float tolerance for probabilities.

---

## 4. Settings & environment detection

A Settings ▸ **ilastik** page with:
- The ilastik executable path, auto-detected in common install locations (`C:\Program Files\ilastik-*`, `~/ilastik-*`, `/Applications/ilastik-*.app`). Shows the detected version, and **blocks the incompatible versions** listed in Section 2 with a clear message.
- **Threads** (−1 for unlimited) and **RAM limit** in MB, passed to ilastik as `LAZYFLOW_THREADS` / `LAZYFLOW_TOTAL_RAM_MB` for headless runs and through "Configure ilastik executable location" for Fiji runs. Verify these variable names against the installed version's docs.
- The Fiji path and whether the ilastik update site is installed. Detect this by finding `ilastik4ij*.jar` in `Fiji.app/jars` or `plugins`.
- The maximum number of parallel ilastik processes, which defaults to 1 because ilastik is already multithreaded and memory-heavy.
- A "Test installation" button that runs a bundled tiny project on a tiny image through every available runner and reports pass/fail.

---

## 5. Project (.ilp) management

- An **ilastik project library** panel. Register `.ilp` files with a name, workflow type, cell line or assay, channels used and notes. The app reads the `.ilp` with `h5py`, read-only and never modified, to extract:
  - the workflow type (Pixel Classification, Autocontext, Object Classification, Multicut, Tracking),
  - the training data's **dimensionality, channel count, and axis tags**, and the label class names and colors,
  - the ilastik version that saved it.
  Store this in the project library. Ask me for sample `.ilp` files to confirm where these fields live in the HDF5 structure before relying on them.
- **Compatibility check** before any run: compare the project's dimensionality (2D/3D), channel count and order, and dtype with the selected data and processing stage (e.g. "max projection, background-removed, channels DAPI + CellMask"). Block on a mismatch and offer fixes: select or reorder channels, project Z (Phase 12), or convert dtype.
- Record the `.ilp` **SHA-256 hash** and ilastik version in `run_manifest.json`, so results are traceable to the exact trained classifier.

---

## 6. Training round-trip ("Train in ilastik")

- From the viewer or plate map, select representative fields, ideally covering every cell line, control and top-dose well, using annotation filters from Phase 11. Click **"Export training set for ilastik"**:
  - Writes ilastik-compatible **HDF5** (dataset `/data`, gzip, chunked, with correct `axistags` attributes, e.g. `zyxc` or `yxc`), at the processing stage chosen (raw, AF-corrected, deconvolved, projected, background-removed). The stage must match what the pipeline will feed ilastik later.
  - Writes a `training_set.json` listing the source wells, fields, stage and channels.
- **"Open in ilastik"** launches the ilastik GUI (non-headless) for the user to create or continue a project with those files.
- When ilastik is closed, offer to register the saved `.ilp` in the project library and run a quick preview on the current field.
- Users can also send the current image to Fiji (Export HDF5 → open in Fiji) when they want to train or check in Fiji.

---

## 7. Workflows inside HCS Studio

For each workflow: a pipeline step, a parameter panel, live single-field preview, batch run, and a loader that turns outputs into HCS Studio layers and tables.

### 7.1 Pixel Classification / Autocontext
- Output choice: **Probabilities** (float32, one channel per class) and/or **Simple Segmentation** (class values 1..N).
- Show probability maps as napari image layers (one per class, named from the project's class names) and segmentation as a labels layer.
- **Probabilities → objects** post-processing step, configurable: threshold plus hysteresis, smoothing, size filter, and seeded watershed (seeds from nuclei or distance transform), producing instance labels for the object analyzer.
- **Hybrid option**: feed an ilastik probability or foreground map as an extra input or mask to Cellpose (e.g. restrict Cellpose to ilastik "cell" foreground, or remove ilastik "debris" class areas). Each combination is a pipeline preset.
- Class-area measurements per field and well (e.g. % area "tumor" vs "stroma", "spheroid core" vs "necrotic"), available as plate readouts and in pharmacology (Phase 13).

### 7.2 Object Classification
- Inputs: raw image plus probabilities **or** segmentation. The segmentation can come from ilastik pixel classification **or from Cellpose/SAM masks** exported as HDF5, so users can classify Cellpose objects with an ilastik object classifier.
- Load the predicted-class image and map each object to its class by majority vote within each HCS Studio object ID. Add `ilastik_class` and `ilastik_class_name` columns to `objects.parquet`. If ilastik's own object feature table export is enabled, join its features by matching object IDs or centroids.
- Class counts and fractions per well go to the plate map, results report and pharmacology readouts. Apply a categorical LUT (Glasbey-style) for display, because raw class-value images look black.

### 7.3 Multicut (boundary-based segmentation)
- Inputs: raw image plus a **boundary probability map**, usually from a Pixel Classification or Autocontext project trained with a "boundary" class.
- Pipeline preset: `Pixel Classification (boundary) → Multicut → labels`. The result is a unique-ID label image used like any other segmentation, useful for dense tissue or touching cells.

### 7.4 Tracking
- Only enabled when data has a **time axis** (live-cell plates). Inputs: raw image with T, plus probabilities or segmentation with the same dimensions.
- Load the lineage-ID stack and build a tracks table (track ID, parent, frame, centroid, division events). Show it in a napari tracks layer.
- Per-well readouts: number of divisions, division time, cell-death events (if a class exists), speed and displacement. Available to pharmacology for time-resolved dose response.

---

## 8. Data handling & performance

- **HDF5 conversion is part of the pipeline.** Before calling ilastik, write the input stage as chunked, compressed HDF5 with correct axistags into a cache folder inside the run folder (`ilastik/inputs/`), unless the source is already ilastik-ready HDF5. Delete the cache at the end if the user chooses.
- **Axis-order handling in one tested utility.** It converts HCS Studio's internal `TCZYX` to the axis string ilastik expects (`tzyxc`, `zyxc`, `yxc` …), drops singleton axes consistently with how the project was trained, and converts results back. Tests cover 2D single-channel, 2D multichannel, 3D multichannel, and 2D+T.
- **Batching:** pass many files to one headless call when the installed version supports multiple `--raw_data` inputs. This avoids ilastik's startup cost for every field. Otherwise use one call per well with all fields. Tune the batch size so peak RAM stays below the configured limit.
- **Resume:** skip fields whose output exists with a matching (input hash + `.ilp` hash + parameters) key.
- Large wells: tile with overlap and stitch for pixel classification when a field exceeds the RAM budget. This isn't allowed for object classification or tracking, which need whole objects.
- Output storage: probabilities as float16/float32 HDF5 or OME-Zarr (user choice), segmentations as uint16/uint32 labels.

Run-folder additions:
```
<run>/ilastik/
├── projects_used.json         # name, path, sha256, workflow, ilastik version, runner
├── inputs/                    # cached HDF5 inputs (optional, deletable)
└── logs/
<run>/wells/<well>/ilastik/
├── probabilities.h5           # pixel classification / autocontext
├── segmentation.h5            # simple segmentation / multicut labels
├── object_classes.h5          # object classification
├── tracks.h5 + tracks.csv     # tracking
└── class_summary.json
```

---

## 9. Pipelines, CLI, UI, tests

- **Pipeline builder**: new steps "ilastik Pixel Classification", "ilastik Autocontext", "ilastik Object Classification", "ilastik Multicut", "ilastik Tracking", "Probabilities → Objects", and "Export HDF5 for ilastik/Fiji". Presets:
  - `Projection → Background removal → ilastik Pixel Classification → Probabilities → Objects → Object analyzer`
  - `Cellpose nuclei → ilastik Object Classification (phenotype) → Pharmacology`
  - `ilastik boundary PC → Multicut → Object analyzer`
- **CLI**: `hcs-studio ilastik --project model.ilp --workflow pixel --output probabilities --input <plate> --wells A01-P24 --stage projected_bgsub --runner headless`
- **Fiji macro export**: "Export as Fiji macro" generates an `.ijm` that reproduces the ilastik step with ilastik4ij commands for a given folder of HDF5 inputs, so collaborators can run it in Fiji.
- **UI**: an ilastik section in the segmentation dock showing project selection with a compatibility badge, runner, output type, preview, and a "Train in ilastik" button; ilastik class layers in the viewer; a class-fraction heat map on the plate.
- **Tests**:
  - axis-order and HDF5 round-trip unit tests,
  - `.ilp` metadata-reader tests on sample projects,
  - a bundled **tiny 2D pixel classification project** and image for an end-to-end headless test (skipped automatically if ilastik isn't installed, with a clear note),
  - a headless vs Fiji **parity test** (skipped if Fiji or ilastik4ij is missing),
  - an incompatible-version detection test,
  - a resume test (kill mid-plate, rerun, only remaining wells processed).

*Done when*:
- A 384-well projected plate runs through ilastik Pixel Classification → objects → object analyzer unattended, resumes after a forced kill, and produces the run-folder structure above.
- Results from `HeadlessRunner` and `FijiRunner` agree on the sample.
- Object Classification on Cellpose masks produces per-well class fractions on the plate map and in Phase 13 dose-response curves.
- `run_manifest.json` records the `.ilp` hash, ilastik version and runner.

---

## 10. Message to send Claude Code

> Read PROMPT.md, PROMPT_ADDON.md and PROMPT_ILASTIK.md. Then read https://www.ilastik.org/documentation/fiji_export/plugin, https://github.com/ilastik/ilastik4ij (including the `examples/` macros) and the ilastik headless-mode documentation. Run `--help` on the installed ilastik to confirm the exact headless flags and environment variables for this version.
> 1. Tell me which ilastik version and Fiji/ilastik4ij setup you detect on this machine, and list any differences between this prompt and the real flags or behaviour.
> 2. Update CLAUDE.md with the ilastik backend architecture (Section 3) and propose any changes, with reasons. Wait for my approval.
> 3. After approval, build **Phase 16** in this order, stopping for review after each step: (a) settings, detection and HDF5/axis utilities; (b) HeadlessRunner plus Pixel Classification/Autocontext with Probabilities → Objects; (c) Object Classification, including on Cellpose masks; (d) Multicut and Tracking; (e) FijiRunner, macro export and the parity test.
> Ask me for sample `.ilp` projects (one per workflow I use) and matching sample images before step (b).
