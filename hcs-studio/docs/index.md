# HCS Studio user guide

## Start

`python -m hcs_studio` opens the application. The status bar shows the detected GPU, or
"CPU only" with the reason. Help → Show test image displays a synthetic image to confirm
that the viewer works.

## Open a plate

Use the **Data browser** (left), File → Open plate folder, or File → Open image file.
Folders recognised as plates are marked `▦ name [Vendor]`. Double-click one to open it.
Selected folders can be kept under **Favourites**.

| Source | What to select |
|---|---|
| Nikon | a `.nd2` file, or a folder of `.nd2` files |
| Operetta CLS | the export folder (the one containing `Images/Index.idx.xml`) |
| ImageXpress | the folder containing the `.HTD` file |
| OME-TIFF / OME-Zarr | the `.ome.tif` file or the plate `.zarr` folder |
| Other TIFF folders | the folder; you are asked for a filename pattern |

For Nikon data, well names are read from the XY position names, or from the file names
(`WellB02_…`). If neither contains well names, each file is shown as one well in file-name
order and a warning says so.

## Plate map and fields

The **Plate** tab shows the plate with its real geometry. Wells that contain images are
filled. Hover over a well for a preview; click it to list its fields; click a field to
open it in the **Viewer** tab with channel names, colours, a micrometre scale bar and a
Z slider. Previews are contrast-stretched for orientation only.

Warnings under the plate title list anything the reader had to assume.

## Metadata

The **Metadata** dock shows what was read from the files. Pixel size, Z step, NA,
magnification, immersion RI, pinhole, modality and channel names can be corrected and
saved with **Save overrides**. Overrides are stored in `~/.hcs_studio/overrides/`
(or `HCS_STUDIO_HOME`), never in the raw data folder.

## Plate layout

File → Import plate layout reads a CSV with a `well` column and optional `compound`,
`concentration`, `treatment` and `control_type` (`negative`, `positive`, `unstained`,
`blank`). Annotations appear in the well tooltips; control wells are coloured.

## Checking a reader against the vendor software

`hcs-studio inspect <path>` prints the detected format, plate geometry, dimensions,
optics, channels and the first wells. Compare these with the values shown by
NIS-Elements, Harmony or MetaXpress.

## Correction and enhancement

The **Pipeline** dock (tab next to Metadata) holds an ordered list of steps. Choose a
module, click **Add step**, and edit its parameters below the list. Untick a step to
skip it. **Save…** and **Load…** store the pipeline as a YAML file that can be re-run
on another plate. See [processing.md](processing.md) for every module.

**Preview on current field** runs the steps up to the selected one on the field shown
in the viewer. **Show** switches between After, Before and Side by side, and the
histograms compare one channel before and after. Tick **Live** to re-run the preview
whenever a parameter changes. Previews estimate plate-level corrections from at most
24 fields and show the first timepoint only.

### Display-only results

Steps marked **◐ display only** (contrast, CLAHE, sharpen, clarity) change intensities
non-linearly. Their output is for viewing and as segmentation input. Each run writes
two images per field: `*_corrected.ome.tif` (linear data from before the first ◐ step:
use this for measurements) and `*_enhanced.ome.tif` (display only).

### Running a pipeline on a plate

Until the job queue arrives in Phase 6, plate runs are started from the command line:

```bash
hcs-studio run --pipeline pipe.yaml --input <plate folder> --wells A01-H12
```

Options: `--output <folder>` (default: next to the raw data, never inside it),
`--layout layout.csv`, `--no-checksums` (skip SHA-256 of the input files).
The run folder is named `<Plate>__<Pipeline>__<date-time>` and contains
`run_manifest.json`, `pipeline.yaml`, `logs/run.log` and, per well,
`enhanced/`, `previews/` and `well_summary.json`. A well that fails is recorded in
the manifest and the run continues.

## Limitations in this phase

- Readers have been tested on synthetic plates only, not yet on real instrument exports.
- RGB `.nd2` files are not supported.
- ImageXpress does not record confocal vs widefield; set the modality manually.
- ImageXpress field positions are nominal (site grid), not measured stage positions.
- Operetta flat-field profiles are not read yet.
- The correction and enhancement modules are verified on synthetic images only.
- Noise2Void and BM3D denoising are not included yet.
- Every field is processed whole in memory; tiling for very large fields comes later.
