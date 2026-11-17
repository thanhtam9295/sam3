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

## Deconvolution

Add a **Deconvolution** step to the pipeline. By default it computes a theoretical
PSF for each channel from the metadata (objective NA, immersion and sample refractive
index, emission wavelength, pixel size, Z step, confocal or widefield). It stops with
a message naming anything that is missing; fill it in on the Metadata panel. Choose
the iteration count on a representative field and keep it for the whole plate.

To use a measured PSF, open the bead stack (or the field with beads) in the viewer,
then **Tools → PSF tool…**, choose *Measured (bead stack)* and **Compute PSF**. The
dialog shows XY and XZ projections and the FWHM. **Save PSF…** writes an OME-TIFF and
sets it as the PSF of the Deconvolution step. The same dialog shows the theoretical
PSF for comparison.

The preview reports, per channel, the iterations, the residual (how far the
re-blurred result is from the data), the runtime and the number of tiles. A GPU is
used automatically when PyTorch can see one. Images too large for the memory budget
are split into overlapping tiles automatically; set `max_memory_gb` to change the
budget. To see how long deconvolution takes on your machine:

```bash
hcs-studio benchmark --shape 30 2048 2048 --iterations 40
```

Deconvolution output stays linear (usable for measurements). It is classic RL/TV
deconvolution: validate it against Huygens on bead and cell images before using it
for quantitative claims. See [validation.md](validation.md).

## Segmentation

Segmentation steps go in the same pipeline as the corrections. They read the image as it
is at that point, so contrast or clarity steps placed before them improve detection,
while measurements (Phase 5) still use the linear data. Each step creates a named
**label set** (`nuclei`, `cells`, `spots`, …):

| Step | Use it for |
|---|---|
| Segmentation: Cellpose | nuclei and whole cells, high throughput (default `cpsam_v2`) |
| Segmentation: threshold + watershed | well-separated nuclei; fast fallback without a model |
| Segmentation: SAM 3 (text concept) | organoids, spheroids, unusual objects, by describing them |
| Cells from nuclei (expansion) | cells as nuclei grown outward; cell *n* contains nucleus *n* |
| Spot detection (LoG / DoG) | sub-cellular spots, each linked to its cell |
| Filter objects | remove edge objects, wrong sizes, shapes or intensities |

A typical hierarchy is *Cellpose (nuclei) → Cells from nuclei, or Cellpose on the
cytoplasm channel with `parent: nuclei` → Spot detection (`parent: cells`) → Filter*.
Every object records its parent ID.

After **Preview**, each label set appears in the viewer as an editable labels layer.

### Editing and SAM 3 in the viewer

The **Segmentation** dock (tab next to Pipeline) works on the selected labels layer:

- **Paint, erase, fill**: use napari's own tools in the viewer's layer controls.
- **Merge / Delete**: type object IDs (hover over an object to see its ID).
- **Split**: *Add seed layer*, click two or more points inside the object, enter its
  ID, *Split*.
- **SAM 3**: type a concept (e.g. "mitotic cell") and *Segment all instances*, or
  *Add prompt layers*, click positive and negative points or draw boxes, then
  *Segment from points/boxes*. SAM works on the displayed Z plane.

### Training data and fine-tuning Cellpose

Correct a segmentation, then **Save as training pair**: the image and its masks are
written to the training folder as `*_img.tif` / `*_masks.tif` (Cellpose's format).
With enough pairs, **Fine-tune Cellpose** trains a new model from the chosen base model.
The result is set as the model of the pipeline's Cellpose step. From the command line:

```bash
hcs-studio train-cellpose --data <training folder> --base cpsam_v2 --epochs 100
```

Fine-tuning needs a GPU in practice. Validate a fine-tuned model on fields that were not
used for training.

### Setting up SAM 3

SAM 3 needs its own Python environment because it requires numpy < 2:

```bash
uv venv --python 3.12 .venv-sam3
```

```bash
uv pip install --python .venv-sam3/bin/python torch torchvision einops pycocotools psutil "setuptools<81" -e <path to the sam3 repository>
```

(`einops`, `pycocotools`, `psutil` and `setuptools` are imported by SAM 3 but not declared
by it.) The weights are gated: request access to **facebook/sam3** on Hugging Face and
wait for approval, then log in with `huggingface-cli login` in that environment, or
download `sam3.pt` yourself. In **Tools → SAM 3 settings**, choose that environment's
Python (and the checkpoint, if downloaded) and click *Check environment*.

SAM 3.1 is a video-tracking release; still images use the SAM 3 image model. A GPU is
strongly recommended. On the CPU, encoding an image takes about 5-10 s and each prompt
about 1 s.

**Choosing between Cellpose and SAM 3.** In our tests SAM 3's concept prompt found every
nucleus in a confocal field of large nuclei, but almost none in a dense field of 300+
small nuclei, where Cellpose found them all. Use Cellpose for batch segmentation of
nuclei and cells; use SAM 3 to correct objects with a click, and for spheroids,
organoids and unusual objects. Concept prompts depend strongly on wording ("cell
nucleus" worked where "nucleus" found nothing), so try a few phrasings. SAM 3 does not
tell phenotypes apart: "mitotic cell" returns ordinary nuclei too. See
[validation.md](validation.md).

### Output

Each run writes, per field, `masks/<field>_<label set>.ome.tif` (label images),
`masks/<field>_<label set>_parents.csv` (object → parent links), and the object counts in
`well_summary.json`. The PNG previews show outlines: nuclei yellow, cells magenta, spots
cyan.

## Limitations in this phase

- Readers have been tested on synthetic plates only, not yet on real instrument exports.
- RGB `.nd2` files are not supported.
- ImageXpress does not record confocal vs widefield; set the modality manually.
- ImageXpress field positions are nominal (site grid), not measured stage positions.
- Operetta flat-field profiles are not read yet.
- The correction, enhancement and segmentation modules are verified on synthetic images only.
- SAM 3 is tested with random weights only (its gated weights were not available); the
  real model's segmentation quality on your images is unverified.
- SAM 3 segments 2D planes; 3D SAM 3 segmentation is not available.
- Noise2Void and BM3D denoising are not included yet.
- Apart from deconvolution, every field is processed whole in memory.
