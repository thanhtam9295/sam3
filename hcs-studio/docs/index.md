# HCS Studio user guide

## Start

Install HCS Studio with the Windows installer or from Python (see [install.md](install.md)),
then start **HCS Studio** from the Start menu, or `python -m hcs_studio`. The status bar
shows the detected GPU, or "CPU only" with the reason. Help → Show test image displays a
synthetic image to confirm that the viewer works; `hcs-studio selftest` checks the whole
installation in about a minute. If something goes wrong, see
[troubleshooting.md](troubleshooting.md). To validate HCS Studio on your own instruments
(counts, colocalization, deconvolution, unattended plates), follow
[acceptance.md](acceptance.md).

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

## Plate layout (Layout tab)

The **Layout** tab annotates every well: cell line, treatments (compounds with
concentrations, units and vehicles; combinations), role (vehicle / positive control,
unstained, blank, empty, excluded), passage, cells per well, time point and custom fields,
with dilution-series and combination-matrix helpers, validation, versions, templates and
CSV/XLSX import and export. The full guide: [annotation.md](annotation.md).

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

### Z projection and background removal

`scripts/pipelines/projection_background.yaml` is the recommended order: flat-field →
(autofluorescence) → (deconvolution) → **projection** → **background removal**. Load it
in the Pipeline dock (Load…) and untick what you do not need; steps can be reordered.

**Projection** (`projection`) works on every field, channel and well of a run:

| Option | Choices |
|---|---|
| method | **max** (default), mean, sum, min, std, edf (extended depth of field: `variance` = sharpest plane per pixel, `pyramid` = multi-scale fusion, smoother at boundaries) |
| planes | **all**; range (first/last plane, counted from 1); auto (the `n_planes` planes around the best focus by normalized variance or Brenner; per channel, or chosen by one `focus_channel` for all) |

The planes used for each field and channel (and the focus score of every plane, for
auto) are stored in `well_summary.json` → `fields[].reports`.

**Background removal** (`background`), one method per step (add several steps with
different `channels` to use different methods per channel):

| Method | What it removes | Use |
|---|---|---|
| tophat | morphological opening with a disk of `radius_um` | fast default |
| rolling_ball / paraboloid | a ball / paraboloid rolled under the image (ImageJ-style, shrunk for large radii) | smooth backgrounds |
| gaussian / median | a large blur (bright objects ignored), subtracted or divided | uneven, slowly varying background |
| polynomial / spline | a smooth surface fitted to background pixels only (objects left out iteratively), subtracted or divided | illumination gradients; divide for multiplicative shading |
| constant | a percentile of each field, or the mean of blank (no-cell) wells of the plate layout | camera offset, medium autofluorescence |
| basic | flat-field and dark-field estimated from many (projected) fields across the plate (BaSiC-style: I = S·B + D) | shading not corrected before projection |

Morphological methods (tophat, rolling ball, paraboloid) are estimated on a 3 × 3 mean of
the image so that noise does not leave a positive floor. Negative values are clipped to
0; the output stays float and linear, so intensities are measured on background-removed
data, never on contrast-stretched images.

For every field and channel, the report records the change of the mean intensity inside
objects and in the background, and the typical object diameter. If the radius is less
than 2× the typical object diameter (estimated from the image, or `object_diameter_um`),
a warning appears in the preview, the log and `well_summary.json` → `warnings`: such a
radius removes signal.

**Checking the result.** Preview the step, choose **Side by side**, then click **Line
profile…** and draw a line in the viewer: the Pipeline dock plots the intensity along it
before and after (raw data as maximum over Z), for the channel chosen for the histograms.

**Outputs.** Projected fields are written as 2D OME-TIFFs (`*_corrected.ome.tif`) with
8-bit preview PNGs. In the run dialog (or `--image-dtype original`), *Saved images* can
be the camera's bit depth instead of 32-bit float: values are rounded and clipped (the
number of clipped pixels is recorded and logged); measurements always use the float
values. `stitch_well` with `save_mosaic: true` also writes each well as one linear
image (`wells/<W>/mosaic.ome.tif`, all channels, overlaps averaged, downsampled by
`mosaic_downsample`).

### Display-only results

Steps marked **◐ display only** (contrast, CLAHE, sharpen, clarity) change intensities
non-linearly. Their output is for viewing and as segmentation input. Each run writes
two images per field: `*_corrected.ome.tif` (linear data from before the first ◐ step:
use this for measurements) and `*_enhanced.ome.tif` (display only).

### Running a pipeline on wells and plates

Build the pipeline in the **Pipeline** dock: add steps, drag them into order, untick a
step to skip it. **Preview up to selected step** runs the steps up to the selected one
on the field in the viewer; **Preview all steps** runs them all. **Save…** stores the
pipeline as a YAML file that can be loaded again or used from the command line.

**Run on plate…** opens the run dialog. Choose the scope:

| Scope | Runs on |
|---|---|
| Current field | the field shown in the viewer |
| Current well | every field of that well |
| Selected wells | the wells selected on the plate map |
| Whole plate | every well |
| Whole plate, and other plates | this plate plus the plate folders you add (one job each) |

Select wells on the plate map by clicking, **Ctrl+click** (add/remove one well),
**Shift+click** (a rectangle from the last clicked well) or by dragging a rectangle.

The run then waits in the **Jobs** dock (tab next to Log). Jobs run one after another,
each in its own process, so the application stays responsive. Within a job, wells are
processed by several worker processes (*auto* picks the number from the CPU cores and
memory). Steps that use the GPU run one at a time across workers; a pipeline with SAM 3
uses one worker, since each worker would load its own copy of the model.

The Jobs dock shows the overall progress with an estimate of the time left, and the
status of every well. **Pause** stops starting new wells; **Cancel** stops after the
wells in progress. **Retry failed wells** reruns only the wells that failed, in the same
run folder. **Resume a run folder…** continues a run that was cancelled, interrupted or
cut short by a crash or power cut. Completed wells are skipped, and the plate-level
estimates (flat-field and similar) are reused, so the result is identical to an
uninterrupted run. A well is only complete once all its fields are written; a well
interrupted part-way is done again.

Resuming refuses a run folder that was made with a different pipeline, plate,
metadata or plate layout: start a new run instead.

### From the command line

The same runs can be started without the application, for example overnight:

```bash
hcs-studio run --pipeline pipe.yaml --input <plate folder> --wells A01-H12 --workers auto
```

| Option | Meaning |
|---|---|
| `--input` | plate folder or file; repeat it to run several plates |
| `--wells` | `all` (default), `A01-H12`, `B02,B03`, … |
| `--workers` | number of worker processes, or `auto` (default 1) |
| `--output` | output root (default: next to the raw data, never inside it) |
| `--layout` | plate layout (CSV, XLSX, plate-map XLSX or JSON) |
| `--no-checksums` | record input file size and time instead of SHA-256 |
| `--no-images` | do not save corrected images; masks, previews and all tables are still written (run folders shrink to about a quarter) |
| `--image-dtype` | `float32` (default) or `original`: save corrected images in the camera's bit depth (rounded, clipped; half the size) |
| `--max-memory-gb` | RAM the whole run may use; limits the number of workers and the deconvolution tile size |
| `--resume <run folder>` | continue an interrupted run (plate and pipeline are read from the folder) |
| `--retry-failed <run folder>` | rerun only the wells that failed |

The exit code is 0 when every well succeeded, 1 when some wells failed, 2 when the run
could not start, and 130 when it was stopped with Ctrl+C (it stops after the current
field; resume it later with `--resume`).

Before it starts, a run checks that the output disk has room for its outputs and refuses
otherwise. If the disk fills up during a run anyway, the run stops at once (instead of
failing every remaining well) and can be resumed once space is freed. Each run records
where its time went (`performance` in `run_manifest.json`: time per step summed over all
fields, reading, writing, fields per minute, peak memory per worker, output size) and
each well its own timings in `well_summary.json`. Each run folder is named `<Plate>__<Pipeline>__<date-time>` and
contains `run_manifest.json` (with the history of every attempt), `pipeline.yaml`,
`logs/run.log`, `wells/<well>/…` and `tables/`.

## Results

When a job finishes, the **Results** tab opens with the run's plate. Open any earlier run
with **File → Open results…**.

- **Readout**: any per-well value (counts, density per mm², feature means and medians,
  colocalization medians, QC counts). Type in the filter box to narrow the list.
- **% positive**: choose an object type, a feature and a threshold, then
  **Show % positive** (e.g. % nuclei with `ki67_mean` > 800). Fields excluded by QC are
  left out.
- **Scale**: the values, or robust Z-scores, (x − median) / (1.4826 × MAD), either over
  the whole plate or over the negative-control wells of the plate layout.
- **Z′** is shown when the layout marks at least two positive and two negative control
  wells: ≥ 0.5 excellent, 0–0.5 acceptable, < 0 no separation.

Click a well to see its annotation, QC flags, key statistics and fields. **Show** switches
the previews between *Enhanced*, *Corrected* (linear, after flat-field, autofluorescence
and deconvolution) and *Raw*; the tick boxes show or hide the outlines of each
segmentation. **Open in viewer** opens the field's image and masks in the viewer;
**Compare raw / corrected / enhanced** places the three side by side (the raw image is
read again from the plate folder, so it must still be available).

### Shareable report

Each run folder contains `index.html`: the same results plate, readouts, robust Z,
Z′ and well panels, in one file that opens in any browser. It needs no network
connection or software. Zip the run folder (or at least `index.html` and `wells/`) to
share it; clicking a well shows its previews with the stage and outline switches.
Arrow keys move between wells, and `index.html#C07` opens well C07 directly.

**Colour by** switches the plate between a readout and the annotation (compound,
concentration, cell line, role). When pharmacology has run, its normalized readouts
(e.g. `pharmacology · nuclei_count · percent inhibition`) are readouts too, and a
**Pharmacology** section shows each dose–response curve (click a point to open its well;
hollow points were excluded) and each combination's response and synergy matrices (click
a cell to outline and open its replicate wells). **Reports and figures** links the
pharmacology and experiment reports, the per-compound figures and the montages.
Regenerate it with:

```bash
hcs-studio report <run folder>
```

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

## Measurements, colocalization and QC

Measurement steps go after the segmentation steps. They do not change the image; they
produce tables. They always measure the linear image, whatever display-only steps come
before them, using the segmentations made so far.

| Step | Produces |
|---|---|
| Measure: objects | `objects.parquet` per well: shape, intensity, texture, spots per cell, neighbours ([columns](measurements.md)) |
| Measure: colocalization | `coloc.csv` per well: Pearson, Manders M1/M2 and tM1/tM2 with Costes thresholds, ICQ, optional randomization p-value |
| Measure: field QC and counts | focus and saturation per field; flags and counts per well |
| Register channels | (correction) removes chromatic shift before measuring |

A typical end of pipeline: *Cellpose (nuclei) → Cells from nuclei → Spot detection →
Filter objects → Measure: objects → Measure: colocalization → Measure: field QC*.

After the run, `tables/` holds `per_object_all.parquet` (every object of the plate),
`per_well_summary.csv` (counts, densities per mm², feature means/medians/SDs,
colocalization medians), `per_field.csv`, `qc_flags.csv` and `coloc_all.csv`. Parquet
files open in Python (pandas), R (arrow), KNIME, Spotfire or Excel (via Power Query).

### QC and counting

Fields whose focus score is below half the plate median are flagged *out of focus*;
fields with more than 0.1 % saturated pixels are flagged *saturated*; wells with fewer
than 30 % of the plate's median count are flagged *low count*. Out-of-focus fields are
left out of the counts by default; edge fields (outermost row and column of a well's
grid of 9 or more fields) can be left out too. All thresholds are parameters of the QC
step.

### Colocalization in the viewer

The **Measurements** dock shows the tables of the last preview and analyses
colocalization on the field in the viewer: choose two channels and a region (whole
field, all cells, or one cell by its ID), then **Cytofluorogram** to see the 2D
intensity histogram with the Costes regression line and thresholds.
**Colocalization map** adds an nMDP image (positive where both channels are above their
mean). The analysis uses the linear, background-subtracted data, never the enhanced
display.

Warnings appear when pixels are saturated in the raw data or when the channels look
shifted (chromatic shift). Image multicolour beads and add **Register channels** with
that bead image to correct the shift.

### Comparing with Fiji Coloc 2

**Export for Fiji…** writes the exact inputs of the current region (background-subtracted
channel A and B and the mask), HCS Studio's results, and a macro `coloc2_compare.ijm`.
Run the macro in Fiji and compare Coloc 2's values with `hcs_studio_results.json`. The
mapping is in the macro's comments. Coloc 2 reports the Costes p-value as 1 − p.

## Whole wells, classification and clustering

### Stitching wells

Add **Stitch well** to a pipeline to place each well's fields in one coordinate system.
The stage positions come from the plate files. When fields overlap, HCS Studio works out
which way the stage axes run (instruments differ) and refines the positions by matching
the overlaps (phase correlation, sub-pixel), so each field lands within a pixel of where
it really was. Fields that do not overlap keep their stage positions.

Each object then has whole-well coordinates (`well_x_um`, `well_y_um`). An object imaged
twice in an overlap is kept once: the copy in the field whose centre is nearer counts, and
the other is marked `in_overlap_duplicate`. Well counts and densities count each object
once and use the area the fields cover together. **Results → Whole-well mosaic** shows the
well at preview resolution. Images are not fused: measurements stay on the original
fields.

### Table steps

Gates, classifiers, phenotypic clusters and spatial statistics are *table steps*. They run
once every well is done, on the object tables, so they can see the whole plate. Their
results go into `objects_annotations.parquet` per well (the measurements are never
rewritten), the merged `tables/per_object_all.parquet`, and new per-well columns in
`tables/per_well_summary.csv`, so they appear as readouts on the results plate and in the
report. What each step did is recorded in `tables/analysis_summary.json`.

Table steps can also be applied to a finished run without reprocessing any image: from the
**Classify** dock (**Apply to run**) or with

```bash
hcs-studio analyze <run folder> --pipeline steps.yaml
```

The run's own table steps run first; add `--only` to run just the file's steps.

### The Classify dock

The dock works on the run open in the Results tab. Choose the object type at the top.

![Classify dock: labelled examples (blue positive, red negative) and the cross-validated result](images/phase8_classify.png)

**Gate.** Pick a feature for a histogram, or two for a scatter plot, and drag the shaded
region (or type the limits). The dock shows how many sampled objects fall inside. **Add
gate to pipeline** or **Apply gate to run**; each object gets `gate_<name>` and each well
the count and fraction inside.

**Classifier.** Name your classes, choose a well and field and press **Show field**. Click
objects in the viewer to label them with the selected class (click again to undo); labelled
objects are coloured. Label examples in several wells, including controls, then **Train**.
The dock reports the cross-validated accuracy and the confusion matrix (rows: the class
you gave; columns: the class predicted). **Suggest uncertain objects** lists the objects the
classifier is least sure about; double-click one to see it, label it, and train again. A
few rounds of this usually beat labelling many easy objects. **Save classifier…** keeps it
as a `.joblib` file with its feature list, normalisation and accuracy; **Add to pipeline**
or **Apply to run** classifies every object (`class_<name>`, its probability, and every
class probability) and adds class counts and fractions per well.

With *plate robust Z* normalisation (the default) every plate's features are first turned
into robust Z-scores of that plate, which removes plate-to-plate brightness differences.
It assumes most objects on a plate are similar; on a plate where nearly all objects belong
to one class, choose *none*. Classifiers are pickle files: only open ones you trust.

From the command line, label objects in a CSV (`well, field, object_id, label`) and run

```bash
hcs-studio train-classifier --run <run folder> --labels labels.csv --output model.joblib
```

**Colour.** Colours the objects of the shown field by any class, gate, cluster or colony
column, with a legend.

### Phenotypic clusters

**Phenotypic clusters** groups objects by their features without labels (k-means, HDBSCAN,
or Leiden if `leidenalg` and `igraph` are installed). Clusters are learned on objects
sampled across the plate, then every object is assigned. `cluster_<name>` holds the
cluster (1 = largest, -1 = noise); wells get the fraction in each cluster; the analysis
summary lists the features that set each cluster apart.

### Spatial statistics

**Spatial clusters** describes how objects are arranged, over the whole well (needs
**Stitch well**) or per field: colonies (DBSCAN with a distance `eps_um`, or HDBSCAN),
nearest-neighbour distances, Clark–Evans ratio, Voronoi area and Delaunay neighbours per
object, and Ripley's H(r) and pair correlation g(r) compared with randomly placed objects
in the same area. `spatial_pattern_<r>um` is +1 where objects are more clustered than
random at that distance, -1 where more regular. Full curves are in
`wells/<W>/analysis.json`. See [measurements.md](measurements.md) for every column.

## Phenotypic profiles (PCA, UMAP, dose-response)

The **Profiles** tab compares wells and objects by all their measured features at once. It
works on the run open in the Results tab; **Add plate (run folder)…** adds other plates of
the same screen. Choose the objects, then **Compute profiles** (it runs in the background
and is saved in the run's `advanced/` folder, so it is there next time).

![Profiles: UMAP of nuclei, positive controls selected with the lasso and marked on the plate](images/phase9_profiles.png)

How the profiles are made:

1. **Features.** Every measured feature present on all plates, minus those missing in more
   than 10 % of objects, those where 95 % of objects share one value, and then one of
   each pair correlating above |r| = 0.95 (the one more correlated with the rest). The
   **Scree and loadings** tab lists what was dropped and why.
2. **Normalisation.** Per plate, each feature becomes a robust Z-score relative to the
   objects in the **negative-control wells** of the plate layout:
   (x − median) / (1.4826 · MAD). This removes plate-to-plate differences in staining and
   imaging. Without negative controls in the layout, all wells are used (with a note).
   *Centre each plate* additionally subtracts each plate's median profile.
3. **Embedding.** PCA (scree plot, loadings) and UMAP (`n_neighbors`, `min_dist`, metric
   and seed are recorded; the same seed gives the same picture). *Embed objects* uses up
   to `Objects sampled` objects spread evenly over the wells; *Embed wells* uses one
   profile per well.
4. **Well profiles and distance to control.** A well's profile is the median of its
   objects' Z-scores. `profile_distance_to_control` is the distance of that profile from
   the median negative-control profile (in the principal components that explain 90 % of
   the variation), in units of object-to-object variation: a well whose objects shifted by
   2 SD in one feature scores about 2. `profile_distance_ratio` divides it by the typical
   distance of the control wells themselves (noisy with few control wells). Both appear as
   readouts on the results plate.

In the plot, **Colour by** controls/compounds, concentration, well, class, gate or cluster.
Click a point to see that object (a crop of its field) and **Open selected object in
viewer**; switch on **Lasso** and draw around points to mark their wells on the results
plate. Clicking a well on the results plate enlarges its points. **Export PNG/SVG** saves
the plot as shown; **Export coordinates** saves the table behind it.

The **Dose-response** tab plots any per-well readout against concentration for each
compound in the layout (mean ± SD of replicate wells, negative-control median dashed) and
fits a four-parameter logistic curve: EC50, Hill slope, bottom, top and R². A fit needs at
least 4 concentrations; an EC50 outside the tested range is marked as extrapolated.

![Dose-response of the distance to control](images/phase9_dose_response.png)

From the command line:

```bash
hcs-studio profile <run folder> [<run folder of another plate> ...] --object-type nuclei
```

writes `advanced/` (PCA and UMAP coordinates, loadings, variance, well profiles,
`dose_response.csv`, `figures/` as SVG and PNG, and `profiling.json` with every option and
the features used) and updates the run's `index.html`, which then has a linked PCA/UMAP
panel: click a point to open its well, drag a rectangle to mark wells on the plate.

## Pharmacology (dose–response and combinations)

The **Pharmacology** tab works on finished runs: the run open in the Results tab, plus
any others (**Add run…**; plates are merged by their annotation). Tick one or more
readouts (e.g. `nuclei_count` for viability, a `class_…_fraction` for a phenotype),
choose the normalization (% of control, % inhibition, GR), the curve model (4PL, 5PL,
3PL), the synergy models (Bliss, HSA, Loewe, ZIP) and, optionally, outlier exclusion,
then **Analyse**. Results need the plate layout: cell lines, compounds with
concentrations, vehicle (negative) and positive controls (Layout tab).

- The list groups results by readout → cell line → compound or combination. A curve
  shows every well (hollow = excluded), mean ± SD and the fit; the line below gives EC50
  with its 95 % interval, absolute IC50, Hill slope, AUC, Emax and flags.
- Click a point to select its well (outlined on the Results plate; **Show in Results**
  opens it there). **Ctrl+click** a point, or **Exclude / include selected wells**,
  excludes it (or includes it again) and refits.
- A combination shows the response matrix and the synergy matrix of the chosen model
  (red synergy, blue antagonism; × = missing well). Click a cell to select its replicate
  wells.
- Tables: IC50/AUC across compounds and cell lines (coloured by EC50), synergy ranking,
  plate QC (Z′ flagged below the threshold).
- **Write report** writes `pharmacology/` in the first run folder: CSV tables, matrices,
  Prism XY tables, figures (PNG 300 dpi + SVG), `pharmacology_report.html` and `.pdf`.

From the command line:

```bash
hcs-studio pharm <run folder> --readout nuclei_count --normalize percent_inhibition --models bliss,hsa,loewe,zip
```

Other options: `--curve 5pl|3pl_bottom|3pl_top`, `--robust`, `--bootstrap 200`,
`--outliers 4`, `--exclude PLATE:WELL`, `--doubling-time 24 --duration 72` (GR),
`--time-zero <run>`, `--surfaces`, `--output <folder>`. Formulas, definitions and
references: [pharmacology.md](pharmacology.md). Synergy labels follow SynergyFinder's
conventions and are a guide, not proof.

## Run steps and the experiment report

Three pipeline steps run once, after all wells, the tables and the report (marked
**▶ after the run** in the Pipeline dock): **Pharmacology** (`pharmacology`), **Figures: a
montage per compound** (`compound_figures`) and **Experiment report** (`experiment_report`).
They do not change images or the pipeline hash (adding them never prevents a resume), a
failure is logged and recorded without failing the run, and their outcome is in
`run_manifest.json` → `run_steps`. Apply them to a finished run with
`hcs-studio analyze <run> --pipeline steps.yaml --only` (a pipeline that contains only
them), or write the experiment report from **Tools → Experiment report for the run in
Results**.

The **experiment report** (`experiment_report/experiment_report.html` and `.pdf`) has:
provenance (software version, git commit, dates, wells, checksum mode), plate layout
figures (compound, concentration, cell line, role), QC (field QC flags, plate Z′/SSMD/CVs
and the control figure), the dose–response table and curves, the combination scores and
heat maps, representative image montages per compound (the field closest to each well's
median, linked contrast, scale bar), and **methods text generated from the manifest**:
acquisition (vendor, objective, NA, pixel size, channels), every pipeline step with what
it does and all its parameters, analyses applied later, the pharmacology methods and
options, and the software environment. Edit it before using it in a paper; it states only
what the run recorded.

## Comparing fields and making figures

The **Compare** tab shows fields of a run side by side (wells from the Results plate, a
Pharmacology point or matrix cell, a Profiles lasso or the annotation filter), arranged
as a grid, by concentration × cell line, or as a combination matrix. Zoom, pan, Z and
contrast are linked (an unlinked panel is marked), outlines can be shown, and crops apply
to all panels. Export figures (PNG/TIFF 300–600 dpi, SVG, PDF) with scale bars, headers,
legends and split channels, the panels' data (OME-TIFF + CSV), Z movies (GIF, MP4), or a
figure per compound. Every export writes a JSON sidecar from which
`hcs-studio figure --regenerate` re-creates it exactly. Details:
[figures.md](figures.md).

## Experiments across runs

The **Experiments** tab is a searchable catalog of every plate, imaging session and run:
plate pages, EC50s of a compound across runs, a readout followed over time, compounds and
cell lines. Runs enter it with **Scan folder…** (or `hcs-studio db scan`); scanning only
reads and can be repeated. Data roots let every computer find the same files; every change
is recorded with who made it and why. Details: [registry.md](registry.md).

## ilastik

Projects trained in ilastik (`.ilp`) will run as pipeline steps, like Cellpose and SAM 3.
Set up ilastik under **Tools → ilastik settings…** (or `hcs-studio ilastik --detect`): HCS
Studio finds the installation, checks its version and blocks releases known not to work.
Running projects follows in the next build steps. Details: [ilastik.md](ilastik.md).

## Limitations in this phase

- Readers have been tested on synthetic plates only, not yet on real instrument exports.
- RGB `.nd2` files are not supported.
- ImageXpress does not record confocal vs widefield; set the modality manually.
- ImageXpress field positions are nominal (site grid), not measured stage positions.
- Operetta flat-field profiles are not read yet.
- The correction, enhancement, segmentation and measurement modules are verified on synthetic images only;
  colocalization has not yet been compared with Fiji Coloc 2 on real fields.
- SAM 3 was checked with its trained weights on one public image set only (see
  validation.md); its quality on your images is unverified.
- SAM 3 segments 2D planes; 3D SAM 3 segmentation is not available.
- Noise2Void and BM3D denoising are not included yet.
- Apart from deconvolution, every field is processed whole in memory.
- Stitched mosaics are previews; fields are not blended into one full-resolution image.
- Classification, clustering and spatial statistics are verified on synthetic data only.
- Dose–response fits and synergy scores are verified on synthetic data; comparison with
  GraphPad Prism and SynergyFinder on reference data is pending (see validation.md).
- Profiles of several plates are saved with the first plate's run; the other runs' reports
  do not show them.
