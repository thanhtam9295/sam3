# Image QC

Automatic image QC finds bad fields, wells and plates from the raw images, before they
distort results. It measures every field, applies a rules file and writes the decisions;
in a pipeline run it runs before segmentation, and its exclusions reach every result:
counts and measurements, IC50 fits, plate maps and reports. Object-level QC removes likely
segmentation errors from feature statistics. The **QC** tab (and `qc_report.html`) shows
what was flagged; every decision can be accepted or overridden with a reason, and a learned
classifier can be trained from those reviews.

## QC in a pipeline run

Tick **Image QC before processing** in the pipeline panel (and load a rules file with
**QC rules…**, or keep the defaults), or add a `qc:` section to the pipeline file — the
rules are stored with the pipeline:

```yaml
name: viability
steps:
  - {module: segment_cellpose, params: {channel: 1}}
  - {module: measure_objects}
  - {module: qc}
qc:
  schema: hcs-studio-qc-rules/1
  name: lab-default
  version: 2
  settings: {content_channel: 1, nucleus_diameter_um: 9}
  field: [...]
```

Every run of that pipeline (from the app, `hcs-studio run`, or the job queue) then checks
all fields of the selected wells from the raw images first, writes `<run>/qc/`, and only
then estimates plate-level corrections and segments. The run's manifest records the QC
outcome and rules, and the time QC took (`performance.wall_s.image_qc`). A resumed run
reuses the QC of the first attempt when the rules are unchanged. Changing the QC rules does
not invalidate a run's processed wells: QC decides what is counted, not what was processed.

With `plate: {stop_run: true}` in the rules, a plate that fails QC stops the run before any
processing (status `qc_failed`) — useful for unattended runs.

## Where exclusions show

| Where | What happens |
|---|---|
| Counts and measurements | excluded fields are left out of counts, densities and per-well statistics (like out-of-focus fields); `tables/qc_flags.csv` has `image_qc_excluded`, `image_qc_flags`, `image_qc_reasons` per field |
| Per-well summary | `qc_failed`, `qc_flags`, `qc_reasons` |
| Pharmacology | wells that failed QC are left out of fits (hollow points, `exclusion_reason` "image QC: …") and of the vehicle / positive-control levels; compound artifacts are added to the curve's flags |
| Plate maps | failed wells are hatched (Results tab and `index.html`), with the reason in the tooltip |
| Reports | `index.html` lists the plate decision, every failed well and excluded field with its reason, and compound artifacts; the pharmacology report lists excluded wells with reasons; the methods text states the QC rules and counts |
| Registry | each run keeps its QC summary; wells that failed QC (`qc_flag = qc_failed`) are left out of EC50s and trends across runs |

You can still include a QC-failed well in a pharmacology analysis by hand (it is recorded).

## Object-level QC

The `object_qc` table step flags objects that are likely segmentation errors — merged cells
(large, low solidity), fragments (small), out-of-focus or debris-like objects (intensity) —
by robust outlier detection: it learns the median and robust spread of size (log area),
shape (solidity) and mean intensity (log, first channel) from objects sampled across the
plate, and flags objects beyond `threshold` (robust Z, default 5) in any of them. Choose
other features with `features`.

Columns per object: `qc_outlier`, `qc_excluded`, `qc_reason` (e.g. "area_um2 high; solidity
low"); per well: `qc_<type>_outlier_count` and `_fraction`. Excluded objects are left out
of per-well feature statistics (means, medians, SDs) and of later table steps (gates,
classifiers, clusters) — put `object_qc` first among the table steps. Counts are not
changed (a merged object still holds cells).

## Compound artifacts

With a plate layout, QC also looks for artifacts caused by the compounds themselves:

- **Autofluorescence**: the background outside cells (per channel) is raised in a well —
  robust Z above `compound.z` (default 5) against the plate's vehicle wells — and rises
  with concentration along that compound's dose series (Spearman ρ ≥ `compound.min_rho`,
  0.8, over at least 3 concentrations).
- **Precipitation**: the same test on `particle_fraction` — small refractile particles in
  the brightfield / digital phase contrast channel (`settings.brightfield_channel`) —
  limited to the upper half of the series' concentrations.

A raised well whose series does not rise with the dose is not called a compound artifact.
The flags appear per well (`well_qc.csv`, per-well summary, report) and on the compound's
dose–response curve. They do not exclude wells unless `compound.exclude: true`.

```yaml
settings: {content_channel: 1, brightfield_channel: 3}
compound: {z: 5, min_rho: 0.8, min_concentrations: 3, autofluorescence: true,
           precipitation: true, exclude: false}
```


## Running QC

```bash
hcs-studio qc <plate> --output <folder>
```

```bash
hcs-studio qc <plate> --output <folder> --rules my_rules.yaml --wells A01-H12 --workers 8 --reason "new plate"
```

QC reads the raw images (first timepoint), one field at a time, and never writes into the
raw data folder. It writes `<folder>/qc/`:

| File | Content |
|---|---|
| `field_qc.parquet` | one row per field: every metric, `flag_<name>` columns, `flags`, `severity`, `excluded`, `reasons` |
| `well_qc.csv` | one row per well: fields, good fields, cells, flags, `failed`, `reasons` |
| `plate_qc.json` | plate decision (`passed`, `reasons`), counts per flag, controls, the rules' name, version and SHA-256, plate path, time taken |
| `rules.yaml` | the rules that were used |
| `qc_report.html` | the QC on one self-contained page: decision, heat maps (failed wells hatched), failed wells, excluded fields, a gallery of flagged fields by severity, compound artifacts, reviews |
| `thumbs/` | a thumbnail of every field (content channel) |
| `overrides.json` | the reviews (after the first review) |

The command exits with status 1 when the plate fails. Each QC run is recorded (who, when,
why, which rules, the outcome).

## What is measured

Per channel (columns `c1_…`, `c2_…`, channels from 1), on the raw image:

| Metric | Meaning |
|---|---|
| `plls` | power log-log slope (Bray et al. 2012), as CellProfiler's MeasureImageQuality computes it: more negative = blurrier |
| `normalized_variance` | variance / mean (CellProfiler's FocusScore) |
| `laplacian_variance` | fine detail: variance of the Laplacian / image variance; blur lowers it |
| `saturated_fraction` | fraction of pixels at the detector maximum (bit depth from the metadata; a 12-bit camera stored as 16 bit is recognised by its maximum, 4095) |
| `background`, `p50`, `p99`, `p999` | background level and intensity percentiles |
| `gradient` | illumination variation across the field: (max − min) / mean of a smooth surface fitted to the background, ignoring bubbles and debris |
| `best_z`, `best_z_offset` | best-focus plane of a Z stack, and its distance (in planes) from the plate's trend |

On the content channel (usually the nuclear stain, `settings.content_channel`):

| Metric | Meaning |
|---|---|
| `object_count`, `area_fraction` | bright objects (cells or nuclei) and the area they cover |
| `largest_gap_fraction` | largest cell-free region (scratches, cell-free bands) |
| `debris_count`, `debris_area_fraction` | objects much brighter and larger than the cells |
| `fiber_count` | long thin bright objects (fibers, lint) |
| `bubble_fraction` | round regions inside the field whose background differs by > 20 % |
| `out_of_well_fraction` | dark regions at the field border (well edge, meniscus shadow) |
| `particle_fraction` | brightfield channel only: area of small high-contrast spots (precipitate, refractile particles) |

Focus metrics are computed on the image clipped at its 99th percentile, so a bright
speck does not count as "sharp" and debris does not count as "blurred".

## Rules

A rules file (YAML) holds field thresholds and well / plate decisions. Write the default
rules, edit them, and keep the file with the pipeline:

```bash
hcs-studio qc --write-rules qc_rules.yaml
```

```yaml
schema: hcs-studio-qc-rules/1
name: default
version: 1            # raise it when you change the rules; the SHA-256 is recorded too
settings:
  content_channel: 1
  nucleus_diameter_um: 10.0
field:
- flag: blur
  metric: laplacian_variance
  channel: content
  rel_below: 0.5      # below half of the plate's median
  exclude: true
- flag: saturation
  metric: saturated_fraction
  channel: any        # any channel
  above: 0.002
  exclude: true
well:
  min_good_fields: 1
  min_good_field_fraction: 0.5
  min_cells: 0
plate:
  max_failed_well_fraction: 0.1
  max_failed_control_fraction: 0.5
  controls: [negative_control, positive_control]
  min_z_prime: 0.5
  stop_run: false     # true: a plate that fails stops the run before processing
```

Conditions of a field rule (a rule fires when any of its conditions holds):

| Condition | Fires when |
|---|---|
| `above`, `below` | the value is above / below a fixed limit |
| `z_above`, `z_below` | the robust Z of the value among the plate's fields (median, 1.4826 × MAD) is above / below |
| `rel_above`, `rel_below` | value / the plate's median is above / below |

Plate-relative conditions adapt to each plate (stain, exposure, instrument); fixed limits
catch problems that affect the whole plate. `exclude: true` removes the field; other flags
are shown but keep the field.

Default rules:

| Flag | Rule | Excludes the field |
|---|---|---|
| blur | `laplacian_variance` < 0.5 × plate median | yes |
| saturation | > 0.2 % of pixels saturated in any channel | yes |
| autofocus | best plane > 1.5 planes from the plate trend | yes |
| bubble | `bubble_fraction` > 2 % | yes |
| out_of_well | `out_of_well_fraction` > 2 % | yes |
| gradient | illumination varies > 30 % in any channel | no (flat-field correction handles it) |
| debris | debris covers > 0.2 % of the field | no (object-level QC removes debris objects, step 18b) |
| fiber | any fiber | no |
| scratch | largest cell-free region > 8 % of a field with cells | no |
| empty | no cells | no — few cells may be biology (potent compounds) |
| background | background robust Z beyond ±6 in any channel | no |
| particles | `particle_fraction` robust Z above 8 (brightfield channel) | no |

**Wells** fail with fewer good fields than `min_good_fields`, a smaller share of good fields
than `min_good_field_fraction`, or fewer cells than `min_cells`. Keep `min_cells` at 0 for
dose–response plates: wells with potent compounds legitimately have few cells.

**Plates** fail when more than `max_failed_well_fraction` of the wells failed, when more than
`max_failed_control_fraction` of the wells of a control role failed (with a layout), or when
Z′ is below `min_z_prime` (when Z′ is known).

## Reviewing QC (the QC tab)

Opening a run with image QC fills the **QC** tab:

- a plate heat map of the chosen metric (worst severity, excluded fields, or any QC metric);
  failed wells are hatched; click a well to list all its fields, click it again for the
  flagged fields of the plate;
- a gallery of flagged fields, most severe first (filter by flag), with thumbnails, flags,
  reasons and whether the field is excluded;
- **Confirm flag**, **Include field**, **Exclude field** for the selected field, and **Pass
  well** / **Fail well** for its well. Each needs a reason and is recorded (who, when,
  why, before and after).

After a review the decisions are recomputed (the rules' own decision is kept as
`rule_excluded`), the `qc/` files and `qc_report.html` are rewritten, and in a run folder the
tables, per-well summary, plate maps and `index.html` are rebuilt. Run pharmacology again to
update fits. From the command line:

```bash
hcs-studio qc-review <run> --well C05 --field 3 --action include --reason "in focus on inspection"
```

```bash
hcs-studio qc-review <run> --well C05 --action exclude --reason "bubble seen at the microscope"
```

| Action | Field | Well |
|---|---|---|
| `confirm` | the flag is right (decision stands) | decision stands |
| `include` | keep a field the rules excluded | pass a well the rules failed |
| `exclude` | leave out a field the rules kept | fail the well |

## Learned QC

Reviews teach a classifier what your reviewers call a bad field — artifacts the rules miss.
Confirmed and excluded fields are "bad", included fields "good", and (unless
`--reviewed-only`) unflagged fields nobody overrode count as "good" (up to 200 per plate).
Inputs are every QC metric and a 28-number image embedding (block means and radial power
spectrum of the field), as robust Z within each plate. The classifier is the Phase 8 random
forest (balanced classes, cross-validated accuracy reported):

```bash
hcs-studio qc-train <run 1> <run 2> --output learned_qc.joblib --reason "trained on October screens"
```

Then add it to the rules (the command prints the line, with the file's SHA-256):

```yaml
learned: {model: 'learned_qc.joblib', sha256: 3f1c…, threshold: 0.5, exclude: false}
```

Fields with P(bad) ≥ `threshold` get the flag `learned` (and are excluded with `exclude:
true`). A changed model file is refused (SHA-256). It needs at least 3 bad and 3 good
reviewed fields; more and more varied reviews make it better.

## Tuning on your data

The default thresholds were set on synthetic artifacts (see the Validation page). Before
relying on them, run QC on 20–50 fields you consider bad (with the reason) and about 50 good
fields per instrument, and adjust the thresholds in your rules file.
