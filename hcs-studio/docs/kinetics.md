# Live-cell kinetics

Time-lapse plates (Incucyte-style live imaging, or one plate imaged repeatedly by the scheduler)
are analysed on **real times**, never on frame numbers. This guide covers the time-course data
(Phase 22a): where the times come from, the treatment time, drift correction, confluence and the
time-course table (Phase 22a), the kinetic analysis: growth curves, doubling times, IC50
over time and GR values (Phase 22b), the views and quality control of a time-lapse plate
(Phase 22c), and cell tracking (Phase 22d).

## Times

Each time point's time, in seconds since the first, is taken from:

| Source | How |
|---|---|
| ImageXpress | the `acquisition-time-local` of the first image of each `TimePoint_n` folder |
| Nikon ND2 | the frame events (`T Index`, `Time [s]`): the earliest frame of each time point |
| OME-Zarr | the `t` axis scale (with its unit), or the `hcs_studio.time_points_s` plate attribute |
| Metadata panel | **Time interval (s)** when the files carry no times: time point *n* at *n* × interval |

The run manifest records `time_points_s`, `time_source` (`instrument`, `time interval
(metadata)`, `single time point` or `unknown`) and `acquired_at` (the start of the
acquisition). A time-lapse plate without times gets no time columns, and the manifest says
so — set the interval in the Metadata panel and run again.

## Treatment time

`hours_since_treatment` needs to know when the compounds were added. In order of precedence:

1. given explicitly (`hcs-studio kinetics table … --treated-at 2026-01-05T08:30`);
2. the plate layout's **treatment time** (`treated_at`, kept in JSON, CSV and Excel layouts);
3. the latest accepted `plate.dispensed` event for the plate's barcode in the registry (sent by
   the dispenser through lab_events).

Times with a time zone are converted to this computer's local time (instruments write local
times without a zone).

## Drift correction (`register_timepoints`)

Stage drift between time points makes cells appear to move and corrupts anything compared over
time. Put **Register time points** first in the pipeline: for each field it measures the shift
between consecutive time points by phase cross-correlation (mean-subtracted, Hann-windowed,
sub-pixel) on the chosen channel, accumulates the shifts, moves every time point back onto the
first and crops the field to the area seen at **every** time point — so counts and confluence
always cover the same area. Consecutive time points are compared (not each with the first)
because cells move and divide: neighbouring frames look most alike.

A jump between two time points larger than `max_shift_um` (default 50 µm) is a stage move, not
drift: it is not corrected and the well gets a warning. The shifts (pixels and µm) and the crop
are recorded per field in `well_summary.json` (`drift`). Drift correction cannot be combined
with stitching (the crop changes the field size).

## Confluence (`confluence`)

% of the field covered by cells, per field and time point (`confluence_pct`,
`confluence_area_um2` in `fields.csv`; per-well means in `per_well_summary.csv`):

- **texture** (default; brightfield or phase contrast): cells are textured, the background is
  not. The texture is the local SD (window σ = `texture_um`, default 1 µm) of the image minus
  its median filter; the median keeps steps, so the strong outlines of cells do not leave a wide
  halo. The threshold is Otsu's level of the log texture of fields sampled across the plate
  **and its time points** (a single field may be all background or all cells). The window still
  spreads texture past the cell edge by a computable distance (from the background noise and
  cell texture, the medians of the two classes); the mask is eroded by it. A plate whose
  calibration shows no texture above the noise is reported, and its threshold is put at 3 × the
  noise so that empty wells read ≈ 0 %.
- **threshold**: a fluorescent cytoplasmic stain (or a nuclear stain at low magnification).
- **labels**: the area of a label set (e.g. Cellpose cells).
- **probability**: the area where an earlier **ilastik** pixel classification step's cell class
  (`class_channel`) is ≥ 0.5 (`threshold` to change it); no plate calibration. Preset:
  `hcs-studio ilastik --preset <project> --for confluence` (docs/ilastik.md).

Limit: gaps between cells narrower than about 3 × `texture_um` are counted as covered, so
near-confluent fields read a few percentage points high (see Validation).

## The time-course table

A run of a time-lapse plate writes `tables/time_course.parquet` (and `.csv`): one row per well
and time point, with `timepoint`, `time_s`, `time_h`, `hours_since_treatment` (when the
treatment time is known), every object statistic of `per_well_summary.csv`
(`nuclei_objects_measured`, means and medians of features …) computed per time point, the
per-field values (confluence, field QC metrics) averaged per time point, and the layout's
annotations. Fields left out by QC are left out here too.

`per_well_summary.csv` of a time-lapse run still pools every time point (counts are summed over
time points, means are over all of them): use the time course for anything that changes with
time.

### Separate acquisitions of one plate

When the scheduler images a plate several times as separate acquisitions (one run each), join
them on their acquisition times:

```bash
hcs-studio kinetics table runs/P001_0h runs/P001_24h runs/P001_48h --output P001_course.csv
```

Runs are ordered by `acquired_at`; times count from the first acquisition; each row records its
`acquisition` and `run`. Runs of different plates (barcode, else plate name) are refused.
For one time-lapse run, `hcs-studio kinetics table <run>` rewrites its `time_course.csv` (e.g.
with another `--treated-at`).

## Kinetic analysis (`kinetics`)

Add the **kinetics** table step to a time-lapse pipeline (after `measure_objects` or
`confluence`), or run it on finished runs:

```bash
hcs-studio kinetics analyse runs/P001_timelapse
hcs-studio kinetics analyse runs/P001_0h runs/P001_24h runs/P001_48h --output P001_kinetics
```

The readout is the object count (`nuclei_objects_measured`) by default, else confluence; set
`readout` for anything else in the time course. Times are hours since treatment when the
treatment time is known (otherwise since the first time point, with a note).

### Growth curves and doubling time

Every well is fitted on the **log** of its readout (counting noise and growth are
multiplicative, so each time point weighs the same) with three models:

| Model | Curve | Use |
|---|---|---|
| exponential | ln y = ln N₀ + μt | unlimited growth |
| lag_exponential | flat until the lag λ, then exponential | recovery after seeding or treatment |
| logistic | y = K / (1 + (K/N₀ − 1)·e^(−μt)) | growth slowing towards a capacity K (confluence) |

The model with the lowest AICc wins (plain AIC when there are too few time points for AICc, e.g.
three acquisitions); a flat line must do worse, otherwise the well is reported as `constant`
(not growing). μ is the specific growth rate per hour; the doubling time is ln 2/μ (for the
logistic model the initial doubling time, while cells are sparse). Wells whose readout falls are
flagged `shrinking`, with no doubling time. `fit_window_h` restricts the fit to a time range.

Model-free values: AUC (trapezoid, readout × h), AUC as % of the vehicle wells, mean fold change,
final fold change, the time to reach `fold` × the first value (default 2: an empirical doubling
time) and the time to reach a `level` (e.g. 80 % confluence). In `per_well_summary.csv` they
carry the prefix `kn_` (`kn_doubling_h`, `kn_model`, `kn_auc_vs_vehicle_pct` …).

### IC50 over time

At each time point after treatment the readout is expressed as % of the median of the vehicle
wells **at the same time point** (same cell line), and a 4PL is fitted per compound (as in the
pharmacology module). Potency usually grows with exposure: the IC50 of a cytostatic drug falls
with time even though the drug does the same thing all along — one reason to prefer GR.

### GR values

GR (Hafner et al. 2016) separates the drug's effect from the cells' division rate:

GR(c, t) = 2^( log2(x(c, t)/x₀) / log2(x_ctrl(t)/x₀,ctrl) ) − 1

GR = 1: no effect; 0: complete cytostasis; < 0: cell death. On a live-cell plate every well is
imaged before the drug is added, so x₀ is **each well's own value** at the last time point at or
before treatment (no separate time-zero plate), and the control term is the median over the
vehicle wells. GR values from treatment are reported for every later time point, with GR50,
GRmax, GR_AOC and GRinf per compound and time.

**GR over intervals** apply the same formula with the start of each interval as time zero; for
an interval [t₁, t₂] this is GR = 2^(k(c)/k_ctrl) − 1 with k = log2(x(t₂)/x(t₁))/(t₂ − t₁), the
growth rate during the interval. They show *when* a drug acts — an effect that starts late or
wears off — which GR from treatment averages out. Intervals are consecutive time points, or with
`interval_h` the first later time point at least that far (non-overlapping); each is reported at
its midpoint.

GR is undefined when the controls do not grow: no GR where the vehicle wells grew less than
`min_control_doublings` (0.25) since treatment, or `min_interval_doublings` (0.1) in an interval.
A time point at which no concentration differs from the others by more than 3 × the replicate SD
is reported as flat and not fitted.

The endpoint formula is checked against the GR reference code (GRmetrics, gr_metrics); the
interval form follows from it algebraically. The paper's own time-dependent definition could
not be read from this environment, so the choice of intervals is HCS Studio's convention (see
Validation).

### Kinetic QC

Problems that only show over time are checked every time `kinetics` runs; the status of each
check is in `kinetics.json` and at the top of the kinetics report, and wells get flags
(`kn_qc_flags` in `per_well_summary.csv`, `qc_wells.csv`):

| Check | What it finds | Rule |
|---|---|---|
| Time points | the instrument skipped acquisitions; wells without a value at some time point | an interval > 1.5 × the median interval; a missing value where the plate has one |
| Focus drift | autofocus failing at some time points, or the plate drifting out of focus over days | each field's focus score (the `qc` step) relative to the same well's median over time: < 0.5 flags the time point; the plate's median relative focus in the last third < 0.8 × the first third warns |
| Edge evaporation | outer wells losing medium: their cells grow differently as time goes on | for each condition imaged both on the edge and inside, the log2 fold change since the first time point of the edge wells minus the interior wells' median; the median over conditions beyond log2 1.25 (±25 %) at the end warns, and edge wells that far from their condition are flagged. Not assessed when no condition is on both (edges left empty on purpose) |
| Control growth | vehicle wells that do not grow, or grow unlike the others or unlike earlier runs | no growing vehicle well fails; a vehicle doubling time > 1.5 × off the vehicle median flags the well; the vehicle median outside the median ± 3 robust SD (at least ±15 %) of earlier runs of the same cell line in the registry warns (needs ≥ 3 earlier runs indexed with `hcs-studio db scan`) |

Zero objects in an imaged field count as 0 (`fields_imaged` in the time course), so an empty
well at the end of a cytotoxic series is not mistaken for a missing time point.

### The kinetics report

`kinetics/report.html` (linked from the run's `index.html`) is self-contained like `index.html`:

- **QC**: each check with its status, and the focus and edge-effect timelines;
- **plate map with sparklines**: every well's readout over time on one shared scale (log when all
  values are positive), coloured by compound with darker shades for higher concentrations and
  vehicle in black; wells with QC flags have a red border (hover for the flags and the doubling
  time); clicking wells adds them to the overlay;
- **curves**: the median curve of each concentration of a compound, vehicle in black;
- **conditions over time**: a heat map with one row per condition (vehicle, then each compound
  by concentration) and one column per time point — log2 vs the vehicle at the same time point,
  or log2 fold change since the first time point.

`heatmap.png` is the same heat map for documents.

### Files

`kinetics/` (in the run folder, or `--output`): `growth_fits.csv` (one row per well),
`ic50_over_time.csv`, `gr_values.csv` and `gr_metrics_over_time.csv` (from treatment),
`gr_interval_values.csv` and `gr_interval_metrics.csv`, `kinetics.json` (readout, times,
treatment time and its source, notes), `growth_curves.png` (each compound with vehicle wells in
black, concentrations light to dark) and `potency_over_time.png` (IC50 and GR50 against time),
`heatmap.png`, `qc_wells.csv` (flags per well), `qc_timeline.csv` (relative focus and edge
effect per time point) and `report.html`.

## Cell tracking (`tracking`)

The **tracking** table step follows each nucleus from time point to time point within a field
(after `segment_classical`/Cellpose and `measure_objects`; put `register_timepoints` first so
stage drift is not read as movement). It uses scipy only:

1. **Linking**: optimal assignment between consecutive time points (`linear_sum_assignment`) on
   the squared distance from each nucleus's *predicted* position — its last displacement carried
   on, since cells move persistently — to the nuclei of the next time point. Each nucleus may
   instead end or start a track at the cost of the gate (Jaqaman et al. 2008), so a far pair is
   left unlinked when that is cheaper overall. Pairs farther than `max_speed_um_per_h` × the
   interval (at least `min_gate_um`) are never linked.
2. **Divisions**: a nucleus that starts a track within reach of a nucleus at the previous time
   point whose track continues into a sister, both smaller than `max_daughter_ratio` (0.8) × the
   mother's area, is a daughter: the mother's track ends and both daughters start tracks with the
   mother as parent.
3. **Gap closing**: a track ending early and one starting up to `max_gap` (2) time points later
   within the gate are joined (a nucleus missed or out of focus at one time point).

Two touching nuclei merged by segmentation for a time point, then separate again, look like a
division; two guards keep them apart: a nucleus appearing within the gate of a track lost in the
last `max_gap` time points is that track coming back, and a cell born less than `min_cycle_h`
(4 h) ago does not divide.

Per object: `trk_id` (unique in the well: field × 100 000 + track), `trk_parent`, `trk_divided`.
`tracking/tracks.csv`: one row per track — start, end, frames, duration, displacement, path,
speed, straightness, parent, whether it divides, how it ends (`division`, `end`, `lost`) and the
cell-cycle time of tracks from one division to the next (birth and division placed halfway
between the time points around them). Per well (`per_well_summary.csv`): `trk_tracks`,
`trk_speed_um_per_h`, `trk_straightness`, `trk_divisions`, `trk_division_rate_per_h`,
`trk_cycle_h`, `trk_lost_fraction`.

**When tracking works**: the movement between time points must be small next to the spacing of
the nuclei. On synthetic cells (persistent random walk, nuclei ~40 µm apart) hourly frames at
10 µm/h keep 98.5 % of the links and find 94 % of the divisions; at 20 µm/h (half the spacing per
frame) 86 % / 81 %; at 2-hour frames 77 % / 71 %. Image more often rather than raising
`max_speed_um_per_h`. Cell-cycle times are complete only for cells born and dividing within the
movie: in a movie not much longer than the cycle, the faster cycles are over-represented.

## Synthetic data

`hcs_studio.synthetic.kinetics.GrowingCells` draws cells growing with a known count per time
(e.g. `exponential(n0, doubling_h)`), nuclei on one channel and textured cell bodies on a
brightfield-like channel, with an optional drift per time; the truth per time point is the
count, the confluence and the drift. `SyntheticPlate(timepoints=…, time_points_s=…,
acquired_at=…)` writes a time-lapse OME-Zarr plate with those times. `response_course` writes
a time-course table of a dose–response plate with known IC50(t) and GR50 (a drug that lowers the
growth rate, optionally after a delay). `MovingCells` / `simulate_lineages` draw nuclei that move
(persistent random walk) and divide (area doubling over the cycle), with every cell's id, parent
and position in the truth.
