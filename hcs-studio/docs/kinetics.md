# Live-cell kinetics

Time-lapse plates (Incucyte-style live imaging, or one plate imaged repeatedly by the scheduler)
are analysed on **real times**, never on frame numbers. This guide covers the time-course data
(Phase 22a): where the times come from, the treatment time, drift correction, confluence and the
time-course table. Growth-curve fits, IC50 over time and GR values follow in Phase 22b.

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

## Synthetic data

`hcs_studio.synthetic.kinetics.GrowingCells` draws cells growing with a known count per time
(e.g. `exponential(n0, doubling_h)`), nuclei on one channel and textured cell bodies on a
brightfield-like channel, with an optional drift per time; the truth per time point is the
count, the confluence and the drift. `SyntheticPlate(timepoints=…, time_points_s=…,
acquired_at=…)` writes a time-lapse OME-Zarr plate with those times.
