# Changelog

## Unreleased

### Phase 23a — primary screens: scoring and hit calling
- Library plate maps (CSV/XLSX, flexible columns and units) into the registry (`hcs-studio screen
  library`); assay-plate layouts from a control template by stamping, 4 × 96 → 384 quadrants or
  an Echo transfer file, optionally pre-registered by barcode (`hcs-studio screen layouts`).
- Per plate and readout: % activity, robust Z, B-score (guard rail: robust Z on non-randomised
  plates), SSMD*; Z′ and robust Z′; SSMD (MM, UMVUE, robust) over replicates.
- Hit rules as safe expressions over the scores (multi-readout, e.g. active and not
  cytotoxic), image-QC artefact filters, replicate rules, ranked hits, manual include/exclude
  with a reason (`hcs-studio screen score|review`; services `screen.score`, `screen.review`).
- The positional guard rail skips single-concentration compounds (a 384-well screen plate is
  scored 6 × faster). Synthetic screen with seeded hits, cytotoxic and autofluorescent compounds
  and a positional artefact (`synthetic.screening`). Docs: screening.md.

### Phase 22d — live-cell kinetics: cell tracking
- `tracking` (table step, scipy only): nuclei linked between time points by optimal assignment
  on the distance to their predicted positions (persistent motion; track ends/starts at the
  cost of a speed gate), divisions (daughters smaller than the mother; guards against merges
  splitting up), gap closing; `trk_id`/`trk_parent`/`trk_divided` per object,
  `tracking/tracks.csv`, per-well speed, straightness, divisions, division rate, cell-cycle
  time, lost tracks.
- Synthetic moving and dividing nuclei with known lineages (`MovingCells`, `simulate_lineages`).

### Phase 22c — live-cell kinetics: views and QC
- Kinetic QC in the `kinetics` step: acquisition gaps and missing time points, focus drift (per
  well and plate-wide), edge evaporation (edge vs interior wells of the same condition over
  time), vehicle growth (non-growing and outlying wells, and against earlier runs of the cell
  line in the registry). Statuses in `kinetics.json`, flags in `kn_qc_flags` / `qc_wells.csv`.
- `kinetics/report.html` (linked from `index.html`): plate map with a sparkline per well,
  condition × time heat map, curve overlays per compound, QC panel; `heatmap.png`.
- Time courses count 0 objects in imaged fields (`fields_imaged`) instead of a missing value.

### Phase 22b — live-cell kinetics: kinetic analysis
- `kinetics` (table step; CLI `hcs-studio kinetics analyse` for one run or separate
  acquisitions): growth fits per well on the log scale (exponential, lag-exponential, logistic;
  AICc, AIC with few time points) → doubling time, growth rate, lag, capacity; AUC (and vs
  vehicle), fold changes, time to a fold change or level; `kn_` per-well readouts.
- IC50 at each time point (% of the same-time vehicle median); GR values from treatment with each
  well's own pre-treatment value as time zero; GR over intervals; GR50/GRmax/GR_AOC per compound
  and time. Flat time points are reported without a fit.
- `kinetics/` files: growth fits, IC50 and GR tables, `kinetics.json`, growth-curve and
  potency-over-time figures. Synthetic `response_course` (known IC50(t) and GR50).

### Phase 22a — live-cell kinetics: time-course data
- Real times per time point from the readers (ImageXpress image timestamps, ND2 frame events,
  OME-Zarr `t` scale / plate attributes), or the Metadata panel's **time interval**; recorded in
  the run manifest (`time_points_s`, `time_source`, `acquired_at`). Frame numbers are never used
  as times.
- Treatment time: given, the layout's `treated_at`, or the plate's `plate.dispensed` event;
  `hours_since_treatment`.
- `register_timepoints` (image step): drift between time points corrected per field (phase
  cross-correlation of consecutive time points, windowed, sub-pixel), stage jumps reported, fields
  cropped to the area seen at every time point.
- `confluence` (measurement step): texture (median-residual local SD, pooled calibration, edge
  correction), threshold or label-set confluence per field and time point; per-well means.
- `tables/time_course.parquet/.csv`: readouts per well and time point; `hcs-studio kinetics
  table` joins separate acquisitions of a plate on their times.
- Plate-level estimates (`fit`) sample fields across time points too.
- Synthetic time-lapse plates (`SyntheticPlate(timepoints=…)`, `GrowingCells`). Docs: kinetics.md.

### Phase 21c — confocal 3D spheroids
- `attenuation` (image step): depth-dependent intensity loss corrected per plane (robust
  exponential fit or per-plane normalisation, signal above background only, gain cap, warnings).
- `detect_spheroids`: 3D detection of fluorescent spheroids in Z stacks (half-height surface).
- `measure_spheroids` on Z stacks: 3D volume, surface, sphericity, Z extent and coverage, 3D core /
  live / dead volumes, counts and density of child objects (Cellpose 3D nuclei with
  `parent: spheroids`), `axial_scale`.
- `spheroid`: `volume: auto` (3D when measured), flag `incomplete_z`; preset for Z stacks
  (`z_planes`, nuclei role). Stain thresholds at half height above the noise floor (2D and 3D).
- Synthetic confocal spheroids (`ConfocalSpheroid`); `SyntheticPlate.z_step_um`.

### Phase 21b — spheroids: viability, necrotic core, invasion, organoids
- `measure_spheroids`: radial profiles in concentric shells; necrotic core at half height between
  rim and centre (brightfield dark core or dead-stain bright core): area, fraction, viable rim;
  live/dead stain areas and dead fraction; invasion (body by morphological opening vs
  protrusions: count, reach, area, skeleton length and branches).
- `detect_spheroids`: `protrusion_reach_um` follows invading cells; small objects' outlines
  fixed (interior from the dark class of the region).
- `spheroid`: organoid mode (count, size distribution, density, organoid-forming efficiency from
  the layout's seeding density); `radial_profiles.csv/png`, `organoid_sizes.csv`,
  `size_distribution.png`; live/dead roles in the wizard; preset options `invasion`, `organoids`.

### Phase 21a — spheroids: detection, size, volume, QC, gallery
- `detect_spheroids` (image step): brightfield texture detection that ignores well-edge and
  meniscus shading, outline at half height (blur-independent size); dark and fluorescence modes.
- `measure_spheroids` (measurement step): size, shape, sphere and ellipsoid volume, darkness,
  outline blur (focus), position.
- `spheroid` (assay step): one spheroid per well, `sp_` readouts, volume vs vehicle, Z′, QC
  flags (no spheroid, multiple, touching edge, off-centre, out of focus, debris, irregular);
  `spheroids/` with per-spheroid and per-well tables, crops and a sortable gallery (linked from
  `index.html`). Readouts for pharmacology (% of control on volume).
- Brightfield/phase channels are previewed in gray. Synthetic brightfield spheroids
  (`synthetic/spheroids.py`). Docs: spheroids.md.

### Phase 20e — proliferation markers, micronuclei, multinucleation
- `proliferation` (assay step): % positive nuclei for Ki67, EdU/BrdU and pH3 with thresholds
  from secondary-only wells (role `unstained`), a two-population fit to the vehicle (equal
  likelihood: independent of the positive share), robust SD (pH3), Otsu or fixed values;
  micronuclei per 100 nuclei, multinucleated/binucleated cells, fold vs vehicle, Z′; report with
  control histograms and per-well intensity distributions (`assays/proliferation/`).
- `mitotic_errors` (image step): micronuclei next to nuclei (segmented ones taken out of the
  nuclei, smaller ones found by threshold; blebs and debris rejected), and nuclei linked to cells
  segmented on a cell stain.
- Preset (`hcs-studio assay preset proliferation`), readouts, docs, synthetic data
  (`synthetic/proliferation.py`).

### Phase 20d — nuclear translocation
- `cytoplasm` (image step): a cytoplasm ring per nucleus (gap and width in µm, assigned to the
  nearest nucleus, optionally kept inside its cell) or the cell minus its nucleus; nucleus IDs.
- `translocation` (assay step): N/C mean-intensity ratio and log2 ratio per cell; responding
  cells above a threshold from the unstimulated controls (robust SD, percentile, Otsu over
  both controls, or fixed); agonist and inhibitor modes; % responding, median N/C, activation
  between the controls, Z′; background warnings; report `assays/translocation/`. EC50/IC50
  through the pharmacology module (% inhibition).
- Preset (`hcs-studio assay preset translocation`: constant background, Otsu nuclei, ring),
  readouts, wizard preview, docs, synthetic data (`synthetic/translocation.py`).
- `AssayModule.all_objects(roles)`: all object types of control wells.

### Phase 20c — apoptosis and cytotoxicity
- `apoptosis` (assay step): apoptosis marker (caspase-3/7, cleaved caspase-3, annexin V on the
  cells) and dead-cell dye (PI, SYTOX, DRAQ7) thresholds calibrated on the controls (vehicle
  robust SD by default, vehicle percentile, or Otsu over vehicle and positive control);
  condensed nuclei by size and brightness vs vehicle (or a classifier's classes); fragments of
  fragmented nuclei counted apart; categories viable / early / late apoptotic / necrotic;
  viable count; GR from time-zero wells (0 h) or a fixed count separates cytostatic from
  cytotoxic; Z′ of % apoptotic; report `assays/apoptosis/`.
- Preset (`hcs-studio assay preset apoptosis`), readouts, wizard preview, docs, synthetic data
  (`synthetic/apoptosis.py`).

### Phase 20b — DNA-damage foci
- `detect_foci` (image step): foci inside nuclei (LoG or white top-hat, local-maxima split,
  size filter), threshold calibrated on the vehicle wells (median + k robust SD of the response
  over vehicle nuclei), 3D detection with Z stacks; label set `foci` with the nuclei as parents.
- `foci` (assay step): foci per nucleus, % foci-positive, pan-nuclear nuclei apart (calibrated
  on vehicle nuclear intensity), focus area and intensity, fold vs vehicle, breakdown by
  cell-cycle phase; projection warning; report `assays/foci/`.
- Preset (`hcs-studio assay preset foci`), readouts, docs. The assay wizard handles any assay
  (channel roles per assay, generic calibration preview).

### Phase 20a — assay framework and cell cycle
- `assays/`: assay modules are table steps with channel roles, a preset pipeline, calibration
  on the run's control wells (from the layout; stored in the run), registered readouts, QC
  warnings and a report page (`assays/<name>/report.html`, linked from `index.html`). Table
  steps now get the run folder and a `finish_table` hook.
- **Cell cycle**: DNA content scaled so the vehicle G1 peak = 2N; Dean–Jett–Fox-style fit per
  well (% sub-G1, G1, S, G2/M, > 4N); EdU 2D gating; pH3 mitotic index; G2/M arrest index;
  warnings for max projections and single confocal planes. Preset pipeline with a triangle
  threshold (whole-nucleus masks).
- **Tools → Assay wizard…** (map channels, check controls, preview calibration, apply or add to
  the pipeline) and `hcs-studio assay list | preset | run`. Docs: assays.md.

### Phase 19 — started, paused
- Work in progress (not yet active): export completeness checks (ImageXpress, Harmony, nd2),
  automation settings, lab-event log and auto-run tables (migration 0005), event intake.

### Phase 18c — QC review, QC tab, learned QC, QC report
- Reviews (`qc/review.py`): confirm / include / exclude a field, pass / fail a well, each with
  a reason through the recorded service `qc.review` (`hcs-studio qc-review`); decisions are
  recomputed (`rule_excluded` keeps the rules' decision), and a run's tables, plate maps and
  report are rebuilt (`executor.refresh_run_tables`).
- **QC** tab: heat map per QC metric with failed wells hatched, gallery of flagged fields by
  severity with thumbnails, review buttons.
- Thumbnails (`qc/thumbs/`) and a 28-number image embedding per field; **learned QC**
  (`hcs-studio qc-train`): the Phase 8 classifier trained on reviewed fields, used through
  `learned:` in the rules (SHA-256 checked) as the flag `learned`.
- `qc/qc_report.html`: self-contained QC report (decision, heat maps, failed wells, excluded
  fields, flagged-field gallery, compound artifacts, reviews).

### Phase 18b — image QC in runs, object QC, compound artifacts, exclusions everywhere
- Pipelines carry QC rules (`qc:` section; pipeline panel: **Image QC before processing**,
  **QC rules…**). Runs check every field from the raw images before fitting and
  segmentation (also in the job queue; events `qc_started` / `qc_finished`), write
  `<run>/qc/`, record the outcome in the manifest and performance, reuse QC on resume, and
  can stop a failing plate (`plate.stop_run`, status `qc_failed`). QC rules are not part of
  the pipeline hash.
- Exclusions propagate: excluded fields leave counts and statistics (`qc_flags.csv` →
  `image_qc_*`); per-well `qc_failed`, `qc_flags`, `qc_reasons`; pharmacology leaves failed
  wells out of fits (hollow points, "image QC: …" reasons) and control levels; failed wells
  hatched on the Results plate map and in `index.html`; reports list every exclusion; the
  methods text states the QC rules.
- Object-level QC table step `object_qc` (robust outliers in size, shape, intensity →
  `qc_outlier`, `qc_excluded`, `qc_reason`; left out of statistics and later table steps).
- Compound artifacts: autofluorescence (background rising with dose) and precipitation
  (brightfield `particle_fraction` at the top doses), flagged per well and on curves.
- Registry: `run.image_qc` (migration 0004); QC-failed wells (`qc_flag = qc_failed`) are left
  out of cross-run queries. QC metrics 5× faster on large fields.

### Phase 18a — automatic image QC: field metrics, rules and decisions
- `qc/` (headless): field metrics on raw data — focus (power log-log slope as in CellProfiler,
  normalized variance, Laplacian variance), saturation (detector maximum from metadata or a
  recognised camera maximum), background and percentiles, illumination gradient (robust
  surface), objects and area, cell-free gaps, debris, fibers, bubbles, out-of-well regions,
  best-focus plane against the plate trend.
- Versioned YAML rules (`hcs-studio-qc-rules/1`): absolute and plate-relative conditions
  (robust Z, ratio to the plate median), field exclusion, well decisions (good fields, cells)
  and plate decisions (failed wells, failed controls, Z′). Outputs `qc/field_qc.parquet`,
  `qc/well_qc.csv`, `qc/plate_qc.json`, `qc/rules.yaml`.
- `hcs-studio qc` (recorded as `qc.run`); labeled synthetic artifact plates
  (`synthetic.artifacts`). Docs: image_qc.md, validation.md.

### Phase 17c — assay monitoring
- `monitoring/` (headless): Levey–Jennings **control charts** with configurable rules
  (`1_3s`, `2of3_2s`, `7x`, `6t` and general forms), limits from a baseline of runs, and
  **event annotations** (reagent lot, cell vial, instrument service/change) filtered per
  assay/instrument/cell line; **MSR** of a reference compound (10^(2√2·s)), test–retest MSR,
  MR and limits of agreement, rolling MSR (Eastwood et al. 2006; not yet checked against the
  NCATS Assay Guidance Manual text); **instrument bridging** (per-plate correlation and
  Bland–Altman, EC50 ratios with 95 % CIs against the MSR, equivalence verdict) with an HTML
  report, CSVs and JSON.
- Registry series `db/monitoring.py` (Z′, control means/CVs, reference log EC50 per run);
  recorded services `registry.bridging_report` (and `registry.event_annotation`).
- Experiments → **Monitoring** tab and `hcs-studio db chart | msr | bridge | event`.
  Docs: registry.md, validation.md.

### Phase 17b — normalization across plates and replicate concordance
- `normalization/` (headless): % of control, % inhibition, robust Z and **B-score** (median
  polish); a **guard rail** refuses B-score on dose–response series along rows/columns and on
  combination matrices, with the reason; **uniformity maps** from vehicle-only plates
  (robust polynomial surface + edge-ring step; edge effect, CV and trends before/after),
  stored in the registry (`uniformity_map`, migration 0003) through a recorded service;
  **day and plate effects** (R² of the vehicle wells raw / same-day / per plate);
  **replicate concordance** (Pearson, Spearman, Bland–Altman, per-condition CV, IC50 fold,
  discordant flags).
- Experiments → **Normalization** tab (plate heat maps raw/normalized, save a uniformity map,
  compare replicates, day and plate effects) and `hcs-studio db normalize | uniformity | maps
  | batch | concordance`. Docs: registry.md, validation.md.

### Platform foundations (PROMPT_PLATFORM.md §2)
- **Registry database** (`hcs_studio/db/`): SQLAlchemy 2 models with Alembic migrations
  shipped in the package; SQLite at `<settings>/registry.sqlite` (WAL, foreign keys) and a
  refusal to open it on a network share; PostgreSQL URL for server mode (Phase 27).
- **Data roots** (`hcs_studio/paths.py`): paths stored as `ROOT:relative` and resolved per
  operating system; run manifests record `plate.rooted_path`.
- **Service layer** (`hcs_studio/services/`): every state change (layout versions,
  re-annotation, pharmacology with exclusions, run start/cancel/retry/resume, analyses,
  profiles, model training, settings, registry scans and edits) records who, when, what,
  before/after and why; manual exclusions need a reason (`--reason` on the command line).
  A test keeps the GUI and the CLI on the services.
- **Job queue** (`pipeline/queue.py`): `JobQueue` interface with a headless local queue
  (priorities interactive > auto > batch, GPU jobs alone, jobs stored in the registry,
  interrupted jobs recovered); the Jobs dock shows it.
- **Readout registry** (`hcs_studio/readouts.py`): name, unit, level, direction and default
  normalization of every readout of Phases 5–15 (patterns for generated names); the report,
  the Results tab and pharmacology use it.
- **`lab_events`** (repository root): versioned JSON Schemas for the scheduler events.
- **Settings and secrets**: versioned `settings.json`; secrets in the OS keychain
  (`keyring`), redacted from logs.
- **Synthetic data** (`hcs_studio/synthetic/`): composable plate builder writing
  `truth.json`; multi-run registry builder.
- New dependencies: sqlalchemy, alembic, jsonschema, keyring, lab-events. `selftest` checks
  the registry (10 checks).
- Fixed: spatial statistics failed on wells whose nearest-neighbour distances were all equal.

### Phase 17a — experiment catalog and browser
- Catalog: projects, experiments, plates (with well annotations), acquisitions, runs,
  per-well results (long form), dose–response fits, combination scores, plate QC,
  compounds, library plates, cell lines, event annotations, saved searches.
- Scanner: indexes run folders (and raw plates with `--raw`); re-running adds nothing; a
  changed run replaces its own results; unregistered readouts are stored and reported.
- **Experiments** tab and `hcs-studio db info | scan | search | potency | ic50s | plate`:
  plate search and pages, potency across runs, readout over time, compound and cell-line
  pages, CSV/XLSX export. Docs: `docs/registry.md`.

### Phase 16 (a) — ilastik: settings, detection, HDF5 and axis order
- Tools → **ilastik settings…** and `hcs-studio ilastik --detect / --save`: finds ilastik
  (install folders, settings, `HCS_ILASTIK`; also the Python of an ilastik conda environment)
  and Fiji with the ilastik4ij plugin, reads ilastik's version, blocks incompatible releases
  (1.3.3b2 … 1.3.3post2, 1.4.0b27, older than 1.3.3post3), warns on pre-releases; threads,
  RAM limit (≥ 500 MB), processes at once, default runner.
- `segmentation/ilastik_backend/axes.py` (TCZYX ⇄ ilastik axis strings) and `h5io.py`
  (ilastik-ready HDF5 with vigra `axistags`), checked against ilastik 1.4.2 itself.
- New dependency: `h5py`. Docs: `docs/ilastik.md`.

### QC review of Phases 0–15
- `docs/qc_review.md`: per-phase QC results (tests and every command-line tool run end to
  end on synthetic data), open issues, the real data needed per phase and how to send it.
- Fixed: layout validation reported a plate without any cell line (one-line screens, older
  layout files) as an error; it is now a warning. Partly missing cell lines stay an error.

### Hardening pass (after Phase 15)
- Compare: time-lapse movies (GIF/MP4 over timepoints from raw or recomputed sources,
  with a time stamp); a combination picker when the layout has several drug pairs; the
  panel cache is bounded (2 GB, LRU) and PNG/TIFF exports over 250 megapixels are refused
  with an actionable message; raw/`step:` panels of a run share one reopened plate.
- Figure sidecars fingerprint folder sources (OME-Zarr), so regeneration refuses changed
  stores too; annotation labels follow layout file changes.
- Experiment report: field QC flags summarised per flag (out of focus, saturated, edge,
  excluded, low-count well); the layout is read once.
- Pharmacology: % of control refuses a vehicle median ≤ 0 with a message (use %
  inhibition); AUC works with numpy versions without `trapezoid`; the pharmacology run
  step no longer rewrites `index.html` twice; concentration columns of mixed unit kinds
  are grouped by kind.
- Tests: pytest dumps all thread stacks after 180 s (`faulthandler_timeout`) to diagnose
  hangs; an intermittent GUI-suite hang seen once did not recur in three repeated GUI runs.

### Phase 15 — integration, reports and regression tests
- Run steps in the pipeline builder (marked "▶ after the run"): **pharmacology**,
  **compound_figures** and **experiment_report** run once after all wells, the tables and
  the report; outside the pipeline hash; recorded in `run_manifest.json` → `run_steps`;
  a failing run step never fails the run. `hcs-studio analyze` applies them to finished runs.
- **Experiment report** (HTML + PDF, `experiment_report/`): provenance, plate layout figures,
  QC, dose–response, combinations, representative image montages per compound, and methods
  text generated from the manifest (acquisition, every step with its parameters, analyses,
  pharmacology, software). Also from Tools → Experiment report.
- `index.html`: colour the plate by annotation (compound, concentration, cell line, role);
  pharmacology normalized readouts; a Pharmacology section with curves (click a point to
  open its well) and response/synergy matrices (click a cell to open its wells); links to
  reports, figures and montages. Refreshed after pharmacology and run steps.
- Regression suite: a synthetic 384-well image plate with known dose–response, a
  synergistic and a Bliss-additive combination, run end to end in CI with a stored baseline.
- Docs: `docs/annotation.md` (annotation guide), run steps and the experiment report in
  the user guide; pharmacology methods and the comparison/export guide (Phases 13–14).
- Fixed: pharmacology on real runs (per-well summaries carry annotation columns, which
  were mistaken for readouts); per-compound montages used the 0–1 default contrast instead
  of linked auto contrast; long column headers overlapped.

### Phase 14 — side-by-side comparison and export
- New **Compare** tab: wells from the Results plate, Pharmacology points or matrix cells,
  Profiles lassos or the annotation filter; arranged as a grid, by concentration × cell
  line, as a combination matrix (6 × 6 image grid) or for one compound.
- Linked zoom, pan, Z and contrast (identical LUTs, low/high and gamma per channel);
  unlinked panels are marked; per-panel zoom lock; outlines (and category colours in
  figures); label template from the annotation; synchronized crop.
- Image source per panel: corrected, enhanced, raw, or the field recomputed up to any
  pipeline step with the run's saved estimates. Representative field by a readout.
- Figure export: PNG/TIFF (300/600 dpi), SVG/PDF with vector scale bars and editable text;
  journal widths (85/120/180 mm), headers, channel legend, split channels, Z stamp,
  presets, and a figure per compound.
- Data export (OME-TIFF per panel + readouts CSV) and Z movies (GIF, MP4).
- Every export writes a JSON sidecar (sources with SHA-256, steps, display, geometry);
  `hcs-studio figure --regenerate` re-creates the figure pixel-identically.
- New dependencies: imageio, imageio-ffmpeg (MP4).

### Phase 13 — pharmacology: dose–response, combination matrices, synergy
- New **Pharmacology** tab and `hcs-studio pharm`: one or more readouts of one or more
  runs, normalized per plate and cell line (% of control, % inhibition, GR with time-zero
  wells, a time-zero plate or a doubling time).
- Plate QC: Z′, SSMD, signal window, control CVs, flag below a Z′ threshold, control and
  plate-trend figure.
- Optional robust outlier exclusion (median ± k·MAD within replicate groups) and manual
  exclusion by clicking a point; excluded wells stay visible (hollow) and are listed.
- Dose–response: 4PL (default), 5PL, 3PL in GraphPad Prism's form; robust fitting;
  relative and absolute IC50 with 95 % intervals (covariance or bootstrap), Hill, Top,
  Bottom, Emax, AUC, activity area, GR50/GRmax/GR_AOC/GI50, R², RMSE, flags; EC50 shifts
  between cell lines with an F test.
- Combination matrices detected from the annotation (any N × M size, 0 row/column
  included, missing wells marked); Bliss, HSA, Loewe (+ combination index) and ZIP with
  mean, best 3 × 3 window, maximum cell, replicate t test and SynergyFinder labels.
- Outputs in `pharmacology/`: normalized wells, QC, fits, comparisons, scores, matrices,
  Prism XY tables, figures (PNG 300 dpi + SVG), HTML and PDF report.
- Synthetic 384-well combination plate with known answers (`pharmacology/synthetic.py`).
- New dependency: reportlab (PDF).
- Methods and references: `docs/pharmacology.md`.

### Phase 12 — Z projection and background removal after projection
- Projection: max (default), mean, sum, min, standard deviation, extended depth of field
  (variance or multi-scale pyramid); all planes, a plane range, or automatically chosen
  in-focus planes (normalized variance or Brenner; per channel or from one channel). The
  planes used per field and channel are stored in `well_summary.json`.
- Background removal: top-hat, rolling ball, sliding paraboloid, Gaussian or median
  background (subtract or divide), polynomial or spline surface fitted to background
  pixels, constant (percentile or blank wells of the plate layout), and BaSiC-style flat-
  and dark-field estimated across the plate. Negatives clipped, float and linear output.
- Guard rails: warning when the radius is under 2× the typical object diameter; change
  of the mean inside objects and in the background reported per field and channel;
  warnings appear in the preview, the log and `well_summary.json`.
- Line-profile tool: draw a line in the viewer to plot intensity before and after.
- Saved corrected images in float32 or the camera's bit depth (`--image-dtype`, run dialog).
- `stitch_well` can write each well as one linear image (`mosaic.ome.tif`).
- Recommended pipeline `scripts/pipelines/projection_background.yaml`.
- OME-Zarr synthetic plate writer (`write_ome_zarr_plate`) for tests.
- Changed: the morphological background step now estimates on a 3 × 3 mean (less noise
  bias) and its title is "Background removal". Runs with a `background` or `projection`
  step from earlier versions get a new pipeline hash and cannot be resumed.

### Phase 11 — plate annotation (PROMPT_ADDON.md)
- New **Layout** tab: annotate cell line, treatments (single agents and combinations, each
  with concentration, unit and vehicle), role, passage, cells per well, time point, notes
  and custom fields; select by click, Shift+click, drag, Ctrl+click or row/column header;
  edit only the ticked fields of a selection; undo/redo; copy/paste.
- Fill helpers: dilution series (1:2, 1:3, 1:10, ½-log, custom; either direction;
  replicates), combination matrix (6 × 6 by default, single agents and vehicle corner,
  replicate matrices), vehicle/positive controls and blanks, edge-well exclusion.
- Colour the plate by cell line, compound, concentration (log), role or replicate group.
- Units: molar, mass and percent units stored in base units and shown as entered
  (`model/units.py`); molar ↔ mass only with a molecular weight.
- Validation with clickable problems (missing cell line, untreated samples, treated empty
  wells, missing or too few vehicle controls, mixed units, combination matrices without
  single agents).
- Import long CSV/XLSX (column-mapping dialog for unfamiliar headers), plate-map XLSX and
  JSON; export long CSV/XLSX, plate-map XLSX, PNG/PDF figure and JSON. CSV round trips are
  byte-identical. Older layout CSVs (`compound`, `concentration`, `control_type`) still load.
- Versions of each plate's layout (`~/.hcs_studio/layouts/`), templates with renaming of
  cell lines and compounds; runs copy the version used and record version + SHA-256.
- Annotation filter bar (cell line, compound, concentration range, role) on the Layout,
  Plate and Results tabs.
- `per_well_summary` carries the annotation columns; the report shows them.
- CLI: `hcs-studio layout <file> [--validate] [--export <file> [--plate-map] [--layer …]]`
  and `hcs-studio annotate <run> --layout <file>` (re-annotate a finished run without
  reprocessing images).
- New dependency: openpyxl.
- Changing sample names, doses or roles in a layout no longer changes the pipeline hash
  (only unstained/blank wells do), so such runs can be resumed.

### Acceptance-test toolkit (PROMPT.md §10)
- `hcs-studio acceptance counts | coloc | run | kill-resume | beads`: one command per §10
  criterion, PASS/FAIL per item and a JSON report; protocol in `docs/acceptance.md`.
- The Fiji export macro saves Coloc 2's results (`coloc2_log.txt`) for the comparison.
- Bead check: FWHM per isolated bead at the same voxel before and after deconvolution,
  without resampling (synthetic confocal beads: −47 % lateral, −59 % axial).

### Phase 10 — hardening and packaging
- Performance record: time per step (summed over fields), reading, writing, fields per
  minute, peak memory per worker and output size in `run_manifest.json` → `performance`;
  per well in `well_summary.json` → `timing`. Profiled on a synthetic 384-well plate
  (`scripts/make_test_plate.py`, `scripts/pipelines/perf_384.yaml`); see validation.md.
- Faster granularity features (OpenCV morphology with cached footprints; identical
  openings): object measurement about 40 % faster.
- Plate summaries are computed well by well; the plate object table and per-object
  colocalization rows are never loaded at once (10⁷-object plates). Training a classifier
  from the command line reads only the labelled wells and a sample.
- Run options: do not save corrected images (`--no-images`; masks, previews and tables are
  still written) and a RAM limit for the whole run (`--max-memory-gb`, also in the run
  dialog), which limits the worker count and the deconvolution tile size.
- Error handling: free disk space is checked before a run starts; a full disk stops the run
  at once (resumable) instead of failing every well; clear out-of-memory messages; Ctrl+C
  in the command line stops after the current field and prints the resume command (exit
  code 130); the application logs to `~/.hcs_studio/logs/hcs-studio.log` and shows
  unexpected errors without closing.
- `hcs-studio selftest` checks an installation end to end; `hcs-studio gui --smoke-test`.
- Windows installer: PyInstaller bundle (`HCS Studio.exe` + `hcs-studio.exe`) and Inno
  Setup (`packaging/`), CPU and CUDA variants, built and self-tested by the
  `hcs-studio-windows` GitHub workflow, which also runs the test suite on Windows.
- User guide shipped as HTML (`scripts/build_docs.py`); new `docs/install.md` and
  `docs/troubleshooting.md`.
- Flat-field: the profile is now a fitted 2D polynomial by default (accurate up to the
  border: corner error 3.8 % → 0.2 %); Gaussian smoothing remains an option. Changes the
  pipeline hash of runs with flat-field (they cannot be resumed with this version).
- Plate summary of a 384-well plate: 132 → 46 s, 1.4 → 0.36 GB peak, identical results.
- Fixed: profiling failed on integer features (counts); synthetic test plates now shade
  the background like real optics; the frozen bundle misses no matplotlib backend or
  napari resources (found by the self-test).

### Phase 9 — phenotypic profiles
- Profiling of one or more runs: feature selection (missing, near-constant, |r| > 0.95),
  robust Z per plate relative to negative-control objects, optional plate centring, PCA
  (scree, loadings) and seeded UMAP of objects or wells.
- Per-well profiles and distance to the negative controls (object SD units), shown as
  readouts on the results plate and in the report.
- Dose-response: four-parameter logistic fits (EC50, Hill, bottom, top, R²) of any per-well
  readout per compound.
- Profiles tab: PCA/UMAP coloured by controls, compound, concentration, class or cluster;
  lasso → wells marked on the results plate; click → object crop and viewer; plate click →
  points highlighted; scree and loadings; dose-response; SVG/PNG/CSV export.
- `advanced/` folder with coordinates, loadings, well profiles, `dose_response.csv`,
  figures (SVG, PNG) and `profiling.json`; `hcs-studio profile <runs…>`.
- `index.html`: PCA/UMAP panel linked to the plate (click a point → well; drag → mark wells;
  selected well highlighted).
- umap-learn and matplotlib are now dependencies.
- Fixed: rebuilding the results plate while wells were selected raised an error.

### Phase 8 — classification, clustering, whole-well analysis
- **Stitch well** step: field positions from stage coordinates, stage orientation found
  from the overlaps, sub-pixel refinement by phase correlation and global least squares;
  whole-well object coordinates, overlap duplicates counted once, well area from the union
  of fields, `previews/mosaic.png` (Results → Whole-well mosaic).
- Table steps, which run on the object tables after all wells: **Gate objects**,
  **Classify objects**, **Phenotypic clusters**, **Spatial clusters**. Annotations go to
  `objects_annotations.parquet` and the plate table; per-well values to the summary and
  the results plate; `tables/analysis_summary.json` records parameters, classifier hash
  and accuracy.
- Object classifier: random forest or histogram gradient boosting, plate robust-Z
  normalisation, stratified cross-validation with confusion matrix, active learning
  (least certain objects), saved with its feature list.
- Phenotypic clusters: k-means, HDBSCAN, Leiden (optional), with per-cluster feature
  profiles.
- Spatial statistics: DBSCAN/HDBSCAN colonies, Ripley K/L/H and pair correlation with
  border correction and CSR envelopes, nearest neighbours, Clark–Evans, Voronoi and
  Delaunay local density.
- Classify dock: gates drawn on histograms and scatter plots, examples labelled by clicking
  objects in the viewer, train / cross-validate / suggest, apply to a finished run, colour
  objects by class, gate, cluster or colony.
- CLI: `hcs-studio analyze <run> --pipeline steps.yaml`, `hcs-studio train-classifier`.
- The report lists the analysis steps and links each well's mosaic.
- scikit-learn is now a direct dependency.
- Fixed: the main window could open taller than the screen (about 2000 px); it now opens
  at 1500 × 950 or the screen size, whichever is smaller.

### Phase 7 — results plate and report
- Results tab: plate coloured by any per-well readout or by % positive above a threshold;
  values or robust Z-scores (plate, or negative controls); colour bar; Z′ from the layout's
  controls. Opens automatically when a job finishes; File → Open results….
- Well panel: annotation, QC flags, key statistics, field previews switchable between
  enhanced, corrected and raw, with toggleable outlines; Open in viewer (image and masks);
  raw / corrected / enhanced side by side.
- `index.html` in every run folder: the same results plate in a single self-contained file
  (works offline, zipped and shared); `hcs-studio report <folder>` regenerates it.
- Run folders now hold raw, corrected and enhanced previews per field and transparent
  outline overlays per segmentation (outlines are no longer burned into the preview).
- Fixed: the focus score depended on how many cells a field held, so sparse fields could
  be flagged out of focus and excluded from counts; it now measures edge sharpness.

### Phase 6 — pipelines and batch runs
- Pipeline builder: drag-and-drop step order, preview up to a step or all steps,
  **Run on plate…**.
- Plate map selection: click, Ctrl+click, Shift+click range and rubber-band drag.
- Run dialog: current field, current well, selected wells, whole plate, or several plates;
  output folder, worker processes (auto), checksums.
- Jobs dock: a queue of jobs, each in its own process; overall progress with ETA, status of
  every well, pause, cancel, retry failed wells, resume a run folder, open folder;
  interrupted jobs detected.
- Batch engine: wells processed by a pool of worker processes; GPU steps serialised by a
  shared lock; memory budget shared between workers; SAM 3 runs with one worker.
- Resume: a well is complete when its summary is written with the run's pipeline hash;
  interrupted wells are redone; plate-level estimates are saved and reused; resuming a
  folder made with a different pipeline, plate, metadata or layout is refused.
- An unreadable image fails only its own well; plate-level estimates skip it and log a
  warning.
- CLI: `--workers N|auto`, several `--input`, `--resume <folder>`, `--retry-failed <folder>`;
  exit codes 0 / 1 (some wells failed) / 2 (could not start).
- Run manifest records the plate source, pipeline hash and every attempt (history).
- Verified on a synthetic 384-well plate: killed part-way, resumed, identical to a clean run.

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
