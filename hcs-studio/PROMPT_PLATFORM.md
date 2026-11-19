# HCS Studio — Add-on Prompt: Phases 17–28
## From analysis tool to screening platform: experiment database, automatic QC, scheduler auto-run, oncology assay modules, 3D spheroids, live-cell kinetics, hit calling, morphological profiling, model management, accounts and audit trail, server mode, and a Claude lab assistant

> **How to use this file**
> 1. Save it in the project folder as `PROMPT_PLATFORM.md`, next to `PROMPT.md`, `PROMPT_ADDON.md` and `PROMPT_ILASTIK.md`.
> 2. In Claude Code, select **Claude Opus 5.5** and send the message in **Section 5**.
> 3. These phases are large. Build **one phase per Claude Code session**. At the end of each phase Claude Code updates `CHANGELOG.md` and `CLAUDE.md`, so the next session starts with the right context. Have it plan each phase and show you the plan before it writes code.
> 4. **Section 2 lists architecture decisions that affect everything from Phase 17 on.** Have Claude Code apply them first, even if you later build the phases in a different order.

All project rules in `PROMPT.md` Section 2 still apply: raw data read-only, provenance for every output, tests with known answers, lazy I/O, the UI never blocks, and pipelines are data. In addition, every phase must update the installer (Phase 10), the user docs in `docs/`, and `CHANGELOG.md`.

---

## 1. Feature → phase map and build order

| Phase | Feature | Needs | Earliest start |
|---|---|---|---|
| 17 | Experiment database, cross-plate normalization, assay monitoring | 6, 11, 13 | after Phase 15 |
| 18 | Automatic image QC | 5, 6, 12, 13 | after Phase 17 |
| 19 | Scheduler integration and automatic runs | 17, 18 | after Phase 18 |
| 20 | Oncology assay modules (cell cycle, DNA-damage foci, apoptosis, translocation, proliferation) | 5, 8, 11, 13 | after Phase 13 |
| 21 | 3D spheroid and organoid analysis | 4, 12, 13, 16 | after Phase 16 |
| 22 | Live-cell kinetics | 13, 16, 17 | after Phase 17 |
| 23 | Hit calling and cherry-picking for primary screens | 13, 14, 17, 18, 19 | after Phase 19 |
| 24 | Morphological profiling (Cell Painting style) | 5, 9, 17 | after Phase 17 |
| 25 | Model training, validation and registry | 4, 8, 16, 17 | after Phase 17 |
| 26 | User accounts, audit trail, run locking and sign-off | 17 | after Phase 17 |
| 27 | Server mode | 17, 26 | after Phase 26 |
| 28 | Claude lab assistant | 17 (26 and 27 for logging and server-held key) | after Phase 17 (local), full after Phase 27 |

**Why this order:** the database comes first because auto-run, hit calling, kinetics, profiling, model management, the audit trail, server mode and the assistant all read from or write to it. QC comes before auto-run so unattended runs can fail fast and request re-imaging. Accounts come before server mode because a shared server needs logins. The assistant comes last because it is only as good as the database it queries.

---

## 2. Architecture decisions to lock in before Phase 17

1. **Storage behind interfaces.** SQLAlchemy 2.x models and Alembic migrations from the very first table. SQLite in local single-user mode; PostgreSQL in server mode (Phase 27) with the same code. **SQLite must never sit on a network share used by several PCs** (file locking can corrupt it). Shared use means server mode.
2. **Large data stays in files.** Per-object tables stay as Parquet in run folders, and DuckDB queries them across runs. The database holds the catalog, per-well results, fits, scores and QC.
3. **Paths relative to named data roots.** Store `(root_name, relative_path)` rather than absolute paths. For example, root `SCREENING` = `\\labserver\screening` on Windows and `/mnt/screening` on Linux, so the same registry works on every PC and on the server.
4. **One service layer for every change.** Every state change (layout edit, exclusion, QC override, run start, model promotion, approval) goes through a service function that takes `actor` and `reason` and emits an event. In local mode the actor is the OS user. Phase 26 turns these events into the tamper-evident audit trail without touching the callers.
5. **Job queue behind an interface.** `JobQueue.submit / status / cancel / retry` with a local implementation now and a remote one in Phase 27. Jobs carry priority, owner and resource needs (CPU or GPU, RAM).
6. **Readout registry.** Every module registers its readouts: ID, display name, unit, level (object / field / well), direction of effect, default normalization and a one-line description. Plate maps, pharmacology, QC, hit calling, profiling and the assistant discover readouts from the registry instead of hard-coded column names. **Retrofit the readouts from Phases 5, 13, 16 and earlier into the registry.**
7. **Lab event contract.** A small, versioned JSON Schema package, `lab_events`, defines the events exchanged with the in-house scheduler (Phase 19). Both projects import the same package.
8. **Settings and secrets.** Settings live in a versioned config file. Secrets (API keys, SMTP and webhook credentials, database passwords) go in the OS keychain via `keyring`, never in config files, logs or the repo.
9. **Synthetic data for every phase.** Extend the synthetic plate generator from Phase 15 so every new phase has end-to-end tests with known answers: image artifacts, time courses, spheroids, screening plates with seeded hits, and profiling phenotypes.

---

## 3. Phases

### Phase 17 — Experiment database, cross-plate normalization and assay monitoring

**Goal:** one searchable catalog of every plate, imaging session, layout, analysis run and result, plus normalization across plates and days, replicate checks, and long-term assay monitoring. Build in three steps (17a data model and browser, 17b normalization and concordance, 17c monitoring), stopping after each.

#### 17.1 Data model
- **Project / Experiment:** assay, purpose, owner, dates, notes.
- **Plate:** barcode, plate type, layout version (Phase 11), cell line(s), seeding and treatment date-times, experiment.
- **Acquisition:** one imaging session of a plate (instrument, protocol name, start/end time, time-point index, raw data location, status). One plate can have many acquisitions (time courses, re-imaging).
- **Run:** one analysis of one or more acquisitions (pipeline and parameter hash, software version, models and their hashes, status, output folder, actor).
- **Results:** per-well readouts in long format `(run, well, readout_id, value, n_objects, qc_flag)`, dose–response fits, combination scores and QC metrics.
- **Compound:** ID, name, synonyms, mechanism of action, target, optional SMILES, supplier, lot, stock concentration, solvent.
- **Library plate:** barcode, well → compound, concentration, volume (used by hit calling).
- **Cell line:** name, Cellosaurus RRID, cancer type, passage used, authentication date, mycoplasma test date.
- **Event annotations:** reagent lot change, new cell vial, instrument service, instrument change. These are shown on control charts.
- **Indexing and backfill:** a scanner that indexes existing run folders (manifests from Phase 6) and raw plate folders into the registry. It must be re-runnable and idempotent.

#### 17.2 Experiment browser
- Search by barcode, experiment, cell line, compound, assay, date, instrument and run status, with saved searches.
- Plate page: acquisitions, runs, layout, QC and reports.
- Cross-experiment views:
  - a potency table of compound × cell line across experiments (latest, all runs, mean ± SD, n)
  - a "readout over time" plot for any compound or cell line
  - export to CSV/XLSX
- Compound and cell-line pages showing every result for that compound or line.

#### 17.3 Cross-plate normalization and replicate concordance
- Per-plate methods: control-based (Phase 13), robust Z-score (median and MAD of sample wells), and **B-score** (two-way median polish of rows and columns, residuals scaled by MAD; Brideau et al. 2003).
- **Guard rail:** B-score and other row/column corrections assume compounds are placed randomly with respect to rows and columns. **Block them, with an explanation, on dose–response and combination layouts** where concentration follows rows or columns, because they would remove real signal. Use uniformity-plate correction for those plates instead.
- **Uniformity plates** (vehicle-only plates): build spatial correction maps per instrument + plate type + assay (2D LOESS or polynomial surface), quantify edge effects, and let the user apply a map to later plates.
- **Day and batch effects:** normalize to same-day controls, and show plate-level and day-level effects before and after correction.
- **Replicate concordance:** replicate-plate correlation (Pearson and Spearman), Bland–Altman plots, per-condition CV and IC50 fold difference between replicates. Flag discordant replicates.

#### 17.4 Assay monitoring
- **Control charts** (Levey–Jennings style) per assay / cell line / instrument for Z′, control means and CVs, and log IC50 of reference compounds.
  - Rules are configurable, e.g. a point beyond 3 SD, 2 of 3 points beyond 2 SD, or 7 consecutive points on one side of the mean.
  - Event annotations from 17.1 appear on the charts.
- **Minimum significant ratio (MSR)** for reference-compound potency across runs: MSR = 10^(2√2·s), where s is the SD of log10 IC50 between runs (NCATS Assay Guidance Manual; verify the formula there before implementing).
- **Instrument bridging report:** compare the same plates or reference compounds imaged on two instruments (e.g. Operetta CLS vs Nikon Ji). It shows per-readout agreement (Bland–Altman, correlation), IC50 ratios with confidence intervals, and whether differences fall within the MSR.

#### Done when
- The scanner indexes every existing run folder, and re-running it adds nothing.
- The query "all IC50s for compound X in cell line Y" returns correct values across ≥ 3 runs (test on a synthetic registry).
- B-score removes a synthetic row/column artifact while keeping seeded hits (test), and is blocked on a dose–response layout (test).
- Control charts flag a seeded drift (test), and MSR matches a hand calculation.
- **Ask me for:** our data roots, reference compounds per assay, our compound list and library plate maps, and our cell-line list.

---

### Phase 18 — Automatic image QC

**Goal:** find bad fields, wells and plates automatically before they distort results. Users review the flags, override them with a reason, and see exclusions everywhere downstream.

#### 18.1 Field-level metrics
These are computed per channel on raw data, early in the pipeline.
- **Focus:** power log-log slope (Bray et al. 2012), normalized variance and Laplacian variance, compared with the plate's own distribution (robust Z) and not only with absolute thresholds.
- **Saturation:** % of pixels at the detector maximum (bit depth from metadata), per channel.
- **Intensity:** background level, signal percentiles, and illumination gradient (low-order surface fit).
- **Content:** object count and % area covered, empty-field / no-cell detection, and scratch or cell-free band detection.
- **Artifacts:** debris (very bright, large or irregular objects), bubbles (large ring-like or low-contrast circular regions), out-of-well imaging (field crossing the well edge or meniscus shadow), fibers and lint.
- **Compound-related artifacts**, which matter in drug screens:
  - **compound autofluorescence:** background outside cells rising with concentration in a channel
  - **precipitation:** refractile particles in brightfield or digital phase contrast at high concentrations
  Flag both on the well and carry the flags into pharmacology.
- **Autofocus failure:** best-focus Z position (from metadata or a per-plane focus metric) deviating from the plate trend.

#### 18.2 Object-level QC
- Detect outlier objects that are likely segmentation errors (merged cells, fragments, out-of-focus nuclei) using robust outlier detection on size, shape and intensity. Optionally use coSMicQC-style single-cell QC.
- Flags are stored per object, and flagged objects are left out of aggregates by default.

#### 18.3 Rules and decisions
- A versioned rules file (YAML) is stored with each pipeline. It holds metric thresholds (absolute and plate-relative) and decisions at three levels:
  - **field:** exclude
  - **well:** fail if fewer than N good fields or fewer than M cells
  - **plate:** fail if more than X% of wells fail, if controls fail, or if Z′ falls below threshold
- **Learned QC (optional):** a field-level classifier trained on fields users marked good or bad, reusing the Phase 8 classifier on QC metrics plus a small image embedding. It catches artifacts the rules miss.

#### 18.4 Review UI and propagation
- A QC dashboard per plate with a heat map per metric and a gallery of flagged fields sorted by severity. One click accepts or overrides a flag, and every override needs a reason, recorded through the service layer.
- **Exclusions propagate:**
  - excluded fields and objects are left out of measurements
  - failed wells show as hollow points on dose–response plots and as hatched wells on plate maps
  - reports list every exclusion and its reason
- Outputs: `qc/field_qc.parquet`, `qc/well_qc.csv`, `qc/plate_qc.json` and `qc/qc_report.html`. QC results are also stored in the registry.

#### Done when
- On a labeled set of synthetic artifacts (blur, saturation, bubble, debris, empty field, gradient), each class is detected with ≥ 95% sensitivity and ≤ 5% false positives on clean fields.
- QC on a 384-well plate runs from raw data in the queue and finishes before segmentation starts.
- A test proves that excluded wells are removed from IC50 fits and listed in the report.
- **Ask me for:** 20–50 real fields we consider bad (with reasons) and about 50 good fields per instrument, to tune thresholds.

---

### Phase 19 — Scheduler integration and automatic runs

**Goal:** when a plate finishes imaging, HCS Studio picks the right layout and pipeline, runs QC and analysis, and notifies people with no clicks. It also talks back to our in-house scheduler (PF400 plate handler, washer, imager). It must be **instrument-agnostic**: supporting a new imager means adding a reader and a completeness checker, nothing else.

#### 19.1 Triggers (support both)
- **Scheduler event (preferred):** the scheduler sends a `plate.imaged` event, either as an HTTP POST to HCS Studio's local API or as a JSON file dropped in an events folder. It carries barcode, instrument, protocol, acquisition start/end, raw data location and time-point index.
- **Watch folder (fallback):** watch instrument export locations with `watchdog`. Detect completion safely: no file changes for a configurable quiet period, plus format-specific completeness checks:
  - **ImageXpress:** the number of TIFFs expected from the `.HTD` file
  - **Operetta / Harmony export:** the index XML is present and every referenced image is readable
  - **Nikon `.nd2`** (e.g. the Ji): the file is closed and readable to the last frame
  Never start on a half-copied plate.
- **Ask me** how plates leave each instrument (Harmony export, MetaXpress export, the NIS-Elements save path) before building the watchers.

#### 19.2 Routing rules
- A rules table maps an incoming plate to a pipeline by barcode pattern (prefix or regex), registry lookup (the experiment or assay linked to the barcode), instrument + protocol name, or a default.
- The plate layout is attached by barcode (Phase 11). If no layout is found, image analysis still runs, pharmacology waits, and the owner is asked to annotate the plate.
- **Idempotency:** each acquisition is processed once (key = barcode + acquisition ID or time). Duplicate events are logged and ignored, and a re-imaged plate creates a new acquisition.
- Time courses: an acquisition with a time-point index joins the plate's kinetic dataset (Phase 22).

#### 19.3 Automatic flow
- `event → register acquisition → QC (Phase 18) → if the plate passes QC: pipeline → pharmacology / hit calling as configured → report → notify`.
- If QC fails, notify people and optionally send a `reimage.request` event (wells or fields, plus the reason) to the scheduler.
- Interactive jobs from users jump ahead of auto-runs. Overnight auto-runs may use all GPUs.

#### 19.4 Notifications
- Email (SMTP), Microsoft Teams or Slack incoming webhooks, and desktop notifications.
- Messages include barcode, assay, QC verdict, key numbers (e.g. Z′, mean cell count) and a link to the report. Failures notify the run owner and an admin list.
- **Ask me** which channels the lab uses.

#### 19.5 Event contract with the scheduler (shared `lab_events` package)
- Into HCS Studio: `plate.imaged`, and `plate.dispensed` (treatment time, if the scheduler knows the dispense step).
- Out of HCS Studio: `analysis.started`, `analysis.completed` (QC verdict, report path), `analysis.failed`, `reimage.request` and `cherrypick.request` (Phase 23).
- JSON Schema with a version field. Both projects validate every message, and unknown versions are rejected with a clear error. Include an event-log viewer.

#### Done when
- Dropping a sample plate export into a watched folder, or posting a sample event, triggers QC → analysis → report → notification with no clicks.
- A simulated slow copy is not processed until it is complete, and a duplicate event doesn't start a second run.
- A plate that fails QC produces a `reimage.request` event that validates against the schema.

---

### Phase 20 — Oncology assay modules

**Goal:** one-click, validated analyses for common oncology readouts, with sensible defaults calibrated on each plate's own controls. Build one module per step (20a–20e) and stop after each.

#### 20.1 Assay module framework
Each module is a package in `assays/` containing:
- the required channels, with a channel-mapping step
- a preset pipeline (YAML)
- a **calibration step** that sets thresholds from the plate's control wells (e.g. the 95th–99th percentile of vehicle wells, or a secondary-only control) and stores them in the run
- its registered readouts and module-specific QC checks
- a report page, a docs page, and tests on synthetic data with known answers

A wizard runs it: choose module → map channels → confirm control wells from the layout → preview gating on the controls → run on a single plate or as an auto-run pipeline.

#### 20a Cell cycle (DNA content, optionally with EdU and pH3)
- DNA content = integrated nuclear intensity of DAPI or Hoechst on flat-field-corrected, unsaturated data.
- **Warn when the input is a single confocal plane or a max projection:** neither represents total DNA. Use widefield, a sum projection, or Z coverage of the whole nucleus.
- Per-well histogram normalized so the vehicle G1 peak = 2N. Fit with a Gaussian mixture or a Dean–Jett–Fox-style model, and report % sub-G1, G1, S, G2/M and > 4N (polyploid).
- EdU: 2D gating (EdU vs DNA content) for G1 / S / G2-M. pH3: mitotic index.
- Readouts: phase fractions, G2/M arrest index, % polyploidy, % sub-G1 and mitotic index.

#### 20b DNA-damage foci (γH2AX, 53BP1, RAD51 and others)
- Detect spots inside each nucleus (white top-hat or LoG plus local maxima, with a size filter), with the threshold calibrated on vehicle wells.
- Per nucleus: foci count, total foci area and intensity. Per well: mean foci per nucleus and % of nuclei with ≥ N foci (N configurable).
- Separate **pan-nuclear γH2AX** (bright, diffuse nuclei, often apoptotic or under replication stress) from focal staining.
- Optional cell-cycle-aware analysis combined with 20a (RAD51 foci are expected in S/G2).
- Count foci in 3D when Z-stacks exist, and warn that a max projection merges foci at different depths.

#### 20c Apoptosis and cytotoxicity
- Markers: caspase-3/7 dyes, cleaved caspase-3 immunofluorescence, annexin V, and dead-cell dyes (PI, SYTOX, DRAQ7 or similar).
- Nuclear morphology: condensation (small, bright nuclei) and fragmentation, either rule-based or with the Phase 8 classifier.
- Thresholds come from vehicle and positive-control wells (e.g. staurosporine).
- Readouts: % apoptotic, % dead and viable count. With a time-zero count (Phase 13 GR metrics), separate cytostatic from cytotoxic responses.

#### 20d Nuclear translocation (NF-κB p65, YAP/TAZ, TFEB, β-catenin, FOXO and others)
- Nuclear mask plus a cytoplasm ring of configurable width (µm) that excludes neighboring nuclei and other cells, or the full cytoplasm from cell segmentation.
- Per cell: nuclear/cytoplasmic mean-intensity ratio (and log ratio). Per well: % responding cells above a threshold set from unstimulated controls.
- Agonist EC50 and inhibitor IC50 against stimulated controls, through Phase 13.

#### 20e Proliferation markers (Ki67, EdU, BrdU, pH3)
- % positive by a nuclear intensity threshold set from controls (secondary-only or the vehicle distribution), with intensity distribution plots.
- Optional extras: micronuclei and multinucleation (mitotic errors after antimitotic drugs), using small-object detection near nuclei.

#### Done when (each module)
- Synthetic tests pass, for example:
  - a known G1/G2 ratio is recovered within 2 percentage points
  - known foci counts are recovered within ±1 per nucleus for well-separated foci
  - a known N/C ratio is recovered within 5%
- On a real control plate, thresholds calibrate automatically and positive controls separate from vehicle, with Z′ reported.
- **Ask me for:** one real control plate per assay (vehicle plus positive control), and our current analysis results for the same plate for comparison.

---

### Phase 21 — 3D spheroid and organoid analysis

**Goal:** per-spheroid and per-organoid measurements and dose–response, from brightfield and fluorescence, in 2D and confocal 3D.

#### 21.1 Detection
- Two modes: one spheroid per well (ULA 96/384 plates), and multi-organoid (many objects per well in matrix).
- Brightfield or digital phase contrast detection with ilastik pixel classification (Phase 16), a texture or variance threshold, or SAM 3.1 with a "spheroid" text prompt (Phase 4). Handle well-edge shadows and the meniscus.
- 3D fluorescence segmentation of whole spheroids and, optionally, of nuclei inside them (Cellpose 3D).
- Correct depth-dependent intensity loss (per-plane normalization) in confocal stacks, and warn when a stack doesn't cover the whole spheroid.

#### 21.2 Measurements
- **Size and shape:** area, equivalent diameter, volume estimate, circularity, solidity, compactness and brightfield darkness. Volume comes from a sphere using the equivalent diameter, from an ellipsoid V = π/6 · L · W², or from 3D segmentation when available.
- **Viability:** live/dead staining (e.g. calcein/EthD-1 or Hoechst/PI): live and dead areas or volumes, and dead fraction.
- **Necrotic core:** radial intensity profiles in concentric shells from edge to center, core area and fraction, and viable rim thickness.
- **Invasion** (Matrigel or collagen): core vs total area, invasion area, number and length of protrusions (skeleton analysis), and maximum invasion distance.
- **Organoid mode:** count, size distribution, per-organoid readouts and organoid-forming efficiency.
- **Over time** (with Phase 22): growth curves, growth inhibition, and regrowth after treatment.

#### 21.3 QC and outputs
- QC flags: no spheroid, multiple spheroids, off-center or touching the well edge, out of focus, attached debris, and incomplete Z coverage.
- Readouts are registered for pharmacology (e.g. volume change vs day 0, dead fraction) and for hit calling.
- Outputs under `spheroids/`: per-spheroid tables, radial profiles, overlays, and a gallery that can be sorted by any readout.

#### Done when
- On synthetic spheroids with known radius, core size and invasion area, volume is within ±5% (sphere) and core fraction within ±5 percentage points.
- A 384-well ULA plate runs in batch, with per-well galleries and a dose–response on volume.
- **Ask me for:** one brightfield and one confocal spheroid plate with treatment, and an invasion-assay example if we run them.

---

### Phase 22 — Live-cell kinetics

**Goal:** time-resolved analysis of plates imaged repeatedly.

#### 22.1 Time-course data
- Sources:
  - a single file with a time axis (e.g. `.nd2`)
  - ImageXpress `TimePoint_n` folders
  - separate acquisitions of the same barcode (scheduler re-imaging), linked in the registry (Phase 17)
- Use real timestamps (hours since treatment, from the layout or the `plate.dispensed` event), never frame numbers. Handle missing or irregular time points.
- Correct drift between time points with phase cross-correlation per field.

#### 22.2 Readouts per time point
- Confluence (% area) from label-free brightfield or digital phase contrast (ilastik or Cellpose), nuclear counts (nuclear dye), dead-cell and caspase-positive counts, and any registered readout.

#### 22.3 Kinetic analysis
- Growth curves with exponential or logistic fits: growth rate, doubling time and lag. Also AUC over time, and time to event (e.g. time to half-maximal caspase signal).
- Kinetic dose–response: IC50 at each time point, an IC50(t) curve, and GR values over each time interval using counts at the interval start (Hafner et al. 2016; verify the time-resolved formula against the GR metrics papers).
- Views: per-well sparklines on the plate map, condition × time heat maps, and curve overlays by concentration.

#### 22.4 Single-cell tracking (optional)
- Uses Phase 16 ilastik tracking or a built-in tracker. Ask before adding a tracking library.
- Outputs: division and death events per lineage, cell-cycle duration and fate maps. For motility assays: migration speed and persistence.

#### 22.5 QC
- Focus drift over time, missing time points, evaporation trends in edge wells, and control growth below its historical range (a possible phototoxicity or culture problem).

#### Done when
- On a synthetic time series with a known growth rate, doubling time is recovered within 5%.
- IC50(t) and GR values over time are computed and plotted for a real kinetic plate, with timestamps checked against the instrument metadata.
- **Ask me for:** a kinetic plate from each instrument we use for time-lapse, with the treatment time.

---

### Phase 23 — Hit calling and cherry-picking for primary screens

**Goal:** go from single-concentration screening plates to a reviewed hit list and a cherry-pick file for confirmation.

#### 23.1 Inputs
- Screening plates annotated from library plate maps (Phase 17): source plate barcode + well → compound ID and concentration, mapped to assay-plate wells using the Echo transfer file or plate-reformat rules.

#### 23.2 Scoring
- Per plate: % activity vs on-plate controls, robust Z-score on sample wells, and B-score for positional effects (randomized libraries only; the Phase 17 guard rail applies).
- For replicate screens: SSMD (Zhang 2007) and its robust variant.
- Multi-readout rules, configurable, e.g. "hit on readout A and not cytotoxic by cell count".
- Artifact filters from QC (Phase 18): autofluorescence, precipitation, focus failure and low cell count.
- Optional chemistry filters (e.g. PAINS substructures via RDKit) if SMILES are available. Ask before adding RDKit.

#### 23.3 Hit selection and review
- Thresholds per readout (e.g. robust Z ≤ −3 or ≥ 50% inhibition), replicate rules (e.g. hit in both replicates), and a ranked hit list.
- Views:
  - plate heat maps
  - screen-wide score vs plate order (to catch drift)
  - score distribution with the threshold marked
  - replicate scatter with the hit region
  - a **hit gallery** comparing each hit with neighboring vehicle wells in the linked-contrast comparison view (Phase 14)
- Manual include or exclude, with a reason.

#### 23.4 Cherry-picking and confirmation
- Echo pick list CSV with configurable columns to match our Echo Cherry Pick protocol. Typical columns: Source Plate Name/Barcode, Source Well, Destination Plate Name/Barcode, Destination Well, Transfer Volume (nL).
- Auto-generate confirmation plate layouts (replicate single point, or n-point dose–response via Phase 11 templates, with controls). **Pre-register these layouts** so confirmation plates are analyzed automatically when imaged.
- Optionally send a `cherrypick.request` event to the scheduler (Phase 19).
- Screen summary report: plates screened, Z′ per plate, hit rate, hit list with images, and flags.

#### Done when
- In a synthetic screen with seeded hits and a positional artifact, ≥ 95% of seeded hits are recovered at the chosen threshold with B-score on, and the artifact creates no false hits.
- The generated pick list matches a sample Echo file I provide, and the confirmation layouts are pre-registered.
- **Ask me for:** a library plate map, a sample Echo pick list we use, and our current hit-calling rules.

---

### Phase 24 — Morphological profiling (Cell Painting style)

**Goal:** rich per-cell features → normalized well and treatment profiles → phenotypic activity, similarity and clustering of compounds.

#### 24.1 Feature extraction
- Compartments: Nuclei, Cells, and Cytoplasm (cells minus nuclei).
- Feature groups: AreaShape, Intensity, RadialDistribution, Texture (Haralick at several scales), Granularity, Correlation between channels, Neighbors and Location.
- **CellProfiler-compatible naming** (`Compartment_Category_Feature_Channel[_Scale]`, e.g. `Nuclei_Intensity_MeanIntensity_DNA`) and `Metadata_` columns, so pycytominer infers features without manual mapping.
- A Cell Painting preset (DNA, ER, RNA, AGP, Mito) plus custom channel sets.
- Import of CellProfiler outputs (CSV, SQLite or Parquet; CytoTable optional), so existing CellProfiler data can be profiled here.
- Optional deep-learning embeddings as an alternative feature space. Ask before adding a model and its dependencies.

#### 24.2 Profiling pipeline (pycytominer)
- `aggregate` (median per well) → `annotate` (layout) → `normalize` → `feature_select` → `consensus` (per treatment).
  - `normalize`: `mad_robustize` against vehicle wells per plate by default, with a `spherize` option for batch effects.
  - `feature_select`: variance threshold, correlation threshold, drop NA columns, drop outliers and a blocklist.
- Single-cell QC before aggregation (Phase 18 flags; coSMicQC optional).

#### 24.3 Evaluation and similarity
- **Phenotypic activity:** mean average precision (mAP) of each treatment's replicates against vehicle wells, with p-values and FDR (copairs).
- **Consistency:** mAP within a compound across doses, or within a mechanism of action.
- Similarity:
  - cosine or correlation matrices and clustered heat maps
  - UMAP of consensus profiles, linked to the Phase 9 views
  - "find compounds that look like X"
  - k-nearest-neighbor mechanism-of-action suggestions from an annotated reference set
- Batch-effect checks: whether vehicle wells cluster by plate or day, before and after normalization.

#### 24.4 Outputs
- `profiling/` with profiles at each stage (Parquet), a feature-selection log, mAP tables, similarity matrices, figures and a report.

#### Done when
- The pipeline reproduces pycytominer's outputs on its own example data within float tolerance.
- On the synthetic plate, seeded phenotypes are called active (mAP p < 0.05) and vehicle wells are not.
- **Ask me for:** a Cell Painting (or multi-stain) plate with known reference compounds, and mechanism-of-action annotations for our library if we have them.

---

### Phase 25 — Model training, validation and registry

**Goal:** fine-tune segmentation and classification on our own cell lines, measure accuracy on held-out data, and know which model produced every number.

#### 25.1 Ground truth
- Annotation sets are made with the Phase 4 editing tools, with SAM 3.1-assisted labeling (click → mask → correct) to speed things up.
- Each set is versioned: images, masks or classes, and JSON provenance (annotator, date, source run, cell line, stain).
- Split into train / validation / test **by plate or well**, never by field within the same well, to avoid leakage. Per-cell-line splits are optional.

#### 25.2 Training jobs (in the job queue)
- Cellpose fine-tuning from `cpsam_v2` (or another base model), with configurable epochs, learning rate and batch size. A GPU is needed for practical training times.
- Object and QC classifiers (Phases 8 and 18) retrained from new labels.
- ilastik projects are trained in the ilastik GUI (Phase 16) but registered and validated here. micro-SAM fine-tuning is optional; ask first.

#### 25.3 Validation
- **Segmentation:** average precision at IoU 0.5–0.95; precision, recall and F1 at IoU 0.5; merge and split error counts; per-field count error; and a per-cell-line breakdown.
- **Classifiers:** balanced accuracy, per-class precision and recall, confusion matrix and calibration.
- **Downstream impact:** re-run a reference plate with the candidate model and compare cell counts and IC50s with the production model, judging differences against the MSR (Phase 17).

#### 25.4 Model registry
- Each model records: ID, version, type, base model, training-set version and hash, hyperparameters, validation metrics, author, date, status (candidate → validated → production → retired) and notes.
- Pipelines reference models by registry ID + version, and the run manifest records the model hash.
- Promotion to production needs approval. Phase 26 adds roles; until then, a confirm dialog with a reason.

#### 25.5 Ongoing QA
- For each plate, a random sample of fields is queued for a human spot-check (accept or flag), to track real-world accuracy over time.
- **Drift alerts:** when object features on new plates move far from the training data (e.g. a new cell line), suggest annotation and fine-tuning.

#### Done when
- Fine-tuning on about 20 annotated fields of a new cell line improves AP@0.5 on held-out fields vs the base model, and the comparison report is saved.
- A pipeline pinned to a model version reproduces identical masks, and changing the version is recorded in the manifest.
- **Ask me for:** 20–40 annotated (or correctable) fields from our hardest cell line.

---

### Phase 26 — User accounts, audit trail, run locking and sign-off

**Goal:** know who did what, when and why; lock approved results; support review and sign-off.

#### 26.1 Accounts and roles
- Roles: Viewer, Analyst, Reviewer and Admin, with a permission matrix. For example, only Reviewers approve runs or promote models, and only Admins change QC or routing rules.
- **Local mode:** OS identity plus an optional app password.
- **Server mode:** single sign-on via OIDC (e.g. Microsoft Entra ID) or LDAP / Active Directory. **Ask me** which our institution uses. Local accounts are the fallback, with argon2 password hashing, lockout and session timeout.

#### 26.2 Audit trail
- Every service-layer event (Section 2, item 4) becomes an append-only audit record with:
  - UTC timestamp (server time in server mode)
  - user and role
  - action, object type and ID
  - before/after diff
  - reason (required for exclusions, overrides, deletions, re-analysis of locked runs and model promotion)
  - client machine and app version
- **Tamper evidence:**
  - each record stores a SHA-256 hash of its content plus the previous record's hash (a hash chain)
  - the chain head is periodically written to a separate location
  - a verification tool recomputes the chain and reports any break
  - in PostgreSQL, the application role can insert audit rows but not update or delete them
- An audit viewer filters by user, date, plate, run and action, and exports CSV or PDF. Each run report includes an audit section.
- Claude assistant interactions (Phase 28) are audited too.

#### 26.3 Run states, locking and sign-off
- States: Draft → Completed → Reviewed → Approved (locked). Locked runs are read-only, and any change creates a new run version linked to the original.
- Electronic sign-off for Reviewed and Approved requires re-authentication. The report shows the meaning of the signature, the printed name and the timestamp. Unapproved reports carry a "DRAFT" watermark.
- Retention and backup settings for the registry and the audit log.

#### 26.4 Compliance note (put this in the docs)
- These features support regulated work (e.g. 21 CFR Part 11-style controls) but don't make the system compliant by themselves. Compliance also needs validation (IQ/OQ/PQ), SOPs and training.
- **Ask me** whether any of our data is used for GLP or regulatory submissions before deciding how strict the defaults should be.

#### Done when
- Every action in a scripted end-to-end scenario appears in the audit log with user and reason.
- Editing an audit row directly in the database is detected by the verification tool (test).
- A locked run can't be changed, and re-analysis creates a linked new version.

---

### Phase 27 — Server mode

**Goal:** heavy processing on a shared GPU workstation or server, with browsing, submitting and reviewing from any PC.

#### 27.1 Architecture
- Same codebase, with two modes chosen in settings:
  - **Local:** SQLite and an in-process queue
  - **Server:** PostgreSQL, a REST API and remote workers
- Backend:
  - FastAPI (REST, plus WebSocket for progress), with authentication from Phase 26
  - the job-queue interface backed by a persistent queue: propose options (e.g. a PostgreSQL-backed queue or Redis-based workers) and ask me before choosing
  - one worker per GPU plus CPU workers, with priorities and per-user fair share
- Storage: raw data stays on the lab file share, mounted read-only on the server. Results go to a results share, with paths mapped through data roots (Section 2).

#### 27.2 Clients
- **Desktop app in connected mode:** browse plates, submit runs, watch progress and open results. Images stream as OME-Zarr chunks over HTTP rather than whole files.
- **Web portal** (mostly read-only): experiment browser, run status, HTML reports (Phases 7 and 15), pharmacology tables, and a simple multichannel OME-Zarr image viewer. Propose a viewer library and ask before adding it.
- The scheduler (Phase 19) sends its events to the server instead of a desktop PC.

#### 27.3 Operations
- Deployment with Docker Compose on a Linux GPU server (NVIDIA Container Toolkit), or as a Windows service. **Ask me** which our IT supports.
- HTTPS through a reverse proxy with institutional certificates, internal network only, and secrets via the environment or a secret store.
- An admin page showing queues, workers, GPU and disk usage, and failed jobs.
- Database backups, and upgrades through Alembic migrations with a documented rollback.
- An IT security checklist in `docs/` (ports, accounts, data flows, backups) for review before deployment.

#### Done when
- Two clients submit jobs at the same time and both complete. A worker killed mid-job is restarted and the job resumes.
- A 384-well plate is processed on the server while the submitting laptop is closed, and the results are visible in both the web and desktop clients.

---

### Phase 28 — Claude lab assistant

**Goal:** ask questions about results in plain language and get answers grounded in the database, plus help with pipelines and methods text, with clear control over what data leaves the building.

#### 28.1 Integration
- Uses the **Claude API** through the official Anthropic Python SDK (`anthropic`).
  - It needs an API key from the Claude Console (platform.claude.com), and API usage is billed separately from Claude app subscriptions.
  - In server mode, every call goes through the server, so the key lives only there. In local mode, the key is stored in the OS keychain.
- Model is configurable. Current API model IDs are `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` and `claude-fable-5-2`. Read the "Choosing a model" page, pick a sensible default for routine questions, and allow switching to a stronger model per question.
- **Read these docs before building** (all linked from https://platform.claude.com/llms.txt): tool use overview, define tools, how tool use works, prompt caching, structured outputs, token counting, rate limits, vision, and API data retention.

#### 28.2 Tools the assistant can call
These are implemented in HCS Studio and are read-only.
- `get_schema`: curated, documented views of the registry (not the raw tables).
- `run_query`: read-only SQL against those views, with a row limit and timeout. Writes are blocked by a read-only connection, not just by the prompt.
- `get_plate_layout`, `get_run_manifest`, `get_dose_response`, `get_combination_scores`, `get_qc_summary`, `list_readouts`.
- `make_chart`: returns a chart spec that HCS Studio renders locally.
- `highlight_wells` / `open_comparison`: UI actions only (select wells, open the comparison view), with no data changes.
- `propose_pipeline_change`: returns a proposed parameter diff that the user applies manually. **The assistant never changes pipelines, layouts, exclusions or approvals itself.**
- `draft_methods_text`: a methods paragraph generated from run manifests (structured output), for the user to edit.
- `get_image_thumbnail`: **disabled by default.** When enabled for a question, it sends a downsampled 8-bit preview only after the user confirms.

#### 28.3 Grounding and transparency
- Every number in an answer must come from a tool result, and the UI shows the query or tool call behind each answer ("show work").
- When data is missing, the assistant says so instead of estimating.
- A "data that will be sent" preview panel. Curated views replace file paths and user names with IDs. Free-text notes are left out unless the user includes them.
- Compound names and structures may be confidential, so add an option to send compound IDs only.

#### 28.4 Governance
- The assistant is off until an admin enables it.
- Per-user monthly token budget, with usage shown in settings, and prompt caching for the system prompt and schema to cut cost.
- Every question, tool call and answer is recorded in the audit trail (Phase 26).
- Docs note: check our institution's data policy and the Anthropic API data-retention terms before enabling, including whether an enterprise agreement or zero-data-retention terms are needed.

#### 28.5 Tests
- Tool plumbing tested against a mocked client.
- An opt-in evaluation set of about 30 questions with known answers on the synthetic registry (e.g. "Which combinations had a ZIP score above 10 in both cell lines?"), run against the live API with my key. Answers are checked against SQL ground truth.
- Security tests: write statements via `run_query` are rejected, and the image tool can't run without confirmation.

#### Done when
- ≥ 90% of answers on the evaluation set match the numbers from SQL ground truth, and every answer shows its supporting queries.
- A test that logs outgoing payloads proves nothing beyond the curated views leaves the app.

---

## 4. New dependencies (ask before installing; pin versions)

| Purpose | Packages |
|---|---|
| Database and migrations | `sqlalchemy`, `alembic`, `duckdb`, `psycopg` (server mode) |
| Watching and events | `watchdog`, `jsonschema`, `httpx` |
| Statistics | `statsmodels` (LOESS, mixed models) |
| Profiling | `pycytominer`, `copairs`; optional `coSMicQC`, `cytotable` |
| Tracking (optional) | a tracking library, proposed in Phase 22 |
| Chemistry (optional) | `rdkit` (PAINS filters) |
| Server | `fastapi`, `uvicorn`, a queue library (chosen in Phase 27), `authlib` or `ldap3`, `argon2-cffi` |
| Secrets | `keyring` |
| Assistant | `anthropic` |

---

## 5. Message to send Claude Code

> Read PROMPT.md, PROMPT_ADDON.md, PROMPT_ILASTIK.md and PROMPT_PLATFORM.md, and check the current code against them.
> 1. Tell me which earlier phases are complete, and confirm or adjust the build order in Section 1 of PROMPT_PLATFORM.md based on what exists.
> 2. Apply the architecture decisions in Section 2. Propose the concrete changes (database layer, service layer with actor and reason, job-queue interface, readout registry and retrofit, `lab_events` schema, settings and secrets, synthetic data extensions), update CLAUDE.md, and wait for my approval.
> 3. After I approve, plan **Phase 17** and show me the plan. Then build step 17a, run the full test suite, and report what you built, how to try it, and what you need from me. Stop there.

For each later phase, start a new session with:

> Read CLAUDE.md, CHANGELOG.md and PROMPT_PLATFORM.md. Plan Phase N and show me the plan. Then build it, run the full test suite, update CHANGELOG.md and CLAUDE.md, and stop for my review.
