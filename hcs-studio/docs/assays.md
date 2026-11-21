# Oncology assays

One-click analyses of common oncology readouts, calibrated on each plate's own control wells.
Each assay is a step that runs after segmentation and measurement (it reads the object tables,
so it can be applied to a finished run without reprocessing images). It needs a plate layout
with **vehicle** (negative-control) wells; without one it calibrates on all wells and says so.

Available: **cell cycle** (20a), **DNA-damage foci** (20b), **apoptosis and cytotoxicity**
(20c), **nuclear translocation** (20d). Coming: proliferation markers (20e).

## The assay wizard

**Tools → Assay wizard…** (with a run open in Results):

1. choose the assay;
2. map its channels to the plate's channels (e.g. DNA → DAPI, EdU → AF488);
3. check the control wells it found in the plate layout (fix them in the Layout tab);
4. **Preview calibration on the controls**: the vehicle histogram and the calibration;
5. **Apply to this run** (the run's tables, `index.html` and an assay report are updated; the
   change is recorded with your reason), or **Add to the pipeline** for future runs.

From the command line:

```bash
hcs-studio assay list
```

```bash
hcs-studio assay preset cell_cycle --map dna=DAPI --map edu=EdU --channels DAPI,EdU,pH3 --z-planes 1 --output cell_cycle.yaml
```

```bash
hcs-studio assay run <run folder> cell_cycle --map dna=DAPI --map edu=EdU --map ph3=pH3 --reason "first look"
```

Each assay writes `assays/<name>/` in the run: `calibration.json` (thresholds and how they were
set), a per-well CSV and `report.html` (linked from `index.html`).

## Cell cycle (DNA content, EdU, pH3)

**Imaging.** DNA content is the *integrated* nuclear intensity of DAPI or Hoechst, on
background-corrected, unsaturated images. It needs the whole nucleus: use widefield images, or a
**sum** projection of Z planes covering the nucleus. A single confocal plane or a maximum
projection does not represent total DNA; the step warns when the run used one.

**Preset pipeline** (`hcs-studio assay preset cell_cycle`): sum projection (with Z planes),
top-hat background removal (radius 2.5 × the nucleus diameter), nuclear segmentation with a
*triangle* threshold, measurement, and the cell-cycle step. The low threshold matters: masks must
contain the whole nucleus, and Otsu's threshold rises on G2/M-rich (arrested) fields and trims
dim G1 nuclei, which biases their DNA content (seen on synthetic plates: G1 CV 13 % instead of
4 %, and 9 % false sub-G1).

**Calibration** (vehicle wells): the G1 peak of the vehicle nuclei is set to 2N (`cc_dna_n` per
nucleus). EdU-positive = above Otsu's threshold of log EdU intensity in the vehicle wells; pH3-
positive = above median + 5 robust SD of log pH3 intensity in the vehicle wells. Give fixed
thresholds with `edu_threshold` / `ph3_threshold` (e.g. from a secondary-only control).

**Per well:**

- a Dean–Jett–Fox-style fit to the nuclei between `sub_g1_below` (1.5N) and `polyploid_above`
  (5N): Gaussian G1 and G2/M peaks (G2/M at 1.85–2.15 × G1) with the same CV, and an S phase
  uniform between them, broadened by that CV; maximum likelihood on a 140-bin histogram;
- % sub-G1 (< 1.5N), % G1, % S, % G2/M, % > 4N (> 5N);
- with EdU, 2D gating (`cc_edu_<phase>_pct`): EdU+ = S, EdU− below the midpoint of the peaks
  = G1, above = G2/M;
- G2/M arrest index = % G2/M of the well / % G2/M of the vehicle wells (EdU gating when present);
- mitotic index = % pH3-positive nuclei;
- the well's G1 peak (`cc_g1_peak_n`, 2 = as the vehicle) and CV (`cc_g1_cv_pct`).

**Warnings** (in `cc_warnings` and the report): G1 CV above 8 % (broad peaks: staining, focus,
segmentation), the G1 peak far from 2N (staining differs from the vehicle wells), the G2/G1
ratio at its limit, fewer than `min_nuclei` (200) nuclei.

| Readout | Meaning |
|---|---|
| `cc_sub_g1_pct`, `cc_g1_pct`, `cc_s_pct`, `cc_g2m_pct`, `cc_polyploid_pct` | phase fractions (model fit) |
| `cc_edu_<phase>_pct`, `cc_edu_pct` | phase fractions by EdU gating; % EdU-positive |
| `cc_g2m_arrest_index` | % G2/M relative to the vehicle wells |
| `cc_mitotic_index_pct` | % pH3-positive |
| `cc_g1_peak_n`, `cc_g1_cv_pct`, `cc_nuclei`, `cc_warnings` | fit quality |
| per nucleus: `cc_dna_n`, `cc_phase`, `cc_edu_positive`, `cc_ph3_positive` | |

## DNA-damage foci (γH2AX, 53BP1, RAD51 …)

Two steps: **`detect_foci`** finds the foci in the images (it must run before
`measure_objects`, so it is part of the pipeline — a finished run without it cannot be
analysed for foci), and **`foci`** summarises them per nucleus and well.

```bash
hcs-studio assay preset foci --map dna=DAPI --map foci=gH2AX --channels DAPI,gH2AX --output foci.yaml
```

**Detection** (`detect_foci`): the foci channel is band-pass filtered — Laplacian of Gaussian
scaled by σ² (default) or a white top-hat — with σ from `diameter_um` (typical focus diameter,
default 0.7 µm). Pixels above the threshold inside nuclei are split at local maxima (seeded
watershed) and kept when their size is plausible (`min_area_um2`; `max_area_um2`, by default a
disc twice the focus diameter). Each focus belongs to its nucleus, so `measure_objects` reports
`foci_count`, `foci_mean_area_um2` and the foci's total intensity per nucleus.

**Threshold**: calibrated on the vehicle wells — median + `threshold_sd` (5) × robust SD
(1.4826 × MAD) of the filter response over the vehicle nuclei's pixels, from up to
`calibration_fields` (24) fields; the MAD ignores the few foci vehicle nuclei have. It is stored
in the run (manifest → `fitted_state`). Set `threshold` to use a fixed value instead.

**Z stacks**: without a projection the detection is 3D (the Z sigma follows the pixel size /
Z step), so a focus spanning several planes counts once and two foci above each other count
twice. A maximum projection merges foci at different depths into one: the foci step warns when
the run used a projection.

**Per well** (`foci` step):

| Readout | Meaning |
|---|---|
| `dd_foci_per_nucleus`, `dd_foci_per_nucleus_median` | foci per nucleus (pan-nuclear nuclei left out) |
| `dd_positive_pct` | % nuclei with ≥ `positive_min_foci` (5) foci |
| `dd_pan_nuclear_pct` | % pan-nuclear nuclei: mean nuclear intensity of the foci channel ≥ `pan_factor` (4) × the vehicle nuclei's median — bright, diffuse γH2AX (apoptosis, replication stress), not counted as focal |
| `dd_focus_area_um2`, `dd_foci_intensity` | mean focus area; summed foci intensity per nucleus (nuclei with foci) |
| `dd_foci_fold_vs_vehicle` | foci per nucleus / vehicle foci per nucleus |
| `dd_foci_per_nucleus_<g1|s|g2m>`, `dd_positive_<g1|s|g2m>_pct` | by cell-cycle phase, when the `cell_cycle` step runs first (RAD51 foci are expected in S/G2) |
| per nucleus: `dd_foci`, `dd_positive`, `dd_pan_nuclear` | |

For cell-cycle-aware foci, put `cell_cycle` before `foci` among the table steps.

## Apoptosis and cytotoxicity (caspase-3/7, annexin V, PI, SYTOX, DRAQ7)

One table step, **`apoptosis`**, on the nuclei. Channels: the DNA stain (required: nuclei and
their morphology), an **apoptosis marker** (caspase-3/7 dye, cleaved caspase-3, annexin V;
optional) and a **dead-cell dye** (PI, SYTOX, DRAQ7; optional). It runs on a finished run.

```bash
hcs-studio assay preset apoptosis --map dna=Hoechst --map apoptosis=Casp37 --map dead=DRAQ7 \
    --channels Hoechst,Casp37,DRAQ7 --output apoptosis.yaml
hcs-studio assay run runs/plate1 apoptosis --map dna=Hoechst --map apoptosis=Casp37 --map dead=DRAQ7
```

The preset segments nuclei with a **triangle** threshold (dim healthy nuclei and bright
condensed ones alike) and measures them; add a cell segmentation step if the marker is a
membrane stain (below).

**Thresholds** — calibrated on the plate's controls and stored in the run
(`assays/apoptosis/calibration.json`):

| `threshold_method` | Threshold |
|---|---|
| `vehicle_robust` (default) | median + `threshold_sd` (5) robust SD (1.4826 × MAD) of log intensity of the vehicle nuclei — unaffected by the few apoptotic cells every vehicle well has |
| `vehicle_percentile` | the `vehicle_percentile` (99) of the vehicle nuclei — simple, but with more than 1 % positive vehicle cells it falls inside the positives |
| `controls` | Otsu's threshold of log intensity over the vehicle and positive-control nuclei (staurosporine, etc.) |

Fixed values: `apoptosis_threshold`, `dead_threshold`.

**Annexin V** and other membrane stains: segment cells too (cells grown from the nuclei share the
nucleus's ID) and set `marker_object_type: cells`; each nucleus takes its cell's intensity.

**Nuclear morphology**: a nucleus is *condensed* (pyknotic) when its area is ≤ `condensed_area`
(0.6) × and its DNA brightness ≥ `condensed_brightness` (1.5) × the vehicle medians.
Brightness is the nucleus's 95th-percentile DNA intensity (`dna_feature`; the mean of a small
nucleus is diluted by its dim edge pixels). Objects ≤ `fragment_area` (0.25) × the vehicle area
are *fragments* of fragmented nuclei: reported (`ap_fragments`) but not counted as cells. Or use
a trained classifier (Phase 8): `morphology_column: class_morphology` and `apoptotic_classes`.
`morphology: false` uses the markers alone.

**Categories** per nucleus:

| Category | Dead-cell dye | Apoptosis marker or condensed |
|---|---|---|
| viable | − | − |
| early apoptotic | − | + |
| late apoptotic | + | + |
| necrotic | + | − |

**Cytostatic vs cytotoxic**: give a **time-zero count** — wells fixed at the time of treatment
with time point 0 h in the plate layout (their median viable count; they are never used as
vehicle), or `time_zero_count`. Then GR = 2^(log2(x/x₀) / log2(x_vehicle/x₀)) − 1 on viable
counts (Hafner et al. 2016): **GR < 0 cytotoxic** (fewer viable cells than at treatment),
**0 ≤ GR < `no_effect_gr` (0.8) cytostatic**, above: no effect.

**Per well**:

| Readout | Meaning |
|---|---|
| `ap_viable_pct`, `ap_early_apoptotic_pct`, `ap_late_apoptotic_pct`, `ap_necrotic_pct` | categories (% of nuclei, fragments apart) |
| `ap_apoptotic_pct`, `ap_dead_pct` | early + late apoptotic; late apoptotic + necrotic |
| `ap_condensed_pct` | condensed nuclei (or the classifier's apoptotic classes) |
| `ap_viable_count`, `ap_viable_vs_vehicle` | viable cells; / the vehicle median |
| `ap_gr`, `ap_response` | GR; cytotoxic / cytostatic / no effect |
| `ap_nuclei`, `ap_fragments` | nuclei analysed; fragments |
| per nucleus: `ap_category`, `ap_condensed`, `ap_marker_positive`, `ap_dead_positive` | |

QC: the report (`assays/apoptosis/report.html`) gives the thresholds, the time-zero count and
Z′ of % apoptotic (positive control vs vehicle, with ≥ 2 wells each); a warning when neither
marker is mapped.

## Nuclear translocation (NF-κB p65, YAP/TAZ, TFEB, β-catenin, FOXO …)

Two steps: **`cytoplasm`** builds a cytoplasm compartment for each nucleus (an image step, before
`measure_objects`), and **`translocation`** compares the reporter's mean intensity in the
nucleus and in its cytoplasm.

```bash
hcs-studio assay preset translocation --map dna=Hoechst --map reporter=p65 --channels Hoechst,p65 --output p65.yaml
```

**Cytoplasm** (`cytoplasm` step):

| `mode` | Cytoplasm |
|---|---|
| `ring` (default) | pixels between `gap_um` (0.5) and `gap_um + width_um` (2) from the nuclear edge, assigned to the **nearest** nucleus — a ring never enters a neighbouring nucleus or the half of the gap that belongs to the neighbour. With a cell segmentation (`cells`: cells grown from the nuclei, e.g. `expand_cells` or Cellpose cells linked to nuclei) the ring also stays inside its own cell. |
| `cells` | the whole cell minus its nucleus (and the gap) — needs `cells` |

The gap keeps the blurred nuclear edge out of the cytoplasm. Cells whose cytoplasm is smaller
than `min_cytoplasm_um2` (2 µm²) — crowded or cut by the image edge — are left out and counted
(`tl_no_cytoplasm_pct`).

**Intensities must be linear and background-subtracted.** An offset left in the image pulls every
N/C ratio towards 1. The preset subtracts a **constant**: the mean of blank (no-cell) wells
(`blank_wells=True` in `preset`, wells marked blank in the layout), else the 5th percentile of
each field. The step warns when the reporter channel had no background subtraction, a top-hat or
rolling ball with a radius under 40 µm (flattens the cytoplasm), or a polynomial/spline surface
(rises into the cytoplasm under dense cells and inflates the ratios — on the synthetic test plate
the median N/C came out 7× too high). The preset segments nuclei with **Otsu** (mask edge at half
height); the triangle threshold of the other assays puts the mask edge in the cytoplasm.

**Controls and threshold** — `mode` follows the % inhibition convention of the pharmacology
module (negative control = 0 % effect, positive control = 100 %):

| `mode` | Unstimulated | Stimulated | Dose-response |
|---|---|---|---|
| `agonist` (default) | vehicle (negative control) | positive control, e.g. TNF-α | agonist EC50 |
| `inhibitor` | positive control (no stimulus, or a reference inhibitor) | vehicle = stimulus alone | inhibitor IC50 |

`unstimulated_wells` overrides the layout. A cell **responds** when its N/C ratio is above:

| `threshold_method` | Threshold |
|---|---|
| `unstimulated_robust` (default) | median + `threshold_sd` (3) robust SD of log N/C of the unstimulated cells |
| `unstimulated_percentile` | the `percentile` (95) of the unstimulated cells |
| `controls` | Otsu's threshold of log N/C over unstimulated and stimulated cells |

or a fixed `ratio_threshold`. The calibration is stored in `assays/translocation/calibration.json`;
the report shows the N/C histograms of both controls with the threshold.

**Per well**:

| Readout | Meaning |
|---|---|
| `tl_responding_pct` | % cells with N/C above the threshold |
| `tl_nc_ratio`, `tl_nc_ratio_mean`, `tl_log2_nc` | median and mean N/C; mean log2 N/C |
| `tl_activation_pct` | % responding scaled from unstimulated (0 %) to stimulated (100 %) controls |
| `tl_fold_vs_unstimulated` | median N/C / median N/C of the unstimulated wells |
| `tl_cells`, `tl_no_cytoplasm_pct` | cells with a usable ratio; nuclei without usable cytoplasm |
| per cell: `tl_nuc_intensity`, `tl_cyto_intensity`, `tl_nc_ratio`, `tl_log2_nc`, `tl_responding` | |

Z′ of % responding and of mean log2 N/C (stimulated vs unstimulated, ≥ 2 wells each) is in the
report, with a warning below 0.5.

**EC50 / IC50**: annotate the dose series in the layout and run the pharmacology module on
`tl_responding_pct` (or `tl_log2_nc`) with **% inhibition** normalization:

```bash
hcs-studio pharm runs/plate1 --readout tl_responding_pct --normalize percent_inhibition
```

In `agonist` mode the curve rises from the vehicle (0 %) to the stimulus (100 %) and its
midpoint is the EC50; in `inhibitor` mode it is the IC50 against the stimulated controls.

## Validating on your plates

Run each assay on a control plate (vehicle plus a positive control, e.g. nocodazole for a
G2/M arrest) and compare with your current analysis of the same plate. The thresholds calibrate
on the vehicle wells automatically; the positive control should separate from the vehicle
(Z′ of the readout in the Results tab).
