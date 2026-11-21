# Oncology assays

One-click analyses of common oncology readouts, calibrated on each plate's own control wells.
Each assay is a step that runs after segmentation and measurement (it reads the object tables,
so it can be applied to a finished run without reprocessing images). It needs a plate layout
with **vehicle** (negative-control) wells; without one it calibrates on all wells and says so.

Available: **cell cycle** (20a), **DNA-damage foci** (20b). Coming: apoptosis (20c), nuclear
translocation (20d), proliferation markers (20e).

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

## Validating on your plates

Run each assay on a control plate (vehicle plus a positive control, e.g. nocodazole for a
G2/M arrest) and compare with your current analysis of the same plate. The thresholds calibrate
on the vehicle wells automatically; the positive control should separate from the vehicle
(Z′ of the readout in the Results tab).
