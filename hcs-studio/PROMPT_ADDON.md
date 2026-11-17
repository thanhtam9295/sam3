# HCS Studio — Add-on Prompt: Phases 11–15
## Plate annotation, pharmacology (dose–response, 6×6 combination matrices, synergy scoring), max projection + background removal, side-by-side comparison and export

> **How to use this file**
> 1. Save it in the project folder as `PROMPT_ADDON.md`, next to `PROMPT.md`.
> 2. Start Claude Code in that folder and select **Claude Opus 5.5** as the model.
> 3. Send the message in **Section 9** of this file. Claude Code reviews it against the existing code and asks before changing architecture. Then it builds one phase at a time and stops after each phase for your review.
> 4. If Phases 0–10 aren't finished yet, these phases can be slotted in earlier: Phase 11 (annotation) and Phase 12 (projection) can be built right after Phase 2. The dependencies are listed in Section 8.

All rules in `PROMPT.md` Section 2 (raw data read-only, provenance, tests with known answers, lazy I/O, UI never blocks, pipelines as data) still apply.

---

## 1. Shared data model additions

Extend the `model/` package with:

```
PlateLayout
 ├─ plate_id / barcode, plate type (96/384/1536), assay name, date, operator
 └─ WellAnnotation (one per well)
     ├─ well            "C07"
     ├─ cell_line       "MCF7"   (+ optional passage, seeding density cells/well)
     ├─ treatments[]    list → supports single agents AND combinations
     │    └─ { compound, concentration, unit (nM/µM/mM/ng/mL/%), vehicle, role }
     ├─ well_role       sample | negative_control (vehicle/DMSO) | positive_control (kill/max effect)
     │                  | untreated | blank (no cells) | empty | excluded
     ├─ replicate_group  auto-derived from (cell_line + treatments), or user override
     ├─ time_point       optional (h)
     └─ notes / custom key–value fields (user-definable columns)
```

- Concentrations are always stored in **molar base units internally** (float, M) and displayed in the user's unit. Unit conversion lives in one tested utility.
- A well with two treatments is a combination well. Order is irrelevant: (A,B) is the same as (B,A).
- The layout is saved as a sidecar `plate_layout.json` (with a CSV/XLSX export) next to the output, never inside raw data, and is linked to the plate by barcode or path hash.
- The layout is **versioned**: edits create a new version, and every analysis run records which layout version it used in `run_manifest.json`.

---

## Phase 11 — Plate annotation (cell lines, drugs, concentrations)

### 11.1 Plate layout editor (GUI)
- Interactive plate grid (96/384/1536) beside the image plate map. Click, shift-click, drag-rectangle and row/column-header selection.
- An **annotation panel** for the selected wells: cell line, compound(s), concentration, unit, role, custom fields. Apply to the selection.
- **Layer views**: color the plate by cell line, compound, concentration (log color scale), role, or replicate group. Show a legend. Hover tooltip shows the full annotation.
- **Fill helpers**:
  - *Dilution series*: start concentration, dilution factor (e.g. 1:3, half-log), number of points, direction (left→right / top→bottom / reverse), optional replicates per point.
  - *Combination matrix*: drug A along rows and drug B along columns, N×M concentrations (**default 6×6, including the 0 concentration for each drug** so the single-agent curves sit on the matrix edges). Supports replicate matrices and placement anywhere on the plate.
  - *Controls*: mark vehicle/DMSO, positive control and blank wells. Edge-well exclusion toggle.
  - Copy/paste annotations between regions or plates. Undo/redo.
- **Templates**: save the layout as a reusable template (e.g. "384w 8-drug 10-pt DRC", "384w 4× 6×6 combo") and apply it to new plates. Change only the cell line or compound names when needed.
- **Validation** before analysis: wells missing a cell line or role, samples without matching vehicle controls on the plate, inconsistent units, combinations missing single-agent points, fewer than 3 controls. Show warnings with the affected wells highlighted.

### 11.2 Import / export
- Import from CSV/XLSX in two formats:
  - *Long format*: one row per well, with columns `well, cell_line, compound_1, conc_1, unit_1, compound_2, conc_2, unit_2, role, ...`.
  - *Plate-map format*: a grid sheet per field (e.g. sheet "compound" and sheet "concentration", laid out as a plate).
- A column-mapping dialog for unfamiliar files.
- Optional import of an **Echo / D300 dispense file** (CSV with source compound, destination well, transfer volume) to compute final concentrations from stock concentration and assay volume. Ask me for a sample file before building this parser.
- Export: long CSV/XLSX, plate-map XLSX, and a printable PNG/PDF plate-layout figure.

### 11.3 Integration
- The annotation appears in the image browser plate map, in well tooltips, in the results `index.html`, in every per-well table (annotation columns are joined automatically), and in PCA/UMAP coloring.
- Plate selection in all views can filter by annotation (e.g. "all MCF7 wells treated with drug X ≥ 100 nM").

*Done when*: a 384-well 6×6 combo layout and a 10-point DRC layout can each be built in under 2 minutes with the helpers; round-trip CSV import → export → import gives an identical layout (test); validation catches each seeded error in a test layout.

---

## Phase 12 — Maximum projection for all planes + background removal after projection

### 12.1 Projection module
- Projection over Z for **every field, channel and well** in the scope (field / well / selection / plate), run in the batch queue.
- Methods: **maximum intensity (default)**, mean, sum, min, standard deviation, and **extended depth of field** (wavelet- or variance-based focus stacking) as an option.
- Plane selection: all planes (default), a range, or **auto-select in-focus planes** using a focus metric (e.g. normalized variance or Brenner) with a user-set number of planes around the best focus. Store the chosen planes per field.
- Order of operations is configurable in the pipeline. Default: `flat-field → (optional) AF subtraction → (optional) deconvolution → projection → background removal`.
- Output: per-field 2D OME-TIFF (float32 or the original bit depth, user-selectable) plus 8-bit preview PNGs. Optional **well mosaic** of projected fields (using the stitching from Phase 8).

### 12.2 Background removal after projection
Methods, selectable per channel, with live before/after preview and a line-profile tool:
- **Rolling ball / sliding paraboloid** (radius in µm, converted using pixel size).
- **Morphological top-hat** (white top-hat, disk radius in µm).
- **Gaussian / median background** (large-kernel blur subtracted or divided).
- **Polynomial / spline surface fit** to background pixels (for illumination gradients).
- **Constant offset**: percentile of background or mean of blank (no-cell) wells from the plate annotation.
- **BaSiC-style flat-field + dark-field** estimated from projected images across the plate (if not already done before projection).
- Clip negatives to 0 and keep float output. Record the method and parameters in the manifest.
- Guard rails: warn if the rolling-ball radius is smaller than about 2× the typical object diameter (it would erase signal). Measure and report the % change in mean intensity inside objects versus background. Default to measuring intensities on background-subtracted linear data, never on contrast-stretched images.

*Done when*: synthetic Z-stack tests pass (MIP equals the known maximum; a background-removal test on a synthetic gradient plus spots keeps spot-integrated intensity within ±5%); a full 384-well 3-channel 20-plane plate projects in the batch queue with resume.

---

## Phase 13 — Pharmacology: dose–response, 6×6 heat maps, combination scoring

This phase uses per-well readouts from the existing analysis (cell/nuclei count, % positive class, mean/integrated intensity, colocalization, cluster fraction, PCA/UMAP distance-to-control, or any column in `per_well_summary`) together with the Phase 11 annotation.

### 13.1 Readout selection and normalization
- The user picks one or more readouts (e.g. "nuclei count" for viability, "% cleaved-caspase-3 positive" for apoptosis).
- Normalization per plate, per cell line:
  - **% of vehicle control** (vehicle = 100%, or 0% effect).
  - **% inhibition** using vehicle (0%) and positive control (100%).
  - **GR metrics** (growth-rate inhibition, Hafner et al. 2016) when the user supplies time-zero cell counts (a t=0 plate or wells) or a doubling time. Recommended for proliferation assays because it corrects for different division rates between cell lines.
- **Plate QC**: Z′-factor, SSMD, signal window, CV of controls, and a control-well scatter plot. Flag plates with Z′ < 0.5 (threshold configurable). Show a row/column trend heat map to spot edge or dispense effects.
- Outlier handling: optional robust exclusion (e.g. median ± 3 MAD within a replicate group). Excluded points stay visible as hollow markers and are listed in the report. Manual exclude/include by clicking a point.

### 13.2 Dose–response curves (single agents)
- Fit **4-parameter logistic (4PL)** by default, with optional 3PL (fixed bottom or top) and 5PL (asymmetric). Use `scipy.optimize.least_squares` with bounds or `lmfit`, fitting on log10(concentration). Use robust loss as an option.
- Report per (cell line × compound × readout):
  - **IC50/EC50** (relative) and **absolute IC50** (concentration at 50% of control), each with **95% CI** (from the covariance matrix, or bootstrap when chosen)
  - **Hill slope**, top, bottom, **Emax** (max observed and fitted effect)
  - **AUC** (trapezoidal, on log-concentration, normalized to the tested range) and **activity area**
  - **GI50, GR50, GRmax, GR_AOC** when GR is enabled
  - R², RMSE, number of points and replicates, and fit flags (e.g. "IC50 extrapolated beyond tested range", "curve not converged", "flat response")
- Plots: mean ± SD (or SEM) per concentration with individual replicate points and the fitted curve on a log x-axis. Overlay multiple compounds or cell lines on one plot. Optional small-multiples grid (compound × cell line).
- Comparison: log-IC50 shift between cell lines or conditions, with an extra-sum-of-squares F-test for whether the IC50s differ.

### 13.3 Combination matrices (6×6 and other N×M sizes)
- Detect combination blocks automatically from the annotation (drug A × drug B × cell line), including the single-agent rows/columns at concentration 0.
- **6×6 heat maps** for each combination:
  1. **Response matrix** (% inhibition or viability) with values printed in each cell.
  2. **Synergy matrix** for the selected model, using a diverging color map centered on 0 (synergy red, antagonism blue, or a user-chosen palette).
  3. Optional 3D synergy surface and a matrix of the replicate SD.
- Axis labels in display units with ascending concentration. Show the 0 row/column. Missing wells are marked, not interpolated silently.

### 13.4 Combination (synergy) scoring
Implement these reference models, each with a per-cell matrix and summary scores:
- **Bliss independence**: expected E_AB = E_A + E_B − E_A·E_B (effects as fractions); excess = observed − expected.
- **Highest Single Agent (HSA)**: excess over max(E_A, E_B).
- **Loewe additivity**: expected from the single-agent 4PL fits; per-point **Combination Index (Chou–Talalay CI)** where the fit allows; flag points where Loewe is undefined (e.g. Emax not reached).
- **ZIP (Zero Interaction Potency)**: δ-score as in SynergyFinder (Yadav et al. 2015).
- Optional **MuSyC** (separate potency vs efficacy synergy) using the `synergy` Python package, if it installs cleanly. Ask before adding the dependency.

Summary per combination: mean synergy score over the full matrix, **mean over the most synergistic 3×3 window** (as SynergyFinder reports), max single-cell synergy with its concentrations, and a **statistical test** (replicate-based or bootstrap) against zero. Include a short interpretation label using SynergyFinder conventions (e.g. δ < −10 likely antagonistic, −10 to 10 likely additive, > 10 likely synergistic), and note in the UI that this is a guide, not proof.

**Validation**: reproduce published example values from SynergyFinder test data or a reference matrix I will supply, within a stated tolerance, and keep this as a regression test. Unit tests: a matrix built to be exactly Bliss-additive gives a Bliss score of ≈ 0; a matrix built from two identical drugs (sham combination) gives a Loewe score of ≈ 0.

### 13.5 Pharmacology workspace UI
- A new **"Pharmacology" tab**: choose plates (one or many, merged by annotation), readout, normalization and model. The tree of results groups by cell line → compound / combination.
- Click a curve point or heat map cell → highlight the wells on the plate map and open those wells' images in the comparison view (Phase 14).
- Summary tables: an IC50/AUC table across compounds × cell lines (sortable, with a heat-map style), and a synergy ranking table across all combinations.

### 13.6 Outputs
In the run folder add `pharmacology/`:
```
pharmacology/
├── normalized_well_data.csv        # annotation + raw + normalized readouts per well
├── plate_qc.csv                    # Z′, SSMD, CVs per plate
├── dose_response_fits.csv          # all parameters, CIs, flags
├── combination_scores.csv          # summary per combination, all models
├── combination_matrices/<cellline>_<drugA>_x_<drugB>_{response,bliss,hsa,loewe,zip}.csv
├── figures/ (SVG + PNG, 300 dpi): curves, 6×6 heat maps, synergy surfaces, QC plots
└── pharmacology_report.html (+ PDF export)
```
Also export a **Prism-compatible XY table** (CSV laid out as concentration × replicate columns) for each curve, so results can be checked or re-plotted in GraphPad Prism.

*Done when*: fits match GraphPad Prism 4PL (IC50 within 5%, Hill within 0.1) on 5 curves I provide; Bliss/HSA/Loewe/ZIP match SynergyFinder on a reference 6×6 matrix within the stated tolerance; the full report generates from a 384-well combo plate in under a minute after image analysis is done.

---

## Phase 14 — Side-by-side image comparison & export

### 14.1 Comparison view
- Pick wells (and fields) from the plate map, from an annotation filter (e.g. "MCF7, drug X, all concentrations"), from a dose–response point or a heat map cell, or from a PCA/UMAP lasso.
- Layouts: 1×2, 2×2, 1×N strip, N×M grid. **Auto-arrange by annotation** (e.g. concentration ascending left→right, cell line by row; or recreate the 6×6 combo matrix as a 6×6 image grid).
- Linked navigation: **synchronized zoom, pan, Z-plane and channel toggles** across all panels (lockable per panel).
- **Linked display settings (default ON)**: identical LUTs, min/max and gamma for each channel across all panels, so images are directly comparable. Show a visible warning badge if the user unlinks contrast for one panel.
- Image source per panel: raw, AF-corrected, deconvolved, enhanced, max projection, background-removed, with optional overlays (masks, class colors, cluster colors, object outlines).
- A **label bar** on each panel built from the annotation (well, cell line, compound + concentration), with a template the user can edit.
- Optional field picker per well: auto-pick the field closest to the well median of the selected readout (a "representative field"), or pick manually.
- Synchronized crop: draw an ROI in one panel → the same region is cropped in all panels (useful for zoom-in insets).

### 14.2 Export
- **Figure montage** export: PNG/TIFF (choice of 300/600 dpi), **SVG/PDF** (vector labels and scale bars over raster images), with:
  - scale bar (length in µm, position, color, thickness), optional time/Z stamp
  - channel legend with names and colors, per-panel labels, column/row headers from the annotation
  - panel spacing, background color, font and size settings, and journal-friendly presets (single column ~85 mm, double column ~180 mm)
  - split-channel montage option (each channel in grayscale + merge), as in a typical figure panel
- **Data export** for the compared panels: 16-bit/float OME-TIFF per panel with the same display metadata, plus a CSV of the shown wells' readouts.
- **Movie export** (MP4/GIF) for Z-scroll or time series with synchronized panels, if the source is 3D/time-lapse.
- **Export presets**: save and reuse settings. Every export writes a small JSON sidecar recording the source files, processing steps and display settings, so a figure can be regenerated exactly.
- Batch export: "export a montage for every compound in this plate" (one figure per compound, concentrations as columns, cell lines as rows).

*Done when*: a 2×6 comparison with linked contrast exports to SVG and PNG with correct scale bars (checked against known pixel size in a test); the export sidecar regenerates a pixel-identical PNG.

---

## Phase 15 — Integration, reports and regression tests

- Add all new modules to the **pipeline builder** and **CLI** (e.g. `hcs-studio pharm --run <run_folder> --readout nuclei_count --normalize percent_inhibition --models bliss,loewe,zip`).
- Extend the clickable results `index.html`: plate map colored by annotation or normalized readout; a pharmacology section with curves and 6×6 heat maps; clicking a matrix cell or curve point opens that well's images; links to the comparison montages.
- A **single experiment report** (HTML + PDF): plate layout figure, QC, dose–response table and curves, combination heat maps and scores, representative image montages, and methods text auto-generated from the manifest (software version, algorithms, parameters).
- A regression test suite with a synthetic 384-well plate (synthetic images + known dose–response + a known-additive and a known-synergistic combination) that runs end to end in CI.
- Update the user docs in `docs/`: annotation guide, pharmacology methods (formulas and references for 4PL, GR, Bliss, HSA, Loewe, ZIP, Z′), comparison and export guide.

---

## 8. Dependencies between phases

| New phase | Needs | Can start after |
|---|---|---|
| 11 Annotation | Phase 1 (plate model, plate map) | Phase 1 |
| 12 Projection + background | Phases 1–2 (readers, flat-field) | Phase 2 |
| 13 Pharmacology | 11 + per-well readouts (Phase 5) + run folder (Phase 6) | Phase 6 |
| 14 Comparison/export | 11 (labels), 12 (projected images), viewer from Phase 1 | Phase 12 |
| 15 Integration | 11–14 + results report (Phase 7) | Phase 14 |

Suggested new library additions (ask before installing anything else): `lmfit` (curve fitting), `openpyxl` (XLSX import/export), `synergy` (optional, MuSyC), `reportlab` or `weasyprint` (PDF reports), `imageio[ffmpeg]` (movie export).

---

## 9. Message to send Claude Code

> Read PROMPT.md and PROMPT_ADDON.md. Check the current state of the code against both. Then:
> 1. Tell me which phases from PROMPT.md are complete, and propose where Phases 11–15 should fit, using the dependency table in Section 8 of PROMPT_ADDON.md.
> 2. Update CLAUDE.md with the new data model (Section 1) and the new dependencies, and propose any architecture changes, with reasons. Wait for my approval.
> 3. After I approve, build **Phase 11 only**. Write the tests listed under "Done when", run the full test suite, and report what you built, how to try it, and what sample files you need from me (e.g. an existing plate-layout spreadsheet, Echo/D300 dispense file, Prism reference curves, a reference 6×6 combo matrix).
