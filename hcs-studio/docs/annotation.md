# Plate annotation (Layout tab)

The **Layout** tab describes what is in every well: cell line, treatments (one compound,
or several for combinations, each with a concentration, unit and vehicle), role, passage,
cells per well, time point, notes and custom fields. Opening a plate loads the latest saved
layout of that plate; the plate map, results, report and tables then show it.

**Selecting wells.** Click a well; Shift+click or drag for a rectangle; Ctrl+click to add;
click a row letter or column number for the whole row or column. The filter bar selects
wells by annotation, e.g. cell line *MCF7*, compound *Drug X*, ≥ 100 nM; the same bar is
on the Plate and Results tabs.

**Editing.** The form on the right shows the first selected well. Editing a field ticks
its box; **Apply ticked fields to selection** changes only those fields in every selected
well (so you can set the cell line of a whole plate without touching its doses). Undo and
redo (Ctrl+Z / Ctrl+Y), copy and paste (Ctrl+C / Ctrl+V, pasted at the first selected
well) work on whole annotations.

**Fill helpers.**

| Helper | What it does |
|---|---|
| Dilution series | from the first selected well, left → right or top → bottom: top concentration, dilution (1:2, 1:3, 1:10, ½-log or a custom factor), number of points, replicates in the next rows/columns, highest concentration first or last |
| Combination matrix | at the first selected well: drug A down the rows, drug B across the columns (6 × 6 by default), each with its own top concentration and dilution; with 0 included, the first row and column are the single agents and the corner is the vehicle control; replicate matrices to the right or below, with a gap |
| Controls | marks the selection as vehicle controls (e.g. "DMSO 0.1"), positive controls or blanks (no cells) |
| Exclude edge wells | marks the outer ring as excluded, and restores it when unticked |

A 384-well 6 × 6 combination plate (three replicate matrices plus control columns) takes
three helper actions; a plate of eight 10-point compounds in duplicate takes nine.

**Colour by** cell line, compound, concentration (log scale; single agents in molar
units — combinations and % vehicles stay grey), role or replicate group;
tooltips show the full annotation of each well.

**Concentrations and units.** Enter any unit (M, mM, µM, nM, pM, g/L, mg/mL, µg/mL,
ng/mL, %, …). Values are stored in base units (mol/L, g/L, fraction) and shown in the unit
you entered; molar and mass units are never mixed unless a molecular weight is given.
Treatments are order-free: *A + B* and *B + A* are the same combination. A concentration
of 0 means the drug is absent (the single-agent edge of a matrix).

**Replicate groups** are derived from cell line + treatments (and role for controls), so
replicate wells group automatically; type a group name to override.

**Validate** lists problems; click one to select its wells:

| Problem | Level |
|---|---|
| wells with cells but no cell line | error |
| sample wells without a treatment | warning |
| treated wells marked empty or blank | error |
| no vehicle (negative) controls, or fewer than three | error / warning |
| one compound entered in different unit kinds (molar and mass) | error |
| one compound shown in different units | warning |
| a combination matrix without the single-agent rows/columns | warning |

**Versions.** **Save version** stores the layout as version 1, 2, … for this plate in
`~/.hcs_studio/layouts/` (or `HCS_STUDIO_HOME`); versions are never overwritten. A new run
uses the layout you see: unsaved edits are saved as a new version when the run starts. The
run folder keeps a copy (`plate_layout.json`, `plate_layout.csv`) and `run_manifest.json`
records its version and SHA-256. Changing names, doses or roles does not prevent resuming
a run; only unstained/blank wells (used by autofluorescence correction) do.

**Templates.** Templates → Save as template keeps the layout pattern; Apply template
asks for the new cell line and compound names (e.g. the same combination layout for a
different drug pair).

**Import and export.** Import… (or File → Import plate layout) reads:

- a long table (CSV or XLSX), one row per well: `well`, `cell_line`, `role`,
  `compound_1`, `conc_1`, `unit_1`, `vehicle_1`, `compound_2`, … , `passage`,
  `seeding_density`, `replicate_group`, `time_point_h`, `notes`; unknown columns become
  custom fields. Common names are recognised (`drug`, `dose`, `concentration`, `Drug A`,
  `conc_b`, `cells_per_well`, …); if the columns are not clear, a dialog asks what each
  column holds. Older layouts with `compound`, `concentration`, `treatment` and
  `control_type` (`negative`, `positive`, `unstained`, `blank`) still work.
- a plate-map workbook: one sheet per field (`cell_line`, `compound_1`, `conc_1`, …), each
  laid out like the plate (row letters down, column numbers across);
- a saved layout (`.json`).

Export writes the long CSV or XLSX (metadata in `# key: value` lines or a `plate` sheet),
the plate-map XLSX, a figure of the current colouring (PNG or PDF) or JSON. CSV → import →
export gives the identical file. Two examples to try: `docs/examples/combo_6x6_384.csv`
(three 6 × 6 matrices with control columns) and `docs/examples/drc_10pt_384.csv` (eight
compounds, 10 points in duplicate).

From the command line:

```bash
hcs-studio layout layout.xlsx --validate
```

```bash
hcs-studio layout layout.csv --export platemap.xlsx --plate-map
```

```bash
hcs-studio layout layout.csv --export layout.pdf --layer concentration
```

```bash
hcs-studio annotate <run folder> --layout corrected_layout.csv
```

`annotate` applies a corrected layout to a finished run: tables, profiles and the report
are rebuilt without reprocessing any image (it refuses if unstained/blank wells changed).

## How the annotation is used

- **Runs** copy the version they used (`plate_layout.json`, `plate_layout.csv`) and record
  its version and SHA-256 in `run_manifest.json`; `per_well_summary.csv` gets the
  annotation columns right after `well` (see [measurements.md](measurements.md)).
- **Controls**: wells marked *unstained* drive the autofluorescence-reference step;
  *negative control* (vehicle) and *positive control* wells drive robust Z, Z′ and the
  pharmacology normalization; *blank* wells can set a constant background.
- **Pharmacology** reads cell lines, compounds, concentrations (base units) and roles
  ([pharmacology.md](pharmacology.md)); combination matrices are found from wells with two
  drugs. Vehicles in % (v/v) and the treatments of control wells are not counted as drugs.
- **Comparison and figures** build headers and labels from the annotation
  ([figures.md](figures.md)).
- A corrected layout can be applied to a finished run with `hcs-studio annotate`.
