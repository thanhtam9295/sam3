# Primary screens: hit calling

From single-concentration screening plates to a reviewed hit list (Phase 23a), the views to
review it (Phase 23b), and cherry-picking the hits onto confirmation plates (Phase 23c).

## 1. Library plate maps and assay layouts

A **library plate map** (CSV or XLSX, one row per well) lists what each library (source) plate
holds: plate barcode, well, compound, concentration and optionally volume and SMILES. Column
names are recognised loosely (`Plate Barcode`, `Source Plate`, `Compound ID`, …); the
concentration unit may be in its own column, in the header (`conc_uM`) or in the value
(`10 mM`).

```bash
hcs-studio screen library library_map.xlsx --name "Kinase set"
```

stores the plates, wells and compounds in the registry (a plate imported again is replaced).

**Assay layouts** put the library compounds into the sample wells of a **template** that holds the
controls (save one in the layout editor, e.g. vehicle in columns 1–2 and the positive control in
23–24):

| Rule | Use |
|---|---|
| `stamp` | library well → same assay well; `--dest LIB=ASSAY` names the assay plate; `--dilution` = final / library concentration (10 mM stock at 1:1000 → `--dilution 0.001`) |
| `quadrants` | four 96-well library plates → one 384-well plate (`--sources A,B,C,D`): plate k goes to quadrant (k // 2, k % 2), i.e. 96-well (r, c) → 384-well (2r + k // 2, 2c + k % 2) |
| `echo` | an Echo transfer file (`--transfers`, `--assay-volume-ul`): final concentration = stock × transferred volume / (assay volume + transferred volume); lines before the header are skipped |

```bash
hcs-studio screen layouts library_map.xlsx --template controls.json --rule stamp \
  --dest LIB001=SCR001 --dilution 0.001 --output layouts --register
```

`--register` saves each layout as a version keyed by the assay-plate barcode, so the plate is
annotated automatically when it is imaged and analysed. A library compound that would land on a
control well of the template is refused.

The Echo column names follow Labcyte's transfer files (`Source Plate Barcode`, `Source Well`,
`Destination Plate Barcode`, `Destination Well`, `Transfer Volume`); exports differ between
software versions, so they can be mapped by hand (`read_transfers(..., columns=…)`). This has not
yet been checked against a real Echo file.

## 2. Scores

Every plate and readout is scored separately (`screening/scores.py`):

| Score | Column | Definition |
|---|---|---|
| % activity | `<alias>_pct` | % inhibition between the vehicle (0) and positive-control (100) medians when the plate has positive controls; else % of the vehicle median (`activity: percent_of_control` — use it for readouts the positive control does not change, e.g. cell count) |
| robust Z | `<alias>_z` | (x − median of the sample wells) / (1.4826 · MAD of the sample wells) |
| B-score | `<alias>_b` | median polish of the sample wells (row and column effects removed), residuals / (1.4826 · MAD) (Brideau et al. 2003) |
| SSMD* | `<alias>_ssmd_single` | (x − median) / (√2 · 1.4826 · MAD) of the sample wells (Zhang 2011; single replicate) |
| threshold score | `<alias>_score` | the B-score (`score: b`), robust Z (`z`) or % activity (`pct`) |

**B-score only for randomised libraries**: the guard rail of the normalization module blocks it on
plates with dilution series along rows or columns or with combination matrices (the "row effect"
would be the dose effect); the screen then uses the robust Z for that plate and says so.

Plate quality (`plates.csv`): Z′ = 1 − 3(σ₊ + σ₋)/|μ₊ − μ₋| of the controls and a robust Z′
(medians, 1.4826 · MAD), the hit rate and the plate order (acquisition time, to catch drift).

**Replicates**: for a compound tested on n replicate plates, with d = its value minus each plate's
sample median, SSMD = d̄ / s (method of moments), the UMVUE √(2/(n − 1)) · Γ((n − 1)/2) /
Γ((n − 2)/2) · d̄ / s (n ≥ 3) and the robust median(d) / (1.4826 · MAD(d)) (n ≥ 3) (Zhang 2007).

## 3. Hit rules

A screen configuration (YAML or JSON):

```yaml
name: Kinase screen 1
readouts:
  - {column: reporter_mean, alias: rep}
  - {column: nuclei_count, alias: count, activity: percent_of_control}
score: b                       # B-score; z or pct
hit_rule: "rep_score <= -3 and count_pct > 60"   # active, not cytotoxic
replicate_rule: all            # all / any / majority / at least n replicates
artifact_filters: [qc_failed, low_count, autofluorescence, precipitation, out_of_focus]
rank_by: rep_score             # most negative first (rank_descending: false)
```

The rule is an expression over the score columns (`<alias>_pct`, `_z`, `_b`, `_score`,
`_ssmd_single`, and the raw value under the alias): comparisons (chained, e.g. `-8 < rep_z <= -3`),
`and`, `or`, `not`, arithmetic and `abs(…)`. Nothing else is accepted; a missing value makes a
comparison false.

**Artefact filters** use image QC (Phase 18): a well that failed image QC (`qc_failed`), has too
few cells (`low_count`) or carries one of the listed QC flags is never a hit, whatever its
score.

A compound is a hit when its replicate wells pass the rule as the replicate rule requires
(artefact wells do not count as replicates). Hits are ranked by `rank_by`.

```bash
hcs-studio screen score runs/SCR* --config screen.yaml --output screens/kinase1
```

writes `wells.csv` (every well and score), `compounds.csv` (per compound: replicates, mean scores,
SSMD, hit), `hits.csv` (ranked), `plates.csv` and `screen.json`.

## 4. Manual review

```bash
hcs-studio screen review screens/kinase1 --compound L01-C07 --action exclude --reason "precipitate"
```

A reason is required; decisions are appended to `overrides.json` (the latest per compound wins)
and the hit list is rewritten. A compound can also be included although it missed the threshold.

## 5. Reviewing the screen

Every scoring run writes **`report.html`** into the screen folder (self-contained; opens in any
browser, also offline):

- the summary (plates, compounds, hits, hit rate, artefact wells, manual decisions) and notes;
- **plates**: Z′ and robust Z′ per readout (red below 0.5), whether B-score was used, hit rate;
- **plate heat maps** of the chosen score (blue below 0, red above; % activity 0–100), hits
  outlined, artefact wells grey with a cross, vehicle (blue dot) and positive control (green
  dot); hovering shows the well, compound and value;
- **score vs plate order**: every sample well, the hits in red, the vehicle and positive-control
  medians per plate as lines, the threshold dashed — drift across the screen shows as a trend;
- **distribution** of the sample wells' scores with the threshold;
- **replicates**: each compound's first two clean replicates against each other, the hit region
  shaded, the hits in red;
- the ranked **hit list** with scores, SSMD, replicates, manual decisions and their reasons.

The **hit gallery** (`--gallery N`: the first N hits) shows each hit well beside its two nearest
vehicle wells on the same plate, every image of a row with the **same contrast** (linked, as in
the comparison view) — so a well with fewer or dimmer cells really looks emptier. The images are
the runs' saved corrected images of each well's representative field (runs analysed without
saving images have none; the gallery says so).

```bash
hcs-studio screen score runs/SCR* --config screen.yaml --output screens/kinase1 --gallery 48
hcs-studio screen report screens/kinase1 --gallery 96     # rewrite report and gallery
```

The **Screen** tab of the app opens a screen folder: plate heat maps by plate and score (artefact
wells hatched), the ranked hits, and **Include as hit / Exclude from hits** for the selected
compound (or a compound clicked on the plate map) with a required reason — the same
`screen.review` service as `hcs-studio screen review`; the hit list, report and table are
rewritten. Buttons open the report and the gallery in the browser.

## 6. Cherry-picking and confirmation plates

```bash
hcs-studio screen cherrypick screens/kinase1 --template confirmation_controls.json \
  --mode dose_response --top-um 30 --points 8 --factor 3 --replicates 2 \
  --register --outbox //scheduler/inbox
```

- **Design**: `replicates` (each hit at its screening concentration, or `--concentration-um`, in
  `--replicates` wells spread over the plate — replicates of a compound never touch) or
  `dose_response` (`--points` concentrations from `--top-um` down by `--factor` along a row,
  each replicate series in another row). Hits fill the free wells of the **template** (a layout
  holding the controls); when a plate is full the next one starts (`CONF-001`, `CONF-002`, …;
  `--prefix`).
- **Volumes**: c · V / stock for a final concentration c in the assay volume V
  (`--assay-volume-ul`, 40), rounded to whole Echo droplets (`--droplet-nl`, 2.5). The layout
  holds the concentration actually reached, stock · v / (V + v). A transfer below one droplet, or
  one whose rounding moves the concentration by more than 10 % (e.g. 0.37 µM from 10 mM into
  40 µL is 1.5 nL; one droplet would give 0.62 µM), is skipped and listed: such points need an
  intermediate dilution plate. `--backfill PLATE:WELL` tops every compound well up with vehicle
  to the plate's largest compound volume (equal DMSO in all wells).
- **Source wells** come from the library map (the registry, or `--library map.csv`).
- **Pick list** (`cherrypick/picklist.csv`): Labcyte's Echo Cherry Pick columns by default
  (`Source Plate Barcode`, `Source Well`, `Destination Plate Barcode`, `Destination Well`,
  `Transfer Volume`). `--format format.yaml` sets the columns to match your protocol — each
  column is a field (`source_plate`, `source_well`, `destination_plate`, `destination_well`,
  `volume_nl`, `compound`, `concentration_um`) or fixed text — plus the well style (`A01` or
  `A1`), the separator, the volume decimals and one file per plate:

  ```yaml
  columns:
    - {header: Source Plate Name, value: "Source[1]"}
    - {header: Source Plate Barcode, field: source_plate}
    - {header: Source Well, field: source_well}
    - {header: Destination Plate Barcode, field: destination_plate}
    - {header: Destination Well, field: destination_well}
    - {header: Transfer Volume, field: volume_nl}
  well_format: A1
  ```

- **Confirmation layouts** (`cherrypick/layouts/<barcode>.json`); `--register` saves each as a
  layout version keyed by its barcode, so the confirmation plates are annotated — and analysed
  with pharmacology — automatically when they are imaged.
- `--outbox` / `--scheduler-url` send one `cherrypick.request` event per plate (lab_events) to
  the scheduler.

`summary.html` (rewritten after every scoring and cherry-pick) is the one-page screen summary:
plates screened with Z′, hit rate, the hit list with gallery images, manual decisions, and the
confirmation plates.

## Not yet included

PAINS and other chemistry filters need RDKit (not installed; ask before adding). The Echo
column names and transfer-file parsing have not yet been checked against real Echo files.
