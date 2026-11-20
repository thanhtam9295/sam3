# Experiment registry

HCS Studio keeps a catalog of plates, imaging sessions, analysis runs and their results, so
you can search across experiments, compare potencies between runs and follow a readout over
time. Large data (images, per-object tables) stays in the run folders; the registry holds
the catalog and per-well results.

## The Experiments tab

| Part | What it does |
|---|---|
| Search bar | plates by barcode, plate name, experiment or assay, cell line and compound; **Save search…** keeps a search for later (the drop-down) |
| **Scan folder…** | index the run folders in a folder and all folders below it |
| Plates | the plates found, newest run first; select one for its page: acquisitions (imaging sessions), runs, layout roles and plate QC (Z′, CV). **Open the latest run in Results** opens it |
| Potency | EC50 per compound × cell line × readout across runs: number of runs, mean and SD of log10 EC50, geometric mean, the latest value and run; **Export…** to Excel or CSV |
| Readout over time | mean ± SD of a per-well readout per run, for wells of a role (e.g. vehicle), a compound (any dose) and a cell line |
| Compounds and cell lines | every compound and cell line in the registry, with their plates and dose–response fits |

## Scanning

Runs enter the registry when they are scanned (Experiments → **Scan folder…**, or the
command line). Scanning only reads; it never changes a run folder. Run it as often as you
like: runs that did not change since the last scan are skipped, a changed run (re-annotated,
re-analysed) replaces its own results, and nothing is ever duplicated.

```bash
hcs-studio db scan "D:\Results" --reason "import of 2026 screens"
```

`--raw` also registers raw plate folders that have not been analysed yet (as a plate and
an acquisition). Numeric per-well columns that the readout registry does not know are stored
as they are and listed in the scan report.

## Command line

```bash
hcs-studio db info
```

```bash
hcs-studio db search --cell-line MCF7 --compound "Drug A"
```

```bash
hcs-studio db ic50s --compound "Drug A" --cell-line MCF7 --readout nuclei_count
```

```bash
hcs-studio db potency --readout nuclei_count --export potency.xlsx
```

```bash
hcs-studio db plate P00123
```

## Normalization across plates (Experiments → Normalization)

| Method | What it does | Use on |
|---|---|---|
| % of control | 100 × value / median of the plate's vehicle wells | any plate with vehicle wells |
| % inhibition | 100 × (vehicle − value) / (vehicle − positive control), medians | plates with both controls |
| Robust Z | (value − median of the sample wells) / (1.4826 × MAD of the sample wells) | screens; any layout |
| B-score | two-way median polish of the sample wells (row and column effects removed), residuals ÷ 1.4826 × MAD (Brideau et al. 2003) | **randomized** screening plates only |
| Uniformity map | divides each well by a correction map from a vehicle-only plate | any layout, including dose–response and combinations |

**Guard rail.** B-score assumes compounds are placed at random with respect to rows and
columns. On dose–response layouts (a compound diluted along a row or a column: ≥ 3
concentrations in one row or column, monotonic) and on combination matrices, the row and
column "effects" are the dose effect itself, and B-score would remove real signal. HCS
Studio refuses it there and says why; use a uniformity map instead.

**Uniformity maps.** Image a plate with vehicle in every well, analyse it, then **Save as
uniformity map…** (or `hcs-studio db uniformity`). The map is a smooth surface (polynomial,
degree 2 by default) plus a step for the outer ring of wells (edge evaporation is a step,
not a smooth trend), fitted robustly so single bad wells do not bend it. The map reports
the edge effect (% edge vs interior), the CV and row/column trends before and after.
Apply it to later plates of the same instrument, plate type and assay (method
*Uniformity map*). Maps are kept in the registry with who made them.

**Day and plate effects.** For the plates listed, the share of the vehicle wells' variance
explained by day and by plate (one-way R²), raw, normalized to same-day controls, and
normalized per plate. A day effect that disappears after same-day normalization is a
day-to-day shift; a plate effect that remains needs per-plate normalization.

**Replicates.** Two runs of the same layout: Pearson and Spearman correlation of the sample
wells, Bland–Altman bias and 95 % limits of agreement, the CV of each condition, and the
fold difference of each IC50. The pair is flagged *discordant* when r < 0.8 or an IC50
differs more than 3-fold (both adjustable).

```bash
hcs-studio db normalize SCREEN-1 --readout nuclei_count --method b_score --export bscores.csv
```

```bash
hcs-studio db uniformity UNIF-1 --readout nuclei_count --name "Operetta 384 viability" --reason "monthly uniformity plate"
```

```bash
hcs-studio db concordance P00123 P00124 --readout nuclei_count
```

```bash
hcs-studio db batch --readout nuclei_count --experiment "DRC week 1"
```

## Assay monitoring (Experiments → Monitoring)

Control charts (Levey–Jennings) follow an assay over time, per assay, cell line and
instrument. Choose a metric and a readout, narrow it with the filters, and press **Show
chart**. Always filter by assay (or experiment) when one instrument runs several plate
types: their Z′ values belong to different charts.

| Metric | One point per run |
|---|---|
| `z_prime` | Z′ of the run's negative and positive controls |
| `vehicle_mean`, `vehicle_cv_pct` | mean and CV % of the negative-control wells |
| `positive_mean`, `positive_cv_pct` | mean and CV % of the positive-control wells |
| `log_ec50` | log10 EC50 (M) of a reference compound in one cell line |

**Limits** come from the first 20 runs (the *baseline*, changeable): centre line = their
mean, ±1, 2 and 3 SD dashed. With fewer runs than the baseline, the limits come from all
runs and the chart says so.

**Rules** (comma-separated; flagged points turn red):

| Rule | Fires when |
|---|---|
| `1_3s` | one point beyond 3 SD |
| `2of3_2s` | 2 of the last 3 points beyond 2 SD on the same side |
| `7x` | 7 consecutive points on one side of the centre |
| `6t` | 6 points steadily rising or falling (optional) |

The general forms are `<n>of<m>_<k>s`, `<n>_<k>s`, `<n>x` and `<n>t`.

**Events** (new reagent lot, cell vial, instrument service, instrument change, other) are
drawn as vertical lines on the charts they concern: an event with an empty instrument,
assay or cell line applies to all of them. Record one with **Record event…** or the
command line; it is kept in the registry with who recorded it.

### Minimum significant ratio (MSR)

For a reference compound the chart shows the MSR of its EC50 across runs:

- single compound tested in several runs: MSR = 10^(2·√2·s), s = SD of log10 EC50;
- two runs of a set of compounds (test–retest): d = log10 EC50 run 1 − run 2 per compound;
  MR = 10^mean(d), MSR = 10^(2·SD(d)), limits of agreement LsA = 10^(mean(d) ± 2·SD(d)).
  Reproducible when MSR < 3 and both limits lie within 1/3 … 3;
- a rolling MSR over the last 6 runs shows whether reproducibility is drifting.

Two EC50 values from different runs differ significantly when their ratio exceeds the MSR.
The formulas follow Eastwood et al. 2006 (J Biomol Screen 11:253). **They have not yet
been checked against the text of the NCATS Assay Guidance Manual** (the manual could not
be read from the build environment); confirm them before relying on MSR decisions.

### Instrument bridging report

When an assay moves to another instrument, image the same plates on both and scan both
runs. **Bridging report…** (or `db bridge`) pairs the plates by barcode and writes a
folder with `bridging_report.html` (self-contained), `bridging.json`, `agreement.csv`
and `potency_ratios.csv`:

- per plate: Pearson and Spearman correlation of the wells, Bland–Altman bias and 95 %
  limits (A − B, in the readout's unit);
- per compound and cell line: EC50 ratio A/B with a 95 % CI (from the two fits' CIs,
  combined in log space) and whether it lies within 1/MSR … MSR;
- overall: mean ratio, limits of agreement and MSR across the pairs.

The instruments are called equivalent when every ratio and both limits of agreement lie
within the MSR (3 by default; use the assay's own MSR when known).

### Command line

```bash
hcs-studio db chart --metric z_prime --readout nuclei_count --assay "Viability 384" --instrument "Operetta CLS"
```

```bash
hcs-studio db chart --metric log_ec50 --compound Staurosporine --cell-line MCF7 --readout nuclei_count --rules 1_3s,2of3_2s,7x,6t --export chart.csv
```

```bash
hcs-studio db msr --compound Staurosporine --cell-line MCF7 --readout nuclei_count --window 6
```

```bash
hcs-studio db event --kind reagent_lot --on 2026-03-21 --instrument "Operetta CLS" --description "new Hoechst lot" --reason "lot change"
```

```bash
hcs-studio db bridge --instrument-a "Operetta CLS" --instrument-b "Nikon Ji" --readout nuclei_count --msr 3 --output bridge/ --reason "instrument change"
```

`db bridge` exits with status 1 when the instruments are not shown to be equivalent.

## Data roots: the same paths on every computer

The same share is `\\labserver\screening` on Windows and `/mnt/screening` on Linux. Define
it once as a **data root** in `~/.hcs_studio/settings.json`:

```json
{
  "config_version": 1,
  "data_roots": [
    {"name": "SCREENING", "windows": "\\\\labserver\\screening", "linux": "/mnt/screening"}
  ]
}
```

The registry and run manifests then store `SCREENING:raw/P00123` instead of an absolute
path, and every computer finds the data at its own location. Paths outside every root are
kept as `ABSOLUTE:…` (they work only on the computer that recorded them).

## Where the registry lives

Locally, the registry is `~/.hcs_studio/registry.sqlite` on this computer's own disk.
**Never put it on a network share used by several computers**: SQLite databases there get
corrupted. HCS Studio refuses to open one on a network drive. A registry shared by several
computers needs server mode (PostgreSQL), which comes in Phase 27.

## Changes are recorded

Every change — saving a layout version, re-annotating a run, writing pharmacology with
exclusions, starting, cancelling or retrying a run, training a model, changing settings,
scanning — is recorded with who made it (your login), when, on which object, the state
before and after, and why. Manual exclusions need a reason (the app asks; on the command
line use `--reason`). Phase 26 turns this record into a tamper-evident audit trail.

## Secrets

Passwords and keys (for e-mail notifications, the database server, the Claude assistant)
are stored in the operating system's keychain (Windows Credential Manager, macOS Keychain,
the Linux secret service), never in settings files, logs or run folders. On a server
without a keychain, set them as environment variables `HCS_SECRET_<NAME>`.
