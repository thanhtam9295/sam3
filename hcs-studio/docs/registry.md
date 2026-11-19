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
