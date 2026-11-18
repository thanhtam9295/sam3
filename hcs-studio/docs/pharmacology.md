# Pharmacology methods

The **Pharmacology** tab and `hcs-studio pharm` turn per-well readouts of finished runs
(any column of `per_well_summary`, e.g. `nuclei_count`, a class fraction, an intensity,
or a profile distance) and the plate layout (Layout tab) into normalized responses, plate
QC, dose–response fits and combination (synergy) scores. Results go to
`<run>/pharmacology/`.

## Wells

- Drugs are the active treatments of a well (concentration > 0). Vehicles in % (v/v),
  e.g. DMSO 0.1 %, and the treatments of control wells are not drugs.
- Vehicle wells: role *negative control* (or *untreated* without drugs). Positive
  controls: role *positive control*. Wells marked *empty*, *blank* or *excluded* are left
  out. Wells with time point 0 h are time-zero references (GR), not samples.
- Several runs (plates) are merged by annotation; normalization stays per plate.

## Normalization (per plate and cell line)

Relative to the median of that cell line's vehicle wells on the same plate (if a cell line
has none, all vehicle wells of the plate are used, with a note):

| Method | Value | Effect (used by synergy models) |
|---|---|---|
| % of control | 100 · x / vehicle | 100 − value |
| % inhibition | 100 · (vehicle − x) / (vehicle − positive) | value |
| GR (Hafner et al. 2016) | 2^(log2(x/x0) / log2(vehicle/x0)) − 1 | 100 · (1 − GR) / 2 |

x0, the time-zero value, comes from time-zero wells of the same cell line on the plate, a
time-zero run (`--time-zero`), or a doubling time Td and assay duration T:
x0 = vehicle / 2^(T/Td). GI (NCI) = 100 · (x − x0)/(vehicle − x0).

## Plate QC

From the raw readout of the controls: Z′ = 1 − 3(σp + σn)/|μp − μn| (Zhang et al. 1999),
SSMD = (μp − μn)/√(σp² + σn²), signal window = (|μp − μn| − 3(σp + σn))/σn, CV = 100 σ/μ.
Plates below the Z′ threshold (default 0.5) are flagged. The QC figure shows the controls
and the normalized values across the plate (edge, gradient and dispense effects).

## Outliers

Optional: wells beyond median ± k·MAD (MAD × 1.4826) of their replicate group (cell line +
treatments, ≥ 3 wells, across plates). The MAD is never taken as smaller than that of the
cell line's vehicle wells, because with three replicates it can be close to 0 by chance.
k = 3 still flags about 2 % of ordinary wells in triplicates; k = 4–5 removes gross
outliers only. Excluded wells stay in all tables (and as hollow points), are listed in
`excluded_wells.csv`, and are left out of fits and matrices. Wells can be excluded or
included by hand (Ctrl+click a point, or `--exclude PLATE:WELL`).

## Dose–response curves

Fitted on X = log10(concentration in the base unit: M, g/L or fraction) with GraphPad
Prism's parametrisation:

- 4PL: Y = Bottom + (Top − Bottom)/(1 + 10^((LogEC50 − X) · Hill))
- 5PL: Y = Bottom + (Top − Bottom)/(1 + 10^((LogXb − X) · Hill))^S, with
  LogEC50 = LogXb + log10(2^(1/S) − 1)/Hill
- 3PL: Bottom (0; −1 for GR) or Top (100; 1 for GR) fixed.

Top ≥ Bottom is reported; a falling curve has a negative Hill slope (as in Prism).
Least squares (`scipy.optimize.least_squares`, several starting points), optionally
robust (soft L1). 95 % intervals: covariance matrix with the t distribution (n − p
degrees of freedom; EC50 intervals computed on log10 and therefore asymmetric), the delta
method for the absolute IC50, or a bootstrap (resampling within concentrations).

| Output | Definition |
|---|---|
| EC50 / IC50 (relative) | the curve's midpoint |
| absolute IC50 | where the curve crosses 50 (% of control or % inhibition; 0.5 for GR) |
| Emax | largest effect of the concentration means (observed) and at the highest tested concentration (fitted) |
| AUC | trapezoidal area of the mean responses over log10 concentration ÷ tested log range (= the average response, in the response's unit) |
| activity area | area between the fitted curve and no effect over the tested range, in effect fraction × log10 units |
| GR50, GRmax, GR_AOC, GEC50, GRinf, h_GR | GR curve with Top = 1: GR50 where GR = 0.5, GRmax = fitted GR at the highest concentration, GR_AOC = area over the mean GR values ÷ range |
| GI50 | where the GI curve crosses 50 |
| R², RMSE, n | fit quality, points, concentrations, replicates per concentration |

Flags: *too few concentrations*, *curve not converged*, *flat response* (the curve is not
better than a horizontal line, F test p > 0.05, or changes by less than 3 × RMSE over the
tested range), *EC50 extrapolated*, *absolute IC50 not reached / extrapolated*, *fewer than
5 concentrations*.

**EC50 shifts** between cell lines (same compound and readout): Δlog EC50 and an
extra-sum-of-squares F test of separate fits against a global fit sharing LogEC50
(`ic50_comparisons.csv`).

## Combination matrices

Blocks are found from the annotation: for each cell line and drug pair with wells
containing both drugs, the matrix spans the drugs' concentrations in those wells plus 0;
the 0 row and column are the single-agent wells at those concentrations, and the 0 × 0
cell the cell line's vehicle wells. Missing wells stay missing (× in heat maps, NaN in
CSVs). The k-th well of each cell forms replicate matrix k.

Responses are % inhibition (effect). Synergy = observed − expected, in % points:

| Model | Expected response E |
|---|---|
| Bliss | yA + yB − yA·yB/100 (observed single agents) |
| HSA | max(yA, yB) (observed single agents) |
| Loewe | solves a/ICA(E) + b/ICB(E) = 1, ICX = inverse of the 4PL fit of drug X's single-agent responses (the 0 point included as a dose far below the lowest); where no solution exists (an effect beyond what the drugs reach), HSA is used and the point counted in a note |
| ZIP δ (Yadav et al. 2015) | each row is fitted as a 4PL in drug B with its lower asymptote fixed at drug A's fitted single-agent response and its top at 100 (columns likewise); the observed-fitted surface is the mean of row and column fits; δ = surface − (yA + yB − yA·yB/100) with fitted single agents |

Combination index (Chou–Talalay, generalised to the 4PL fits): CI = a/ICA(y) + b/ICB(y) at
the observed response y; < 1 synergy, > 1 antagonism; undefined where a drug alone never
reaches y.

Summary per combination and model, over the combination cells (both drugs > 0): mean,
mean of the most synergistic 3 × 3 window, maximum cell (with its concentrations), and,
with replicate matrices, a one-sample t test of the replicate means against 0 (p and 95 %
CI). Labels (SynergyFinder): < −10 likely antagonistic, −10 to 10 likely additive, > 10
likely synergistic — a guide, not proof. MuSyC is not included (it needs the `synergy`
package).

**Differences from SynergyFinder to check with a reference matrix**: SynergyFinder 3 can
apply a baseline correction and fits single agents with `drc` (LL.4); HCS Studio uses
observed single agents for Bliss and HSA and its own 4PL fits for Loewe and ZIP.

## Outputs (`pharmacology/`)

| File | Content |
|---|---|
| `normalized_well_data.csv` | annotation, raw value, normalized value, effect and exclusion per well and readout |
| `plate_qc.csv` | Z′, SSMD, signal window, CVs, flags per plate and readout |
| `dose_response_fits.csv` | all curve parameters, intervals and flags |
| `ic50_comparisons.csv` | EC50 shifts between cell lines (F test) |
| `combination_scores.csv` | per combination: all models' mean, 3 × 3 maximum, maximum cell, p, CI, label |
| `combination_matrices/…_{response,sd,n,ci,bliss,hsa,loewe,zip}.csv` | rows drug A, columns drug B, in display units |
| `prism/<readout>/<cell line>_<compound>.csv` | Prism XY tables: X = log10 concentration (M), Y1…Yn replicates |
| `excluded_wells.csv` | wells left out, with the reason |
| `figures/` | curves (per cell line and a compound × cell line grid), combination heat maps, QC, optional 3D synergy surfaces (PNG 300 dpi + SVG) |
| `pharmacology_report.html` / `.pdf` | the report: options, QC, exclusions, fits, curves, combinations, methods |
| `pharmacology.json` | options, runs, software version, notes, file list |

## References

Hafner M et al. (2016) Growth rate inhibition metrics correct for confounders in
measuring sensitivity to cancer drugs. *Nat Methods* 13:521. — Zhang JH et al. (1999) A
simple statistical parameter for use in evaluation and validation of high throughput
screening assays. *J Biomol Screen* 4:67. — Yadav B et al. (2015) Searching for drug
synergy in complex dose–response landscapes using an interaction potency model. *Comput
Struct Biotechnol J* 13:504. — Ianevski A et al. (2022) SynergyFinder 3.0. *Nucleic
Acids Res* 50:W739. — Chou TC, Talalay P (1984) *Adv Enzyme Regul* 22:27. — Bliss CI (1939)
*Ann Appl Biol* 26:585. — Loewe S (1953) *Arzneimittelforschung* 3:285.
