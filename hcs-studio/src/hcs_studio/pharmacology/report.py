"""Writes ``pharmacology/``: tables, matrices, Prism XY tables, figures and the report.

```
pharmacology/
├── pharmacology.json                  options, runs, software version, files
├── normalized_well_data.csv           annotation + raw + normalized readouts per well
├── plate_qc.csv                       Z′, SSMD, signal window, CVs per plate and readout
├── dose_response_fits.csv             all parameters, CIs, flags
├── ic50_comparisons.csv               log-EC50 shifts between cell lines (F test)
├── combination_scores.csv             summary per combination, all models
├── excluded_wells.csv                 wells left out of fits, with the reason
├── combination_matrices/[<readout>/]<cell line>_<drug A>_x_<drug B>_{response,sd,bliss,…}.csv
├── prism/<readout>/<cell line>_<compound>.csv   X = log10 concentration, Y1…Yn replicates
├── figures/  (PNG 300 dpi + SVG): curves, grids, combination heat maps, QC, surfaces
├── pharmacology_report.html
└── pharmacology_report.pdf
```
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.model import units
from hcs_studio.pharmacology.analysis import (
    CurveResult,
    PharmResult,
    ReadoutResult,
    combination_tables,
    curves_table,
    options_dict,
)
from hcs_studio.pharmacology.combination import THRESHOLD, CombinationResult, Matrix
from hcs_studio.pharmacology.normalize import LABELS, plate_trends

COLOURS = (
    "#2a6fdb",
    "#e4572e",
    "#29a36a",
    "#8e44ad",
    "#f2a541",
    "#17becf",
    "#7f7f7f",
    "#bcbd22",
    "#d62728",
    "#1f77b4",
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "x"


def _display(values: Any, unit: str) -> np.ndarray:
    try:
        return np.array([units.from_base(float(v), unit) for v in np.atleast_1d(values)])
    except (ValueError, KeyError):
        return np.asarray(values, dtype=float)


# ------------------------------------------------------------------- tables


def normalized_wells(result: PharmResult) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for readout, rr in result.readouts.items():
        frame = rr.data.wells.copy()
        frame["treatments"] = frame["treatments"].map(
            lambda ts: " + ".join(f"{c} {units.format_concentration(v, u)}" for c, v, u in ts)
        )
        keep = [
            "plate",
            "run",
            "well",
            "row",
            "col",
            "cell_line",
            "role",
            "treatments",
            "replicate_group",
            "time_point_h",
        ]
        extra = {
            "value": readout,
            "normalized": f"{readout}__{rr.data.method}",
            "effect": f"{readout}__effect_pct",
            "excluded": f"{readout}__excluded",
            "exclusion_reason": f"{readout}__exclusion_reason",
        }
        if "gi" in frame:
            extra["gi"] = f"{readout}__gi"
        part = frame[keep + list(extra)].rename(columns=extra)
        merged = part if merged is None else merged.merge(part, on=keep, how="outer")
    return merged if merged is not None else pd.DataFrame()


def matrix_frame(m: Matrix, values: np.ndarray) -> pd.DataFrame:
    """Rows: drug A (display unit), columns: drug B; the 0 row and column included."""
    a = _display(m.conc_a, m.unit_a)
    b = _display(m.conc_b, m.unit_b)
    frame = pd.DataFrame(
        values,
        index=pd.Index([f"{v:.4g}" for v in a], name=f"{m.drug_a} ({m.unit_a})"),
        columns=[f"{m.drug_b} {v:.4g} {m.unit_b}" for v in b],
    )
    return frame


def prism_table(curve: CurveResult) -> pd.DataFrame:
    """Prism XY layout: X = log10 concentration (base unit, e.g. M), one Y column per
    replicate; excluded points left empty."""
    points = curve.points.copy()
    points["log10_conc"] = np.round(np.log10(points["conc"].astype(float)), 6)
    points.loc[points["excluded"], "normalized"] = np.nan
    points["rank"] = points.groupby("log10_conc").cumcount() + 1
    table = points.pivot(index="log10_conc", columns="rank", values="normalized")
    table.columns = [f"Y{c}" for c in table.columns]
    base = units.parse_unit(curve.unit).kind if curve.unit else "molar"
    name = {"molar": "M", "mass": "g_per_L", "percent": "fraction"}.get(base, "base")
    table.index.name = f"log10_conc_{name}"
    return table


# ------------------------------------------------------------------- figures


def _save(fig: Any, folder: Path, stem: str, files: list[str]) -> None:
    for suffix, extra in (("png", {"dpi": 300}), ("svg", {})):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **extra)
        files.append(str(path))


def _curve_axes(ax: Any, curves: list[CurveResult], method: str) -> None:
    unit_set = {c.unit for c in curves}
    shared = unit_set.pop() if len(unit_set) == 1 else None
    for k, c in enumerate(curves):
        colour = COLOURS[k % len(COLOURS)]
        unit = shared or ("M" if units.parse_unit(c.unit).kind == "molar" else c.unit)
        used = c.points[~c.points["excluded"]]
        excluded = c.points[c.points["excluded"]]
        x = _display(used["conc"], unit)
        ax.scatter(x, used["normalized"], s=9, color=colour, alpha=0.5, linewidths=0)
        if len(excluded):
            ax.scatter(
                _display(excluded["conc"], unit),
                excluded["normalized"],
                s=22,
                facecolors="none",
                edgecolors=colour,
                linewidths=1.0,
            )
        grouped = used.groupby("conc")["normalized"]
        levels = np.sort(used["conc"].unique().astype(float))
        means, sds = grouped.mean().loc[levels], grouped.std().loc[levels]
        ax.errorbar(
            _display(levels, unit),
            means,
            yerr=sds.fillna(0),
            fmt="o",
            ms=4,
            color=colour,
            capsize=2,
            label=f"{c.compound} ({c.cell_line})",
        )
        if math.isfinite(c.fit.top) and levels.size:
            xs = np.linspace(np.log10(levels.min()) - 0.3, np.log10(levels.max()) + 0.3, 200)
            ax.plot(_display(10**xs, unit), c.fit.predict(xs), color=colour, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(f"concentration ({shared or 'base unit'})")
    ax.set_ylabel(LABELS.get(method, method))
    ax.grid(alpha=0.2)


def curve_figures(rr: ReadoutResult, folder: Path, files: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    stems = []
    by_line: dict[str, list[CurveResult]] = {}
    for c in rr.curves:
        by_line.setdefault(c.cell_line, []).append(c)
    for line, curves in sorted(by_line.items()):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        _curve_axes(ax, curves, rr.data.method)
        ax.set_title(f"{rr.readout} — {line or 'no cell line'}")
        ax.legend(fontsize=7, frameon=False)
        stem = f"curves_{_slug(rr.readout)}_{_slug(line)}"
        _save(fig, folder, stem, files)
        plt.close(fig)
        stems.append(stem)
    compounds = sorted({c.compound for c in rr.curves})
    lines = sorted(by_line)
    if rr.curves and (len(compounds) > 1 or len(lines) > 1):
        fig, axes = plt.subplots(
            len(lines),
            len(compounds),
            figsize=(2.6 * len(compounds), 2.2 * len(lines)),
            squeeze=False,
            sharey=True,
        )
        lookup = {(c.cell_line, c.compound): c for c in rr.curves}
        for i, line in enumerate(lines):
            for j, compound in enumerate(compounds):
                ax = axes[i][j]
                if (line, compound) in lookup:
                    _curve_axes(ax, [lookup[(line, compound)]], rr.data.method)
                else:
                    ax.set_axis_off()
                ax.set_title(f"{compound} · {line}", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.xaxis.label.set_size(6)
                ax.yaxis.label.set_size(6)
        fig.tight_layout()
        stem = f"curves_{_slug(rr.readout)}_grid"
        _save(fig, folder, stem, files)
        plt.close(fig)
        stems.append(stem)
    return stems


def _label_colour(image: Any, value: float) -> str:
    """Black or white, whichever reads better on the cell's colour."""
    r, g, b, _ = image.cmap(image.norm(value))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


def _heatmap(
    ax: Any, m: Matrix, values: np.ndarray, title: str, diverging: bool, response_scale: bool = True
) -> Any:
    import matplotlib.colors as mcolors

    norm: mcolors.Normalize
    if diverging:
        bound = max(THRESHOLD, float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
        cmap = "RdBu_r"
    else:
        finite = values[np.isfinite(values)]
        low = min(0.0, float(np.min(finite))) if finite.size else 0.0
        high = float(np.max(finite)) if finite.size else 100.0
        if response_scale:
            high = max(100.0, high)
        norm = mcolors.Normalize(vmin=low, vmax=high)
        cmap = "viridis"
    image = ax.imshow(values, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            text = "×" if not math.isfinite(v) else f"{v:.0f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=6,
                color="#888" if not math.isfinite(v) else _label_colour(image, v),
            )
    ax.set_xticks(
        range(m.conc_b.size),
        [f"{v:.3g}" for v in _display(m.conc_b, m.unit_b)],
        rotation=45,
        fontsize=6,
    )
    ax.set_yticks(
        range(m.conc_a.size), [f"{v:.3g}" for v in _display(m.conc_a, m.unit_a)], fontsize=6
    )
    ax.set_xlabel(f"{m.drug_b} ({m.unit_b})", fontsize=7)
    ax.set_ylabel(f"{m.drug_a} ({m.unit_a})", fontsize=7)
    ax.set_title(title, fontsize=8)
    return image


def combination_figure(
    rr: ReadoutResult, c: CombinationResult, folder: Path, files: list[str], surfaces: bool
) -> str:
    import matplotlib.pyplot as plt

    m = c.matrix
    panels = [("response (% inhibition)", m.response, False)]
    for name, res in c.models.items():
        panels.append(
            (f"{name.upper()} synergy (mean {res.mean:.1f}, {res.label})", res.synergy, True)
        )
    if np.isfinite(m.sd).any():
        panels.append(("replicate SD", m.sd, False))
    cols = min(3, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.9 * rows), squeeze=False)
    for ax, (title, values, diverging) in zip(axes.flat, panels, strict=False):
        image = _heatmap(ax, m, values, title, diverging, response_scale=title.startswith("resp"))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(f"{m.name} — {rr.readout}", fontsize=9)
    fig.tight_layout()
    stem = f"combination_{_slug(rr.readout)}_{m.slug}"
    _save(fig, folder, stem, files)
    plt.close(fig)
    if surfaces:
        fig = plt.figure(figsize=(3.6 * len(c.models), 3.2))
        for k, (name, res) in enumerate(c.models.items(), start=1):
            ax = fig.add_subplot(1, len(c.models), k, projection="3d")
            yy, xx = np.meshgrid(range(m.conc_a.size), range(m.conc_b.size), indexing="ij")
            ax.plot_surface(xx, yy, res.synergy, cmap="RdBu_r", vmin=-30, vmax=30)
            ax.set_title(name.upper(), fontsize=8)
            ax.set_xlabel(m.drug_b, fontsize=6)
            ax.set_ylabel(m.drug_a, fontsize=6)
        _save(fig, folder, f"surface_{_slug(rr.readout)}_{m.slug}", files)
        plt.close(fig)
    return stem


def qc_figures(rr: ReadoutResult, folder: Path, files: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    stems = []
    wells = rr.data.wells
    for plate in sorted(wells["plate"].unique()):
        mine = wells[wells["plate"] == plate]
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4), width_ratios=[1, 1.4])
        for label, mask, colour in (
            ("vehicle", mine["role"] == "negative_control", "#2a6fdb"),
            ("positive control", mine["role"] == "positive_control", "#e4572e"),
        ):
            sub = mine[mask]
            left.scatter(range(len(sub)), sub["value"], s=12, color=colour, label=label)
        left.set_xlabel("control well")
        left.set_ylabel(rr.readout)
        row = rr.qc[rr.qc["plate"] == plate]
        z = float(row["z_prime"].iloc[0]) if len(row) else math.nan
        left.set_title(f"Controls — Z′ = {z:.2f}" if math.isfinite(z) else "Controls", fontsize=9)
        left.legend(fontsize=7, frameon=False)
        grid, _, _ = plate_trends(rr.data, str(plate))
        image = right.imshow(grid, cmap="viridis", aspect="auto")
        right.set_xticks(
            range(grid.shape[1]), [str(c + 1) for c in range(grid.shape[1])], fontsize=5
        )
        right.set_yticks(
            range(grid.shape[0]), [chr(65 + r) for r in range(grid.shape[0])], fontsize=5
        )
        right.set_title(
            f"{LABELS.get(rr.data.method, rr.data.method)} across the plate", fontsize=9
        )
        fig.colorbar(image, ax=right, fraction=0.03)
        fig.tight_layout()
        stem = f"qc_{_slug(rr.readout)}_{_slug(str(plate))}"
        _save(fig, folder, stem, files)
        plt.close(fig)
        stems.append(stem)
    return stems


# ---------------------------------------------------------------------- write


def write_outputs(result: PharmResult, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from hcs_studio import __version__

    folder = Path(folder)
    figures = folder / "figures"
    for sub in ("", "figures", "combination_matrices", "prism"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    multiple = len(result.readouts) > 1

    def table(name: str, frame: pd.DataFrame, index: bool = False) -> None:
        path = folder / name
        frame.to_csv(path, index=index)
        files.append(str(path))

    table("normalized_well_data.csv", normalized_wells(result))
    table("plate_qc.csv", pd.concat([rr.qc for rr in result.readouts.values()], ignore_index=True))
    fits = pd.concat(
        [curves_table(rr.curves) for rr in result.readouts.values()], ignore_index=True
    )
    table("dose_response_fits.csv", fits)
    table(
        "ic50_comparisons.csv",
        pd.concat([rr.comparisons for rr in result.readouts.values()], ignore_index=True),
    )
    scores = pd.concat(
        [combination_tables(rr) for rr in result.readouts.values()], ignore_index=True
    )
    table("combination_scores.csv", scores)
    excluded = pd.concat(
        [
            rr.data.wells.loc[
                rr.data.wells["excluded"],
                ["plate", "well", "cell_line", "treatment", "exclusion_reason"],
            ].assign(readout=r)
            for r, rr in result.readouts.items()
        ],
        ignore_index=True,
    )
    table("excluded_wells.csv", excluded)

    figure_stems: dict[str, dict[str, Any]] = {}
    for readout, rr in result.readouts.items():
        matrices = folder / "combination_matrices" / (_slug(readout) if multiple else "")
        matrices.mkdir(parents=True, exist_ok=True)
        for c in rr.combinations:
            m = c.matrix
            kinds = {"response": m.response, "sd": m.sd, "n": m.n, "ci": c.combination_index}
            kinds.update({name: res.synergy for name, res in c.models.items()})
            for kind, values in kinds.items():
                path = matrices / f"{m.slug}_{kind}.csv"
                matrix_frame(m, values).to_csv(path)
                files.append(str(path))
        prism = folder / "prism" / _slug(readout)
        prism.mkdir(parents=True, exist_ok=True)
        for curve in rr.curves:
            path = prism / f"{_slug(curve.cell_line)}_{_slug(curve.compound)}.csv"
            prism_table(curve).to_csv(path)
            files.append(str(path))
        stems: dict[str, Any] = {"curves": [], "combinations": [], "qc": []}
        if result.options.figures:
            stems["curves"] = curve_figures(rr, figures, files)
            stems["combinations"] = [
                combination_figure(rr, c, figures, files, result.options.surfaces)
                for c in rr.combinations
            ]
            stems["qc"] = qc_figures(rr, figures, files)
        figure_stems[readout] = stems

    data = interactive_data(result)
    (folder / "report_data.json").write_text(json.dumps(data, separators=(",", ":")), "utf-8")
    files.append(str(folder / "report_data.json"))

    info = {
        "software": "HCS Studio",
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": result.runs,
        "options": options_dict(result.options),
        "notes": {r: rr.notes for r, rr in result.readouts.items()},
    }
    report = folder / "pharmacology_report.html"
    report.write_text(
        report_html(result, figures, figure_stems, info, fits, scores), encoding="utf-8"
    )
    files.append(str(report))
    if result.options.pdf:
        pdf = folder / "pharmacology_report.pdf"
        report_pdf(result, figures, figure_stems, info, fits, scores, pdf)
        files.append(str(pdf))
    info["files"] = sorted(str(Path(f).relative_to(folder)) for f in files)
    (folder / "pharmacology.json").write_text(
        json.dumps(info, indent=2, default=str), encoding="utf-8"
    )
    result.folder = folder
    result.files = {Path(f).name: f for f in files}


# ------------------------------------------------- interactive data (index.html)


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return float(f"{v:.6g}") if math.isfinite(v) else None


def _factor(unit: str) -> float:
    try:
        return float(units.from_base(1.0, unit))
    except (ValueError, KeyError):
        return 1.0


def interactive_data(result: PharmResult) -> dict[str, Any]:
    """Curves (points with their wells, fitted parameters) and matrices (values, synergy,
    wells per cell) for the run's clickable ``index.html``; concentrations in display units."""
    out: dict[str, Any] = {}
    for readout, rr in result.readouts.items():
        runs = dict(zip(rr.data.wells["plate"], rr.data.wells["run"], strict=False))
        curves = []
        for c in rr.curves:
            f, factor = c.fit, _factor(c.unit)
            center = f.log_xb if f.log_xb is not None else f.log_ec50
            curves.append(
                {
                    "cell_line": c.cell_line,
                    "compound": c.compound,
                    "unit": c.unit,
                    "points": [
                        {
                            "run": runs.get(p, ""),
                            "plate": p,
                            "well": w,
                            "x": _num(x * factor),
                            "y": _num(y),
                            "excluded": bool(e),
                        }
                        for x, y, e, p, w in c.points[
                            ["conc", "normalized", "excluded", "plate", "well"]
                        ].itertuples(index=False)
                    ],
                    "fit": None
                    if not math.isfinite(f.top)
                    else {
                        "bottom": _num(f.bottom),
                        "top": _num(f.top),
                        "hill": _num(f.hill),
                        "s": _num(f.asymmetry),
                        "center": _num(center + math.log10(factor)),
                    },
                    "ec50": _num(f.ec50 * factor),
                    "abs_ic50": _num(f.abs_ic50 * factor),
                    "r2": _num(f.r2),
                    "flags": f.flags,
                }
            )
        combos = []
        for combo in rr.combinations:
            m = combo.matrix

            def grid(values: Any) -> list[list[float | None]]:
                return [[_num(v) for v in row] for row in values]

            combos.append(
                {
                    "cell_line": m.cell_line,
                    "drug_a": m.drug_a,
                    "drug_b": m.drug_b,
                    "unit_a": m.unit_a,
                    "unit_b": m.unit_b,
                    "conc_a": [_num(v) for v in _display(m.conc_a, m.unit_a)],
                    "conc_b": [_num(v) for v in _display(m.conc_b, m.unit_b)],
                    "response": grid(m.response),
                    "synergy": {k: grid(r.synergy) for k, r in combo.models.items()},
                    "scores": {
                        k: {
                            "mean": _num(r.mean),
                            "window": _num(r.window_mean),
                            "p": _num(r.p_value),
                            "label": r.label,
                        }
                        for k, r in combo.models.items()
                    },
                    "wells": {
                        f"{i},{j}": [[runs.get(p, ""), p, w] for p, w in ws]
                        for (i, j), ws in m.wells.items()
                    },
                }
            )
        out[readout] = {
            "label": LABELS.get(rr.data.method, rr.data.method),
            "curves": curves,
            "combinations": combos,
        }
    return out


# --------------------------------------------------------------------- report

GUIDE = "Synergy labels follow SynergyFinder conventions: a guide, not proof."
METHODS_TEXT = """
<p><b>Normalization</b> per plate and cell line, relative to the median of that cell
line's vehicle wells: % of control = 100·x/vehicle; % inhibition =
100·(vehicle − x)/(vehicle − positive control); GR = 2^(log2(x/x0)/log2(vehicle/x0)) − 1
(Hafner et al., Nat Methods 2016), x0 = time-zero value.
<b>Plate QC</b>: Z′ = 1 − 3(σp + σn)/|μp − μn| (Zhang et al. 1999); SSMD =
(μp − μn)/√(σp² + σn²); signal window = (|μp − μn| − 3(σp + σn))/σn.</p>
<p><b>Curves</b>: Y = Bottom + (Top − Bottom)/(1 + 10^((LogEC50 − X)·Hill)) on X =
log10(concentration) (GraphPad Prism's variable-slope model; 5PL: asymmetric sigmoidal),
least squares (optionally robust), 95 % intervals from the covariance matrix (t
distribution) or a bootstrap. Absolute IC50: where the curve crosses 50 (0.5 for GR).
AUC: trapezoidal area of the mean responses over log10 concentration divided by the
tested range. Activity area: area between the fitted curve and no effect, in effect
fraction × log10 units. EC50 shifts between cell lines: extra-sum-of-squares F test
(shared vs separate LogEC50).</p>
<p><b>Combinations</b> on the % inhibition scale. Bliss: E = yA + yB − yA·yB/100; HSA:
E = max(yA, yB) (observed single agents); Loewe: a/ICA(E) + b/ICB(E) = 1 with 4PL single-agent
fits (HSA where undefined), with the combination index of Chou &amp; Talalay; ZIP δ
(Yadav et al., Comput Struct Biotechnol J 2015) as in SynergyFinder (Ianevski et al.).
Scores: mean over the combination cells, mean of the most synergistic 3 × 3 window,
maximum cell; replicate matrices give a t test against 0. Labels follow SynergyFinder:
&lt; −10 likely antagonistic, −10 to 10 likely additive, &gt; 10 likely synergistic — a
guide, not proof.</p>
"""


def _img(figures: Path, stem: str, width: str = "100%") -> str:
    path = figures / f"{stem}.png"
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    alt = html.escape(stem)
    return f'<img src="data:image/png;base64,{data}" style="max-width:{width}" alt="{alt}">'


def _table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    if frame.empty:
        return "<p><i>none</i></p>"
    shown = frame[[c for c in (columns or list(frame.columns)) if c in frame]]

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.{digits}g}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


FIT_COLUMNS = [
    "readout",
    "cell_line",
    "compound",
    "ec50",
    "ec50_ci_low",
    "ec50_ci_high",
    "abs_ic50",
    "hill",
    "top",
    "bottom",
    "emax_observed",
    "auc",
    "activity_area",
    "r2",
    "n_points",
    "n_excluded",
    "flags",
]


def report_html(
    result: PharmResult,
    figures: Path,
    stems: dict[str, dict[str, Any]],
    info: dict[str, Any],
    fits: pd.DataFrame,
    scores: pd.DataFrame,
) -> str:
    o = result.options
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Pharmacology report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#222}"
        "table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ddd;"
        "padding:3px 6px;text-align:right}th{background:#f3f3f3}td:first-child,th:first-child"
        "{text-align:left}.note{color:#a05a00}.flag{color:#b4232c}h2{margin-top:32px}</style>",
        "</head><body><h1>Pharmacology report</h1>",
        f"<p>HCS Studio {html.escape(info['version'])} · {html.escape(info['created'])} · "
        f"runs: {html.escape(', '.join(Path(r).name for r in result.runs) or '—')}</p>",
        f"<p>Readouts: {html.escape(', '.join(o.readouts))} · normalization: "
        f"{html.escape(LABELS.get(o.normalization, o.normalization))} · curve model: "
        f"{html.escape(o.curve_model)}{' (robust)' if o.robust else ''} · synergy models: "
        f"{html.escape(', '.join(o.models) or 'none')}"
        + (f" · outliers: median ± {o.outlier_mad:g} MAD" if o.outlier_mad else "")
        + "</p>",
    ]
    for readout, rr in result.readouts.items():
        parts.append(f"<h2>{html.escape(readout)}</h2>")
        for note in rr.notes:
            parts.append(f"<p class='note'>{html.escape(note)}</p>")
        parts.append("<h3>Plate QC</h3>")
        parts.append(
            _table(
                rr.qc,
                [
                    "plate",
                    "n_vehicle",
                    "n_positive",
                    "z_prime",
                    "ssmd",
                    "signal_window",
                    "cv_vehicle_pct",
                    "cv_positive_pct",
                    "flag",
                ],
            )
        )
        for stem in stems.get(readout, {}).get("qc", []):
            parts.append(_img(figures, stem))
        excluded = rr.data.wells[rr.data.wells["excluded"]]
        if len(excluded):
            parts.append("<h3>Excluded wells</h3>")
            parts.append(
                _table(
                    excluded,
                    ["plate", "well", "cell_line", "treatment", "normalized", "exclusion_reason"],
                )
            )
        parts.append("<h3>Dose–response</h3>")
        parts.append(
            _table(fits[fits["readout"] == readout] if "readout" in fits else fits, FIT_COLUMNS)
        )
        if len(rr.comparisons):
            parts.append("<h4>EC50 shifts between cell lines</h4>")
            parts.append(_table(rr.comparisons))
        for stem in stems.get(readout, {}).get("curves", []):
            parts.append(_img(figures, stem, "80%"))
        if rr.combinations:
            parts.append(
                "<h3>Combinations</h3><p><i>Synergy labels are a guide (SynergyFinder "
                "conventions), not proof.</i></p>"
            )
            mine = scores[scores["readout"] == readout] if "readout" in scores else scores
            cols = ["cell_line", "drug_a", "drug_b", "size", "replicates", "missing_cells"]
            for model in o.models:
                cols += [f"{model}_mean", f"{model}_max_3x3", f"{model}_p", f"{model}_label"]
            parts.append(_table(mine, cols))
            for stem in stems.get(readout, {}).get("combinations", []):
                parts.append(_img(figures, stem))
    parts.append(f"<h2>Methods</h2>{METHODS_TEXT}")
    parts.append(
        "<h3>Options</h3><pre>"
        + html.escape(json.dumps(info["options"], indent=1, default=str))
        + "</pre>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def report_pdf(
    result: PharmResult,
    figures: Path,
    stems: dict[str, dict[str, Any]],
    info: dict[str, Any],
    fits: pd.DataFrame,
    scores: pd.DataFrame,
    path: Path,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small", fontSize=7, leading=9)
    page = landscape(A4)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=page,
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Pharmacology report",
        author="HCS Studio",
    )
    width = page[0] - 2.4 * cm
    story: list[Any] = [
        Paragraph("Pharmacology report", styles["Title"]),
        Paragraph(
            f"HCS Studio {info['version']} · {info['created']} · runs: "
            f"{', '.join(Path(r).name for r in result.runs) or '—'} · normalization: "
            f"{LABELS.get(result.options.normalization, result.options.normalization)} · "
            f"models: {', '.join(result.options.models) or 'none'}",
            small,
        ),
    ]

    def table(frame: pd.DataFrame, columns: list[str]) -> Any:
        cols = [c for c in columns if c in frame]
        if frame.empty or not cols:
            return Paragraph("none", small)

        def cell(v: Any) -> str:
            if isinstance(v, float):
                return "" if math.isnan(v) else f"{v:.3g}"
            return str(v)[:40]

        data = [cols] + [[cell(v) for v in row] for row in frame[cols].itertuples(index=False)]
        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ]
            )
        )
        return t

    def image(stem: str, max_width: float) -> Any:
        png = figures / f"{stem}.png"
        if not png.exists():
            return Spacer(1, 1)
        from reportlab.lib.utils import ImageReader

        w, h = ImageReader(str(png)).getSize()
        scale = min(max_width / w, (page[1] - 4 * cm) / h)
        return Image(str(png), width=w * scale, height=h * scale)

    for readout, rr in result.readouts.items():
        story += [Spacer(1, 8), Paragraph(readout, styles["Heading2"])]
        story += [Paragraph(n, small) for n in rr.notes]
        story += [
            Paragraph("Plate QC", styles["Heading3"]),
            table(
                rr.qc,
                [
                    "plate",
                    "z_prime",
                    "ssmd",
                    "signal_window",
                    "cv_vehicle_pct",
                    "cv_positive_pct",
                    "flag",
                ],
            ),
        ]
        mine = fits[fits["readout"] == readout] if "readout" in fits else fits
        story += [
            Paragraph("Dose–response", styles["Heading3"]),
            table(
                mine,
                [
                    "cell_line",
                    "compound",
                    "ec50",
                    "ec50_ci_low",
                    "ec50_ci_high",
                    "abs_ic50",
                    "hill",
                    "emax_observed",
                    "auc",
                    "r2",
                    "flags",
                ],
            ),
        ]
        for stem in stems.get(readout, {}).get("curves", []):
            story.append(image(stem, width * 0.7))
        if rr.combinations:
            sc = scores[scores["readout"] == readout] if "readout" in scores else scores
            cols = ["cell_line", "drug_a", "drug_b"] + [
                f"{m}_{k}" for m in result.options.models for k in ("mean", "max_3x3", "p")
            ]
            story += [
                Paragraph("Combinations", styles["Heading3"]),
                table(sc, cols),
                Paragraph(GUIDE, small),
            ]
            for stem in stems.get(readout, {}).get("combinations", []):
                story.append(image(stem, width))
        for stem in stems.get(readout, {}).get("qc", []):
            story.append(image(stem, width))
    methods = (
        re.sub(r"<[^>]+>", "", METHODS_TEXT)
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )
    story += [Paragraph("Methods", styles["Heading2"])] + [
        Paragraph(html.escape(p.strip()), small) for p in methods.split("\n\n") if p.strip()
    ]
    doc.build(story)
