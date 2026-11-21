"""Cell cycle from DNA content, optionally with EdU and pH3 (Phase 20a).

**DNA content** is the integrated nuclear intensity of a DNA stain (DAPI, Hoechst) measured on
background-corrected, unsaturated data. A single confocal plane or a maximum projection does
not contain the whole nucleus and cannot give DNA content: the step warns (use widefield, a sum
projection, or Z planes covering the whole nucleus).

**Calibration** (vehicle wells of the run's layout): the G1 peak of the vehicle nuclei is found
on a smoothed histogram (the tallest peak, or half of it when a peak at half the position is
substantial), and every nucleus is scaled so that this peak is 2N (``cc_dna_n``). EdU and pH3
thresholds come from the vehicle nuclei too: EdU by Otsu's threshold on log intensity (vehicle
wells hold EdU-negative and EdU-positive cells), pH3 as median + 5 robust SD of log intensity
(mitotic cells are few).

**Per well**, a Dean–Jett–Fox-style model is fitted by binned maximum likelihood to the nuclei
between ``sub_g1_below`` and ``polyploid_above`` (in N): a Gaussian G1 peak at μ₁, a Gaussian
G2/M peak at r·μ₁ (r within 1.85–2.15), both with the same CV, and an S phase that is uniform
between the peaks and broadened by the same CV. Nuclei below ``sub_g1_below`` are sub-G1, above
``polyploid_above`` > 4N (polyploid). With EdU, phases come from 2D gating instead: EdU+ = S,
EdU− below the midpoint of the peaks = G1, above = G2/M.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field
from scipy import optimize, stats

from hcs_studio.analysis.table_steps import TableParams, WellContext
from hcs_studio.assays.base import (
    POSITIVE_ROLES,
    VEHICLE_ROLES,
    AssayModule,
    ChannelRole,
    channel_column,
)
from hcs_studio.processing.base import MODULES, ModuleError

ROBUST = 1.4826
PHASES = ("sub_g1", "g1", "s", "g2m", "polyploid")
FloatArray = NDArray[np.float64]


# ------------------------------------------------------------ calibration


def g1_peak(values: NDArray[Any]) -> float:
    """Position of the G1 peak of DNA-content values (raw units)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 30:
        raise ModuleError(f"too few nuclei ({len(x)}) to find the G1 peak")
    low, high = np.percentile(x, [0.5, 99.5])
    edges = np.linspace(0, high * 1.1, 241)
    counts, _ = np.histogram(x, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(counts, np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2), mode="same")
    tallest = int(np.argmax(smooth))
    peak = float(centres[tallest])
    # a substantial peak at half the position is G1 (the tallest was G2/M: an arrested plate)
    half = int(np.argmin(np.abs(centres - peak / 2)))
    window = slice(max(0, half - 8), half + 9)
    if smooth[window].max() > 0.35 * smooth[tallest] and peak / 2 > low:
        local = window.start + int(np.argmax(smooth[window]))
        peak = float(centres[local])
    # refine on the raw values near the peak
    near = x[np.abs(x - peak) < 0.12 * peak]
    return float(np.median(near)) if len(near) >= 10 else peak


def otsu_log(values: NDArray[Any]) -> float:
    """Otsu's threshold of log10 intensities, returned in linear units."""
    from skimage.filters import threshold_otsu

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 20 or np.ptp(np.log10(x)) == 0:
        raise ModuleError("too few or identical values to set a threshold")
    return float(10 ** threshold_otsu(np.log10(x)))


def robust_upper_log(values: NDArray[Any], k: float = 5.0) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = np.log10(x[np.isfinite(x) & (x > 0)])
    if len(x) < 20:
        raise ModuleError("too few values to set a threshold")
    median = float(np.median(x))
    return float(10 ** (median + k * ROBUST * float(np.median(np.abs(x - median)))))


# ------------------------------------------------------------------ model


@dataclass
class CycleFit:
    n: int = 0
    fractions: dict[str, float] = field(default_factory=dict)  # % of all nuclei
    g1_peak_n: float = math.nan
    g2_g1_ratio: float = math.nan
    cv_pct: float = math.nan
    converged: bool = False
    warnings: list[str] = field(default_factory=list)
    method: str = "dean-jett-fox"


def _components(
    centres: FloatArray, mu1: float, cv: float, ratio: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    mu2 = mu1 * ratio
    g1 = stats.norm.pdf(centres, mu1, cv * mu1)
    g2 = stats.norm.pdf(centres, mu2, cv * mu2)
    t = np.linspace(mu1, mu2, 64)
    s = stats.norm.pdf(centres[:, None], t[None, :], cv * t[None, :]).mean(axis=1)
    return g1, s, g2


def fit_dna(
    dna_n: NDArray[Any], *, sub_g1_below: float = 1.5, polyploid_above: float = 5.0, bins: int = 140
) -> CycleFit:
    """Phase fractions of DNA contents in N units (vehicle G1 = 2N); see the module docs."""
    x = np.asarray(dna_n, dtype=np.float64)
    x = x[np.isfinite(x)]
    fit = CycleFit(n=len(x))
    if len(x) < 50:
        fit.warnings.append(f"only {len(x)} nuclei: no fit")
        return fit
    inside = x[(x >= sub_g1_below) & (x <= polyploid_above)]
    edges = np.linspace(sub_g1_below, polyploid_above, bins + 1)
    counts, _ = np.histogram(inside, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    # start: the G1 peak of this well near 2N
    near = inside[(inside > 1.6) & (inside < 2.6)]
    mu_start = float(np.median(near)) if len(near) > 10 else 2.0

    def unpack(p: FloatArray) -> tuple[float, float, float, FloatArray]:
        mu1, cv, ratio, a, b = p
        logits = np.array([a, b, 0.0])
        weights = np.exp(logits - logits.max())
        return float(mu1), float(cv), float(ratio), weights / weights.sum()

    def nll(p: FloatArray) -> float:
        mu1, cv, ratio, w = unpack(p)
        g1, s, g2 = _components(centres, mu1, cv, ratio)
        density = w[0] * g1 + w[1] * s + w[2] * g2
        total = density.sum()
        if not total > 0:
            return 1e12
        prob = np.clip(density / total, 1e-300, None)
        return float(-(counts * np.log(prob)).sum())

    best = None
    for start in ((0.6, -0.2), (0.0, 0.0), (-0.8, 0.3)):  # G1-rich, even, G2-rich starts
        p0 = np.array([mu_start, 0.05, 2.0, start[0] + 0.6, start[1]])
        result = optimize.minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=[(1.6, 2.6), (0.01, 0.15), (1.85, 2.15), (-8, 8), (-8, 8)],
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    mu1, cv, ratio, w = unpack(best.x)
    share = len(inside) / len(x)
    fit.fractions = {
        "sub_g1": 100 * float((x < sub_g1_below).mean()),
        "g1": 100 * share * float(w[0]),
        "s": 100 * share * float(w[1]),
        "g2m": 100 * share * float(w[2]),
        "polyploid": 100 * float((x > polyploid_above).mean()),
    }
    fit.g1_peak_n, fit.g2_g1_ratio, fit.cv_pct = mu1, ratio, 100 * cv
    fit.converged = bool(best.success)
    if cv > 0.08:
        fit.warnings.append(
            f"G1 CV {100 * cv:.1f} % (> 8 %): broad peaks, check staining and focus"
        )
    if abs(mu1 - 2.0) > 0.25:
        fit.warnings.append(
            f"G1 peak at {mu1:.2f}N, not 2N: staining differs from the vehicle wells"
        )
    if ratio in (1.85, 2.15) or abs(ratio - 2.15) < 1e-3 or abs(ratio - 1.85) < 1e-3:
        fit.warnings.append(f"G2/G1 ratio {ratio:.2f} at its limit: check the DNA measurement")
    return fit


def phase_of(
    dna_n: FloatArray, fit: CycleFit, sub_g1_below: float, polyploid_above: float
) -> NDArray[Any]:
    """Most likely phase of each nucleus under the fitted model."""
    out = np.full(len(dna_n), "", dtype=object)
    if not fit.fractions or not math.isfinite(fit.g1_peak_n):
        return out
    cv = fit.cv_pct / 100
    g1, s, g2 = _components(np.asarray(dna_n, float), fit.g1_peak_n, cv, fit.g2_g1_ratio)
    total = fit.fractions["g1"] + fit.fractions["s"] + fit.fractions["g2m"] or 1.0
    scores = (
        np.vstack([g1 * fit.fractions["g1"], s * fit.fractions["s"], g2 * fit.fractions["g2m"]])
        / total
    )
    out[:] = np.array(["g1", "s", "g2m"], dtype=object)[np.argmax(scores, axis=0)]
    out[dna_n < sub_g1_below] = "sub_g1"
    out[dna_n > polyploid_above] = "polyploid"
    out[~np.isfinite(dna_n)] = ""
    return out


def gate_edu(
    dna_n: FloatArray, edu_positive: NDArray[np.bool_], fit: CycleFit,
    sub_g1_below: float, polyploid_above: float,
) -> NDArray[Any]:  # fmt: skip
    """2D gating: EdU+ = S; EdU− below the midpoint of the G1 and G2 peaks = G1, above = G2/M."""
    mu1 = fit.g1_peak_n if math.isfinite(fit.g1_peak_n) else 2.0
    mid = mu1 * (1 + (fit.g2_g1_ratio if math.isfinite(fit.g2_g1_ratio) else 2.0)) / 2
    out = np.where(edu_positive, "s", np.where(dna_n < mid, "g1", "g2m")).astype(object)
    out[dna_n < sub_g1_below] = "sub_g1"
    out[dna_n > polyploid_above] = "polyploid"
    return out


# ----------------------------------------------------------------- module


class CellCycleParams(TableParams):
    dna_channel: str = Field(default="DAPI", description="DNA stain channel (DAPI, Hoechst).")
    edu_channel: str = Field(default="", description="EdU channel; empty = none.")
    ph3_channel: str = Field(default="", description="Phospho-histone H3 channel; empty = none.")
    dna_feature: str = Field(
        default="integrated", description="DNA content feature (integrated nuclear intensity)."
    )
    sub_g1_below: float = Field(default=1.5, gt=0, description="Sub-G1 below this DNA content (N).")
    polyploid_above: float = Field(
        default=5.0, gt=0, description="> 4N above this DNA content (N)."
    )
    edu_threshold: float | None = Field(
        default=None, gt=0, description="EdU mean intensity threshold; empty = from vehicle wells."
    )
    ph3_threshold: float | None = Field(
        default=None, gt=0, description="pH3 mean intensity threshold; empty = from vehicle wells."
    )
    min_nuclei: int = Field(default=200, ge=1, description="Warn below this many nuclei per well.")


@MODULES.register("cell_cycle")
class CellCycle(AssayModule):
    """Cell cycle from DNA content (and EdU, pH3), calibrated on the vehicle wells."""

    name = "cell_cycle"
    title = "Assay: cell cycle (DNA content, EdU, pH3)"
    Params = CellCycleParams
    prefix: ClassVar[str] = "cc_"
    roles: ClassVar[tuple[ChannelRole, ...]] = (
        ChannelRole("dna", "DNA", "DAPI or Hoechst: integrated nuclear intensity"),
        ChannelRole("edu", "EdU", "EdU click chemistry: S phase", required=False),
        ChannelRole("ph3", "pH3", "phospho-histone H3 (Ser10): mitosis", required=False),
    )

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        return preset(mapping, **options)

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.wells: dict[str, dict[str, Any]] = {}
        self.histograms: dict[str, list[int]] = {}

    @property
    def p(self) -> CellCycleParams:
        params: CellCycleParams = self.params
        return params

    def columns(self) -> dict[str, str]:
        out = {"dna": channel_column(self.p.dna_channel, self.p.dna_feature)}
        if self.p.edu_channel:
            out["edu"] = channel_column(self.p.edu_channel, "mean")
        if self.p.ph3_channel:
            out["ph3"] = channel_column(self.p.ph3_channel, "mean")
        return out

    def run_warnings(self) -> list[str]:
        manifest = self.manifest()
        warnings = []
        steps = (manifest.get("pipeline") or {}).get("steps", [])
        for step in steps:
            if step.get("module") == "projection" and step.get("enabled", True):
                method = (step.get("params") or {}).get("method", "max")
                if method != "sum":
                    warnings.append(
                        f"the images were Z-projected with '{method}': only a sum projection "
                        "keeps the total DNA of each nucleus"
                    )
        metadata = (manifest.get("plate") or {}).get("metadata") or {}
        confocal = str(metadata.get("modality", "")).lower() in ("confocal", "spinning_disk")
        if confocal and not any(s.get("module") == "projection" for s in steps):
            warnings.append(
                "confocal images without a sum projection: a single optical section does "
                "not hold the whole nucleus (use widefield, or a sum over Z planes covering "
                "the nucleus)"
            )
        return warnings

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        columns = self.columns()
        vehicle = self.control_objects(VEHICLE_ROLES)
        source = "vehicle wells"
        if vehicle.empty:
            vehicle = sample[sample["object_type"] == str(self.params.object_type)]
            source = "all wells (no vehicle wells in the plate layout)"
        self.require(vehicle, list(columns.values()))
        raw = vehicle[columns["dna"]].to_numpy(float)
        peak = g1_peak(raw)
        scale = 2.0 / peak
        state: dict[str, Any] = {
            "calibrated_on": source,
            "vehicle_nuclei": len(vehicle),
            "dna_column": columns["dna"],
            "g1_peak_raw": peak,
            "scale_to_n": scale,
            "warnings": self.run_warnings(),
        }
        vehicle_fit = fit_dna(raw * scale, sub_g1_below=self.p.sub_g1_below,
                              polyploid_above=self.p.polyploid_above)  # fmt: skip
        state["vehicle_fit"] = asdict(vehicle_fit)
        if "edu" in columns:
            state["edu_column"] = columns["edu"]
            state["edu_threshold"] = self.p.edu_threshold or otsu_log(
                vehicle[columns["edu"]].to_numpy(float)
            )
        if "ph3" in columns:
            state["ph3_column"] = columns["ph3"]
            state["ph3_threshold"] = self.p.ph3_threshold or robust_upper_log(
                vehicle[columns["ph3"]].to_numpy(float)
            )
        if "edu" in columns:
            edu_pos = vehicle[columns["edu"]].to_numpy(float) > state["edu_threshold"]
            phases = gate_edu(raw * scale, edu_pos, vehicle_fit, self.p.sub_g1_below,
                              self.p.polyploid_above)  # fmt: skip
            state["vehicle_g2m_pct"] = 100 * float((phases == "g2m").mean())
        else:
            state["vehicle_g2m_pct"] = vehicle_fit.fractions.get("g2m", math.nan)
        return state

    def describe_state(self, state: Any) -> dict[str, Any]:
        return {k: v for k, v in state.items() if k != "vehicle_fit"} | {
            "vehicle_fractions_pct": state["vehicle_fit"]["fractions"]
        }

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        rows = self.selected(objects)
        dna = pd.to_numeric(objects[state["dna_column"]], errors="coerce").to_numpy(float)
        dna_n = np.where(rows.to_numpy(bool), dna * state["scale_to_n"], np.nan)
        fit = fit_dna(dna_n[np.isfinite(dna_n)], sub_g1_below=self.p.sub_g1_below,
                      polyploid_above=self.p.polyploid_above)  # fmt: skip
        out = pd.DataFrame(index=objects.index)
        out["cc_dna_n"] = dna_n
        edu_pos = None
        if "edu_column" in state:
            edu = pd.to_numeric(objects[state["edu_column"]], errors="coerce").to_numpy(float)
            edu_pos = edu > state["edu_threshold"]
            out["cc_edu_positive"] = pd.Series(edu_pos, index=objects.index).where(rows, pd.NA)
        if "ph3_column" in state:
            ph3 = pd.to_numeric(objects[state["ph3_column"]], errors="coerce").to_numpy(float)
            out["cc_ph3_positive"] = pd.Series(
                ph3 > state["ph3_threshold"], index=objects.index
            ).where(rows, pd.NA)
        phases = (
            gate_edu(dna_n, edu_pos, fit, self.p.sub_g1_below, self.p.polyploid_above)
            if edu_pos is not None
            else phase_of(dna_n, fit, self.p.sub_g1_below, self.p.polyploid_above)
        )
        out["cc_phase"] = pd.Series(phases, index=objects.index).where(rows, None)
        self.wells[well.well] = {"fit": fit}
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        fit: CycleFit = self.wells[well.well]["fit"]
        rows = objects[objects["cc_dna_n"].notna()]
        values: dict[str, Any] = {"cc_nuclei": len(rows)}
        for phase in PHASES:
            values[f"cc_{phase}_pct"] = fit.fractions.get(phase, math.nan)
        values["cc_g1_peak_n"] = fit.g1_peak_n
        values["cc_g1_cv_pct"] = fit.cv_pct
        warnings = list(fit.warnings)
        if len(rows) < self.p.min_nuclei:
            warnings.append(f"{len(rows)} nuclei (< {self.p.min_nuclei})")
        if "cc_edu_positive" in rows:
            phases = rows["cc_phase"]
            values["cc_edu_pct"] = 100 * float(rows["cc_edu_positive"].astype(bool).mean())
            for phase in PHASES:
                values[f"cc_edu_{phase}_pct"] = 100 * float((phases == phase).mean())
        g2m = values.get("cc_edu_g2m_pct", values["cc_g2m_pct"])
        reference = state.get("vehicle_g2m_pct", math.nan)
        values["cc_g2m_arrest_index"] = g2m / reference if reference and reference > 0 else math.nan
        if "cc_ph3_positive" in rows:
            values["cc_mitotic_index_pct"] = 100 * float(
                rows["cc_ph3_positive"].astype(bool).mean()
            )
        values["cc_warnings"] = "; ".join(warnings)
        dna = rows["cc_dna_n"].to_numpy(float)
        histogram, _ = np.histogram(dna, np.linspace(0, 8, 161))
        self.histograms[well.well] = histogram.tolist()
        self.wells[well.well]["values"] = values
        return values, {"fit": asdict(fit)}

    def finish_table(self, state: Any, folder: Path, per_well: dict[str, dict[str, Any]]) -> None:
        out = self.output_folder(folder)
        calibration = {"module": self.name, "params": self.p.model_dump(mode="json"),
                       **self.describe_state(state)}  # fmt: skip
        (out / "calibration.json").write_text(
            json.dumps(calibration, indent=1, default=float), "utf-8"
        )
        table = pd.DataFrame(
            [
                {"well": w, **{k: v for k, v in d.get("values", {}).items()}}
                for w, d in sorted(self.wells.items())
            ]
        )
        table.to_csv(out / "cell_cycle_wells.csv", index=False)
        write_report(out, calibration, table, self.histograms, self.well_roles())


# ----------------------------------------------------------------- report


def _histogram_png(histograms: dict[str, list[int]], groups: dict[str, list[str]]) -> str:
    import base64
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 3.2), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    centres = (np.linspace(0, 8, 161)[:-1] + np.linspace(0, 8, 161)[1:]) / 2
    for label, wells in groups.items():
        counts = np.sum([histograms[w] for w in wells if w in histograms], axis=0)
        if np.ndim(counts) == 0 or counts.sum() == 0:
            continue
        ax.plot(centres, counts / counts.sum(), label=f"{label} ({len(wells)} wells)")
    for x in (2, 4):
        ax.axvline(x, color="#999", lw=0.7, ls="--")
    ax.set_xlabel("DNA content (N; vehicle G1 = 2N)")
    ax.set_ylabel("fraction of nuclei")
    ax.legend(fontsize=8)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_report(
    folder: Path,
    calibration: dict[str, Any],
    table: pd.DataFrame,
    histograms: dict[str, list[int]],
    roles: dict[str, str],
) -> Path:
    groups: dict[str, list[str]] = {}
    for well in histograms:
        role = roles.get(well, "sample")
        label = (
            "vehicle"
            if role in VEHICLE_ROLES
            else "positive control"
            if role in POSITIVE_ROLES
            else "samples"
        )
        groups.setdefault(label, []).append(well)
    image = _histogram_png(histograms, groups)
    shown = [
        c
        for c in table.columns
        if c == "well" or (c.endswith(("_pct", "_index", "_n")) or c == "cc_nuclei")
    ]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in shown)

    def cell(value: Any) -> str:
        if isinstance(value, float) and math.isfinite(value):
            return f"<td>{value:.1f}</td>"
        return f"<td>{html.escape(str(value))}</td>"

    body = "".join(
        "<tr>" + "".join(cell(row[c]) for c in shown) + "</tr>" for row in table.to_dict("records")
    )
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in calibration.get("warnings", []))
    vehicle = calibration.get("vehicle_fractions_pct", {})
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cell cycle</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;max-width:1200px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}th{{background:#f3f3f3}}.warn{{color:#b00020}}</style></head><body>
<h1>Cell cycle</h1>
<p>Calibrated on {html.escape(str(calibration.get("calibrated_on", "")))} ({calibration.get("vehicle_nuclei", 0)} nuclei):
vehicle G1 peak = {calibration.get("g1_peak_raw", 0):.4g} ({html.escape(str(calibration.get("dna_column", "")))}) = 2N.
Vehicle phases (%): {", ".join(f"{k} {v:.1f}" for k, v in vehicle.items())}.
{f"EdU threshold {calibration['edu_threshold']:.4g}." if "edu_threshold" in calibration else ""}
{f"pH3 threshold {calibration['ph3_threshold']:.4g}." if "ph3_threshold" in calibration else ""}</p>
{f'<ul class="warn">{warnings}</ul>' if warnings else ""}
<img alt="DNA content histograms" src="data:image/png;base64,{image}">
<h2>Per well</h2><table><tr>{head}</tr>{body}</table>
</body></html>"""
    target = folder / "report.html"
    target.write_text(page, encoding="utf-8")
    return target


# ----------------------------------------------------------------- preset


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str] | None = None,
    nucleus_diameter_um: float = 10.0,
    z_planes: int = 1,
) -> Any:
    """Background removal, nuclear segmentation on the DNA channel, intensity measurement and
    the cell-cycle step (a sum projection first when the plate has Z planes). ``mapping``: role
    → channel name; ``channels``: the plate's channel names (to find the DNA channel's number)."""
    from hcs_studio.pipeline.schema import Pipeline

    dna = mapping.get("dna", "DAPI")
    number = channels.index(dna) + 1 if channels and dna in channels else 1

    steps: list[dict[str, Any]] = []
    if z_planes > 1:
        steps.append({"module": "projection", "params": {"method": "sum"}})
    steps += [
        {
            "module": "background",
            "params": {"method": "tophat", "radius_um": 2.5 * nucleus_diameter_um},
        },
        # DNA content needs the whole nucleus in the mask: a low (triangle) threshold, because
        # Otsu's threshold rises on G2/M-rich fields and trims dim G1 nuclei (biased contents)
        {
            "module": "segment_classical",
            "params": {
                "channel": number,
                "method": "triangle",
                "smoothing_um": 0.5,
                "min_distance_um": nucleus_diameter_um / 3,
            },
        },
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {
            "module": "cell_cycle",
            "params": {
                "dna_channel": dna,
                "edu_channel": mapping.get("edu", ""),
                "ph3_channel": mapping.get("ph3", ""),
            },
        },
    ]
    return Pipeline.model_validate({"name": "cell-cycle", "steps": steps})
