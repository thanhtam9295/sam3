"""Downstream impact of a candidate model on a reference plate (Phase 25).

A finished reference run is re-run with the candidate model in place of the one it used (same
plate, same pipeline otherwise, same layout). The two runs are then compared where it matters:

- **counts** per well (``<labels>_count``): difference and % difference per well, mean and SD,
  Bland–Altman limits of agreement (mean ± 1.96 SD of the % differences), largest change;
- **potency**: the IC50 of every compound × cell line fitted in both runs (pharmacology, %
  inhibition of the count), ratio candidate / reference; a ratio beyond the assay's minimum
  significant ratio (MSR, Phase 17 — from the registry's history of the compound when known,
  else ``msr``, 3 by default) counts as a significant change.

Verdict: *equivalent* when no potency changes significantly and the mean count difference is
within ``count_tolerance_pct`` (10 %); otherwise *different*, with the reasons.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.training.registry import MODEL_PARAMS, is_ref, parse_ref, resolve


@dataclass
class ImpactResult:
    reference_run: Path
    candidate_run: Path
    candidate: str
    replaced: str
    readout: str
    counts: pd.DataFrame
    counts_summary: dict[str, float]
    potency: pd.DataFrame
    verdict: dict[str, Any] = field(default_factory=dict)


def swap_model(pipeline: Any, candidate: str, step: int | None = None) -> tuple[Any, str, str]:
    """The pipeline with the candidate in place of the model it replaces; returns it, a
    description and the label set the swapped step makes. ``step`` (from 1) picks the step;
    without it, the steps referencing another version of the candidate's name are swapped."""
    model = resolve(candidate)
    data = pipeline.model_dump(mode="json")
    steps = data["steps"]
    chosen: list[int] = []
    if step is not None:
        if not 1 <= step <= len(steps):
            raise ValueError(f"The pipeline has no step {step}")
        chosen = [step - 1]
    else:
        for i, s in enumerate(steps):
            for param in MODEL_PARAMS:
                value = s["params"].get(param)
                if is_ref(value) and parse_ref(value)[0] == model.name:
                    chosen.append(i)
        if not chosen:
            raise ValueError(f"No step uses a version of {model.name}; choose the step to replace")
    described = []
    labels = "nuclei"
    for i in chosen:
        old = steps[i]
        if model.kind == "steps":
            steps[i] = {"module": "segment_model", "enabled": True, "params": {"model": model.ref}}
            stored = json.loads(model.path.read_text(encoding="utf-8"))["steps"]
            labels = stored[-1]["params"].get("output", labels)
        elif model.kind == "cellpose" and old["module"] == "segment_cellpose":
            steps[i] = {**old, "params": {**old["params"], "model": model.ref}}
            labels = old["params"].get("output", labels)
        else:
            raise ValueError(f"Cannot put a {model.kind} model into step {i + 1} ({old['module']})")
        described.append(f"step {i + 1} ({old['module']} {_model_of(old)}) → {model.ref}")
    return type(pipeline).model_validate(data), "; ".join(described), labels


def _model_of(step: dict[str, Any]) -> str:
    for param in MODEL_PARAMS:
        if step["params"].get(param):
            return str(step["params"][param])
    return "defaults"


def rerun(reference: Path, pipeline: Any, out: Path, workers: int = 1) -> Path:
    from hcs_studio.io.base import reopen_plate
    from hcs_studio.model.layout import load_plate_layout, run_layout_file
    from hcs_studio.model.plate import parse_well_name
    from hcs_studio.pipeline.executor import run_pipeline

    manifest = json.loads((reference / "run_manifest.json").read_text(encoding="utf-8"))
    plate = reopen_plate(manifest["plate"]["source"])
    layout_path = run_layout_file(reference)
    wells = [parse_well_name(w) for w in manifest.get("wells", [])] or None
    result = run_pipeline(
        plate,
        pipeline,
        output_root=out,
        wells=wells,
        layout=load_plate_layout(layout_path) if layout_path else None,
        layout_path=layout_path,
        workers=workers,
        checksums=False,
        save_images=False,
    )
    return Path(result.folder)


def compare_counts(
    reference: Path, candidate: Path, readout: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    a, b = (_with_count(run, readout) for run in (reference, candidate))
    if readout not in a or readout not in b:
        raise ValueError(f"{readout} is not in both runs' per-well tables")
    table = a[["well", readout]].merge(
        b[["well", readout]], on="well", suffixes=("_reference", "_candidate")
    )
    table = table.rename(
        columns={f"{readout}_reference": "reference", f"{readout}_candidate": "candidate"}
    )
    table["difference"] = table["candidate"] - table["reference"]
    mean = (table["candidate"] + table["reference"]) / 2
    table["difference_pct"] = np.where(
        mean > 0, 100 * table["difference"] / mean.where(mean > 0, 1), np.nan
    )
    pct = table["difference_pct"].dropna()
    sd = float(pct.std(ddof=1)) if len(pct) > 1 else 0.0
    summary = {
        "wells": len(table),
        "mean_difference": float(table["difference"].mean()),
        "mean_difference_pct": float(pct.mean()) if len(pct) else math.nan,
        "sd_difference_pct": sd,
        "loa_low_pct": float(pct.mean() - 1.96 * sd) if len(pct) else math.nan,
        "loa_high_pct": float(pct.mean() + 1.96 * sd) if len(pct) else math.nan,
        "max_abs_difference_pct": float(pct.abs().max()) if len(pct) else math.nan,
    }
    return table, summary


def _with_count(run: Path, readout: str) -> pd.DataFrame:
    """The run's per-well table; a missing ``<labels>_count`` is counted from the objects
    (pipelines without the ``qc`` step have no count column)."""
    table = pd.read_csv(run / "tables" / "per_well_summary.csv")
    if readout in table or not readout.endswith("_count"):
        return table
    labels = readout.removesuffix("_count")
    counts = {}
    for path in sorted(run.glob("wells/*/objects.parquet")):
        objects = pd.read_parquet(path)
        if "object_type" in objects:
            objects = objects[objects["object_type"] == labels]
        if "in_overlap_duplicate" in objects:
            objects = objects[~objects["in_overlap_duplicate"].fillna(False).astype(bool)]
        counts[path.parent.name] = len(objects)
    table[readout] = table["well"].map(counts).fillna(0)
    return table


def _potencies(run: Path, readout: str) -> pd.DataFrame:
    from hcs_studio.pharmacology.analysis import PharmOptions, curves_table, run_pharmacology

    options = PharmOptions(readouts=[readout], models=(), figures=False, pdf=False)
    try:
        result = run_pharmacology([run], options, write=False)
    except ValueError:
        return pd.DataFrame(columns=["compound", "cell_line", "ec50"])
    curves = result.readouts[readout].curves
    return (
        curves_table(curves)[["compound", "cell_line", "ec50"]]
        if curves
        else pd.DataFrame(columns=["compound", "cell_line", "ec50"])
    )


def registry_msr(compound: str, cell_line: str) -> float | None:
    """The compound's MSR from the registry's potency history (None when unknown)."""
    try:
        from hcs_studio.db import session_scope
        from hcs_studio.db.monitoring import msr_history

        with session_scope() as session:
            value = msr_history(session, compound, cell_line=cell_line)["msr"]
    except Exception:  # no registry, no history
        return None
    return float(value) if value and math.isfinite(value) else None


def compare_potency(reference: Path, candidate: Path, readout: str, msr: float) -> pd.DataFrame:
    a, b = _potencies(reference, readout), _potencies(candidate, readout)
    table = a.merge(
        b, on=["compound", "cell_line"], how="outer", suffixes=("_reference", "_candidate")
    )
    rows = []
    for record in table.to_dict("records"):
        ref, cand = record.get("ec50_reference"), record.get("ec50_candidate")
        known = registry_msr(str(record["compound"]), str(record["cell_line"]))
        limit = known or msr
        ratio = cand / ref if ref and cand and ref > 0 and cand > 0 else math.nan
        fold = max(ratio, 1 / ratio) if math.isfinite(ratio) and ratio > 0 else math.nan
        rows.append(
            {
                **record,
                "ratio": ratio,
                "fold_change": fold,
                "msr": limit,
                "msr_source": "registry" if known else "default",
                "significant": bool(not math.isfinite(fold) or fold > limit),
            }
        )
    return pd.DataFrame(rows)


def downstream_impact(
    reference_run: Path,
    candidate: str,
    out: Path,
    *,
    step: int | None = None,
    readout: str | None = None,
    msr: float = 3.0,
    count_tolerance_pct: float = 10.0,
    workers: int = 1,
) -> ImpactResult:
    from hcs_studio.pipeline.schema import Pipeline

    reference_run = Path(reference_run)
    manifest = json.loads((reference_run / "run_manifest.json").read_text(encoding="utf-8"))
    pipeline, replaced, labels = swap_model(
        Pipeline.model_validate(manifest["pipeline"]), candidate, step
    )
    readout = readout or f"{labels}_count"
    candidate_run = rerun(reference_run, pipeline, Path(out) / "runs", workers)
    counts, summary = compare_counts(reference_run, candidate_run, readout)
    potency = compare_potency(reference_run, candidate_run, readout, msr)
    reasons = []
    if potency["significant"].any() if len(potency) else False:
        names = potency.loc[potency["significant"], "compound"].astype(str).tolist()
        reasons.append(f"potency beyond the MSR for {', '.join(names)}")
    if abs(summary["mean_difference_pct"]) > count_tolerance_pct:
        reasons.append(
            f"mean count difference {summary['mean_difference_pct']:+.1f} % "
            f"(tolerance ±{count_tolerance_pct:g} %)"
        )
    verdict = {
        "equivalent": not reasons,
        "reasons": reasons,
        "msr_default": msr,
        "count_tolerance_pct": count_tolerance_pct,
        "potencies_compared": int(
            potency[["ec50_reference", "ec50_candidate"]].notna().all(axis=1).sum()
        )
        if len(potency)
        else 0,
    }
    return ImpactResult(
        reference_run,
        candidate_run,
        resolve(candidate).ref,
        replaced,
        readout,
        counts,
        summary,
        potency,
        verdict,
    )


def write_impact(result: ImpactResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    result.counts.to_csv(out / "counts.csv", index=False)
    result.potency.to_csv(out / "potency.csv", index=False)
    from hcs_studio.output.manifest import rooted_text

    record = {
        "reference_run": rooted_text(result.reference_run),
        "candidate_run": rooted_text(result.candidate_run),
        "candidate": result.candidate,
        "replaced": result.replaced,
        "readout": result.readout,
        "counts": result.counts_summary,
        "verdict": result.verdict,
        "potency": json.loads(result.potency.to_json(orient="records")),
    }
    (out / "impact.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.6))
    mean = (result.counts["candidate"] + result.counts["reference"]) / 2
    ax.scatter(mean, result.counts["difference_pct"], s=12)
    s = result.counts_summary
    for value, style in (
        (s["mean_difference_pct"], "-"),
        (s["loa_low_pct"], "--"),
        (s["loa_high_pct"], "--"),
    ):
        if math.isfinite(value):
            ax.axhline(value, color="#888888", lw=0.8, ls=style)
    ax.set_xlabel(f"mean {result.readout} (reference, candidate)")
    ax.set_ylabel("difference, % of mean")
    ax.set_title("Bland–Altman: candidate vs reference")
    fig.savefig(out / "counts_bland_altman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    verdict = (
        "equivalent"
        if result.verdict["equivalent"]
        else "different: " + "; ".join(result.verdict["reasons"])
    )

    def to_html(frame: pd.DataFrame) -> str:
        return str(frame.to_html(index=False, border=0, float_format=lambda v: f"{v:.4g}"))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Model impact</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}
td,th{{padding:3px 10px;border-bottom:1px solid #ddd}}
img{{max-width:60%}}</style></head><body>
<h1>Downstream impact of {html.escape(result.candidate)}</h1>
<p>{html.escape(result.replaced)}. Readout <b>{html.escape(result.readout)}</b>.
Verdict: <b>{html.escape(verdict)}</b></p>
<h2>Counts per well</h2>{to_html(pd.DataFrame([s]))}
<img src="counts_bland_altman.png" alt="Bland–Altman">
<h2>Potency (IC50 ratio candidate / reference vs MSR)</h2>{to_html(result.potency)}
</body></html>"""
    path = out / "impact.html"
    path.write_text(page, encoding="utf-8")
    return path
