"""Comparing segmentation models on held-out ground truth (Phase 25).

``compare_candidate`` scores a registered candidate and a baseline (the current production
version of the same name, else the candidate's base model) on a ground-truth split — by default
the **test** split of the set the candidate was trained on, which training never saw — and
saves a comparison report: metrics side by side (overall and per cell line), AP against the IoU
threshold, count errors per field and example fields with the true and predicted outlines.
The candidate's registry metrics get the test results and the report path.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.pipeline.schema import PipelineStep
from hcs_studio.training.groundtruth import load_set, read_item
from hcs_studio.training.registry import ModelRecord, get_model, resolve
from hcs_studio.training.validate import SegmentationValidation, validate_segmentation

COLUMNS = (
    ("ap50", "AP@0.5"),
    ("map", "mAP 0.5–0.95"),
    ("precision50", "precision"),
    ("recall50", "recall"),
    ("f1_50", "F1"),
    ("merges", "merges"),
    ("splits", "splits"),
    ("mean_abs_count_error", "mean |count error|"),
)


@dataclass
class Comparison:
    set_ref: str
    split: str
    results: dict[str, SegmentationValidation]

    def table(self) -> pd.DataFrame:
        rows = []
        for label, result in self.results.items():
            o = result.overall
            rows.append(
                {
                    "model": label,
                    **{name: o[key] for key, name in COLUMNS},
                    "fields": o["fields"],
                    "objects": o["objects_true"],
                }
            )
        return pd.DataFrame(rows)


def model_steps(model: ModelRecord) -> list[PipelineStep]:
    """Pipeline steps that segment with a registered model."""
    if model.kind == "steps":
        return [PipelineStep(module="segment_model", params={"model": model.ref})]
    if model.kind == "cellpose":
        return [_cellpose(model.ref, model.hyperparameters)]
    raise ValueError(f"{model.ref} is a {model.kind} model: only segmentation models compare")


def base_steps(model: ModelRecord) -> list[PipelineStep]:
    """The model a candidate was trained from (defaults for tuned steps)."""
    if model.kind == "steps":
        fixed = dict(model.hyperparameters.get("fixed") or {"channel": 1})
        return [PipelineStep(module="segment_classical", params=fixed)]
    if model.kind == "cellpose":
        return [_cellpose(model.base_model or "cpsam_v2", model.hyperparameters)]
    raise ValueError(f"{model.ref}: no base model to compare with")


def _cellpose(name: str, hyper: dict[str, Any]) -> PipelineStep:
    params: dict[str, Any] = {"channel": int(hyper.get("channel") or 1), "model": name}
    if hyper.get("second_channel"):
        params["second_channel"] = int(hyper["second_channel"])
    for key in ("diameter_um", "flow_threshold", "cellprob_threshold", "min_size_px"):
        if key in hyper:
            params[key] = hyper[key]
    return PipelineStep(module="segment_cellpose", params=params)


def compare_segmentation(
    set_ref: str,
    models: dict[str, list[PipelineStep]],
    split: str | None = "test",
    pixel_size_um: float | None = None,
) -> Comparison:
    ann = load_set(set_ref)
    results = {
        label: validate_segmentation(
            ann, steps, split=split, model=label, pixel_size_um=pixel_size_um, keep_predictions=True
        )
        for label, steps in models.items()
    }
    return Comparison(ann.ref, split or "all", results)


def _figures(comparison: Comparison, out: Path, examples: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    figures = []
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for label, result in comparison.results.items():
        ap = result.overall["ap"]
        ax.plot([float(t) for t in ap], list(ap.values()), marker="o", ms=3, label=label)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP = TP / (TP + FP + FN)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / "ap_by_iou.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append("ap_by_iou.png")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for k, result in enumerate(comparison.results.values()):
        errors = result.fields["count_error"].to_numpy(float)
        ax.scatter(np.full(len(errors), k) + np.linspace(-0.15, 0.15, len(errors)), errors, s=12)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(len(comparison.results)), list(comparison.results), fontsize=8)
    ax.set_ylabel("predicted − true objects per field")
    fig.savefig(out / "count_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append("count_error.png")

    ann = load_set(comparison.set_ref)
    first = next(iter(comparison.results.values()))
    worst = first.fields.sort_values("tp@0.50").head(examples)["item"].tolist()
    items = {i.id: i for i in ann.items}
    if worst:
        cols = 2 + len(comparison.results)
        fig, axes = plt.subplots(
            len(worst), cols, figsize=(2.2 * cols, 2.2 * len(worst)), squeeze=False
        )
        for r, item_id in enumerate(worst):
            image, truth = read_item(ann, items[item_id])
            shown = image[0]
            low, high = np.percentile(shown, (1, 99.5))
            panels = [
                ("image", None),
                ("ground truth", truth),
                *(
                    (label, res.predictions.get(item_id))
                    for label, res in comparison.results.items()
                ),
            ]
            for c, (title, labels) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(shown, cmap="gray", vmin=low, vmax=high)
                if labels is not None:
                    edge = np.ma.masked_where(~find_boundaries(labels), np.ones_like(labels))
                    ax.imshow(edge, cmap="autumn", alpha=0.9, interpolation="nearest")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(title, fontsize=8)
            axes[r][0].text(2, 8, item_id, color="yellow", fontsize=6)
        fig.savefig(out / "examples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append("examples.png")
    return figures


def write_comparison(comparison: Comparison, out: Path, title: str = "", examples: int = 4) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    table = comparison.table()
    table.to_csv(out / "comparison.csv", index=False)
    per_line = []
    for label, result in comparison.results.items():
        for line, values in result.report.get("by_cell_line", {}).items():
            per_line.append(
                {
                    "model": label,
                    "cell line": line or "–",
                    "AP@0.5": values["ap50"],
                    "F1": values["f1_50"],
                    "fields": values["fields"],
                }
            )
        result.fields.to_csv(out / f"fields_{_slug(label)}.csv", index=False)
    record = {
        "set": comparison.set_ref,
        "split": comparison.split,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {
            label: {"model": r.model, **r.report} for label, r in comparison.results.items()
        },
    }
    (out / "comparison.json").write_text(
        json.dumps(record, indent=1, default=float), encoding="utf-8"
    )
    figures = _figures(comparison, out, examples)

    def to_html(frame: pd.DataFrame) -> str:
        return str(
            frame.to_html(float_format=lambda v: f"{v:.3f}", index=False, border=0, classes="t")
        )

    images = "".join(f'<img src="{f}" alt="{f}">' for f in figures)
    lines_html = to_html(pd.DataFrame(per_line)) if per_line else ""
    table_html = to_html(table)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Model comparison</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fff}}
.t{{border-collapse:collapse;font-size:13px}}
.t td,.t th{{padding:3px 10px;border-bottom:1px solid #ddd}}
img{{max-width:48%;margin:6px;vertical-align:top}}.note{{color:#666}}</style></head><body>
<h1>{html.escape(title or "Segmentation model comparison")}</h1>
<p>Ground truth {html.escape(comparison.set_ref)}, split <b>{html.escape(comparison.split)}</b>
(never used for training). Objects are matched one-to-one; AP = TP / (TP + FP + FN).</p>
{table_html}<h2>Per cell line</h2>{lines_html}{images}
<p class=note>Fields with the fewest matches are shown as examples (true outlines and each
model's outlines in orange).</p></body></html>"""
    path = out / "comparison.html"
    path.write_text(page, encoding="utf-8")
    return path


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def compare_candidate(
    candidate: str,
    baseline: str | None = None,
    set_ref: str | None = None,
    split: str = "test",
    out: Path | None = None,
    pixel_size_um: float | None = None,
) -> tuple[Comparison, Path]:
    """Score a registered candidate against a baseline on held-out ground truth and save the
    report (default: next to the candidate in the model store)."""
    from hcs_studio.training.registry import models_root, update_metrics

    model = resolve(candidate)
    models: dict[str, list[PipelineStep]] = {
        f"{model.name}@{model.version} (candidate)": model_steps(model)
    }
    if baseline:
        other = resolve(baseline)
        models[f"{other.name}@{other.version}"] = model_steps(other)
    else:
        try:
            live = get_model(model.name, "production")
        except ValueError:
            live = None
        if live is not None and live.version != model.version:
            models[f"{live.name}@{live.version} (production)"] = model_steps(live)
        else:
            models[f"base: {model.base_model or 'defaults'}"] = base_steps(model)
    target_set = set_ref or model.training_set
    if not target_set:
        raise ValueError(f"{model.ref} has no training set: give the ground truth to compare on")
    comparison = compare_segmentation(target_set, models, split, pixel_size_um)
    out = out or models_root() / _slug(model.name) / f"v{model.version:04d}" / "reports" / _slug(
        f"{comparison.set_ref}-{split}"
    )
    report = write_comparison(
        comparison, out, f"{model.name}@{model.version} vs {', '.join(list(models)[1:])}"
    )
    first = next(iter(comparison.results.values())).overall
    update_metrics(
        model.name,
        model.version,
        {
            "test": {
                "set": comparison.set_ref,
                "split": split,
                "ap50": first["ap50"],
                "map": first["map"],
                "f1_50": first["f1_50"],
                "report": str(report),
                "compared_with": {
                    label: r.overall["ap50"] for label, r in list(comparison.results.items())[1:]
                },
            }
        },
    )
    return comparison, report
