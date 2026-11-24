"""Ongoing model QA (Phase 25): human spot checks and drift alerts.

**Spot checks.** After a run, a random sample of fields (``n`` per plate) is queued in the
``spot_check`` table with the segmentation models the run used. A person looks at the field and
its outlines and accepts or flags it (with a reason); accept rates per model over time show
real-world accuracy, which held-out ground truth alone cannot.

**Drift.** Every trained model keeps quantiles of its training objects (area, intensity, shape,
objects per field). For a new run, the same features of its objects are compared with them:
``shift`` = (run median − training median) / training IQR, and ``spread`` = run IQR / training
IQR. A feature drifts when |shift| > ``threshold`` (1 IQR by default) or the spread leaves
[0.5, 2]. Drift does not mean the masks are wrong — it means the model now sees cells unlike
those it was trained on (a new cell line, stain or magnification): annotate a few fields of
the plate and check or fine-tune.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.training.registry import ModelRecord, get_model, is_ref, parse_ref

SEGMENTATION_MODULES = ("segment_cellpose", "segment_model", "segment_sam3")


def run_fields(folder: Path) -> list[tuple[str, int]]:
    """(well, field) of every field a run wrote."""
    out = []
    for summary in sorted(Path(folder).glob("wells/*/well_summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        for entry in data.get("fields") or []:
            out.append((str(data.get("well", summary.parent.name)), int(entry["field"])))
    return out


def run_models(folder: Path) -> list[dict[str, Any]]:
    """Registered segmentation models of a run (from its manifest)."""
    manifest = json.loads((Path(folder) / "run_manifest.json").read_text(encoding="utf-8"))
    return [
        m
        for m in manifest.get("models") or []
        if m.get("registry") and m.get("module") in SEGMENTATION_MODULES
    ]


def _labels(folder: Path) -> str:
    manifest = json.loads((Path(folder) / "run_manifest.json").read_text(encoding="utf-8"))
    for step in manifest.get("pipeline", {}).get("steps", []):
        if step.get("module") in SEGMENTATION_MODULES or step.get("module") == "segment_classical":
            return str(step.get("params", {}).get("output", "") or "")
    return ""


def queue_spot_checks(folder: Path, n: int = 3, seed: int | None = None) -> list[int]:
    """Queue ``n`` random fields of a run (once per run: later calls return the same ids)."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import SpotCheck
    from hcs_studio.output.manifest import rooted_text

    folder = Path(folder)
    key = rooted_text(folder.absolute())
    with session_scope() as session:
        existing = list(session.scalars(select(SpotCheck.id).where(SpotCheck.run == key)))
        if existing:
            return existing
        fields = run_fields(folder)
        chosen = random.Random(seed).sample(fields, min(n, len(fields)))
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        barcode = str(manifest.get("plate", {}).get("barcode") or "")
        models = ", ".join(
            f"{m['registry']['name']}@{m['registry']['version']}" for m in run_models(folder)
        )
        labels = _labels(folder)
        rows = [
            SpotCheck(run=key, barcode=barcode, well=w, field=f, labels=labels, model=models)
            for w, f in sorted(chosen)
        ]
        session.add_all(rows)
        session.flush()
        return [r.id for r in rows]


def review_spot_check(
    check_id: int, status: str, reviewer: str, reason: str = ""
) -> dict[str, Any]:
    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import SpotCheck

    if status not in ("accepted", "flagged", "pending"):
        raise ValueError("A spot check is accepted, flagged or pending")
    if status == "flagged" and not reason.strip():
        raise ValueError("Say what is wrong when flagging a field")
    with session_scope() as session:
        row = session.get(SpotCheck, check_id)
        if row is None:
            raise ValueError(f"No spot check {check_id}")
        before = row.status
        row.status, row.reviewer, row.reason = status, reviewer, reason
        row.decided = datetime.now(UTC) if status != "pending" else None
        return {
            "id": row.id,
            "before": before,
            "status": status,
            "run": row.run,
            "well": row.well,
            "field": row.field,
        }


def spot_checks(status: str | None = None, model: str | None = None) -> pd.DataFrame:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import SpotCheck

    query = select(SpotCheck).order_by(SpotCheck.created, SpotCheck.id)
    if status:
        query = query.where(SpotCheck.status == status)
    with session_scope() as session:
        rows = [
            {
                "id": r.id,
                "run": r.run,
                "barcode": r.barcode,
                "well": r.well,
                "field": r.field,
                "labels": r.labels,
                "model": r.model,
                "status": r.status,
                "reviewer": r.reviewer,
                "reason": r.reason,
                "created": r.created,
                "decided": r.decided,
            }
            for r in session.scalars(query)
        ]
    table = pd.DataFrame(
        rows,
        columns=[
            "id",
            "run",
            "barcode",
            "well",
            "field",
            "labels",
            "model",
            "status",
            "reviewer",
            "reason",
            "created",
            "decided",
        ],
    )
    if model:
        table = table[table["model"].str.contains(model, regex=False)]
    return table


def spot_check_summary() -> pd.DataFrame:
    """Per model: checks, accepted, flagged, pending and the accept rate of decided ones."""
    table = spot_checks()
    if table.empty:
        return pd.DataFrame(
            columns=["model", "checks", "accepted", "flagged", "pending", "accept_rate"]
        )
    table["model"] = table["model"].replace("", "(no registered model)")
    out = table.groupby("model")["status"].value_counts().unstack(fill_value=0)
    for column in ("accepted", "flagged", "pending"):
        if column not in out:
            out[column] = 0
    out["checks"] = out[["accepted", "flagged", "pending"]].sum(axis=1)
    decided = out["accepted"] + out["flagged"]
    out["accept_rate"] = np.where(
        decided > 0, out["accepted"] / decided.where(decided > 0, 1), np.nan
    )
    return out.reset_index()[["model", "checks", "accepted", "flagged", "pending", "accept_rate"]]


def field_files(folder: Path, well: str, field: int, labels: str = "") -> dict[str, Path]:
    """The preview and outline images of a field for reviewing (those that exist)."""
    base = Path(folder) / "wells" / well / "previews"
    out = {"preview": base / f"f{field:03d}.png", "raw": base / f"f{field:03d}_raw.png"}
    if labels:
        out["outline"] = base / f"f{field:03d}_outline_{labels}.png"
    return {k: v for k, v in out.items() if v.exists()}


# --- drift ---------------------------------------------------------------------------------


def run_object_features(
    folder: Path, labels: str, channel_prefix: str | None, max_objects: int = 20000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Feature values of a run's objects (sampled) and its objects per field."""
    frames, per_field = [], []
    for path in sorted(Path(folder).glob("wells/*/objects.parquet")):
        table = pd.read_parquet(path)
        if "object_type" in table and labels:
            table = table[table["object_type"] == labels]
        if "in_overlap_duplicate" in table:
            table = table[~table["in_overlap_duplicate"].fillna(False).astype(bool)]
        per_field.extend(
            table.groupby("field").size().tolist() if "field" in table else [len(table)]
        )
        frames.append(table)
    if not frames:
        return {}
    objects = pd.concat(frames, ignore_index=True)
    if len(objects) > max_objects:
        objects = objects.sample(max_objects, random_state=seed)
    out: dict[str, np.ndarray] = {"objects_per_field": np.asarray(per_field, float)}
    for name, column in (
        ("area_um2", "area_um2"),
        ("eccentricity", "eccentricity"),
        ("intensity_mean", f"{channel_prefix}_mean" if channel_prefix else ""),
    ):
        if column and column in objects:
            out[name] = objects[column].dropna().to_numpy(float)
    return out


def drift(
    train: dict[str, Any], values: dict[str, np.ndarray], threshold: float = 1.0
) -> dict[str, Any]:
    """Compare run feature values with a model's training quantiles."""
    features = {}
    alerts = []
    for name, q in (train.get("features") or {}).items():
        data = values.get(name)
        if data is None or len(data) < 5:
            continue
        iqr = q["q75"] - q["q25"]
        if iqr <= 0:
            continue
        median = float(np.median(data))
        run_iqr = float(np.subtract(*np.percentile(data, [75, 25])))
        shift = (median - q["q50"]) / iqr
        spread = run_iqr / iqr
        drifted = abs(shift) > threshold or not 0.5 <= spread <= 2.0
        features[name] = {
            "train_median": q["q50"],
            "run_median": median,
            "shift_iqr": shift,
            "spread_ratio": spread,
            "drift": drifted,
            "n": len(data),
        }
        if drifted:
            alerts.append(
                f"{name}: median {median:.3g} vs {q['q50']:.3g} in training "
                f"({shift:+.1f} IQR, spread ×{spread:.2f})"
            )
    return {"features": features, "alerts": alerts, "threshold_iqr": threshold}


def drift_report(folder: Path, model: str | None = None, threshold: float = 1.0) -> dict[str, Any]:
    """Drift of a run's objects against the training data of the model it used (or ``model``)."""
    from hcs_studio.analysis.features import channel_prefixes
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata

    folder = Path(folder)
    record: ModelRecord
    if model:
        name, version = parse_ref(model if is_ref(model) else f"registry:{model}")
        record = get_model(name, version)
    else:
        used = run_models(folder)
        if not used:
            return {
                "model": None,
                "alerts": [],
                "note": "the run used no registered segmentation model",
            }
        reg = used[0]["registry"]
        record = get_model(reg["name"], reg["version"])
    if not record.feature_summary:
        return {
            "model": record.ref,
            "alerts": [],
            "note": "the model has no training feature summary",
        }
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    channels = [c["name"] for c in manifest["plate"]["metadata"].get("channels", [])]
    channel = int(
        record.hyperparameters.get("channel")
        or record.hyperparameters.get("fixed", {}).get("channel")
        or 1
    )
    prefixes = channel_prefixes(ImageMetadata(channels=tuple(ChannelInfo(c) for c in channels)))
    prefix = prefixes[channel - 1] if 0 < channel <= len(prefixes) else None
    labels = _labels(folder) or "nuclei"
    values = run_object_features(folder, labels, prefix)
    result = drift(record.feature_summary, values, threshold)
    result.update({"model": record.ref, "training_set": record.training_set, "labels": labels})
    if result["alerts"]:
        result["suggestion"] = (
            f"Objects on this plate differ from the training data of "
            f"{record.ref}: check a few fields, annotate 10–20 and fine-tune "
            "if the masks are off."
        )
    return result


def model_qa(
    folder: Path, spot_check_fields: int = 3, seed: int | None = None, drift_threshold: float = 1.0
) -> dict[str, Any]:
    """Queue spot checks and check drift for a finished run; writes ``qa/model_qa.json``."""
    record: dict[str, Any] = {"spot_checks": [], "drift": None}
    if spot_check_fields:
        record["spot_checks"] = queue_spot_checks(folder, spot_check_fields, seed)
    record["drift"] = drift_report(folder, threshold=drift_threshold)
    out = Path(folder) / "qa"
    out.mkdir(exist_ok=True)
    (out / "model_qa.json").write_text(
        json.dumps(record, indent=1, default=float), encoding="utf-8"
    )
    return record
