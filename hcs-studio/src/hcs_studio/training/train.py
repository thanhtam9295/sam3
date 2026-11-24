"""Training models on ground-truth sets and registering them (Phase 25).

Three trainers, all ending the same way — validation on the set's **validation** split against
the base model, then registration as a *candidate* with the training set (version and content
hash), hyperparameters, metrics and a feature summary of the training objects (for drift alerts):

- ``steps`` — a classical segmentation (``segment_classical``) tuned by grid search on the
  **train** split (threshold method, smoothing, minimum distance between centres, … — any
  parameter can be searched), scored by AP at IoU 0.5 with the counts summed over fields
  (ties: the higher mAP). Runs on any CPU in seconds; the model is the chosen steps as JSON.
- ``cellpose`` — Cellpose fine-tuning from a base model (``cpsam_v2`` by default) with epochs,
  learning rate and weight decay; practical only on a GPU.
- ``classifier`` — an object classifier (random forest or logistic regression) on a
  classification set (``objects.csv`` with a ``label`` column and numeric features).

``train`` runs in the calling process; ``run_training_job`` is the job-queue entry (it reports
``status``/``log``/``finished`` events like a pipeline job).
"""

from __future__ import annotations

import itertools
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.training.groundtruth import AnnotationSet, load_set, read_item
from hcs_studio.training.registry import ModelRecord, register_model

DEFAULT_GRID: dict[str, list[Any]] = {
    "method": ["otsu", "li", "triangle", "local"],
    "min_distance_um": [2.0, 3.0, 4.0, 5.0, 6.0],
    "smoothing_um": [0.5, 1.0, 2.0],
}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["steps", "cellpose", "classifier"]
    name: str = Field(min_length=1, description="Registry name of the model.")
    set: str = Field(description="Ground-truth set NAME[@VERSION] with train/validation splits.")
    channel: int = Field(default=1, ge=1, description="Channel to segment (from 1).")
    second_channel: int | None = Field(default=None, ge=1)
    base_model: str = Field(default="", description="Cellpose base model (cpsam_v2).")
    pixel_size_um: float | None = Field(default=None, gt=0)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    device: str = "auto"
    author: str = ""
    notes: str = ""


@dataclass
class TrainingResult:
    model: ModelRecord
    validation: dict[str, Any]
    baseline: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)


Log = Callable[[str], None]


def _items_with_masks(ann: AnnotationSet, split: str) -> int:
    return sum(1 for i in ann.subset(split))


def feature_summary(ann: AnnotationSet, split: str | None = "train") -> dict[str, Any]:
    """Quantiles of object features in the training masks (area, intensity, shape, density)."""
    from skimage.measure import regionprops_table

    rows = []
    per_field = []
    for item in ann.subset(split):
        image, masks = read_item(ann, item)
        if masks is None or not masks.any():
            continue
        props = pd.DataFrame(
            regionprops_table(
                masks, image[0], properties=("area", "mean_intensity", "eccentricity")
            )
        )
        if item.pixel_size_um:
            props["area_um2"] = props["area"] * item.pixel_size_um**2
        rows.append(props)
        per_field.append(len(props))
    if not rows:
        return {}
    table = pd.concat(rows, ignore_index=True)
    out: dict[str, Any] = {"objects": len(table), "fields": len(per_field), "features": {}}
    for column, name in (
        ("area_um2", "area_um2"),
        ("area", "area_px"),
        ("mean_intensity", "intensity_mean"),
        ("eccentricity", "eccentricity"),
    ):
        if column in table:
            values = table[column].to_numpy(float)
            out["features"][name] = {
                f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES
            }
    counts = np.asarray(per_field, float)
    out["features"]["objects_per_field"] = {
        f"q{int(q * 100):02d}": float(np.quantile(counts, q)) for q in QUANTILES
    }
    return out


def _segmentation_metrics(
    ann: AnnotationSet, steps: list[Any], split: str, spec: TrainingSpec
) -> dict[str, Any]:
    from hcs_studio.training.validate import validate_segmentation

    result = validate_segmentation(ann, steps, split=split, pixel_size_um=spec.pixel_size_um)
    return {"split": split, **result.report}


def _classical_step(params: dict[str, Any]) -> Any:
    from hcs_studio.pipeline.schema import PipelineStep

    return PipelineStep(module="segment_classical", params=params)


def train_steps(ann: AnnotationSet, spec: TrainingSpec, log: Log) -> TrainingResult:
    """Grid search of a classical segmentation on the train split."""
    from hcs_studio.training.validate import validate_segmentation

    hyper = spec.hyperparameters
    fixed = {"channel": spec.channel, **dict(hyper.get("fixed", {}))}
    grid = {k: list(v) for k, v in dict(hyper.get("grid", DEFAULT_GRID)).items()}
    combos = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    log(f"Searching {len(combos)} parameter sets on {_items_with_masks(ann, 'train')} train fields")
    best: tuple[float, float, dict[str, Any]] | None = None
    for combo in combos:
        params = {**fixed, **combo}
        overall = validate_segmentation(
            ann, [_classical_step(params)], split="train", pixel_size_um=spec.pixel_size_um
        ).overall
        score = (overall["ap50"], overall["map"])
        if best is None or score > best[:2]:
            best = (score[0], score[1], params)
    assert best is not None
    chosen = best[2]
    log(f"Best on train: AP@0.5 {best[0]:.3f} with {json.dumps({k: chosen[k] for k in grid})}")
    steps = [_classical_step(chosen)]
    validation = _segmentation_metrics(ann, steps, "validation", spec)
    baseline = _segmentation_metrics(ann, [_classical_step(fixed)], "validation", spec)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        path.write_text(
            json.dumps({"steps": [s.model_dump(mode="json") for s in steps]}, indent=1),
            encoding="utf-8",
        )
        record = register_model(
            spec.name,
            "steps",
            path,
            base_model="segment_classical (defaults)",
            training_set=ann.ref,
            hyperparameters={
                "fixed": fixed,
                "grid": grid,
                "selected": chosen,
                "train_ap50": best[0],
                "train_map": best[1],
            },
            metrics={"validation": validation, "baseline_validation": baseline},
            feature_summary=feature_summary(ann),
            author=spec.author,
            notes=spec.notes,
        )
    return TrainingResult(record, validation, baseline)


def _cellpose_step(model: str, spec: TrainingSpec) -> Any:
    from hcs_studio.pipeline.schema import PipelineStep

    params: dict[str, Any] = {"channel": spec.channel, "model": model, "device": spec.device}
    if spec.second_channel:
        params["second_channel"] = spec.second_channel
    for key in ("diameter_um", "flow_threshold", "cellprob_threshold", "min_size_px"):
        if key in spec.hyperparameters:
            params[key] = spec.hyperparameters[key]
    return PipelineStep(module="segment_cellpose", params=params)


def train_cellpose(ann: AnnotationSet, spec: TrainingSpec, log: Log) -> TrainingResult:
    from hcs_studio.segmentation.cellpose_backend import fine_tune
    from hcs_studio.training.groundtruth import export_pairs

    base = spec.base_model or "cpsam_v2"
    hyper = {
        "epochs": 100,
        "learning_rate": 1e-5,
        "weight_decay": 0.1,
        "min_train_masks": 5,
        **spec.hyperparameters,
    }
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "train"
        pairs = export_pairs(ann.ref, "train", folder)
        log(f"Fine-tuning {base} on {len(pairs)} fields for {hyper['epochs']} epochs")
        path, history = fine_tune(
            folder,
            base_model=base,
            model_name=spec.name,
            epochs=int(hyper["epochs"]),
            learning_rate=float(hyper["learning_rate"]),
            weight_decay=float(hyper["weight_decay"]),
            min_train_masks=int(hyper["min_train_masks"]),
            device=spec.device,
        )
        log("Validating the fine-tuned and the base model")
        validation = _segmentation_metrics(
            ann, [_cellpose_step(str(path), spec)], "validation", spec
        )
        baseline = _segmentation_metrics(ann, [_cellpose_step(base, spec)], "validation", spec)
        record = register_model(
            spec.name,
            "cellpose",
            Path(path),
            base_model=base,
            training_set=ann.ref,
            hyperparameters={
                **hyper,
                "channel": spec.channel,
                "second_channel": spec.second_channel,
                "final_train_loss": (history.get("train_loss") or [None])[-1],
            },
            metrics={"validation": validation, "baseline_validation": baseline},
            feature_summary=feature_summary(ann),
            author=spec.author,
            notes=spec.notes,
        )
    return TrainingResult(record, validation, baseline)


def train_object_classifier(ann: AnnotationSet, spec: TrainingSpec, log: Log) -> TrainingResult:
    from hcs_studio.analysis.classifier import robust_stats, save_classifier, train_classifier
    from hcs_studio.training.groundtruth import objects_table
    from hcs_studio.training.metrics import classifier_metrics

    table = objects_table(ann)
    if "label" not in table:
        raise ValueError(f"{ann.ref}: objects.csv needs a 'label' column")
    train, valid = table[table["split"] == "train"], table[table["split"] == "validation"]
    if train.empty or valid.empty:
        raise ValueError(f"{ann.ref}: split the set so it has train and validation objects")
    hyper = {
        "algorithm": "random_forest",
        "normalisation": "plate_robust_z",
        "seed": 0,
        **spec.hyperparameters,
    }
    meta = {"item", "label", "split", "plate", "well", "field", "object_id"}
    features = [
        c for c in train.columns if c not in meta and pd.api.types.is_numeric_dtype(train[c])
    ]
    log(f"Training a {hyper['algorithm']} on {len(train)} objects, {len(features)} features")
    classifier = train_classifier(
        train,
        train["label"].astype(str).tolist(),
        features=features,
        algorithm=hyper["algorithm"],
        normalisation=hyper["normalisation"],
        seed=int(hyper["seed"]),
    )
    stats = (
        robust_stats(valid, classifier.features)
        if classifier.normalisation == "plate_robust_z"
        else None
    )
    proba = classifier.probabilities(valid, stats)
    predicted = [classifier.classes[i] for i in proba.argmax(axis=1)]
    validation = {
        "split": "validation",
        **classifier_metrics(valid["label"].astype(str), predicted, proba, classifier.classes),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{spec.name}.joblib"
        save_classifier(classifier, path)
        record = register_model(
            spec.name,
            "classifier",
            path,
            base_model=hyper["algorithm"],
            training_set=ann.ref,
            hyperparameters={**hyper, "features": classifier.features},
            metrics={
                "validation": validation,
                "cv_balanced_accuracy": classifier.cv_balanced_accuracy,
            },
            author=spec.author,
            notes=spec.notes,
        )
    return TrainingResult(record, validation)


TRAINERS: dict[str, Callable[[AnnotationSet, TrainingSpec, Log], TrainingResult]] = {
    "steps": train_steps,
    "cellpose": train_cellpose,
    "classifier": train_object_classifier,
}


def train(spec: TrainingSpec, log: Log | None = None) -> TrainingResult:
    """Train, validate against the base model and register a candidate."""
    messages: list[str] = []

    def note(text: str) -> None:
        messages.append(text)
        if log is not None:
            log(text)

    ann = load_set(spec.set)
    counts = ann.counts()
    if not counts["train"] or not counts["validation"]:
        raise ValueError(
            f"{ann.ref} needs train and validation items (hcs-studio gt split …); it has {counts}"
        )
    if spec.kind in ("steps", "cellpose") and ann.kind != "segmentation":
        raise ValueError(f"{ann.ref} is a {ann.kind} set; {spec.kind} needs segmentation masks")
    result = TRAINERS[spec.kind](ann, spec, note)
    result.log = messages
    note(f"Registered {result.model.ref} (candidate)")
    return result


def run_training_job(spec: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> None:
    """Job-queue entry: train and report events (never raises)."""
    import traceback

    try:
        parsed = TrainingSpec.model_validate(spec)
        emit({"type": "status", "text": f"Training {parsed.name} ({parsed.kind})"})
        result = train(parsed, lambda text: emit({"type": "status", "text": text}))
        emit(
            {
                "type": "finished",
                "status": "completed",
                "model": result.model.ref,
                "validation": result.validation.get("overall", {}).get("ap50"),
            }
        )
    except Exception as exc:
        emit({"type": "log", "level": "error", "text": traceback.format_exc()})
        emit({"type": "finished", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
