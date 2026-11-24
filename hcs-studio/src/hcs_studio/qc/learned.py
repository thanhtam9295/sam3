"""Learned field QC (Phase 18): a classifier trained on fields people reviewed.

Training examples come from the reviews (``qc/overrides.json``) of one or more QC folders:
fields whose flag was confirmed or that were excluded by hand are ``bad``; fields included
against a flag are ``good``; optionally, unflagged fields nobody overrode count as ``good``
too (up to ``max_unflagged`` per folder, sampled). Inputs are the QC metrics of every
channel and the 28-number image embedding (``emb_*``), each turned into a robust Z within
its own plate, so the classifier learns what is unusual on a plate rather than absolute
levels. The Phase 8 classifier (random forest, balanced classes, stratified CV) does the
learning; the file format is the classifier format (``hcs-studio-classifier/1``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hcs_studio.analysis.classifier import (
    TrainedClassifier,
    feature_matrix,
    file_sha256,
    load_classifier,
    robust_stats,
    train_classifier,
)
from hcs_studio.qc.rules import CHANNEL_METRICS, FIELD_METRICS, LearnedQc

#: Per-channel metrics used (positions of the focus plane are not appearance).
_CHANNEL = [m for m in CHANNEL_METRICS if m not in ("best_z", "best_z_offset")]


def learned_features(frame: pd.DataFrame) -> list[str]:
    names = []
    for column in frame.columns:
        name = str(column)
        match = re.match(r"c\d+_(.+)$", name)
        wanted = (match and match.group(1) in _CHANNEL) or name in FIELD_METRICS
        if (wanted or name.startswith("emb_")) and pd.to_numeric(
            frame[name], errors="coerce"
        ).notna().any():
            names.append(name)
    return names


def _plate_z(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    stats = robust_stats(frame, features)
    matrix = feature_matrix(frame, features, stats, {})
    return pd.DataFrame(matrix, columns=features, index=frame.index)


def training_examples(
    folders: Sequence[Path], *, unflagged_good: bool = True, max_unflagged: int = 200, seed: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    """Plate-normalised features and labels (``bad`` / ``good``) from reviewed QC folders."""
    from hcs_studio.qc.review import latest_overrides, load_fields

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    labels: list[str] = []
    for folder in folders:
        fields = load_fields(Path(folder))
        features = learned_features(fields)
        z = _plate_z(fields, features)
        reviewed: dict[int, str] = {}
        for (level, well, field), override in latest_overrides(Path(folder)).items():
            if level != "field":
                continue
            hit = fields.index[(fields["well"] == well) & (fields["field"] == field)]
            for i in hit:
                reviewed[int(i)] = "good" if override.action == "include" else "bad"
        if unflagged_good:
            quiet = [
                int(i)
                for i in fields.index[fields["flags"].fillna("") == ""]
                if int(i) not in reviewed
            ]
            if len(quiet) > max_unflagged:
                quiet = sorted(rng.choice(quiet, size=max_unflagged, replace=False).tolist())
            reviewed.update(dict.fromkeys(quiet, "good"))
        rows = sorted(reviewed)
        parts.append(z.loc[rows])
        labels.extend(reviewed[i] for i in rows)
    if not parts:
        raise ValueError("No QC folders given")
    examples = pd.concat(parts, ignore_index=True)
    return examples, labels


def train_field_qc(
    folders: Sequence[Path], *, unflagged_good: bool = True, max_unflagged: int = 200, seed: int = 0
) -> TrainedClassifier:
    examples, labels = training_examples(
        folders, unflagged_good=unflagged_good, max_unflagged=max_unflagged, seed=seed
    )
    counts = {c: labels.count(c) for c in ("bad", "good")}
    if min(counts.values()) < 3:
        raise ValueError(
            f"Learned QC needs at least 3 bad and 3 good reviewed fields (have {counts}); "
            "review more fields first"
        )
    features = list(examples.columns)
    classifier = train_classifier(
        examples, labels, features=features, normalisation="none", object_type="field", seed=seed
    )
    # inputs are robust Z within each plate: predictions normalise the same way
    classifier.normalisation = "plate_robust_z"
    return classifier


def bad_probability(fields: pd.DataFrame, learned: LearnedQc) -> pd.Series:
    """P(bad) of each field from a learned-QC classifier (checked against its SHA-256)."""
    from hcs_studio.analysis.classifier import registered_file

    path = Path(registered_file(learned.model, "qc_classifier"))
    if learned.sha256 and file_sha256(path) != learned.sha256:
        raise ValueError(f"The learned-QC model {path} changed since the rules were written")
    classifier = load_classifier(path)
    stats = robust_stats(fields, [f for f in classifier.features if f in fields.columns])
    probabilities = classifier.probabilities(fields, stats)
    column = classifier.classes.index("bad")
    return pd.Series(probabilities[:, column], index=fields.index)
