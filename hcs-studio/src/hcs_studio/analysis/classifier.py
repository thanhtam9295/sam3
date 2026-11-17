"""Object classifier trained on example objects (clicked in the viewer).

A classifier stores its feature list, the normalisation it was trained with and its
cross-validated accuracy. With ``plate_robust_z`` normalisation each plate's features are
expressed as robust Z-scores of that plate (median, 1.4826·MAD), which removes plate-to-plate
intensity differences before the model sees them.

Saved classifiers are joblib (pickle) files: only load files you or your colleagues made.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.analysis.table_steps import (
    OBJECT_KEY,
    TableModule,
    TableParams,
    WellContext,
    column_name,
    count_fractions,
    feature_columns,
)
from hcs_studio.processing.base import MODULES, ModuleError

Algorithm = Literal["random_forest", "hist_gradient_boosting"]
Normalisation = Literal["none", "plate_robust_z"]
Stats = dict[str, tuple[float, float]]
FORMAT = "hcs-studio-classifier/1"


def robust_stats(frame: pd.DataFrame, features: list[str]) -> Stats:
    """(median, 1.4826·MAD) per feature; the scale falls back to the SD, then 1."""
    stats: Stats = {}
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            stats[name] = (0.0, 1.0)
            continue
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        if not scale > 0:
            scale = float(values.std()) if values.std() > 0 else 1.0
        stats[name] = (median, scale)
    return stats


def feature_matrix(
    frame: pd.DataFrame, features: list[str], stats: Stats | None, fill: dict[str, float]
) -> NDArray[np.float64]:
    """Features as a float matrix, normalised with ``stats`` (if given); NaN → ``fill``."""
    columns = []
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        if stats is not None:
            median, scale = stats[name]
            values = (values - median) / scale
        values = np.where(np.isfinite(values), values, fill.get(name, 0.0))
        columns.append(values)
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


@dataclass
class TrainedClassifier:
    model: Any
    features: list[str]
    classes: list[str]
    algorithm: str
    normalisation: str
    object_type: str
    #: Normalised value used for missing features (training median).
    fill: dict[str, float]
    #: Plate statistics of the training data (for reference; each plate uses its own).
    reference_stats: Stats
    n_examples: dict[str, int]
    cv_folds: int = 0
    cv_accuracy: float | None = None
    cv_balanced_accuracy: float | None = None
    #: Rows = true class, columns = predicted class, in ``classes`` order.
    confusion: list[list[int]] = field(default_factory=list)
    trained: str = ""
    sklearn_version: str = ""

    def probabilities(self, frame: pd.DataFrame, stats: Stats | None) -> NDArray[np.float64]:
        """Class probabilities (n, len(classes)); ``stats`` = the plate's robust stats."""
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ModuleError(
                f"The classifier needs {len(missing)} feature(s) this table lacks, e.g. "
                f"{', '.join(missing[:3])}; measure the same features as for training"
            )
        use = stats if self.normalisation == "plate_robust_z" else None
        if use is None and self.normalisation == "plate_robust_z":
            raise ModuleError("plate statistics are needed for plate_robust_z normalisation")
        matrix = feature_matrix(frame, self.features, use, self.fill)
        if not len(matrix):
            return np.zeros((0, len(self.classes)))
        return np.asarray(self.model.predict_proba(matrix), dtype=float)

    def summary(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "algorithm": self.algorithm,
            "normalisation": self.normalisation,
            "object_type": self.object_type,
            "features": len(self.features),
            "examples": self.n_examples,
            "cv_folds": self.cv_folds,
            "cv_accuracy": self.cv_accuracy,
            "cv_balanced_accuracy": self.cv_balanced_accuracy,
            "confusion": self.confusion,
            "trained": self.trained,
            "sklearn_version": self.sklearn_version,
        }


def _make_model(algorithm: str, seed: int) -> Any:
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, class_weight="balanced", random_state=seed
        )
    if algorithm == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=seed
        )
    raise ModuleError(f"Unknown algorithm '{algorithm}'")


def train_classifier(
    examples: pd.DataFrame,
    labels: Sequence[str],
    *,
    features: list[str] | None = None,
    algorithm: Algorithm = "random_forest",
    normalisation: Normalisation = "plate_robust_z",
    plate_stats: Stats | None = None,
    object_type: str = "",
    cv_folds: int = 5,
    seed: int = 0,
) -> TrainedClassifier:
    """Train on labelled example objects and estimate accuracy by stratified CV.

    ``plate_stats``: robust statistics of all objects on the training plate (needed for
    ``plate_robust_z``; computed from the examples if not given, which is less reliable).
    """
    import sklearn
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = np.asarray([str(v) for v in labels])
    if len(y) != len(examples):
        raise ValueError("one label per example is needed")
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        raise ModuleError("give examples of at least two classes")
    names = feature_columns(examples, features)
    names = [n for n in names if pd.to_numeric(examples[n], errors="coerce").notna().any()]
    if not names:
        raise ModuleError("the examples have no numeric features")
    stats = None
    if normalisation == "plate_robust_z":
        stats = plate_stats or robust_stats(examples, names)
        missing = [n for n in names if n not in stats]
        if missing:
            stats = {**stats, **robust_stats(examples, missing)}
    raw = feature_matrix(examples, names, stats, {})
    fill = {}
    for k, name in enumerate(names):
        finite = raw[:, k][np.isfinite(raw[:, k])]
        fill[name] = float(np.median(finite)) if len(finite) else 0.0
    x = feature_matrix(examples, names, stats, fill)
    counts = {c: int((y == c).sum()) for c in classes}
    folds = min(cv_folds, min(counts.values()))
    result = TrainedClassifier(
        model=None,
        features=names,
        classes=classes,
        algorithm=algorithm,
        normalisation=normalisation,
        object_type=object_type,
        fill=fill,
        reference_stats=stats or {},
        n_examples=counts,
        trained=datetime.now(UTC).isoformat(),
        sklearn_version=sklearn.__version__,
    )
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        predicted = cross_val_predict(_make_model(algorithm, seed), x, y, cv=splitter)
        result.cv_folds = folds
        result.cv_accuracy = float(accuracy_score(y, predicted))
        result.cv_balanced_accuracy = float(balanced_accuracy_score(y, predicted))
        result.confusion = confusion_matrix(y, predicted, labels=classes).tolist()
    model = _make_model(algorithm, seed)
    model.fit(x, y)
    if list(model.classes_) != classes:
        raise RuntimeError("class order changed during training")
    result.model = model
    return result


def uncertain_objects(
    classifier: TrainedClassifier,
    objects: pd.DataFrame,
    stats: Stats | None,
    n: int = 20,
    exclude: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Active learning: the ``n`` objects the classifier is least sure about (smallest margin
    between the two most likely classes), at most a few per field so suggestions are varied.
    ``exclude``: objects already labelled (key columns)."""
    candidates = objects[objects["object_type"] == classifier.object_type].copy()
    if exclude is not None and len(exclude):
        keys = [c for c in OBJECT_KEY if c in candidates and c in exclude]
        labelled = set(map(tuple, exclude[keys].astype(str).to_numpy().tolist()))
        mask = [tuple(r) not in labelled for r in candidates[keys].astype(str).to_numpy().tolist()]
        candidates = candidates[mask]
    if candidates.empty:
        return candidates.assign(margin=[], predicted=[])
    probabilities = classifier.probabilities(candidates, stats)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - (ordered[:, -2] if ordered.shape[1] > 1 else 0)
    candidates["margin"] = margin
    candidates["predicted"] = np.asarray(classifier.classes)[probabilities.argmax(axis=1)]
    candidates = candidates.sort_values("margin", kind="stable")
    per_field = max(1, math.ceil(n / 5))
    groups = ["well", "field"] if {"well", "field"}.issubset(candidates.columns) else []
    if groups:
        candidates = candidates.groupby(groups, sort=False).head(per_field)
    return candidates.head(n)


def run_objects(folder: Path, object_type: str) -> pd.DataFrame:
    """Objects of one type from a run folder (each object once where fields overlap)."""
    path = Path(folder) / "tables" / "per_object_all.parquet"
    if not path.exists():
        raise ModuleError(f"{folder} has no object table (tables/per_object_all.parquet)")
    import pyarrow.parquet as pq

    frame: pd.DataFrame = pq.read_table(
        path, filters=[("object_type", "=", object_type)]
    ).to_pandas()
    if frame.empty:
        raise ModuleError(f"The run has no '{object_type}' objects")
    if "in_overlap_duplicate" in frame:
        frame = frame[~frame["in_overlap_duplicate"].fillna(False).astype(bool)]
    return frame.reset_index(drop=True)


def labelled_examples(objects: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join example labels (well, field, object_id, label; optional timepoint) to objects."""
    labels = labels.copy()
    if "timepoint" not in labels:
        labels["timepoint"] = 0
    needed = {"well", "field", "object_id", "label"}
    if not needed.issubset(labels.columns):
        raise ModuleError(f"The labels table needs the columns {sorted(needed)}")
    keys = ["well", "field", "timepoint", "object_id"]
    labels = labels[[*keys, "label"]].drop_duplicates(keys, keep="last")
    for key in ("field", "timepoint", "object_id"):
        labels[key] = labels[key].astype(int)
    merged = objects.merge(labels, on=keys, how="inner")
    if len(merged) < len(labels):
        raise ModuleError(f"{len(labels) - len(merged)} labelled object(s) are not in the run")
    return merged


def train_from_run(
    folder: Path,
    labels: pd.DataFrame,
    object_type: str,
    **options: Any,
) -> TrainedClassifier:
    """Train on objects of a run folder labelled in ``labels``; plate statistics come from
    all objects of the run."""
    objects = run_objects(folder, object_type)
    examples = labelled_examples(objects, labels)
    features = feature_columns(objects, options.pop("features", None))
    return train_classifier(
        examples,
        examples["label"].astype(str).tolist(),
        features=features,
        plate_stats=robust_stats(objects, features),
        object_type=object_type,
        **options,
    )


def save_classifier(classifier: TrainedClassifier, path: Path) -> None:
    import joblib

    path = Path(path)
    part = path.with_name(path.name + ".part")
    joblib.dump({"format": FORMAT, **asdict(classifier)}, part)
    part.replace(path)


def load_classifier(path: Path) -> TrainedClassifier:
    """Load a classifier saved by HCS Studio (a pickle: only load trusted files)."""
    import joblib

    try:
        data = joblib.load(Path(path))
    except FileNotFoundError:
        raise ModuleError(f"Classifier file not found: {path}") from None
    if not isinstance(data, dict) or data.pop("format", None) != FORMAT:
        raise ModuleError(f"{path} is not an HCS Studio classifier")
    data["reference_stats"] = {k: tuple(v) for k, v in data["reference_stats"].items()}
    return TrainedClassifier(**data)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ pipeline step


class ClassifyParams(TableParams):
    name: str = Field(
        default="class",
        pattern=r"^[A-Za-z0-9_]+$",
        description="Objects get class_<name>, class_<name>_probability, class_<name>_p_<class>.",
    )
    model_path: str = Field(
        default="", description="Classifier file (.joblib) saved by HCS Studio."
    )
    min_probability: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Objects whose best class is less likely than this are 'unassigned'.",
    )


@MODULES.register("classify_objects")
class ClassifyObjects(TableModule):
    """Machine-learning object classes (random forest or gradient boosting) from a classifier
    trained on example objects; per-well class counts and fractions."""

    name = "classify_objects"
    title = "Classify objects"
    Params = ClassifyParams
    needs_fit = True  # loads the classifier and the plate's normalisation statistics

    def fit_table(self, sample: pd.DataFrame) -> dict[str, Any]:
        params: ClassifyParams = self.params
        if not params.model_path:
            raise ModuleError("choose a classifier file (train one in the Classify panel)")
        classifier = load_classifier(Path(params.model_path))
        if classifier.object_type and classifier.object_type != params.object_type:
            raise ModuleError(
                f"the classifier was trained on '{classifier.object_type}', not "
                f"'{params.object_type}'"
            )
        missing = [f for f in classifier.features if f not in sample.columns]
        if missing:
            raise ModuleError(
                f"the object table lacks {len(missing)} feature(s) the classifier uses, e.g. "
                f"{', '.join(missing[:3])}"
            )
        stats = (
            robust_stats(sample, classifier.features)
            if classifier.normalisation == "plate_robust_z"
            else None
        )
        return {
            "classifier": classifier,
            "stats": stats,
            "sha256": file_sha256(Path(params.model_path)),
        }

    def transform(self, objects: pd.DataFrame, state: Any, well: WellContext) -> pd.DataFrame:
        params: ClassifyParams = self.params
        classifier: TrainedClassifier = state["classifier"]
        prefix = f"class_{params.name}"
        rows = self.selected(objects)
        out = pd.DataFrame(index=objects.index)
        out[prefix] = pd.Series(None, index=objects.index, dtype="object")
        out[f"{prefix}_probability"] = np.nan
        for label in classifier.classes:
            out[f"{prefix}_p_{column_name(label)}"] = np.nan
        if rows.any():
            probabilities = classifier.probabilities(objects[rows], state["stats"])
            best = probabilities.max(axis=1)
            names = np.asarray(classifier.classes, dtype=object)[probabilities.argmax(axis=1)]
            names = np.where(best >= params.min_probability, names, "unassigned")
            out.loc[rows, prefix] = names
            out.loc[rows, f"{prefix}_probability"] = best
            for k, label in enumerate(classifier.classes):
                out.loc[rows, f"{prefix}_p_{column_name(label)}"] = probabilities[:, k]
        return out

    def summarize(
        self, objects: pd.DataFrame, state: Any, well: WellContext
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        classifier: TrainedClassifier = state["classifier"]
        prefix = f"class_{self.params.name}"
        labels = list(classifier.classes)
        if self.params.min_probability > 0:
            labels.append("unassigned")
        return count_fractions(prefix, objects[prefix], labels), {}

    def describe_state(self, state: Any) -> dict[str, Any]:
        classifier: TrainedClassifier = state["classifier"]
        return {
            "model_path": self.params.model_path,
            "model_sha256": state["sha256"],
            **classifier.summary(),
        }
