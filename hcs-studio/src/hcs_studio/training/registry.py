"""The model registry (Phase 25): which model produced every number.

Each registered version is copied into ``<settings>/models/<name>/vNNNN/`` (never changed
afterwards) and recorded in the ``model_version`` table with its kind, base model, training set
(name, version, content hash), hyperparameters, validation metrics, author, date, status and
notes. Statuses move **candidate → validated → production → retired**; promotion to production
needs an explicit approval (a confirmation with a reason until Phase 26 adds roles), and the
previous production version of the same name goes back to *validated*.

Pipelines reference models as ``registry:<name>@<version>`` (Cellpose ``model``, the object
classifier's ``model_path``, learned image QC, and ``segment_model`` for registered step sets);
``@production`` is resolved — and should be pinned with ``pin_pipeline`` — before a run, so a
saved pipeline always names an exact version. ``pipeline_models`` lists every model a pipeline
uses with its file and SHA-256 for the run manifest.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFIX = "registry:"
KINDS = ("cellpose", "steps", "classifier", "qc_classifier", "ilastik", "micro_sam")
STATUSES = ("candidate", "validated", "production", "retired")
TRANSITIONS = {
    "candidate": {"validated", "retired"},
    "validated": {"production", "retired", "candidate"},
    "production": {"validated", "retired"},
    "retired": {"candidate"},
}
#: Step parameters that name a model (file path, built-in name or registry reference).
MODEL_PARAMS = ("model", "model_path", "project", "model_file")


@dataclass(frozen=True)
class ModelRecord:
    name: str
    version: int
    kind: str
    status: str
    path: Path
    sha256: str
    base_model: str = ""
    training_set: str = ""
    training_set_sha256: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    author: str = ""
    notes: str = ""
    created: str = ""

    @property
    def ref(self) -> str:
        return f"{PREFIX}{self.name}@{self.version}"

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "status": self.status,
            "sha256": self.sha256,
            "training_set": self.training_set,
            "base_model": self.base_model,
        }


def models_root() -> Path:
    from hcs_studio.io.sidecar import config_dir

    return config_dir() / "models"


def is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def parse_ref(ref: str) -> tuple[str, str | None]:
    """``registry:name@3`` → ("name", "3"); ``@production`` → ("name", "production")."""
    body = ref.removeprefix(PREFIX)
    name, _, version = body.partition("@")
    if not name:
        raise ValueError(f"Not a model reference: {ref!r} (registry:NAME@VERSION)")
    return name, version or None


_HASHES: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file (cached by path, size and modification time); folders: a tree digest."""
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for sub in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(sub.relative_to(path).as_posix().encode())
            digest.update(file_sha256(sub).encode())
        return digest.hexdigest()
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


def _record(row: Any) -> ModelRecord:
    from hcs_studio.paths import resolve as resolve_rooted

    return ModelRecord(
        name=row.name,
        version=row.version,
        kind=row.kind,
        status=row.status,
        path=Path(resolve_rooted(row.file)),
        sha256=row.sha256,
        base_model=row.base_model,
        training_set=row.training_set,
        training_set_sha256=row.training_set_sha256,
        hyperparameters=dict(row.hyperparameters),
        metrics=dict(row.metrics),
        feature_summary=dict(row.feature_summary),
        author=row.author,
        notes=row.notes,
        created=row.created.isoformat() if row.created else "",
    )


def list_models(name: str | None = None, kind: str | None = None) -> list[ModelRecord]:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import ModelVersion

    query = select(ModelVersion).order_by(ModelVersion.name, ModelVersion.version)
    if name:
        query = query.where(ModelVersion.name == name)
    if kind:
        query = query.where(ModelVersion.kind == kind)
    with session_scope() as session:
        return [_record(r) for r in session.scalars(query)]


def get_model(name: str, version: int | str | None = None) -> ModelRecord:
    """A version by number, ``production``, or the latest (None)."""
    found = list_models(name)
    if not found:
        raise ValueError(f"No registered model named {name!r}")
    if version is None or version == "latest":
        return found[-1]
    if version == "production":
        live = [m for m in found if m.status == "production"]
        if not live:
            raise ValueError(f"{name} has no production version")
        return live[-1]
    try:
        number = int(version)
    except ValueError:
        raise ValueError(f"Version must be a number or 'production', not {version!r}") from None
    for model in found:
        if model.version == number:
            return model
    raise ValueError(f"{name} has no version {number} (versions: {[m.version for m in found]})")


def resolve(ref: str, kind: str | None = None) -> ModelRecord:
    """The model a ``registry:`` reference names; checks the file is intact."""
    name, version = parse_ref(ref)
    model = get_model(name, version)
    if kind and model.kind != kind:
        raise ValueError(f"{ref} is a {model.kind} model, not {kind}")
    if model.status == "retired":
        raise ValueError(f"{model.ref} is retired; choose another version")
    if not model.path.exists():
        raise ValueError(f"{model.ref}: file missing from the model store ({model.path})")
    if file_sha256(model.path) != model.sha256:
        raise ValueError(f"{model.ref}: the stored file was changed (checksum differs)")
    return model


def resolve_path(value: str, kind: str | None = None) -> str:
    """A file path for a step parameter: registry references become their stored file."""
    return str(resolve(value, kind).path) if is_ref(value) else value


def register_model(
    name: str,
    kind: str,
    source: Path,
    *,
    base_model: str = "",
    training_set: str = "",
    hyperparameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    feature_summary: dict[str, Any] | None = None,
    author: str = "",
    notes: str = "",
    root: Path | None = None,
) -> ModelRecord:
    """Copy a model file (or folder) into the store as the next version, status candidate."""
    from sqlalchemy import func, select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import ModelVersion
    from hcs_studio.paths import to_rooted

    if kind not in KINDS:
        raise ValueError(f"Unknown model kind {kind!r} ({', '.join(KINDS)})")
    source = Path(source)
    if not source.exists():
        raise ValueError(f"{source} does not exist")
    set_sha = ""
    if training_set:
        from hcs_studio.training.groundtruth import load_set

        ann = load_set(training_set)
        training_set, set_sha = ann.ref, ann.content_sha256
    with session_scope() as session:
        latest = session.scalar(
            select(func.max(ModelVersion.version)).where(ModelVersion.name == name)
        )
        version = int(latest or 0) + 1
        folder = (root or models_root()) / _safe(name) / f"v{version:04d}"
        if folder.exists():
            raise FileExistsError(f"{folder} exists: model versions are never overwritten")
        folder.mkdir(parents=True)
        target = folder / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        _protect(target)
        row = ModelVersion(
            name=name,
            version=version,
            kind=kind,
            base_model=base_model,
            training_set=training_set,
            training_set_sha256=set_sha,
            hyperparameters=_jsonable(hyperparameters or {}),
            metrics=_jsonable(metrics or {}),
            feature_summary=_jsonable(feature_summary or {}),
            file=str(to_rooted(target)),
            sha256=file_sha256(target),
            author=author,
            notes=notes,
            status="candidate",
        )
        session.add(row)
        session.flush()
        return _record(row)


def _safe(name: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError(f"Unusable model name {name!r}")
    return safe


def _protect(path: Path) -> None:
    """Read-only files in the store (a guard against accidental edits, not a security)."""
    for sub in [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]:
        with contextlib.suppress(OSError):
            os.chmod(sub, 0o444)


def _jsonable(value: Any) -> Any:
    return json.loads(
        json.dumps(value, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    )


def set_status(
    name: str, version: int, status: str, *, approved: bool = False, notes: str = ""
) -> tuple[ModelRecord, list[ModelRecord]]:
    """Change a version's status; returns it and any version demoted by a promotion."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import ModelVersion

    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r} ({', '.join(STATUSES)})")
    with session_scope() as session:
        row = session.scalar(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        )
        if row is None:
            raise ValueError(f"No model {name}@{version}")
        if status == row.status:
            return _record(row), []
        if status not in TRANSITIONS[row.status]:
            raise ValueError(
                f"{name}@{version} is {row.status}: it cannot become {status} "
                f"(allowed: {', '.join(sorted(TRANSITIONS[row.status]))})"
            )
        if status == "production" and not approved:
            raise PermissionError("Promotion to production needs an approval (confirm it)")
        demoted = []
        now = datetime.now(UTC)
        if status == "production":
            for other in session.scalars(
                select(ModelVersion).where(
                    ModelVersion.name == name, ModelVersion.status == "production"
                )
            ):
                other.status, other.status_changed = "validated", now
                demoted.append(_record(other))
        row.status, row.status_changed = status, now
        if notes:
            row.notes = (row.notes + "\n" if row.notes else "") + notes
        return _record(row), demoted


def update_metrics(name: str, version: int, metrics: dict[str, Any]) -> ModelRecord:
    """Merge validation results into a version's metrics (the model file never changes)."""
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.model_registry import ModelVersion

    with session_scope() as session:
        row = session.scalar(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        )
        if row is None:
            raise ValueError(f"No model {name}@{version}")
        row.metrics = {**dict(row.metrics), **_jsonable(metrics)}
        return _record(row)


# --- pipelines -----------------------------------------------------------------------------


def _cellpose_file(name: str) -> Path | None:
    """Where Cellpose keeps a built-in model's weights (if downloaded)."""
    folder = Path(
        os.environ.get("CELLPOSE_LOCAL_MODELS_PATH", Path.home() / ".cellpose" / "models")
    )
    path = folder / name
    return path if path.is_file() else None


def describe_model(step_index: int, module: str, param: str, value: str) -> dict[str, Any]:
    """Manifest entry for one model a step uses."""
    from hcs_studio.output.manifest import rooted_text

    entry: dict[str, Any] = {"step": step_index, "module": module, "param": param, "value": value}
    path: Path | None = None
    if is_ref(value):
        model = resolve(value)
        path = model.path
        entry["registry"] = {
            "name": model.name,
            "version": model.version,
            "kind": model.kind,
            "status": model.status,
            "training_set": model.training_set,
            "training_set_sha256": model.training_set_sha256,
        }
    elif Path(value).exists():
        path = Path(value)
    elif module == "segment_cellpose":
        path = _cellpose_file(value)
        entry["builtin"] = True
    if path is not None:
        entry["path"] = rooted_text(path)
        entry["sha256"] = file_sha256(path)
    else:
        entry["sha256"] = None
        entry["note"] = "model file not found locally (downloaded on first use)"
    return entry


def step_models(step_index: int, module: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for param in MODEL_PARAMS:
        value = params.get(param)
        if isinstance(value, str) and value:
            out.append(describe_model(step_index, module, param, value))
    if module == "qc" or isinstance(params.get("learned"), dict):
        learned = params.get("learned") or {}
        if isinstance(learned, dict) and learned.get("model"):
            out.append(describe_model(step_index, module, "learned.model", learned["model"]))
    return out


def pipeline_models(pipeline: Any) -> list[dict[str, Any]]:
    """Every model a pipeline uses (steps numbered from 1), with files and SHA-256."""
    out = []
    for i, step in enumerate(pipeline.enabled_steps(), start=1):
        out.extend(step_models(i, step.module, dict(step.params)))
    qc = getattr(pipeline, "qc", None)
    learned = getattr(qc, "learned", None) if qc is not None else None
    model = str(getattr(learned, "model", "") or "") if learned is not None else ""
    if model:
        out.append(describe_model(0, "image_qc", "learned.model", model))
    return out


def pin_pipeline(pipeline: Any) -> tuple[Any, list[str]]:
    """The pipeline with ``@production`` / unversioned references replaced by exact versions;
    returns it and the replacements made."""
    changes = []
    data = pipeline.model_dump(mode="json")
    for step in data["steps"]:
        for param in MODEL_PARAMS:
            value = step["params"].get(param)
            if is_ref(value):
                name, version = parse_ref(value)
                if version is None or not version.isdigit():
                    model = get_model(name, version)
                    step["params"][param] = model.ref
                    changes.append(f"{value} → {model.ref}")
    return type(pipeline).model_validate(data), changes
