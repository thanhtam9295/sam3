"""User metadata overrides, stored outside the raw data (raw folders are read-only)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hcs_studio.model.plate import Plate


def config_dir() -> Path:
    """Per-user settings folder; ``HCS_STUDIO_HOME`` overrides the default."""
    root = os.environ.get("HCS_STUDIO_HOME")
    return Path(root) if root else Path.home() / ".hcs_studio"


def sidecar_path(plate_path: Path) -> Path:
    digest = hashlib.sha256(str(plate_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return config_dir() / "overrides" / f"{plate_path.name}__{digest}.json"


def load_overrides(plate_path: Path) -> dict[str, Any]:
    path = sidecar_path(plate_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: malformed overrides file")
    return overrides


def save_overrides(plate_path: Path, overrides: dict[str, Any]) -> Path:
    """Write (or, when empty, remove) the overrides sidecar for a plate."""
    path = sidecar_path(plate_path)
    if not overrides:
        path.unlink(missing_ok=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(plate_path.resolve()), "overrides": overrides}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def apply_overrides(plate: Plate) -> Plate:
    """Return the plate with any saved overrides applied to its metadata."""
    from dataclasses import replace

    overrides = load_overrides(plate.path)
    if not overrides:
        return plate
    return replace(plate, metadata=plate.metadata.with_overrides(overrides))
