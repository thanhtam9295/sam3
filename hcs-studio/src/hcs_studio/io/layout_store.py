"""Versioned plate layouts and layout templates, kept in the user's settings folder.

``<settings>/layouts/<plate key>/v0001.json, v0002.json, …``: every save makes a new,
immutable version. The plate key is the barcode, or else a hash of the plate's path, so
layouts never go into raw data folders. Templates are ``<settings>/layout_templates/
<name>.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from hcs_studio.io.sidecar import config_dir
from hcs_studio.model.annotation import PlateLayout
from hcs_studio.model.layout_tools import as_template


def plate_key(plate_path: Path | None = None, barcode: str = "") -> str:
    """The folder name that links layouts to a plate (barcode, else path hash)."""
    if barcode.strip():
        return "barcode_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", barcode.strip())
    if plate_path is None:
        raise ValueError("A plate key needs a barcode or the plate's path")
    resolved = str(Path(plate_path).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(plate_path).name)[:40]
    return f"{name}__{digest}"


def layout_sha256(layout: PlateLayout) -> str:
    text = json.dumps(layout.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LayoutStore:
    """All saved versions of one plate's layout."""

    def __init__(self, key: str, root: Path | None = None) -> None:
        self.key = key
        self.folder = (root or config_dir() / "layouts") / key

    def versions(self) -> list[int]:
        if not self.folder.exists():
            return []
        found = [re.fullmatch(r"v(\d+)\.json", p.name) for p in self.folder.iterdir()]
        return sorted(int(m.group(1)) for m in found if m)

    def path(self, version: int) -> Path:
        return self.folder / f"v{version:04d}.json"

    def save(self, layout: PlateLayout, note: str = "") -> PlateLayout:
        """Save as the next version; returns the layout with its version numbers."""
        versions = self.versions()
        number = (versions[-1] + 1) if versions else 1
        saved = replace(layout, version=number, parent_version=layout.version or None)
        payload = saved.to_dict()
        payload["saved"] = {"at": datetime.now(UTC).isoformat(), "note": note}
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(number)
        if path.exists():  # never overwrite a version
            raise FileExistsError(path)
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        part.replace(path)
        return saved

    def load(self, version: int | None = None) -> PlateLayout | None:
        """A saved version (default: the latest), or None if none was saved."""
        versions = self.versions()
        if not versions:
            return None
        number = versions[-1] if version is None else version
        data = json.loads(self.path(number).read_text(encoding="utf-8"))
        return PlateLayout.from_dict(data)


def templates_folder() -> Path:
    return config_dir() / "layout_templates"


def _template_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9 _.×x+-]+", "_", name).strip() or "template"
    return templates_folder() / f"{safe}.json"


def save_template(layout: PlateLayout, name: str) -> Path:
    path = _template_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = as_template(layout).to_dict()
    payload["template"] = {"name": name, "saved": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def list_templates() -> list[str]:
    folder = templates_folder()
    if not folder.exists():
        return []
    names = []
    for path in sorted(folder.glob("*.json")):
        try:
            names.append(json.loads(path.read_text(encoding="utf-8"))["template"]["name"])
        except (OSError, ValueError, KeyError):
            continue
    return names


def load_template(name: str) -> PlateLayout:
    path = _template_path(name)
    if not path.exists():
        raise ValueError(f"No layout template named {name!r}")
    return PlateLayout.from_dict(json.loads(path.read_text(encoding="utf-8")))
