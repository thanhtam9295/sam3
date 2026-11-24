"""Versioned ground-truth sets for training and validating models (Phase 25).

A set lives in ``<settings>/annotations/<name>/vNNNN/``:

- ``images/<item>.tif`` — the field as (C, Y, X) float32 (or (Y, X) for one channel);
- ``masks/<item>.tif`` — instance labels (segmentation sets);
- ``objects.csv`` — one row per labelled object with its class and features (classification
  sets; column ``item`` links it to an item, ``label`` is the class);
- ``set.json`` — schema ``hcs-studio-annotation-set/1``: name, version, kind, channels, classes,
  author, notes, the parent version, the split rule, and per item its provenance (plate, well,
  field, z, source run, cell line, stain, annotator, date, pixel size) and its split.

Versions never change once written: every edit (new items, corrected masks, a new split) is a
new version. ``content_sha256`` covers the files and the provenance, not the split, so a
re-split keeps the content hash and a model records exactly what it saw (set, version, hash).

Splits are made **by plate or by well** (or by cell line), never by field: fields of one well
share cells, focus and staining, so a field-level split leaks. Groups are shuffled with a seed
and filled into train / validation / test in turn until each split reaches its share of
items; ``check_leakage`` verifies that no group spans two splits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "hcs-studio-annotation-set/1"
SPLITS = ("train", "validation", "test")
Split = Literal["train", "validation", "test", ""]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    plate: str = ""
    well: str = ""
    field: int | None = None
    z: int | None = None
    run: str = ""
    cell_line: str = ""
    stain: str = ""
    annotator: str = ""
    date: str = ""
    source: str = Field(default="", description="How the labels were made (edited, SAM, …).")
    pixel_size_um: float | None = None
    split: Split = ""

    def group(self, by: str) -> str:
        if by == "plate":
            return self.plate or self.id
        if by == "well":
            return f"{self.plate}|{self.well}" if self.well else self.id
        if by == "cell_line":
            return self.cell_line or self.id
        raise ValueError(f"Split by plate, well or cell_line, not {by!r}")


class AnnotationSet(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=SCHEMA, alias="schema")
    name: str
    version: int = 1
    kind: Literal["segmentation", "classification"] = "segmentation"
    channels: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    author: str = ""
    notes: str = ""
    created: str = ""
    parent_version: int | None = None
    split_rule: dict[str, Any] = Field(default_factory=dict)
    content_sha256: str = ""
    items: list[Item] = Field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def subset(self, split: str | None) -> list[Item]:
        if not split:
            return list(self.items)
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r} ({', '.join(SPLITS)})")
        return [i for i in self.items if i.split == split]

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in (*SPLITS, "unassigned")}
        for item in self.items:
            out[item.split or "unassigned"] += 1
        return out


# --- storage -------------------------------------------------------------------------------


def annotations_root() -> Path:
    from hcs_studio.io.sidecar import config_dir

    return config_dir() / "annotations"


def _safe(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError(f"Unusable name {name!r}")
    return safe


def set_folder(name: str, version: int, root: Path | None = None) -> Path:
    return (root or annotations_root()) / _safe(name) / f"v{version:04d}"


def versions(name: str, root: Path | None = None) -> list[int]:
    folder = (root or annotations_root()) / _safe(name)
    if not folder.exists():
        return []
    found = [
        int(p.name[1:]) for p in folder.glob("v[0-9][0-9][0-9][0-9]") if (p / "set.json").exists()
    ]
    return sorted(found)


def parse_ref(ref: str) -> tuple[str, int | None]:
    """``name`` or ``name@version``."""
    name, _, version = ref.partition("@")
    return name, int(version) if version else None


def load_set(ref: str, root: Path | None = None) -> AnnotationSet:
    name, version = parse_ref(ref)
    found = versions(name, root)
    if not found:
        raise ValueError(f"No ground-truth set named {name!r}")
    version = version or found[-1]
    path = set_folder(name, version, root) / "set.json"
    if not path.exists():
        raise ValueError(f"{name} has no version {version} (versions: {found})")
    return AnnotationSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_sets(root: Path | None = None) -> list[AnnotationSet]:
    base = root or annotations_root()
    if not base.exists():
        return []
    return [
        load_set(f"{p.name}@{versions(p.name, root)[-1]}", root)
        for p in sorted(base.iterdir())
        if versions(p.name, root)
    ]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(folder: Path, ann: AnnotationSet) -> str:
    """Digest of every file and of the provenance (splits excluded)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "set.json"):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(_file_digest(path).encode())
    provenance = [i.model_dump(exclude={"split"}) for i in ann.items]
    meta = {"kind": ann.kind, "channels": ann.channels, "classes": ann.classes, "items": provenance}
    digest.update(json.dumps(meta, sort_keys=True).encode())
    return digest.hexdigest()


def _write(ann: AnnotationSet, staging: Path, root: Path | None) -> AnnotationSet:
    """Finish a staged version folder (hash, set.json) and move it into place."""
    ann.content_sha256 = content_sha256(staging, ann)
    ann.created = ann.created or datetime.now(UTC).isoformat(timespec="seconds")
    (staging / "set.json").write_text(
        json.dumps(ann.model_dump(by_alias=True), indent=1), encoding="utf-8"
    )
    target = set_folder(ann.name, ann.version, root)
    if target.exists():
        raise FileExistsError(f"{target} exists: versions are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    return ann


def _staging(name: str, root: Path | None) -> Path:
    base = (root or annotations_root()) / _safe(name)
    base.mkdir(parents=True, exist_ok=True)
    staging = base / f".staging-{os.getpid()}-{datetime.now(UTC).strftime('%H%M%S%f')}"
    (staging / "images").mkdir(parents=True)
    return staging


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_item_arrays(
    staging: Path, item_id: str, image: NDArray[Any], masks: NDArray[Any] | None
) -> None:
    import tifffile

    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError(f"Item {item_id}: image must be (Y, X) or (C, Y, X), got {image.shape}")
    tifffile.imwrite(staging / "images" / f"{item_id}.tif", image)
    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != image.shape[-2:]:
            raise ValueError(f"Item {item_id}: masks {masks.shape} ≠ image {image.shape[-2:]}")
        from hcs_studio.segmentation.objects import relabel_sequential

        labels = relabel_sequential(masks.astype(np.int64))
        dtype = np.uint16 if int(np.max(labels, initial=0)) < 65536 else np.uint32
        (staging / "masks").mkdir(exist_ok=True)
        tifffile.imwrite(staging / "masks" / f"{item_id}.tif", labels.astype(dtype))


def create_set(
    name: str,
    items: Sequence[tuple[Item, NDArray[Any], NDArray[Any] | None]],
    *,
    kind: Literal["segmentation", "classification"] = "segmentation",
    channels: Sequence[str] = (),
    classes: Sequence[str] = (),
    objects: pd.DataFrame | None = None,
    author: str = "",
    notes: str = "",
    based_on: str | None = None,
    root: Path | None = None,
) -> AnnotationSet:
    """A new version of ``name`` from (item, image, masks) triples; with ``based_on``
    (``name@version``), the items of that version are kept unless replaced by id."""
    existing = versions(name, root)
    version = (existing[-1] + 1) if existing else 1
    staging = _staging(name, root)
    try:
        kept: list[Item] = []
        parent = None
        new_ids = {item.id for item, _, _ in items}
        if len(new_ids) != len(items):
            raise ValueError("Item ids must be unique")
        if based_on:
            base = load_set(based_on, root)
            parent = base.version
            base_folder = set_folder(base.name, base.version, root)
            for item in base.items:
                if item.id in new_ids:
                    continue
                for sub in ("images", "masks"):
                    src = base_folder / sub / f"{item.id}.tif"
                    if src.exists():
                        _link_or_copy(src, staging / sub / src.name)
                kept.append(item)
            if objects is None and (base_folder / "objects.csv").exists():
                objects = pd.read_csv(base_folder / "objects.csv")
            channels = channels or base.channels
            classes = classes or base.classes
        for item, image, masks in items:
            _safe(item.id)
            write_item_arrays(staging, item.id, image, masks)
        if objects is not None:
            objects.to_csv(staging / "objects.csv", index=False)
            if not classes and "label" in objects:
                classes = sorted(objects["label"].dropna().astype(str).unique())
        ann = AnnotationSet(
            name=name,
            version=version,
            kind=kind,
            channels=list(channels),
            classes=list(classes),
            author=author,
            notes=notes,
            parent_version=parent,
            items=[*kept, *(i for i, _, _ in items)],
        )
        return _write(ann, staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_item(
    ann: AnnotationSet, item: Item, root: Path | None = None
) -> tuple[NDArray[np.float32], NDArray[np.int32] | None]:
    """(image as (C, Y, X) float32, masks (Y, X) int32 or None)."""
    import tifffile

    folder = set_folder(ann.name, ann.version, root)
    image = np.asarray(tifffile.imread(folder / "images" / f"{item.id}.tif"), dtype=np.float32)
    if image.ndim == 2:
        image = image[None]
    mask_path = folder / "masks" / f"{item.id}.tif"
    masks = np.asarray(tifffile.imread(mask_path), dtype=np.int32) if mask_path.exists() else None
    return image, masks


def objects_table(ann: AnnotationSet, root: Path | None = None) -> pd.DataFrame:
    path = set_folder(ann.name, ann.version, root) / "objects.csv"
    if not path.exists():
        raise ValueError(f"{ann.ref} has no objects.csv (not a classification set)")
    table = pd.read_csv(path)
    split = {i.id: i.split for i in ann.items}
    table["split"] = table["item"].astype(str).map(split).fillna("")
    return table


# --- importing -----------------------------------------------------------------------------

PROVENANCE_COLUMNS = (
    "plate",
    "well",
    "field",
    "z",
    "run",
    "cell_line",
    "stain",
    "annotator",
    "date",
    "source",
    "pixel_size_um",
)


def import_pairs(
    folder: Path,
    name: str,
    *,
    provenance: pd.DataFrame | None = None,
    defaults: Mapping[str, Any] | None = None,
    channels: Sequence[str] = (),
    author: str = "",
    notes: str = "",
    based_on: str | None = None,
    root: Path | None = None,
) -> AnnotationSet:
    """A set from a folder of Cellpose-style ``<id>_img.tif`` / ``<id>_masks.tif`` pairs (as the
    segmentation editor's *Save as training data* writes). ``provenance``: one row per id
    (column ``id`` or ``file``) with any of the provenance columns; ``defaults`` fill the rest."""
    import tifffile

    folder = Path(folder)
    images = sorted(folder.glob("*_img.tif"))
    if not images:
        raise ValueError(f"No *_img.tif files in {folder}")
    rows: dict[str, dict[str, Any]] = {}
    if provenance is not None:
        key = "id" if "id" in provenance else "file"
        for record in provenance.to_dict("records"):
            stem = re.sub(r"(_img)?\.tiff?$", "", str(record[key]))
            rows[stem] = {
                k: v for k, v in record.items() if k in PROVENANCE_COLUMNS and pd.notna(v)
            }
    triples = []
    for path in images:
        stem = path.name[: -len("_img.tif")]
        masks_path = path.with_name(f"{stem}_masks.tif")
        if not masks_path.exists():
            raise ValueError(f"Masks missing for {path.name}")
        meta = {**(defaults or {}), **rows.get(stem, {})}
        meta.setdefault("well", _well_from_name(stem))
        item = Item(
            id=_safe(stem), **{k: _cast(k, v) for k, v in meta.items() if k in PROVENANCE_COLUMNS}
        )
        triples.append((item, tifffile.imread(path), tifffile.imread(masks_path)))
    return create_set(
        name, triples, channels=channels, author=author, notes=notes, based_on=based_on, root=root
    )


def _well_from_name(stem: str) -> str:
    match = re.search(r"(?:^|_)([A-P])(\d{1,2})(?:_|$)", stem)
    return f"{match.group(1)}{int(match.group(2)):02d}" if match else ""


def _cast(key: str, value: Any) -> Any:
    if key in ("field", "z"):
        return int(value)
    if key == "pixel_size_um":
        return float(value)
    return str(value)


# --- splitting -----------------------------------------------------------------------------


def assign_splits(
    items: Sequence[Item],
    by: str = "well",
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    test_groups: Iterable[str] = (),
) -> list[Item]:
    """Items with ``split`` set, whole groups (plate / well / cell line) at a time.

    ``test_groups`` go to test whatever the fractions (e.g. a held-out cell line)."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or not sum(fractions):
        raise ValueError("fractions: three non-negative numbers (train, validation, test)")
    shares = np.asarray(fractions, float) / sum(fractions)
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item.group(by), []).append(i)
    forced = set(test_groups)
    unknown = forced - set(groups)
    if unknown:
        raise ValueError(f"Unknown {by} groups for test: {sorted(unknown)}")
    names = [g for g in sorted(groups) if g not in forced]
    order = np.random.default_rng(seed).permutation(len(names))
    target = shares * len(items)
    filled = np.zeros(3)
    split: dict[str, str] = {g: "test" for g in forced}
    filled[2] += sum(len(groups[g]) for g in forced)
    for k in order:  # each group to the split furthest below its share
        group = names[k]
        deficit = (target - filled) / np.maximum(target, 1e-9)
        deficit[target == 0] = -np.inf
        j = int(np.argmax(deficit))
        split[group] = SPLITS[j]
        filled[j] += len(groups[group])
    return [item.model_copy(update={"split": split[item.group(by)]}) for item in items]


def check_leakage(items: Sequence[Item], by: str = "well") -> dict[str, list[str]]:
    """Groups found in more than one split (empty = no leakage)."""
    seen: dict[str, set[str]] = {}
    for item in items:
        if item.split:
            seen.setdefault(item.group(by), set()).add(item.split)
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}


def split_set(
    ref: str,
    by: str = "well",
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    test_groups: Iterable[str] = (),
    root: Path | None = None,
) -> AnnotationSet:
    """A new version of a set with splits assigned (files are hard-linked, not copied)."""
    base = load_set(ref, root)
    items = assign_splits(base.items, by, fractions, seed, test_groups)
    leaks = check_leakage(items, by)
    if leaks:  # cannot happen by construction; guard anyway
        raise RuntimeError(f"Split leaks: {leaks}")
    staging = _staging(base.name, root)
    try:
        source = set_folder(base.name, base.version, root)
        for path in source.rglob("*"):
            if path.is_file() and path.name != "set.json":
                _link_or_copy(path, staging / path.relative_to(source))
        ann = base.model_copy(
            update={
                "version": versions(base.name, root)[-1] + 1,
                "parent_version": base.version,
                "items": items,
                "created": "",
                "content_sha256": "",
                "split_rule": {
                    "by": by,
                    "fractions": list(fractions),
                    "seed": seed,
                    "test_groups": sorted(test_groups),
                },
            }
        )
        return _write(ann, staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_pairs(ref: str, split: str | None, out: Path, root: Path | None = None) -> list[Path]:
    """Cellpose training pairs (``<id>_img.tif`` / ``<id>_masks.tif``) of one split."""
    import tifffile

    ann = load_set(ref, root)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for item in ann.subset(split):
        image, masks = read_item(ann, item, root)
        if masks is None:
            continue
        tifffile.imwrite(out / f"{item.id}_img.tif", image if len(image) > 1 else image[0])
        tifffile.imwrite(
            out / f"{item.id}_masks.tif",
            masks.astype(np.uint16 if int(np.max(masks)) < 65536 else np.uint32),
        )
        written.append(out / f"{item.id}_img.tif")
    if not written:
        raise ValueError(f"{ann.ref}: no items with masks in split {split or 'all'!r}")
    return written
