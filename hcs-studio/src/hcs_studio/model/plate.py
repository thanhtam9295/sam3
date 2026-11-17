"""Unified data model: Experiment → Plate → Well → Field → lazy (T, C, Z, Y, X) image."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

Modality = Literal["confocal", "widefield", "brightfield", "unknown"]

#: Standard microplate geometries as (rows, columns), smallest first.
PLATE_FORMATS: dict[int, tuple[int, int]] = {
    6: (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48),
}

_WELL_RE = re.compile(r"^([A-Za-z]{1,2})0*(\d{1,2})$")


def row_label(row: int) -> str:
    """0 → 'A', 25 → 'Z', 26 → 'AA' (1536-well plates use A..AF)."""
    if row < 0:
        raise ValueError("row must be non-negative")
    if row < 26:
        return chr(ord("A") + row)
    return chr(ord("A") + row // 26 - 1) + chr(ord("A") + row % 26)


def well_name(row: int, col: int) -> str:
    """Zero-based (row, col) → 'C07'."""
    if col < 0:
        raise ValueError("col must be non-negative")
    return f"{row_label(row)}{col + 1:02d}"


def parse_well_name(name: str) -> tuple[int, int]:
    """'C7', 'c07' or 'AF48' → zero-based (row, col)."""
    match = _WELL_RE.match(name.strip())
    if not match:
        raise ValueError(f"Not a well name: {name!r}")
    letters = match.group(1).upper()
    row = ord(letters[0]) - ord("A")
    if len(letters) == 2:
        row = (row + 1) * 26 + ord(letters[1]) - ord("A")
    col = int(match.group(2)) - 1
    if col < 0:
        raise ValueError(f"Not a well name: {name!r}")
    return row, col


def infer_plate_shape(n_rows_used: int, n_cols_used: int) -> tuple[int, int]:
    """Smallest standard plate that contains the given used extent."""
    for rows, cols in PLATE_FORMATS.values():
        if n_rows_used <= rows and n_cols_used <= cols:
            return rows, cols
    return n_rows_used, n_cols_used


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    excitation_nm: float | None = None
    emission_nm: float | None = None
    exposure_ms: float | None = None


@dataclass(frozen=True)
class ImageMetadata:
    """Optical metadata shared by the images of a plate. ``None`` means unknown."""

    channels: tuple[ChannelInfo, ...] = ()
    pixel_size_um: float | None = None
    z_step_um: float | None = None
    objective_na: float | None = None
    objective_name: str | None = None
    magnification: float | None = None
    immersion_ri: float | None = None
    modality: Modality = "unknown"
    pinhole_um: float | None = None
    bit_depth: int | None = None

    #: Scalar fields a user may override from the metadata panel.
    OVERRIDABLE = (
        "pixel_size_um",
        "z_step_um",
        "objective_na",
        "magnification",
        "immersion_ri",
        "modality",
        "pinhole_um",
    )

    def with_overrides(self, overrides: dict[str, Any]) -> ImageMetadata:
        """Return a copy with user overrides applied; unknown keys are an error."""
        known = {f.name for f in fields(self)}
        changes: dict[str, Any] = {}
        for key, value in overrides.items():
            if key == "channel_names":
                names = list(value)
                if len(names) != len(self.channels):
                    raise ValueError("channel_names override has the wrong length")
                changes["channels"] = tuple(
                    replace(ch, name=str(n)) for ch, n in zip(self.channels, names, strict=True)
                )
            elif key in known and key in self.OVERRIDABLE:
                changes[key] = value
            else:
                raise ValueError(f"Cannot override metadata field {key!r}")
        return replace(self, **changes)


@dataclass(frozen=True)
class Field:
    """One imaged site of a well. ``load`` returns a lazy dask array (T, C, Z, Y, X)."""

    index: int
    load: Callable[[], Any] = field(repr=False, compare=False)
    x_um: float | None = None
    y_um: float | None = None
    #: Raw files this field is read from (for provenance checksums).
    sources: tuple[Path, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class Well:
    row: int
    col: int
    fields: tuple[Field, ...]
    #: Free-text label when the source had no real well name (e.g. a file name).
    label: str | None = None

    @property
    def name(self) -> str:
        return well_name(self.row, self.col)


@dataclass(frozen=True)
class Plate:
    name: str
    path: Path
    vendor: str
    rows: int
    cols: int
    wells: dict[tuple[int, int], Well]
    metadata: ImageMetadata
    barcode: str | None = None
    #: (T, C, Z, Y, X) of a field, as declared by the source metadata.
    shape_tczyx: tuple[int, int, int, int, int] | None = None
    #: Things the reader had to assume; shown to the user.
    notes: tuple[str, ...] = ()
    #: How to open this plate again (reader name, path, options), e.g. in a worker process.
    source: dict[str, Any] = field(default_factory=dict, compare=False)

    def well(self, name: str) -> Well:
        return self.wells[parse_well_name(name)]

    @property
    def n_fields(self) -> int:
        return sum(len(w.fields) for w in self.wells.values())


@dataclass(frozen=True)
class Experiment:
    name: str
    plates: tuple[Plate, ...]
