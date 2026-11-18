"""Plate layouts as used by runs: well → annotation.

The annotation model is ``model/annotation.py`` (cell lines, single agents and
combinations, roles); reading and writing files is ``io/layout_io.py``. This module keeps
the names earlier code uses.
"""

from __future__ import annotations

from pathlib import Path

from hcs_studio.model.annotation import PlateLayout, WellAnnotation

Layout = dict[tuple[int, int], WellAnnotation]


def read_plate_layout(path: Path) -> Layout:
    """Read a layout file (JSON, long CSV/XLSX, plate-map XLSX, or an older layout CSV)."""
    from hcs_studio.io.layout_io import load_layout

    return dict(load_layout(Path(path)).layout.wells)


def load_plate_layout(path: Path) -> PlateLayout:
    from hcs_studio.io.layout_io import load_layout

    return load_layout(Path(path)).layout


def run_layout_file(folder: Path) -> Path | None:
    """The layout a run used: ``plate_layout.json`` (or the CSV of older runs)."""
    for name in ("plate_layout.json", "plate_layout.csv"):
        if (Path(folder) / name).exists():
            return Path(folder) / name
    return None


def processing_signature(layout: Layout | PlateLayout | None) -> list[tuple[str, str]]:
    """The part of a layout that image processing uses (unstained and blank wells)."""
    from hcs_studio.model.plate import well_name

    wells = layout.wells if isinstance(layout, PlateLayout) else (layout or {})
    return sorted(
        (well_name(*k), a.role) for k, a in wells.items() if a.role in ("unstained", "blank")
    )


__all__ = [
    "Layout",
    "PlateLayout",
    "WellAnnotation",
    "load_plate_layout",
    "processing_signature",
    "read_plate_layout",
    "run_layout_file",
]
