"""Plate layout import (CSV: well, compound, concentration, treatment, control_type)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from hcs_studio.model.plate import parse_well_name

_CONTROL_TYPES = {"", "negative", "positive", "unstained", "blank"}


@dataclass(frozen=True)
class WellAnnotation:
    compound: str = ""
    concentration: float | None = None
    treatment: str = ""
    control_type: str = ""


def read_plate_layout(path: Path) -> dict[tuple[int, int], WellAnnotation]:
    """Read a layout CSV. Only the ``well`` column is required; headers are case-insensitive."""
    layout: dict[tuple[int, int], WellAnnotation] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: empty layout file")
        headers = {h.strip().lower().replace(" ", "_"): h for h in reader.fieldnames}
        if "well" not in headers:
            raise ValueError(f"{path.name}: a 'well' column is required")

        def cell(row: dict[str, str], key: str) -> str:
            return (row.get(headers[key]) or "").strip() if key in headers else ""

        for line, row in enumerate(reader, start=2):
            try:
                position = parse_well_name(cell(row, "well"))
            except ValueError as exc:
                raise ValueError(f"{path.name} line {line}: {exc}") from None
            if position in layout:
                raise ValueError(f"{path.name} line {line}: well listed twice")
            conc_text = cell(row, "concentration")
            try:
                concentration = float(conc_text) if conc_text else None
            except ValueError:
                raise ValueError(
                    f"{path.name} line {line}: concentration {conc_text!r} is not a number"
                ) from None
            control = cell(row, "control_type").lower()
            if control not in _CONTROL_TYPES:
                allowed = ", ".join(sorted(t for t in _CONTROL_TYPES if t))
                raise ValueError(
                    f"{path.name} line {line}: control_type {control!r} (allowed: {allowed})"
                )
            layout[position] = WellAnnotation(
                compound=cell(row, "compound"),
                concentration=concentration,
                treatment=cell(row, "treatment"),
                control_type=control,
            )
    return layout
