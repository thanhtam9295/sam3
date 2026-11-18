"""Reading and writing plate layouts: long CSV/XLSX, plate-map XLSX, JSON, figures.

Long format: one row per well, columns ``well, cell_line, role, compound_1, conc_1,
unit_1, vehicle_1, compound_2, …`` plus ``passage, seeding_density, replicate_group,
time_point_h, notes`` and any user-defined columns. Plate metadata (rows, columns,
barcode, …) is written as ``# key: value`` lines at the top of CSV files, or on a sheet
named ``plate`` in XLSX files. Layout files of earlier versions (``well, compound,
concentration, treatment, control_type``) are read too.

Plate-map format (XLSX): one sheet per field, laid out like the plate (column numbers in
the first row, row letters in the first column).
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcs_studio.model import units
from hcs_studio.model.annotation import (
    PLATE_TYPES,
    ROLES,
    PlateLayout,
    Treatment,
    WellAnnotation,
    legacy_role,
)
from hcs_studio.model.plate import parse_well_name, row_label, well_name

METADATA = ("rows", "cols", "plate_id", "barcode", "assay", "date", "operator")
BASE_FIELDS = (
    "well",
    "cell_line",
    "role",
    "passage",
    "seeding_density",
    "replicate_group",
    "time_point_h",
    "notes",
)
#: Header spellings → field names (lower case, spaces/dashes as underscores).
ALIASES = {
    "well": "well",
    "well_id": "well",
    "position": "well",
    "cell_line": "cell_line",
    "cellline": "cell_line",
    "cell": "cell_line",
    "cells": "cell_line",
    "role": "role",
    "well_role": "role",
    "well_type": "role",
    "type": "role",
    "control_type": "control_type",
    "control": "control_type",
    "passage": "passage",
    "seeding_density": "seeding_density",
    "density": "seeding_density",
    "cells_per_well": "seeding_density",
    "replicate_group": "replicate_group",
    "replicate": "replicate_group",
    "group": "replicate_group",
    "time_point_h": "time_point_h",
    "time_point": "time_point_h",
    "timepoint": "time_point_h",
    "time_h": "time_point_h",
    "time": "time_point_h",
    "notes": "notes",
    "note": "notes",
    "comment": "notes",
    "comments": "notes",
    "compound": "compound_1",
    "drug": "compound_1",
    "concentration": "conc_1",
    "conc": "conc_1",
    "dose": "conc_1",
    "unit": "unit_1",
    "units": "unit_1",
    "vehicle": "vehicle_1",
    "treatment": "custom:treatment",
}
_NUMBERED = re.compile(r"^(compound|drug|conc|concentration|dose|unit|units|vehicle)_?([1-9])$")
_NUMBERED_NAMES = {
    "compound": "compound",
    "drug": "compound",
    "conc": "conc",
    "concentration": "conc",
    "dose": "conc",
    "unit": "unit",
    "units": "unit",
    "vehicle": "vehicle",
}
_LETTERS = {"a": "1", "b": "2", "c": "3"}


def _norm(header: str) -> str:
    return re.sub(r"[\s\-./()]+", "_", header.strip().lower()).strip("_")


def suggest_mapping(columns: Iterable[str]) -> dict[str, str]:
    """For each column, the field it most likely holds (``custom:<name>`` otherwise)."""
    mapping = {}
    for column in columns:
        key = _norm(column)
        key = re.sub(
            r"_?(drug|compound|conc|unit)_?([abc])$",
            lambda m: f"{m.group(1)}_{_LETTERS[m.group(2)]}",
            key,
        )
        key = re.sub(r"^drug_?([abc])$", lambda m: f"compound_{_LETTERS[m.group(1)]}", key)
        numbered = _NUMBERED.match(key)
        if numbered:
            mapping[column] = f"{_NUMBERED_NAMES[numbered.group(1)]}_{numbered.group(2)}"
        elif key in ALIASES:
            mapping[column] = ALIASES[key]
        else:
            mapping[column] = f"custom:{column.strip()}"
    return mapping


@dataclass
class LayoutImport:
    layout: PlateLayout
    mapping: dict[str, str]
    notes: list[str] = field(default_factory=list)


def _plate_size(names: Iterable[str], plate_type: int | None) -> tuple[int, int]:
    if plate_type is not None:
        return PLATE_TYPES[plate_type]
    keys = [parse_well_name(n) for n in names]
    if not keys:
        return PLATE_TYPES[384]
    rows, cols = max(r for r, _ in keys) + 1, max(c for _, c in keys) + 1
    for size in (96, 384, 1536):
        r, c = PLATE_TYPES[size]
        if rows <= r and cols <= c:
            return r, c
    raise ValueError(f"Wells up to {well_name(rows - 1, cols - 1)} exceed a 1536-well plate")


def _number(text: str, what: str, line: str) -> float | None:
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        raise ValueError(f"{line}: {what} {text!r} is not a number") from None


def layout_from_records(
    records: list[dict[str, str]],
    mapping: Mapping[str, str],
    metadata: Mapping[str, str] | None = None,
    *,
    plate_type: int | None = None,
    source: str = "layout",
    default_unit: str = "µM",
) -> LayoutImport:
    """Build a layout from rows of (column → text) with a column → field mapping."""
    metadata = dict(metadata or {})
    notes: list[str] = []
    well_column = next((c for c, f in mapping.items() if f == "well"), None)
    if well_column is None:
        raise ValueError(f"{source}: a 'well' column is required (values like C07)")
    wells: dict[tuple[int, int], WellAnnotation] = {}
    custom_order: list[str] = []
    used_default_unit = False
    zero_concentration: list[str] = []
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(records, start=1):
        line = f"{source} line {index + 1}"
        name = str(record.get(well_column, "")).strip()
        if not name:
            continue
        try:
            key = parse_well_name(name)
        except ValueError as exc:
            raise ValueError(f"{line}: {exc}") from None
        if key in seen:
            raise ValueError(f"{line}: well {name} is listed twice")
        seen.add(key)
        values: dict[str, str] = {}
        custom: dict[str, str] = {}
        for column, target in mapping.items():
            text = str(record.get(column, "") if record.get(column) is not None else "").strip()
            if target == "ignore" or target == "well":
                continue
            if target.startswith("custom:"):
                if text:
                    custom[target[7:]] = text
                    if target[7:] not in custom_order:
                        custom_order.append(target[7:])
                continue
            values[target] = text
        treatments = []
        for n in range(1, 10):
            compound = values.get(f"compound_{n}", "")
            value = _number(values.get(f"conc_{n}", ""), f"concentration {n}", line)
            if not compound:
                if value is not None:
                    raise ValueError(f"{line}: a concentration without a compound (column {n})")
                continue
            if value is None:  # e.g. a vehicle named without a dose: the compound is absent
                value = 0.0
                zero_concentration.append(name)
            unit = values.get(f"unit_{n}", "")
            if not unit:
                unit = default_unit
                used_default_unit = True
            try:
                treatments.append(
                    Treatment.of(compound, value, unit, values.get(f"vehicle_{n}", ""))
                )
            except ValueError as exc:
                raise ValueError(f"{line}: {exc}") from None
        role_text = values.get("role", "")
        if not role_text and values.get("control_type"):
            role_text = values["control_type"]
        try:
            role = legacy_role(role_text) if role_text else "sample"
        except ValueError:
            raise ValueError(
                f"{line}: unknown role/control_type {role_text!r} (allowed: {', '.join(ROLES)})"
            ) from None
        density = _number(values.get("seeding_density", ""), "seeding density", line)
        time_point = _number(values.get("time_point_h", ""), "time point", line)
        annotation = WellAnnotation(
            cell_line=values.get("cell_line", ""),
            treatments=tuple(treatments),
            role=role,
            passage=values.get("passage", ""),
            seeding_density=density,
            time_point_h=time_point,
            notes=values.get("notes", ""),
            custom=tuple(custom.items()),
        )
        group = values.get("replicate_group", "")
        if group and group != annotation.replicate_group:
            annotation = WellAnnotation(**{**annotation.__dict__, "replicate_override": group})
        if annotation != WellAnnotation(role="empty"):
            wells[key] = annotation
    if zero_concentration:
        notes.append(
            "Compounds without a concentration were recorded at 0 (absent), in "
            + ", ".join(zero_concentration[:10])
            + (" …" if len(zero_concentration) > 10 else "")
        )
    if used_default_unit:
        notes.append(f"Concentrations without a unit were read as {default_unit}")
    if "rows" in metadata and "cols" in metadata:
        rows_cols = (int(metadata["rows"]), int(metadata["cols"]))
    else:
        rows_cols = _plate_size([well_name(*k) for k in seen], plate_type)
    declared = [c for c in metadata.get("custom_fields", "").split("|") if c]
    layout = PlateLayout(
        rows=rows_cols[0],
        cols=rows_cols[1],
        wells=wells,
        plate_id=metadata.get("plate_id", ""),
        barcode=metadata.get("barcode", ""),
        assay=metadata.get("assay", ""),
        date=metadata.get("date", ""),
        operator=metadata.get("operator", ""),
        custom_fields=tuple(declared + [c for c in custom_order if c not in declared]),
    )
    return LayoutImport(layout, dict(mapping), notes)


# --------------------------------------------------------------------- reading


def read_records(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    """Rows, columns and plate metadata of a long-format CSV or XLSX file."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return _read_long_xlsx(path)
    text = path.read_text(encoding="utf-8-sig")
    metadata: dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            match = re.match(r"#\s*([\w ]+?)\s*:\s*(.*)$", line)
            if match:
                metadata[_norm(match.group(1))] = match.group(2).strip()
            body_start = i + 1
        else:
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[body_start:])))
    if reader.fieldnames is None:
        raise ValueError(f"{path.name}: no header row")
    columns = [c for c in reader.fieldnames if c is not None]
    records = [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]
    return records, columns, metadata


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _read_long_xlsx(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    metadata: dict[str, str] = {}
    if "plate" in book.sheetnames:
        for key, value, *_ in book["plate"].iter_rows(values_only=True):
            if key:
                metadata[_norm(str(key))] = _cell_text(value)
    sheet = book[next(n for n in book.sheetnames if n != "plate")]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"{path.name}: the first sheet is empty")
    columns = [_cell_text(c) for c in rows[0]]
    records = [
        {columns[i]: _cell_text(v) for i, v in enumerate(row) if i < len(columns) and columns[i]}
        for row in rows[1:]
        if any(v is not None and str(v).strip() for v in row)
    ]
    return records, [c for c in columns if c], metadata


def is_platemap_workbook(path: Path) -> bool:
    """True when the XLSX lays out each field as a plate grid."""
    if Path(path).suffix.lower() not in (".xlsx", ".xlsm"):
        return False
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    for name in book.sheetnames:
        if name == "plate":
            continue
        first = next(book[name].iter_rows(values_only=True), ())
        header = [_cell_text(v) for v in first[1:4]]
        return header[:3] == ["1", "2", "3"]
    return False


def read_platemap_xlsx(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    """Plate-map workbook → long-format records (one per well with any value)."""
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    records: dict[str, dict[str, str]] = {}
    metadata: dict[str, str] = {}
    columns: list[str] = ["well"]
    for name in book.sheetnames:
        sheet = list(book[name].iter_rows(values_only=True))
        if name == "plate":
            metadata.update({_norm(str(r[0])): _cell_text(r[1]) for r in sheet if r and r[0]})
            continue
        header = [_cell_text(v) for v in sheet[0]]
        columns.append(name)
        for row in sheet[1:]:
            if not row or not row[0]:
                continue
            letters = str(row[0]).strip()
            for index, value in enumerate(row[1:], start=1):
                text = _cell_text(value).strip()
                if not text or index >= len(header) or not header[index]:
                    continue
                well = f"{letters}{int(header[index]):02d}"
                records.setdefault(well, {"well": well})[name] = text
    return list(records.values()), columns, metadata


def load_layout(
    path: Path,
    mapping: Mapping[str, str] | None = None,
    *,
    plate_type: int | None = None,
    default_unit: str = "µM",
) -> LayoutImport:
    """Read a layout from JSON, long CSV/XLSX or a plate-map XLSX."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        layout = PlateLayout.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return LayoutImport(layout, {})
    if is_platemap_workbook(path):
        records, columns, metadata = read_platemap_xlsx(path)
    else:
        records, columns, metadata = read_records(path)
    used = dict(mapping) if mapping is not None else suggest_mapping(columns)
    return layout_from_records(
        records, used, metadata, plate_type=plate_type, source=path.name, default_unit=default_unit
    )


# --------------------------------------------------------------------- writing


def _max_treatments(layout: PlateLayout) -> int:
    return max((len(a.treatments) for a in layout.wells.values()), default=0)


def long_columns(layout: PlateLayout) -> list[str]:
    columns = ["well", "cell_line", "role"]
    for n in range(1, max(1, _max_treatments(layout)) + 1):
        columns += [f"compound_{n}", f"conc_{n}", f"unit_{n}", f"vehicle_{n}"]
    columns += ["passage", "seeding_density", "replicate_group", "time_point_h", "notes"]
    return columns + list(layout.custom_fields)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def long_records(layout: PlateLayout, derived_groups: bool = False) -> list[dict[str, str]]:
    """One row per annotated well. ``replicate_group`` holds the user's override (or, with
    ``derived_groups``, the derived group, for tables meant for people)."""
    rows = []
    for key, a in sorted(layout.wells.items()):
        row = {
            "well": well_name(*key),
            "cell_line": a.cell_line,
            "role": a.role,
            "passage": a.passage,
            "seeding_density": _fmt(a.seeding_density),
            "replicate_group": a.replicate_group if derived_groups else a.replicate_override,
            "time_point_h": _fmt(a.time_point_h),
            "notes": a.notes,
        }
        for n, t in enumerate(a.treatments, start=1):
            row.update(
                {
                    f"compound_{n}": t.compound,
                    f"conc_{n}": _fmt(t.value),
                    f"unit_{n}": t.unit,
                    f"vehicle_{n}": t.vehicle,
                }
            )
        row.update(a.custom_dict)
        rows.append(row)
    return rows


def _metadata(layout: PlateLayout) -> dict[str, str]:
    values = {name: str(getattr(layout, name)) for name in METADATA}
    if layout.custom_fields:
        values["custom_fields"] = "|".join(layout.custom_fields)
    return values


def write_long_csv(layout: PlateLayout, path: Path) -> None:
    columns = long_columns(layout)
    buffer = io.StringIO()
    for key, value in _metadata(layout).items():
        buffer.write(f"# {key}: {value}\n")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in long_records(layout):
        writer.writerow({c: row.get(c, "") for c in columns})
    Path(path).write_text(buffer.getvalue(), encoding="utf-8")


def write_long_xlsx(layout: PlateLayout, path: Path) -> None:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "layout"
    columns = long_columns(layout)
    sheet.append(columns)
    for row in long_records(layout):
        sheet.append([row.get(c, "") for c in columns])
    plate = book.create_sheet("plate")
    for key, value in _metadata(layout).items():
        plate.append([key, value])
    book.save(path)


def write_platemap_xlsx(layout: PlateLayout, path: Path) -> None:
    from openpyxl import Workbook

    book = Workbook()
    records = {r["well"]: r for r in long_records(layout)}
    fields = [c for c in long_columns(layout) if c != "well"]
    first = True
    for name in fields:
        if not any(r.get(name) for r in records.values()):
            continue
        sheet = book.active if first else book.create_sheet()
        assert sheet is not None
        first = False
        sheet.title = name[:31]
        sheet.append(["", *list(range(1, layout.cols + 1))])
        for row in range(layout.rows):
            values = [records.get(well_name(row, c), {}).get(name, "") for c in range(layout.cols)]
            sheet.append([row_label(row), *[_typed(v) for v in values]])
    plate = book.active if first else book.create_sheet("plate")
    assert plate is not None
    plate.title = "plate"
    for key, value in _metadata(layout).items():
        plate.append([key, value])
    book.save(path)


def _typed(text: str) -> Any:
    try:
        return float(text) if text and re.fullmatch(r"-?[\d.]+(e-?\d+)?", text) else text
    except ValueError:
        return text


def save_layout(layout: PlateLayout, path: Path) -> None:
    """Write by suffix: .json, .csv (long), .xlsx (long)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(
            json.dumps(layout.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8"
        )
    elif suffix == ".csv":
        write_long_csv(layout, path)
    elif suffix in (".xlsx", ".xlsm"):
        write_long_xlsx(layout, path)
    else:
        raise ValueError(f"Cannot write a layout as {suffix} (use .json, .csv or .xlsx)")


# ----------------------------------------------------------- annotation columns

_ANNOTATION_COLUMN = re.compile(
    r"^(cell_line|role|replicate_group|passage|seeding_density|time_point_h|notes"
    r"|(compound|conc|unit|vehicle)_\d+)$"
)


def is_annotation_column(name: str, layout: PlateLayout | None = None) -> bool:
    """Columns joined from the layout (not measured readouts)."""
    return bool(_ANNOTATION_COLUMN.match(name)) or (
        layout is not None and name in layout.custom_fields
    )


def annotation_rows(layout: PlateLayout) -> dict[str, dict[str, Any]]:
    """Per well: the annotation columns for tables (derived replicate groups; numbers as
    numbers)."""
    out: dict[str, dict[str, Any]] = {}
    columns = [c for c in long_columns(layout) if c != "well"]
    for row in long_records(layout, derived_groups=True):
        values: dict[str, Any] = {}
        for key in columns:
            text = row.get(key, "")
            if key.startswith("conc_") or key in ("seeding_density", "time_point_h"):
                values[key] = float(text) if text else None
            else:
                values[key] = text
        out[row["well"]] = values
    return out


# ------------------------------------------------------------------------- layers

LAYERS = ("cell_line", "compound", "concentration", "role", "replicate_group")


def layer_values(layout: PlateLayout, layer: str) -> dict[tuple[int, int], Any]:
    """The value each well shows in a layer view: text, or log10 molar concentration of
    single-agent wells (None for combinations, vehicles in % and mass units)."""
    import math

    values: dict[tuple[int, int], Any] = {}
    for key, a in layout.wells.items():
        if layer == "cell_line":
            values[key] = a.cell_line or None
        elif layer == "compound":
            values[key] = a.compound or None
        elif layer == "role":
            values[key] = a.role
        elif layer == "replicate_group":
            values[key] = a.replicate_group
        elif layer == "concentration":
            # one molar scale: single agents in molar units (not % vehicles, mass units
            # or combinations)
            active = a.active_treatments
            single = len(active) == 1 and active[0].kind == "molar"
            values[key] = math.log10(active[0].concentration) if single else None
        else:
            raise ValueError(f"Unknown layer {layer!r} (known: {', '.join(LAYERS)})")
    return values


def write_layout_figure(layout: PlateLayout, path: Path, layer: str = "compound") -> None:
    """A printable plate figure (PNG or PDF) coloured by ``layer``, with a legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hcs_studio.analysis.profiling import PALETTE

    values = layer_values(layout, layer)
    size = 0.32 if layout.cols <= 12 else 0.2 if layout.cols <= 24 else 0.12
    fig, ax = plt.subplots(figsize=(layout.cols * size + 3.5, layout.rows * size + 1.2))
    numeric = layer == "concentration"
    finite = [v for v in values.values() if v is not None]
    categories = sorted({str(v) for v in finite}) if not numeric else []
    cmap = plt.get_cmap("viridis")
    lo, hi = (min(finite), max(finite)) if numeric and finite else (0.0, 1.0)
    for r in range(layout.rows):
        for c in range(layout.cols):
            value = values.get((r, c))
            if value is None:
                face = "white"
            elif numeric:
                face = cmap((value - lo) / ((hi - lo) or 1))
            else:
                face = PALETTE[categories.index(str(value)) % len(PALETTE)]
            ax.add_patch(plt.Circle((c, r), 0.42, facecolor=face, edgecolor="#888", linewidth=0.5))
            role = layout.get((r, c)).role
            if role in ("negative_control", "positive_control", "blank", "excluded"):
                ax.text(
                    c,
                    r,
                    {
                        "negative_control": "N",
                        "positive_control": "P",
                        "blank": "B",
                        "excluded": "×",
                    }[role],
                    ha="center",
                    va="center",
                    fontsize=5,
                )
    ax.set_xlim(-0.6, layout.cols - 0.4)
    ax.set_ylim(layout.rows - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(layout.cols), [str(c + 1) for c in range(layout.cols)], fontsize=6)
    ax.set_yticks(range(layout.rows), [row_label(r) for r in range(layout.rows)], fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = " · ".join(x for x in (layout.assay, layout.barcode or layout.plate_id) if x)
    ax.set_title(f"{title + ' · ' if title else ''}{layer.replace('_', ' ')}", fontsize=9)
    if numeric and finite:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        bar = fig.colorbar(mappable, ax=ax, fraction=0.03)
        bar.set_label("log10 concentration (base units)", fontsize=7)
    elif categories:
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=6,
                color=PALETTE[i % len(PALETTE)],
                label=name,
            )
            for i, name in enumerate(categories[:24])
        ]
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, frameon=False
        )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "LAYERS",
    "LayoutImport",
    "is_platemap_workbook",
    "layer_values",
    "layout_from_records",
    "load_layout",
    "long_columns",
    "long_records",
    "read_platemap_xlsx",
    "read_records",
    "save_layout",
    "suggest_mapping",
    "units",
    "write_layout_figure",
    "write_long_csv",
    "write_long_xlsx",
    "write_platemap_xlsx",
]
