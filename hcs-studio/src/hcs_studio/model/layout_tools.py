"""Fill helpers, copy/paste and validation for plate layouts (all return new layouts)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from hcs_studio.model import units
from hcs_studio.model.annotation import PlateLayout, Treatment, WellAnnotation
from hcs_studio.model.plate import well_name

Key = tuple[int, int]
Direction = Literal["right", "down"]

#: Common dilution factors by name.
DILUTIONS = {"1:2": 2.0, "1:3": 3.0, "half-log": math.sqrt(10), "1:10": 10.0}


def dilution_values(
    top: float, factor: float, points: int, include_zero: bool = False
) -> list[float]:
    """``points`` concentrations from ``top`` down by ``factor`` each (plus 0 if asked),
    highest first."""
    if points < 1 or factor <= 1:
        raise ValueError("A dilution series needs at least 1 point and a factor above 1")
    values = [top / factor**i for i in range(points - (1 if include_zero else 0))]
    return [*values, 0.0] if include_zero else values


def _positions(start: Key, length: int, direction: Direction, offset: int) -> list[Key]:
    row, col = start
    if direction == "right":
        return [(row + offset, col + i) for i in range(length)]
    return [(row + i, col + offset) for i in range(length)]


def _check_inside(layout: PlateLayout, keys: Iterable[Key]) -> None:
    outside = [k for k in keys if not (0 <= k[0] < layout.rows and 0 <= k[1] < layout.cols)]
    if outside:
        raise ValueError(
            f"{len(outside)} well(s) fall outside the {layout.rows}×{layout.cols} plate, "
            f"e.g. row {outside[0][0] + 1}, column {outside[0][1] + 1}"
        )


def dilution_series(
    layout: PlateLayout,
    start: Key,
    compound: str,
    top: float,
    unit: str = "µM",
    *,
    factor: float = 3.0,
    points: int = 10,
    direction: Direction = "right",
    reverse: bool = False,
    replicates: int = 1,
    cell_line: str | None = None,
    vehicle: str = "",
) -> PlateLayout:
    """Place a dilution series from ``start``: highest concentration first along
    ``direction`` (last with ``reverse``); replicates go in the next rows (series to the
    right) or columns (series down). Other annotation of the wells is kept; the
    treatments are replaced."""
    values = dilution_values(top, factor, points)
    if reverse:
        values = values[::-1]
    changes: dict[Key, WellAnnotation | None] = {}
    for replicate in range(replicates):
        keys = _positions(start, points, direction, replicate)
        _check_inside(layout, keys)
        for key, value in zip(keys, values, strict=True):
            old = layout.get(key)
            changes[key] = replace(
                old,
                cell_line=old.cell_line if cell_line is None else cell_line,
                treatments=(Treatment.of(compound, value, unit, vehicle),),
                role="sample",
            )
    return layout.updated(changes)


def combination_matrix(
    layout: PlateLayout,
    origin: Key,
    drug_a: str,
    top_a: float,
    drug_b: str,
    top_b: float,
    *,
    unit_a: str = "µM",
    unit_b: str = "µM",
    factor_a: float = 3.0,
    factor_b: float = 3.0,
    size_a: int = 6,
    size_b: int = 6,
    include_zero: bool = True,
    replicates: int = 1,
    replicate_direction: Direction = "right",
    gap: int = 0,
    cell_line: str | None = None,
    vehicle: str = "",
) -> PlateLayout:
    """An N×M combination block: drug A down the rows, drug B across the columns, each
    ascending from 0 (with ``include_zero``) so the single-agent curves lie on the first
    row and column and the 0 × 0 well is the vehicle control. Replicate blocks repeat to
    the right (or down), ``gap`` wells apart."""
    concentrations_a = sorted(dilution_values(top_a, factor_a, size_a, include_zero))
    concentrations_b = sorted(dilution_values(top_b, factor_b, size_b, include_zero))
    changes: dict[Key, WellAnnotation | None] = {}
    for replicate in range(replicates):
        shift = replicate * ((size_b if replicate_direction == "right" else size_a) + gap)
        row0 = origin[0] + (shift if replicate_direction == "down" else 0)
        col0 = origin[1] + (shift if replicate_direction == "right" else 0)
        keys = [(row0 + i, col0 + j) for i in range(size_a) for j in range(size_b)]
        _check_inside(layout, keys)
        for i, a in enumerate(concentrations_a):
            for j, b in enumerate(concentrations_b):
                key = (row0 + i, col0 + j)
                old = layout.get(key)
                treatments = (
                    Treatment.of(drug_a, a, unit_a, vehicle),
                    Treatment.of(drug_b, b, unit_b, vehicle),
                )
                changes[key] = replace(
                    old,
                    cell_line=old.cell_line if cell_line is None else cell_line,
                    treatments=treatments,
                    role="negative_control" if a == 0 and b == 0 else "sample",
                )
    return layout.updated(changes)


def set_role(
    layout: PlateLayout,
    wells: Iterable[Key],
    role: str,
    *,
    vehicle: str = "",
    vehicle_percent: float | None = None,
    cell_line: str | None = None,
    keep_treatments: bool = False,
) -> PlateLayout:
    """Mark wells as controls, blanks etc. ``vehicle`` / ``vehicle_percent`` add the
    vehicle (e.g. DMSO at 0.1 %) as the well's treatment."""
    changes: dict[Key, WellAnnotation | None] = {}
    for key in wells:
        old = layout.get(key)
        treatments = old.treatments if keep_treatments else ()
        if vehicle and vehicle_percent is not None:
            treatments = (Treatment.of(vehicle, vehicle_percent, "%"),)
        changes[key] = replace(
            old,
            role=role,
            treatments=treatments if role not in ("blank", "empty") else (),
            cell_line=old.cell_line if cell_line is None else cell_line,
        )
    return layout.updated(changes)


def edge_wells(layout: PlateLayout) -> list[Key]:
    return [
        (r, c)
        for r in range(layout.rows)
        for c in range(layout.cols)
        if r in (0, layout.rows - 1) or c in (0, layout.cols - 1)
    ]


#: Internal custom fields start with "_" and are not shown as user columns.
EDGE_KEY = "_edge_excluded_role"


def exclude_edges(layout: PlateLayout, exclude: bool = True) -> PlateLayout:
    """Mark the outer wells as excluded (remembering their role), or undo it."""
    changes: dict[Key, WellAnnotation | None] = {}
    for key in edge_wells(layout):
        old = layout.get(key)
        custom = old.custom_dict
        if exclude and old.role != "excluded":
            custom[EDGE_KEY] = old.role
            changes[key] = replace(old, role="excluded", custom=tuple(custom.items()))
        elif not exclude and EDGE_KEY in custom:
            role = custom.pop(EDGE_KEY)
            restored = replace(old, role=role, custom=tuple(custom.items()))
            changes[key] = None if restored == WellAnnotation(role="empty") else restored
    return layout.updated(changes)


def apply_fields(layout: PlateLayout, wells: Iterable[Key], **fields: object) -> PlateLayout:
    """Set the given WellAnnotation fields on every well (others kept). A well without an
    annotation starts as a sample (not "empty") when fields are given to it."""
    changes: dict[Key, WellAnnotation | None] = {
        key: replace(layout.wells.get(key, WellAnnotation()), **fields)  # type: ignore[arg-type]
        for key in wells
    }
    return layout.updated(changes)


def clear(layout: PlateLayout, wells: Iterable[Key]) -> PlateLayout:
    return layout.updated(dict.fromkeys(wells))


# ----------------------------------------------------------------- copy / paste


@dataclass(frozen=True)
class Clipboard:
    """Annotations relative to the top-left of the copied region."""

    cells: Mapping[Key, WellAnnotation]

    @property
    def size(self) -> Key:
        if not self.cells:
            return (0, 0)
        return (max(r for r, _ in self.cells) + 1, max(c for _, c in self.cells) + 1)


def copy_region(layout: PlateLayout, wells: Iterable[Key]) -> Clipboard:
    keys = list(wells)
    if not keys:
        return Clipboard({})
    top, left = min(r for r, _ in keys), min(c for _, c in keys)
    return Clipboard({(r - top, c - left): layout.get((r, c)) for r, c in keys})


def paste(layout: PlateLayout, clipboard: Clipboard, at: Key) -> PlateLayout:
    keys = {(at[0] + r, at[1] + c): a for (r, c), a in clipboard.cells.items()}
    _check_inside(layout, keys)
    return layout.updated(dict(keys.items()))


# -------------------------------------------------------------------- templates


def as_template(layout: PlateLayout) -> PlateLayout:
    """The layout without plate identity (barcode, IDs, dates, version)."""
    return replace(
        layout, plate_id="", barcode="", date="", operator="", version=0, parent_version=None
    )


def apply_template(
    template: PlateLayout,
    *,
    cell_lines: Mapping[str, str] | None = None,
    compounds: Mapping[str, str] | None = None,
    **metadata: object,
) -> PlateLayout:
    """A new layout from a template, renaming cell lines and compounds (old → new)."""
    cell_lines, compounds = cell_lines or {}, compounds or {}
    changes: dict[Key, WellAnnotation | None] = {}
    for key, annotation in template.wells.items():
        treatments = tuple(
            replace(t, compound=compounds.get(t.compound, t.compound))
            for t in annotation.treatments
        )
        changes[key] = replace(
            annotation,
            cell_line=cell_lines.get(annotation.cell_line, annotation.cell_line),
            treatments=treatments,
        )
    return replace(template.updated(changes), version=0, parent_version=None, **metadata)  # type: ignore[arg-type]


# ------------------------------------------------------------------- validation


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    code: str
    message: str
    wells: tuple[Key, ...] = ()

    def text(self) -> str:
        names = ", ".join(well_name(*k) for k in self.wells[:8])
        more = f" … (+{len(self.wells) - 8})" if len(self.wells) > 8 else ""
        return f"{self.message}" + (f": {names}{more}" if self.wells else "")


def validate(layout: PlateLayout, min_controls: int = 3) -> list[Issue]:
    """Problems that would make an analysis wrong or impossible."""
    issues: list[Issue] = []
    wells = dict(layout.wells)
    cells = {k: a for k, a in wells.items() if a.role not in ("blank", "empty", "excluded")}

    missing_line = sorted(k for k, a in cells.items() if not a.cell_line)
    if missing_line:
        issues.append(
            Issue(
                "error",
                "missing_cell_line",
                "Wells with cells but no cell line",
                tuple(missing_line),
            )
        )
    untreated_samples = sorted(
        k for k, a in cells.items() if a.role == "sample" and not a.active_treatments
    )
    if untreated_samples:
        issues.append(
            Issue(
                "warning",
                "sample_without_treatment",
                "Sample wells without a treatment (mark them as untreated or controls)",
                tuple(untreated_samples),
            )
        )
    treated_empty = sorted(
        k for k, a in wells.items() if a.role in ("blank", "empty") and a.treatments
    )
    if treated_empty:
        issues.append(
            Issue(
                "warning",
                "treated_empty",
                "Blank or empty wells with a treatment",
                tuple(treated_empty),
            )
        )

    by_line: dict[str, list[Key]] = defaultdict(list)
    controls: dict[str, list[Key]] = defaultdict(list)
    for key, annotation in cells.items():
        if annotation.role == "sample":
            by_line[annotation.cell_line].append(key)
        if annotation.role == "negative_control":
            controls[annotation.cell_line].append(key)
    for line, samples in sorted(by_line.items()):
        name = line or "(no cell line)"
        if not controls[line]:
            issues.append(
                Issue(
                    "error",
                    "no_vehicle_controls",
                    f"{name}: samples but no negative (vehicle) controls on the plate",
                    tuple(sorted(samples)),
                )
            )
        elif len(controls[line]) < min_controls:
            issues.append(
                Issue(
                    "warning",
                    "few_controls",
                    f"{name}: only {len(controls[line])} negative controls (at least "
                    f"{min_controls} are needed for Z′ and normalisation)",
                    tuple(sorted(controls[line])),
                )
            )

    kinds: dict[str, set[str]] = defaultdict(set)
    unit_wells: dict[str, list[Key]] = defaultdict(list)
    for key, annotation in wells.items():
        for treatment in annotation.treatments:
            kinds[treatment.compound].add(treatment.unit)
            unit_wells[treatment.compound].append(key)
    for compound, used in sorted(kinds.items()):
        if len({units.parse_unit(u).kind for u in used}) > 1:
            issues.append(
                Issue(
                    "error",
                    "inconsistent_units",
                    f"{compound}: concentrations in units of different kinds "
                    f"({', '.join(sorted(used))})",
                    tuple(sorted(unit_wells[compound])),
                )
            )
        elif len(used) > 1:
            issues.append(
                Issue(
                    "warning",
                    "mixed_units",
                    f"{compound}: shown in several units ({', '.join(sorted(used))})",
                    tuple(sorted(unit_wells[compound])),
                )
            )

    singles = {
        (a.cell_line, a.active_treatments[0].compound, a.active_treatments[0].concentration)
        for a in cells.values()
        if len(a.active_treatments) == 1
    }
    missing_single = []
    for key, annotation in cells.items():
        if not annotation.is_combination:
            continue
        for treatment in annotation.active_treatments:
            if (annotation.cell_line, treatment.compound, treatment.concentration) not in singles:
                missing_single.append(key)
                break
    if missing_single:
        issues.append(
            Issue(
                "warning",
                "missing_single_agent",
                "Combination wells without the matching single-agent wells (needed for "
                "Bliss, HSA, Loewe and ZIP)",
                tuple(sorted(missing_single)),
            )
        )
    return issues
