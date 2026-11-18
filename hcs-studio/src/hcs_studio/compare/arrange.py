"""Arrange wells into a comparison grid from the plate annotation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hcs_studio.compare.spec import Comparison, PanelSpec
from hcs_studio.model import units
from hcs_studio.model.annotation import PlateLayout, WellAnnotation
from hcs_studio.model.plate import parse_well_name, well_name


def layout_of(run: Path) -> PlateLayout:
    from hcs_studio.model.layout import load_plate_layout, run_layout_file

    path = run_layout_file(Path(run))
    if path is None:
        raise ValueError(f"{Path(run).name} has no plate layout (annotate it in the Layout tab)")
    return load_plate_layout(path)


def _drugs(a: WellAnnotation) -> list[Any]:
    if a.role in ("negative_control", "positive_control", "untreated"):
        return []
    return [t for t in a.active_treatments if t.kind != "percent"]


def grid(
    run: Path,
    wells: Sequence[str],
    base: Comparison,
    rows: int | None = None,
    cols: int | None = None,
) -> Comparison:
    """Wells in reading order on a rows × cols grid (default: one row, or near-square)."""
    n = len(wells)
    if cols is None:
        cols = n if rows is None else -(-n // rows)
    rows = rows or -(-n // cols)
    panels = [
        PanelSpec(run=str(run), well=w, row=k // cols, col=k % cols, source=_source(base))
        for k, w in enumerate(wells)
    ]
    return base.model_copy(
        update={"panels": panels, "rows": rows, "cols": cols, "row_headers": [], "col_headers": []}
    )


def _source(base: Comparison) -> str:
    return base.panels[0].source if base.panels else "corrected"


def by_annotation(
    run: Path,
    wells: Sequence[str],
    base: Comparison,
    rows_by: str = "cell_line",
    cols_by: str = "concentration",
) -> Comparison:
    """Columns ascending by concentration (or another field), rows by cell line (or
    another field); one well per cell (the first by well name; replicates are dropped)."""
    layout = layout_of(run)

    def key(a: WellAnnotation, by: str) -> tuple[Any, str]:
        drugs = _drugs(a)
        if by == "concentration":
            # sort within a unit kind (molar, mass…): base units of different kinds differ
            # by orders of magnitude and must not be compared
            kinds = " + ".join(sorted({t.kind for t in drugs}))
            order = (kinds, sum(t.concentration for t in drugs))
            text = " + ".join(
                f"{t.compound} {units.format_concentration(t.concentration, t.unit)}" for t in drugs
            )
            return (
                order,
                text or (a.role.replace("_", " ") if a.role != "sample" else "untreated"),
            )
        if by == "compound":
            text = " + ".join(t.compound for t in drugs) or a.role.replace("_", " ")
            return (text, text)
        value = (
            getattr(a, by, "")
            if by in ("cell_line", "role", "passage", "notes")
            else dict(a.custom).get(by, "")
        )
        return (str(value), str(value) or "—")

    annotated = [(w, layout.get(parse_well_name(w))) for w in sorted(wells)]
    row_keys = sorted({key(a, rows_by) for _, a in annotated})
    col_keys = sorted({key(a, cols_by) for _, a in annotated})
    panels: dict[tuple[int, int], PanelSpec] = {}
    for w, a in annotated:
        r, c = row_keys.index(key(a, rows_by)), col_keys.index(key(a, cols_by))
        panels.setdefault(
            (r, c), PanelSpec(run=str(run), well=w, row=r, col=c, source=_source(base))
        )
    return base.model_copy(
        update={
            "panels": list(panels.values()),
            "rows": len(row_keys),
            "cols": len(col_keys),
            "row_headers": [k[1] for k in row_keys],
            "col_headers": [k[1] for k in col_keys],
        }
    )


def compound_grid(run: Path, compound: str, base: Comparison) -> Comparison:
    """Single-agent wells of ``compound`` plus each cell line's vehicle wells:
    concentrations as columns (vehicle first), cell lines as rows."""
    layout = layout_of(run)
    lines = sorted(
        {
            a.cell_line
            for a in layout.wells.values()
            if [t.compound for t in _drugs(a)] == [compound]
        }
    )
    wells = [
        well_name(*k)
        for k, a in layout.wells.items()
        if a.cell_line in lines
        and (
            [t.compound for t in _drugs(a)] == [compound]
            or (a.role == "negative_control" and not _drugs(a))
        )
    ]
    comparison = by_annotation(run, wells, base)
    prefix = f"{compound} "
    headers = [h[len(prefix) :] if h.startswith(prefix) else h for h in comparison.col_headers]
    return comparison.model_copy(update={"title": compound, "col_headers": headers})


def combination_grid(
    run: Path, cell_line: str, drug_a: str, drug_b: str, base: Comparison
) -> Comparison:
    """A combination matrix as an image grid: drug A down (highest at the top, as in the
    heat maps), drug B across (ascending); the 0 row/column are the single agents and the
    corner a vehicle well."""
    layout = layout_of(run)
    cells: dict[tuple[float, float], str] = {}
    a_levels: set[float] = {0.0}
    b_levels: set[float] = {0.0}
    corner_found = False
    for key, a in sorted(layout.wells.items()):
        if a.cell_line != cell_line:
            continue
        drugs = {t.compound: t.concentration for t in _drugs(a)}
        if set(drugs) - {drug_a, drug_b}:
            continue
        if not drugs and a.role != "negative_control":
            continue
        ca, cb = drugs.get(drug_a, 0.0), drugs.get(drug_b, 0.0)
        if ca and cb:
            a_levels.add(ca)
            b_levels.add(cb)
        own_corner = {t.compound for t in a.treatments} >= {drug_a, drug_b}
        if (ca, cb) == (0.0, 0.0) and own_corner and not corner_found:
            cells[(ca, cb)] = well_name(*key)  # the matrix's own vehicle corner first
            corner_found = True
        cells.setdefault((ca, cb), well_name(*key))
    a_sorted = sorted(a_levels, reverse=True)
    b_sorted = sorted(b_levels)
    unit_a = next(
        (t.unit for a in layout.wells.values() for t in a.treatments if t.compound == drug_a), "µM"
    )
    unit_b = next(
        (t.unit for a in layout.wells.values() for t in a.treatments if t.compound == drug_b), "µM"
    )
    panels = [
        PanelSpec(run=str(run), well=cells[(ca, cb)], row=r, col=c, source=_source(base))
        for r, ca in enumerate(a_sorted)
        for c, cb in enumerate(b_sorted)
        if (ca, cb) in cells
    ]
    return base.model_copy(
        update={
            "panels": panels,
            "rows": len(a_sorted),
            "cols": len(b_sorted),
            "row_headers": [f"{drug_a} {units.format_concentration(v, unit_a)}" for v in a_sorted],
            "col_headers": [f"{drug_b} {units.format_concentration(v, unit_b)}" for v in b_sorted],
            "title": f"{cell_line} · {drug_a} × {drug_b}",
        }
    )


def combinations_in_layout(layout: PlateLayout) -> list[tuple[str, str, str]]:
    """(cell line, drug A, drug B) of every drug pair given together in some well."""
    found = set()
    for a in layout.wells.values():
        drugs = sorted(t.compound for t in _drugs(a))
        if len(drugs) == 2:
            found.add((a.cell_line, drugs[0], drugs[1]))
    return sorted(found)
