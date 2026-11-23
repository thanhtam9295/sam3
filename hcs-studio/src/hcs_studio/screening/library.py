"""Library plate maps and their transfer to assay plates (Phase 23a).

A **library plate map** (CSV or XLSX, one row per well) lists what each library (source) plate
holds. Columns are found by name, case- and punctuation-insensitive:

| Meaning | Accepted names |
|---|---|
| library plate | ``library_plate``, ``plate``, ``barcode``, ``source_plate`` (``_barcode``) |
| well | ``well``, ``source_well`` |
| compound | ``compound``, ``compound_id``, ``compound_name``, ``name``, ``id`` |
| concentration | ``concentration`` + a ``unit`` column, a unit in the header (``conc_mM``) or in |
|  | the value (``"10 mM"``) |
| volume (optional) | ``volume_nl``, ``volume`` (nL) |
| SMILES (optional) | ``smiles`` |

Concentrations are stored in mol/L. Empty compound cells are empty wells (DMSO-only wells
should be named, e.g. ``DMSO``, and are then treated as compounds unless the template makes
them controls).

**Assay layouts** (``assay_layouts``) put library compounds into the sample wells of a control
**template** (a ``PlateLayout`` holding the controls, e.g. vehicle in columns 1–2 and positive
control in 23–24):

- ``stamp``: library well → the same assay well (96 → 96, 384 → 384), at
  ``concentration × dilution`` (e.g. 10 mM stock, 1:1000 → 10 µM);
- ``quadrants``: four 96-well library plates → one 384-well assay plate; plate k of the four goes
  to quadrant (k // 2, k % 2): 96-well (r, c) → 384-well (2r + k // 2, 2c + k % 2);
- ``echo``: an Echo transfer file (``read_transfers``: source plate, source well, destination
  plate, destination well, transfer volume in nL; the column names are configurable because
  exports differ between instruments and software versions) — the final concentration is
  stock × transferred volume / (assay volume + transferred volume), summed for several
  transfers of one compound into a well.

A library compound landing on a control well of the template is an error (the transfer would
spoil the control). Layouts are returned by assay-plate barcode; ``save_assay_layouts`` stores
them as layout versions keyed by that barcode, so the plates are annotated automatically when
they are imaged and analysed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcs_studio.model.plate import parse_well_name, well_name

ALIASES = {
    "library_plate": (
        "libraryplate",
        "plate",
        "barcode",
        "sourceplate",
        "sourceplatebarcode",
        "libraryplatebarcode",
        "platebarcode",
    ),
    "well": ("well", "sourcewell", "wellid", "position"),
    "compound": ("compound", "compoundid", "compoundname", "name", "id", "sampleid"),
    "volume_nl": ("volumenl", "volume"),
    "smiles": ("smiles",),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _find(columns: list[str], names: tuple[str, ...]) -> str | None:
    by_norm = {_norm(c): c for c in columns}
    for name in names:
        if name in by_norm:
            return by_norm[name]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, sep=None, engine="python")


def _concentrations(table: pd.DataFrame) -> pd.Series:
    """Concentrations in mol/L from a value + unit column, a unit in the header, or units in
    the values."""
    from hcs_studio.model.units import to_base

    columns = list(table.columns)
    unit_column = _find(columns, ("unit", "concentrationunit", "units"))
    candidates = [
        c
        for c in columns
        if _norm(c).startswith(("concentration", "conc")) and _norm(c) not in ("concentrationunit",)
    ]
    if not candidates:
        raise ValueError("The library map has no concentration column")
    column = candidates[0]
    header_unit = None
    match = re.search(r"(?:\(|_|\s)(pm|nm|um|µm|mm|m)\)?$", str(column).strip().lower())
    if match:
        header_unit = {"pm": "pM", "nm": "nM", "um": "µM", "µm": "µM", "mm": "mM", "m": "M"}[
            match.group(1)
        ]
    out = []
    for i, raw in enumerate(table[column]):
        text = str(raw).strip() if raw is not None else ""
        if not text or text.lower() == "nan":
            out.append(np.nan)
            continue
        found = re.fullmatch(r"([-+0-9.eE]+)\s*([a-zA-Zµμ/]*)", text)
        if not found:
            raise ValueError(f"Cannot read the concentration {text!r} (row {i + 2})")
        unit = found.group(2) or (str(table[unit_column].iloc[i]) if unit_column else header_unit)
        if not unit:
            raise ValueError(
                "Concentration units are missing (add a unit column or put the unit "
                "in the header, e.g. concentration_mM)"
            )
        out.append(to_base(float(found.group(1)), unit.replace("μ", "µ")))
    return pd.Series(out, index=table.index, dtype=float)


def read_library_map(path: Path) -> pd.DataFrame:
    """``library_plate``, ``well`` (canonical, e.g. A01), ``compound``, ``concentration_m``,
    ``volume_nl``, ``smiles`` — one row per filled well."""
    table = _read_table(path)
    columns = list(table.columns)
    found = {key: _find(columns, names) for key, names in ALIASES.items()}
    for required in ("library_plate", "well", "compound"):
        if found[required] is None:
            raise ValueError(
                f"The library map has no {required.replace('_', ' ')} column "
                f"(columns: {', '.join(columns)})"
            )
    out = pd.DataFrame(
        {
            "library_plate": table[found["library_plate"]].astype(str).str.strip(),
            "well": [well_name(*parse_well_name(str(w).strip())) for w in table[found["well"]]],
            "compound": table[found["compound"]].fillna("").astype(str).str.strip(),
            "concentration_m": _concentrations(table),
        }
    )
    out["volume_nl"] = (
        pd.to_numeric(table[found["volume_nl"]], errors="coerce") if found["volume_nl"] else np.nan
    )
    out["smiles"] = table[found["smiles"]].fillna("") if found["smiles"] else ""
    out = out[out["compound"] != ""].reset_index(drop=True)
    duplicated = out.duplicated(["library_plate", "well"])
    if duplicated.any():
        first = out[duplicated].iloc[0]
        raise ValueError(f"Well {first['well']} of {first['library_plate']} is listed twice")
    return out


TRANSFER_COLUMNS = {
    "source_plate": ("sourceplatebarcode", "sourceplatename", "sourceplate"),
    "source_well": ("sourcewell",),
    "destination_plate": ("destinationplatebarcode", "destinationplatename", "destinationplate"),
    "destination_well": ("destinationwell",),
    "volume_nl": ("transfervolume", "actualvolume", "transfervolumenl", "volume"),
}


def read_transfers(path: Path, columns: dict[str, str] | None = None) -> pd.DataFrame:
    """An Echo transfer (or survey/protocol) file: ``source_plate``, ``source_well``,
    ``destination_plate``, ``destination_well``, ``volume_nl``. ``columns`` maps these to the
    file's own headers when the defaults do not match. Lines before the header row (Echo files
    start with run details) are skipped."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines()
    wanted = {k: (columns or {}).get(k) for k in TRANSFER_COLUMNS}
    start = 0
    for i, line in enumerate(raw):
        cells = [_norm(c) for c in line.split(",")]
        if (
            any(c in cells for c in ("sourcewell", _norm(wanted["source_well"] or "")))
            and len(cells) > 3
        ):
            start = i
            break
    from io import StringIO

    table = pd.read_csv(StringIO("\n".join(raw[start:])), dtype=str)
    out = {}
    for key, names in TRANSFER_COLUMNS.items():
        column = wanted[key] if wanted[key] in table.columns else _find(list(table.columns), names)
        if column is None:
            raise ValueError(
                f"The transfer file has no {key.replace('_', ' ')} column (columns: "
                f"{', '.join(table.columns)}); map it with the columns option"
            )
        out[key] = table[column]
    frame = pd.DataFrame(out).dropna(subset=["source_well", "destination_well"])
    frame["volume_nl"] = pd.to_numeric(frame["volume_nl"], errors="coerce")
    for key in ("source_well", "destination_well"):
        frame[key] = [well_name(*parse_well_name(str(w).strip())) for w in frame[key]]
    for key in ("source_plate", "destination_plate"):
        frame[key] = frame[key].astype(str).str.strip()
    return frame.reset_index(drop=True)


def _place(
    template: Any, barcode: str, items: dict[tuple[int, int], tuple[str, float]], unit: str
) -> Any:
    """The template with compounds in the given wells (refused on control wells)."""
    from hcs_studio.model.annotation import Treatment, WellAnnotation

    wells = dict(template.wells)
    for key, (compound, concentration) in items.items():
        if not (0 <= key[0] < template.rows and 0 <= key[1] < template.cols):
            raise ValueError(
                f"{barcode}: well {well_name(*key)} is outside the "
                f"{template.rows}×{template.cols} template"
            )
        current = wells.get(key)
        if current is not None and current.role not in ("sample", "empty"):
            raise ValueError(
                f"{barcode}: a library compound ({compound}) would go into "
                f"{well_name(*key)}, a {current.role} well of the template"
            )
        base = current if current is not None and current.role == "sample" else WellAnnotation()
        treatment = Treatment(compound, float(concentration), unit)
        wells[key] = replace(base, treatments=(*base.treatments, treatment), role="sample")
    return replace(template, wells=wells, barcode=barcode, version=0, parent_version=None)


def assay_layouts(
    library: pd.DataFrame,
    template: Any,
    rule: str,
    *,
    destinations: dict[str, str] | None = None,
    sources: list[str] | None = None,
    dilution: float = 1.0,
    transfers: pd.DataFrame | None = None,
    assay_volume_ul: float | None = None,
    unit: str = "µM",
) -> dict[str, Any]:
    """Assay-plate layouts by barcode (see the module docstring).

    - ``stamp``: ``destinations`` maps each library plate to its assay plate barcode
      (default: the library barcode + ``-A``);
    - ``quadrants``: ``sources`` = the four library plates in quadrant order;
      ``destinations`` = {"<any>": assay barcode} (one entry);
    - ``echo``: ``transfers`` from ``read_transfers`` and ``assay_volume_ul``.
    """
    items: dict[tuple[int, int], tuple[str, float]]
    if rule == "stamp":
        out = {}
        for plate, group in library.groupby("library_plate"):
            barcode = (destinations or {}).get(str(plate), f"{plate}-A")
            items = {
                parse_well_name(w): (c, conc * dilution)
                for w, c, conc in zip(
                    group["well"], group["compound"], group["concentration_m"], strict=True
                )
            }
            out[barcode] = _place(template, barcode, items, unit)
        return out
    if rule == "quadrants":
        if not sources or len(sources) > 4:
            raise ValueError("quadrants needs up to four library plates in quadrant order")
        if (template.rows, template.cols) != (16, 24):
            raise ValueError("quadrants maps 96-well library plates to a 384-well template")
        barcode = next(iter((destinations or {}).values()), "-".join(sources) + "-384")
        items = {}
        for k, plate in enumerate(sources):
            group = library[library["library_plate"] == plate]
            if group.empty:
                raise ValueError(f"No library plate {plate!r} in the map")
            for w, c, conc in zip(
                group["well"], group["compound"], group["concentration_m"], strict=True
            ):
                r, col = parse_well_name(w)
                if r >= 8 or col >= 12:
                    raise ValueError(f"{plate} {w}: quadrants needs 96-well library plates")
                items[(2 * r + k // 2, 2 * col + k % 2)] = (c, conc * dilution)
        return {barcode: _place(template, barcode, items, unit)}
    if rule == "echo":
        if transfers is None or assay_volume_ul is None:
            raise ValueError("echo needs the transfer file and the assay volume (µL)")
        stock = library.set_index(["library_plate", "well"])
        out = {}
        for plate, group in transfers.groupby("destination_plate"):
            total_nl = group.groupby("destination_well")["volume_nl"].sum()
            amounts: dict[tuple[int, int], dict[str, float]] = {}
            for row in group.itertuples():
                key = (row.source_plate, row.source_well)
                if key not in stock.index:
                    raise ValueError(
                        f"Transfer from {row.source_plate} {row.source_well}: not in "
                        "the library map"
                    )
                source = stock.loc[key]  # type: ignore[index]
                moles = float(str(source["concentration_m"])) * float(str(row.volume_nl)) * 1e-9
                cell = amounts.setdefault(parse_well_name(str(row.destination_well)), {})
                cell[str(source["compound"])] = cell.get(str(source["compound"]), 0.0) + moles
            items = {}
            for key, compounds in amounts.items():
                volume_l = assay_volume_ul * 1e-6 + float(total_nl[well_name(*key)]) * 1e-9
                if len(compounds) != 1:
                    raise ValueError(
                        f"{plate} {well_name(*key)}: several compounds transferred "
                        "into one well (combinations are not screened here)"
                    )
                compound, moles = next(iter(compounds.items()))
                items[key] = (compound, moles / volume_l)
            out[str(plate)] = _place(template, str(plate), items, unit)
        return out
    raise ValueError(f"Unknown transfer rule {rule!r} (stamp, quadrants, echo)")
