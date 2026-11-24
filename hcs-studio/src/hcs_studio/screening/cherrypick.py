"""Cherry-picking hits for confirmation (Phase 23c).

``cherry_pick(hits, library, design, template)`` places the hits on confirmation plates and
lists the transfers:

- **design** (``ConfirmationDesign``): ``replicates`` — each hit at its screening concentration
  (or ``concentration_um``) in ``replicates`` wells, spread over the plate (replicate r of hit i
  goes to sample position i + r · n, so replicates of a compound never sit side by side); or
  ``dose_response`` — ``points`` concentrations from ``top_um`` down by ``factor`` in a run of
  sample wells along a row, ``replicates`` runs per compound in consecutive rows (the first run
  of every row is filled before the second). Hits are placed on the sample wells of the
  control **template** (the layout holding vehicle and positive controls); when one plate is
  full the next is started (barcodes ``<prefix>-001``, ``-002`` …);
- **volumes**: the transfer for a final concentration c in an assay volume V from a library
  stock s is c · V / s, rounded to whole Echo droplets (``droplet_nl``, 2.5 nL on the Echo 525);
  the concentration written into the layout is the one achieved, s · v / (V + v). A transfer
  below one droplet, or one whose rounding moves the concentration by more than
  ``max_rounding_pct`` (10 %), is skipped with a note (an intermediate dilution plate is
  needed: 0.37 µM from 10 mM into 40 µL is 1.5 nL, which one droplet would make 0.62 µM). With
  ``backfill`` (a vehicle source well), every compound well is topped up with vehicle to the
  largest compound volume of its plate, so all wells get the same amount of DMSO;
- **source wells** come from the library map (registry or file): the compound's library plate
  and well (the first one when a compound is in several wells, with a note).

The **pick list** is written with a ``PickListFormat``: the column headers, which field or fixed
text each holds, the well format (A01 or A1), the separator and the volume decimals — set it
to match the cherry-pick protocol of the Echo (Labcyte's defaults are used until a sample file
is checked). The confirmation layouts are returned by barcode, to be pre-registered so that the
plates are analysed automatically when they are imaged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.model.plate import parse_well_name, well_name

FIELDS = (
    "source_plate",
    "source_well",
    "destination_plate",
    "destination_well",
    "volume_nl",
    "compound",
    "concentration_um",
)


class PickColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: str
    field: (
        Literal[
            "source_plate",
            "source_well",
            "destination_plate",
            "destination_well",
            "volume_nl",
            "compound",
            "concentration_um",
        ]
        | None
    ) = None
    value: str = Field(default="", description="Fixed text when there is no field.")


class PickListFormat(BaseModel):
    """Columns of the pick list, in order (Labcyte's Echo Cherry Pick names by default)."""

    model_config = ConfigDict(extra="forbid")

    columns: list[PickColumn] = Field(
        default_factory=lambda: [
            PickColumn(header="Source Plate Barcode", field="source_plate"),
            PickColumn(header="Source Well", field="source_well"),
            PickColumn(header="Destination Plate Barcode", field="destination_plate"),
            PickColumn(header="Destination Well", field="destination_well"),
            PickColumn(header="Transfer Volume", field="volume_nl"),
        ]
    )
    well_format: Literal["A01", "A1"] = "A01"
    separator: str = ","
    volume_decimals: int = Field(default=1, ge=0, le=4)
    one_file_per_plate: bool = False

    @classmethod
    def load(cls, path: Path) -> PickListFormat:
        import json

        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            import yaml

            return cls.model_validate(yaml.safe_load(text))
        return cls.model_validate(json.loads(text))


class Backfill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_plate: str
    source_well: str


class ConfirmationDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["replicates", "dose_response"] = "replicates"
    replicates: int = Field(default=3, ge=1, le=12)
    concentration_um: float | None = Field(
        default=None, gt=0, description="replicates: final concentration (default: the screen's)."
    )
    top_um: float | None = Field(default=None, gt=0, description="dose_response: highest.")
    points: int = Field(default=8, ge=2, le=24)
    factor: float = Field(default=3.0, gt=1)
    assay_volume_ul: float = Field(default=40.0, gt=0)
    droplet_nl: float = Field(default=2.5, gt=0)
    max_rounding_pct: float = Field(
        default=10.0,
        gt=0,
        description="Skip transfers whose droplet rounding shifts the concentration more than "
        "this (an intermediate dilution plate is needed).",
    )
    barcode_prefix: str = "CONF"
    backfill: Backfill | None = None
    unit: str = "µM"


@dataclass
class CherryPick:
    picks: pd.DataFrame  # one row per transfer (FIELDS + kind)
    layouts: dict[str, Any]  # barcode → PlateLayout
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _sources(
    hits: pd.DataFrame, library: pd.DataFrame, notes: list[str]
) -> dict[str, tuple[str, str, float]]:
    """compound → (library plate, well, stock concentration in M)."""
    out = {}
    for compound in hits["compound"]:
        rows = library[library["compound"] == compound]
        if rows.empty:
            notes.append(f"{compound}: not in the library map — not picked")
            continue
        if len(rows) > 1:
            notes.append(
                f"{compound}: in {len(rows)} library wells; "
                f"{rows['library_plate'].iloc[0]} {rows['well'].iloc[0]} used"
            )
        first = rows.iloc[0]
        out[str(compound)] = (
            str(first["library_plate"]),
            str(first["well"]),
            float(first["concentration_m"]),
        )
    return out


def _sample_wells(template: Any) -> list[tuple[int, int]]:
    """Template wells that may take compounds: unannotated, empty-role or sample wells, in
    row-major order."""
    out = []
    for r in range(template.rows):
        for c in range(template.cols):
            a = template.wells.get((r, c))
            if a is None or (a.role in ("sample", "empty") and not a.treatments):
                out.append((r, c))
    return out


def _volume(final_m: float, stock_m: float, design: ConfirmationDesign) -> tuple[float, float]:
    """(transfer volume in nL rounded to droplets, achieved final concentration in M)."""
    exact = final_m * design.assay_volume_ul * 1e3 / stock_m  # nL (µL × 1e3)
    volume = round(exact / design.droplet_nl) * design.droplet_nl
    if volume <= 0:
        return 0.0, 0.0
    achieved = stock_m * volume * 1e-9 / (design.assay_volume_ul * 1e-6 + volume * 1e-9)
    return volume, achieved


def cherry_pick(
    hits: pd.DataFrame, library: pd.DataFrame, design: ConfirmationDesign, template: Any
) -> CherryPick:
    """Transfers and confirmation layouts for ``hits`` (a screen's ``hits.csv``)."""
    from hcs_studio.model.annotation import Treatment, WellAnnotation

    notes: list[str] = []
    sources = _sources(hits, library, notes)
    positions = _sample_wells(template)
    if not positions:
        raise ValueError("The template has no free wells for compounds")
    # what goes where: list of (compound, final concentration M) per plate position
    items: list[tuple[str, float]] = []
    compounds = [str(c) for c in hits["compound"] if str(c) in sources]
    screened = dict(zip(hits["compound"].astype(str), hits["concentration"], strict=True))
    plates: list[dict[tuple[int, int], tuple[str, float]]] = []
    if design.mode == "replicates":
        for compound in compounds:
            conc = (
                design.concentration_um * 1e-6
                if design.concentration_um
                else float(screened[compound])
            )
            if not conc > 0:
                notes.append(f"{compound}: no screening concentration — give concentration_um")
                continue
            items.append((compound, conc))
        per_plate = max(len(positions) // design.replicates, 1)
        for start in range(0, len(items), per_plate):
            chunk = items[start : start + per_plate]
            placed: dict[tuple[int, int], tuple[str, float]] = {}
            for r in range(design.replicates):
                for i, item in enumerate(chunk):
                    placed[positions[i + r * len(chunk)]] = item
            plates.append(placed)
    else:
        if design.top_um is None:
            raise ValueError("dose_response needs top_um (the highest concentration)")
        from hcs_studio.model.layout_tools import dilution_values

        values = [v * 1e-6 for v in dilution_values(design.top_um, design.factor, design.points)]
        rows: dict[int, list[tuple[int, int]]] = {}
        for r, c in positions:
            rows.setdefault(r, []).append((r, c))
        # first series slot of every row, then the second …: replicates go to other rows
        slots = max((len(cells) // design.points for cells in rows.values()), default=0)
        runs = [
            cells[k * design.points : (k + 1) * design.points]
            for k in range(slots)
            for cells in rows.values()
            if len(cells) >= (k + 1) * design.points
        ]
        if not runs:
            raise ValueError(f"No row of the template has {design.points} free wells in a row")
        series = [(compound, rep) for compound in compounds for rep in range(design.replicates)]
        for start in range(0, len(series), len(runs)):
            placed = {}
            for run, (compound, _rep) in zip(runs, series[start : start + len(runs)], strict=False):
                for key, value in zip(run, values, strict=True):
                    placed[key] = (compound, value)
            plates.append(placed)
    picks, layouts, skipped = [], {}, []
    for number, placed in enumerate(plates, start=1):
        barcode = f"{design.barcode_prefix}-{number:03d}"
        wells = dict(template.wells)
        volumes: dict[tuple[int, int], float] = {}
        for key, (compound, final) in placed.items():
            plate, source_well, stock = sources[compound]
            volume, achieved = _volume(final, stock, design)
            off = abs(achieved / final - 1) * 100 if volume >= design.droplet_nl else math.inf
            if volume >= design.droplet_nl and off > design.max_rounding_pct:
                skipped.append(
                    f"{compound} at {final * 1e6:.3g} µM: {volume:g} nL gives "
                    f"{achieved * 1e6:.3g} µM ({off:.0f} % off)"
                )
                continue
            if volume < design.droplet_nl:
                skipped.append(
                    f"{compound} at {final * 1e6:.3g} µM: below one droplet "
                    f"({final * design.assay_volume_ul * 1e3 / stock:.3g} nL)"
                )
                continue
            volumes[key] = volume
            base = wells.get(key) or WellAnnotation()
            wells[key] = replace(
                base,
                role="sample",
                treatments=(Treatment(compound, float(f"{achieved:.6g}"), design.unit),),
            )
            picks.append(
                {
                    "source_plate": plate,
                    "source_well": source_well,
                    "destination_plate": barcode,
                    "destination_well": well_name(*key),
                    "volume_nl": volume,
                    "compound": compound,
                    "concentration_um": achieved * 1e6,
                    "kind": "compound",
                }
            )
        if design.backfill is not None and volumes:
            top = max(volumes.values())
            for key, volume in sorted(volumes.items()):
                if top - volume >= design.droplet_nl:
                    picks.append(
                        {
                            "source_plate": design.backfill.source_plate,
                            "source_well": design.backfill.source_well,
                            "destination_plate": barcode,
                            "destination_well": well_name(*key),
                            "volume_nl": top - volume,
                            "compound": "backfill",
                            "concentration_um": 0.0,
                            "kind": "backfill",
                        }
                    )
        layouts[barcode] = replace(
            template, wells=wells, barcode=barcode, version=0, parent_version=None
        )
    if skipped:
        notes.append(
            f"{len(skipped)} transfer(s) skipped: below one droplet or too far off after "
            "rounding to droplets (use an intermediate dilution plate)"
        )
    return CherryPick(pd.DataFrame(picks, columns=[*FIELDS, "kind"]), layouts, notes, skipped)


def _well(text: str, style: str) -> str:
    row, col = parse_well_name(text)
    name = well_name(row, col)
    if style == "A1":
        letters = "".join(ch for ch in name if ch.isalpha())
        return f"{letters}{col + 1}"
    return name


def write_pick_list(picks: pd.DataFrame, fmt: PickListFormat, out: Path) -> list[Path]:
    """The pick list(s) in the format's columns (one file, or one per destination plate)."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in picks.to_dict("records"):
        row = {}
        for column in fmt.columns:
            if column.field is None:
                row[column.header] = column.value
            elif column.field in ("source_well", "destination_well"):
                row[column.header] = _well(str(record[column.field]), fmt.well_format)
            elif column.field == "volume_nl":
                row[column.header] = f"{float(record['volume_nl']):.{fmt.volume_decimals}f}"
            elif column.field == "concentration_um":
                row[column.header] = f"{float(record['concentration_um']):.6g}"
            else:
                row[column.header] = str(record[column.field])
        rows.append(row)
    table = pd.DataFrame(rows, columns=[c.header for c in fmt.columns])
    if not fmt.one_file_per_plate:
        target = out / "picklist.csv"
        table.to_csv(target, index=False, sep=fmt.separator)
        return [target]
    files = []
    for plate, group in table.groupby(picks["destination_plate"].to_numpy()):
        target = out / f"picklist_{plate}.csv"
        group.to_csv(target, index=False, sep=fmt.separator)
        files.append(target)
    return files


def event_payload(
    screen: str, barcode: str, picks: pd.DataFrame, pick_list: Path
) -> dict[str, Any]:
    """A ``cherrypick.request`` payload for one destination plate (lab_events schema)."""
    mine = picks[picks["destination_plate"] == barcode]
    return {
        "screen": screen,
        "destination_plate": barcode,
        "pick_list_path": str(pick_list),
        "picks": [
            {
                "source_plate": r["source_plate"],
                "source_well": r["source_well"],
                "destination_well": r["destination_well"],
                "volume_nl": float(r["volume_nl"]),
                **({"compound_id": r["compound"]} if r["kind"] == "compound" else {}),
            }
            for r in mine.to_dict("records")
        ],
    }


def run_cherry_pick(
    screen: Path,
    design: ConfirmationDesign,
    template: Any,
    *,
    library: pd.DataFrame | None = None,
    fmt: PickListFormat | None = None,
) -> tuple[CherryPick, list[Path]]:
    """Cherry-pick a screen folder's hits into ``<screen>/cherrypick/`` (pick list, layouts,
    design, summary). ``library``: a library map (default: the registry's)."""
    import json

    screen = Path(screen)
    hits = pd.read_csv(screen / "hits.csv")
    if hits.empty:
        raise ValueError("The screen has no hits to pick")
    if library is None:
        from hcs_studio.db.library import library_map

        library = library_map()
    result = cherry_pick(hits, library, design, template)
    out = screen / "cherrypick"
    files = write_pick_list(result.picks, fmt or PickListFormat(), out)
    from hcs_studio.io.layout_io import save_layout

    (out / "layouts").mkdir(parents=True, exist_ok=True)
    for barcode, layout in result.layouts.items():
        save_layout(layout, out / "layouts" / f"{barcode}.json")
    summary = {
        "design": design.model_dump(mode="json"),
        "format": (fmt or PickListFormat()).model_dump(mode="json"),
        "plates": sorted(result.layouts),
        "transfers": int((result.picks["kind"] == "compound").sum()),
        "backfills": int((result.picks["kind"] == "backfill").sum()),
        "compounds": int(
            result.picks.loc[result.picks["kind"] == "compound", "compound"].nunique()
        ),
        "pick_lists": [f.name for f in files],
        "notes": result.notes,
        "skipped": result.skipped,
    }
    (out / "cherrypick.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return result, files


def volume_check(picks: pd.DataFrame, design: ConfirmationDesign) -> list[str]:
    """Problems with the transfers: volumes not whole droplets, wells over 1 % DMSO."""
    problems = []
    for record in picks.to_dict("records"):
        droplets = record["volume_nl"] / design.droplet_nl
        if not math.isclose(droplets, round(droplets), abs_tol=1e-6):
            problems.append(
                f"{record['destination_well']}: {record['volume_nl']} nL is not whole droplets"
            )
    totals = picks.groupby(["destination_plate", "destination_well"])["volume_nl"].sum()
    over = totals[totals > 0.01 * design.assay_volume_ul * 1e3]
    for key, volume in over.to_dict().items():
        plate, well = (str(k) for k in key)  # type: ignore[attr-defined]
        problems.append(f"{plate} {well}: {float(volume):g} nL > 1 % of the assay volume")
    return problems
