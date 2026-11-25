"""Phase 23c: cherry-picking hits onto confirmation plates (Echo pick list, layouts,
pre-registration, cherrypick.request) and the screen summary."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hcs_studio.model.plate import parse_well_name
from hcs_studio.screening.cherrypick import (
    Backfill,
    ConfirmationDesign,
    PickColumn,
    PickListFormat,
    cherry_pick,
    volume_check,
    write_pick_list,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _template() -> Any:
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role

    layout = PlateLayout.for_plate_type(384)
    layout = set_role(layout, [(r, c) for r in range(16) for c in (0, 1)], "negative_control")
    return set_role(layout, [(r, c) for r in range(16) for c in (22, 23)], "positive_control")


def _hits(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [f"CPD-{i:03d}" for i in range(n)]
    hits = pd.DataFrame({"rank": range(1, n + 1), "compound": names, "concentration": 1e-5})
    library = pd.DataFrame(
        {
            "library_plate": [f"LIB{i // 300:02d}" for i in range(n)],
            "well": [f"{'ABCDEFGHIJKLMNOP'[(i // 20) % 16]}{3 + i % 20:02d}" for i in range(n)],
            "compound": names,
            "concentration_m": 0.01,
            "volume_nl": 5000.0,
            "smiles": "",
        }
    )
    return hits, library


def test_replicates_are_spread_and_volumes_exact() -> None:
    """10 µM from a 10 mM stock into 40 µL = 40 nL (16 droplets of 2.5 nL); the layout holds
    the achieved 9.99 µM; replicates of a compound are never neighbours; controls stay."""
    hits, library = _hits(30)
    result = cherry_pick(hits, library, ConfirmationDesign(replicates=3), _template())
    assert list(result.layouts) == ["CONF-001"]
    picks = result.picks
    assert len(picks) == 90 and set(picks["volume_nl"]) == {40.0}
    layout = result.layouts["CONF-001"]
    achieved = 0.01 * 40e-9 / (40e-6 + 40e-9)
    for compound, group in picks.groupby("compound"):
        wells = [parse_well_name(w) for w in group["destination_well"]]
        assert len(wells) == 3
        for a in wells:
            treatment = layout.wells[a].treatments[0]
            assert treatment.compound == compound
            assert treatment.concentration == pytest.approx(achieved, rel=1e-5)
        for a in wells:
            for b in wells:
                if a != b:
                    assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1  # not adjacent
    assert (
        layout.wells[(0, 0)].role == "negative_control"
        and layout.wells[(5, 23)].role == "positive_control"
    )
    assert not set(picks["destination_well"]) & {"A01", "A02", "P24"}  # never on a control
    assert volume_check(picks, ConfirmationDesign()) == []


def test_full_plates_continue_on_the_next() -> None:
    hits, library = _hits(150)  # 150 × 3 = 450 wells > 320 free wells of the 384 template
    result = cherry_pick(hits, library, ConfirmationDesign(replicates=3), _template())
    assert list(result.layouts) == ["CONF-001", "CONF-002"]
    per_plate = result.picks.groupby("destination_plate")["compound"].nunique()
    assert per_plate.sum() == 150 and per_plate["CONF-001"] == 320 // 3
    assert result.picks.groupby("compound").size().eq(3).all()


def test_dose_response_series_and_droplet_limit() -> None:
    """8 points from 30 µM by 3 from a 10 mM stock into 40 µL: in a row, highest first, the two
    replicates in different rows; points whose volume rounds to whole 2.5 nL droplets with more
    than 10 % error (1.11 µM: 4.4 → 5 nL) or to nothing are skipped with a note."""
    hits, library = _hits(4)
    design = ConfirmationDesign(mode="dose_response", top_um=30, points=8, factor=3, replicates=2)
    result = cherry_pick(hits, library, design, _template())
    picks = result.picks
    one = picks[picks["compound"] == "CPD-000"]
    rows = sorted({parse_well_name(w)[0] for w in one["destination_well"]})
    assert len(rows) == 2  # replicate series in different rows
    first = one[[parse_well_name(w)[0] == rows[0] for w in one["destination_well"]]]
    concentrations = first.sort_values("destination_well")["concentration_um"].tolist()
    assert concentrations == sorted(concentrations, reverse=True)
    intended = [30 / 3**i for i in range(8)]
    kept = [
        c for c in intended if abs(round(c * 4 / 2.5) * 2.5 / (c * 4) - 1) <= 0.1 and c * 4 >= 1.25
    ]
    assert len(concentrations) == len(kept) == 3  # 30, 10, 3.3 µM
    for got, want in zip(concentrations, kept, strict=True):
        assert got == pytest.approx(want, rel=0.1)
    assert any("skipped" in n for n in result.notes) and len(result.skipped) == 5 * 2 * 4
    for volume in picks["volume_nl"]:
        assert math.isclose(volume / 2.5, round(volume / 2.5))


def test_backfill_equalises_the_vehicle() -> None:
    hits, library = _hits(3)
    design = ConfirmationDesign(
        mode="dose_response",
        top_um=30,
        points=4,
        factor=2,
        replicates=1,
        backfill=Backfill(source_plate="DMSO-1", source_well="A01"),
    )
    result = cherry_pick(hits, library, design, _template())
    totals = result.picks.groupby("destination_well")["volume_nl"].sum()
    assert totals.nunique() == 1 and totals.iloc[0] == 120.0  # the 30 µM well's volume
    backfill = result.picks[result.picks["kind"] == "backfill"]
    assert set(backfill["source_plate"]) == {"DMSO-1"} and len(backfill) == 3 * 3


def test_pick_list_formats(tmp_path: Path) -> None:
    hits, library = _hits(2)
    result = cherry_pick(hits, library, ConfirmationDesign(replicates=1), _template())
    default = write_pick_list(result.picks, PickListFormat(), tmp_path / "a")[0]
    table = pd.read_csv(default)
    assert list(table.columns) == [
        "Source Plate Barcode",
        "Source Well",
        "Destination Plate Barcode",
        "Destination Well",
        "Transfer Volume",
    ]
    assert table["Source Well"].tolist() == ["A03", "A04"] and table[
        "Transfer Volume"
    ].tolist() == [40.0, 40.0]
    custom = PickListFormat(
        columns=[
            PickColumn(header="Source Plate Name", value="Source[1]"),
            PickColumn(header="Source Plate Barcode", field="source_plate"),
            PickColumn(header="Source Well", field="source_well"),
            PickColumn(header="Destination Plate Name", value="Destination[1]"),
            PickColumn(header="Destination Well", field="destination_well"),
            PickColumn(header="Volume", field="volume_nl"),
            PickColumn(header="Sample", field="compound"),
        ],
        well_format="A1",
        separator=";",
        volume_decimals=0,
        one_file_per_plate=True,
    )
    files = write_pick_list(result.picks, custom, tmp_path / "b")
    assert [f.name for f in files] == ["picklist_CONF-001.csv"]
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert (
        lines[0] == "Source Plate Name;Source Plate Barcode;Source Well;Destination Plate Name;"
        "Destination Well;Volume;Sample"
    )
    assert lines[1].startswith("Source[1];LIB00;A3;Destination[1];") and lines[1].endswith(
        ";40;CPD-000"
    )


def test_cherry_pick_command_registers_and_sends(home: Path) -> None:
    """A synthetic screen's hits picked through the CLI: pick list from the library map file,
    layouts pre-registered by barcode, one cherrypick.request per plate (schema-valid) in the
    outbox, and the screen summary updated."""
    from lab_events import validate

    from hcs_studio.app import main
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.io.layout_store import LayoutStore, plate_key
    from hcs_studio.screening.screen import ScreenConfig, run_screen
    from hcs_studio.synthetic.screening import write_screen

    folders, truth = write_screen(home / "runs", libraries=2)
    config = ScreenConfig.model_validate(
        {
            "readouts": [
                {"column": "reporter_mean", "alias": "rep"},
                {"column": "nuclei_count", "alias": "count", "activity": "percent_of_control"},
            ],
            "hit_rule": "rep_score <= -3 and count_pct > 60",
        }
    )
    screen = home / "screen"
    hits = run_screen(folders, config, screen).hits
    library = home / "library.csv"
    rows = ["plate,well,compound,concentration"]
    for name in pd.read_csv(screen / "compounds.csv")["compound"]:
        lib, well = name.split("-")
        rows.append(f"LIB{lib[1:]},{well},{name},10 mM")
    library.write_text("\n".join(rows), encoding="utf-8")
    template = home / "template.json"
    save_layout(_template(), template)
    outbox = home / "outbox"
    assert (
        main(
            [
                "screen",
                "cherrypick",
                str(screen),
                "--template",
                str(template),
                "--library",
                str(library),
                "--replicates",
                "4",
                "--register",
                "--outbox",
                str(outbox),
                "--prefix",
                "CONFX",
            ]
        )
        == 0
    )
    picks = pd.read_csv(screen / "cherrypick" / "picklist.csv")
    assert len(picks) == 4 * len(hits)
    assert set(picks["Source Plate Barcode"]) <= {"LIB01", "LIB02"}
    first = hits["compound"].iloc[0]
    assert (
        picks[picks["Source Well"] == first.split("-")[1]]["Destination Plate Barcode"].iloc[0]
        == "CONFX-001"
    )
    registered = LayoutStore(plate_key(barcode="CONFX-001")).load()
    assert registered is not None and registered.barcode == "CONFX-001"
    assert sum(len(a.treatments) for a in registered.wells.values()) == 4 * len(hits)
    events = list(outbox.glob("*cherrypick.request*.json"))
    assert len(events) == 1
    event = validate(json.loads(events[0].read_text(encoding="utf-8")))
    assert event["payload"]["destination_plate"] == "CONFX-001" and len(
        event["payload"]["picks"]
    ) == 4 * len(hits)
    summary = (screen / "summary.html").read_text(encoding="utf-8")
    assert (
        "Confirmation" in summary and "CONFX-001 v1" in summary and f"{len(hits)} hits" in summary
    )
    picked = json.loads((screen / "cherrypick" / "cherrypick.json").read_text(encoding="utf-8"))
    assert picked["compounds"] == len(hits) and picked["registered"] == ["CONFX-001 v1"]
    assert set(truth["hits"]) >= set(hits["compound"])
