"""Phase 17a: the experiment catalog — scanning, idempotency, queries, pages, exports."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from hcs_studio.db import session_scope
from hcs_studio.db.catalog import (
    Acquisition,
    CombinationScore,
    DoseResponseFit,
    Plate,
    PlateQCRecord,
    Run,
    WellResult,
)
from hcs_studio.db.queries import (
    PlateQuery,
    cell_line_page,
    compound_page,
    export_table,
    ic50s,
    plate_page,
    potency_table,
    readout_over_time,
    saved_searches,
    search_plates,
)
from hcs_studio.services import context, recent_events
from hcs_studio.services.registry import (
    add_event_annotation,
    save_cell_line,
    save_compound,
    save_experiment,
    save_search,
    scan_folders,
)
from hcs_studio.synthetic import TRUTH, read_truth, registry_runs


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    return registry_runs(tmp_path_factory.mktemp("registry_runs"), n_runs=3)


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def counts() -> dict[str, int]:
    with session_scope() as s:
        return {
            model.__tablename__: int(s.scalar(select(func.count()).select_from(model)) or 0)
            for model in (
                Plate,
                Acquisition,
                Run,
                WellResult,
                DoseResponseFit,
                CombinationScore,
                PlateQCRecord,
            )
        }


def test_scan_indexes_every_run_and_rescanning_adds_nothing(
    runs: list[Path], registry: Path
) -> None:
    report = scan_folders(context("first import"), [runs[0].parent])
    assert len(report.runs_added) == 3 and not report.errors
    assert report.unregistered == {"pct_caspase_positive"}  # stored, and reported
    assert sorted(report.plates_added) == ["REG-001", "REG-002", "REG-003"]
    first = counts()
    assert first["plate"] == 3 and first["run"] == 3 and first["acquisition"] == 3
    summary = pd.read_csv(runs[0] / "tables" / "per_well_summary.csv")
    assert first["well_result"] == 3 * len(summary) * 2  # nuclei_count + pct_caspase_positive
    fits = pd.read_csv(runs[0] / "pharmacology" / "dose_response_fits.csv")
    assert first["dose_response_fit"] == 3 * len(fits)

    again = scan_folders(context(), [runs[0].parent])
    assert again.runs_added == [] and len(again.runs_unchanged) == 3 and not again.changed
    assert counts() == first  # idempotent

    # a changed run is re-indexed in place, the others are left alone
    table = runs[1] / "tables" / "per_well_summary.csv"
    os.utime(table, ns=(time.time_ns(), time.time_ns() + 10**9))
    changed = scan_folders(context(), [runs[0].parent])
    assert len(changed.runs_updated) == 1 and len(changed.runs_unchanged) == 2
    assert counts() == first
    event = recent_events(1, "registry.scan")[0]
    assert event["after"]["runs_updated"] == 1


def test_all_ic50s_of_a_compound_in_a_cell_line_across_runs(
    runs: list[Path], registry: Path
) -> None:
    scan_folders(context(), [runs[0].parent])
    with session_scope() as s:
        found = ic50s(s, "Drug A", "MCF7", "nuclei_count")
        table = potency_table(s, compound="Drug A", readout="nuclei_count")
    assert list(found["barcode"]) == ["REG-001", "REG-002", "REG-003"]  # oldest first
    for run, value in zip(runs, found["ec50_m"], strict=True):
        fits = pd.read_csv(run / "pharmacology" / "dose_response_fits.csv")
        row = fits[
            (fits.compound == "Drug A")
            & (fits.cell_line == "MCF7")
            & (fits.readout == "nuclei_count")
        ]
        assert value == pytest.approx(float(row["ec50"].iloc[0]), rel=1e-12)  # what the run fitted
    truth = read_truth(runs[0].parent)["ec50_m"]["MCF7|Drug A"]
    assert truth == TRUTH[("MCF7", "Drug A")][0]
    assert all(abs(math.log10(v / truth)) < 0.1 for v in found["ec50_m"])  # within 26 %
    mcf7 = table[table.cell_line == "MCF7"].iloc[0]
    assert mcf7["n"] == 3 and mcf7["latest_run"] == runs[2].name
    assert mcf7["mean_log_ec50"] == pytest.approx(np.mean(np.log10(found["ec50_m"])))
    assert mcf7["sd_log_ec50"] == pytest.approx(np.std(np.log10(found["ec50_m"]), ddof=1))
    assert set(table.cell_line) == {"MCF7", "HeLa"}


def test_search_pages_readout_over_time_and_reference_data(
    runs: list[Path], registry: Path, tmp_path: Path
) -> None:
    scan_folders(context(), [runs[0].parent])
    save_experiment(
        context(),
        "DRC week 1",
        project="Kinase panel",
        assay="viability",
        plates=["REG-001", "REG-002"],
    )
    save_compound(context(), "Drug A", moa="kinase inhibitor", target="EGFR")
    save_cell_line(context(), "MCF7", rrid="CVCL_0031", cancer_type="breast")
    add_event_annotation(
        context(),
        "reagent_lot",
        pd.Timestamp("2026-01-06").date(),
        assay="viability",
        description="new caspase dye lot",
    )
    with session_scope() as s:
        everything = search_plates(s)
        assert list(everything.barcode) == ["REG-003", "REG-002", "REG-001"]  # newest first
        assert list(search_plates(s, PlateQuery(experiment="week 1")).barcode) == [
            "REG-002",
            "REG-001",
        ]
        assert len(search_plates(s, PlateQuery(text="viability"))) == 2
        assert len(search_plates(s, PlateQuery(cell_line="HeLa"))) == 3
        assert len(search_plates(s, PlateQuery(compound="Drug C"))) == 3
        assert search_plates(s, PlateQuery(compound="Drug Z")).empty
        dated = PlateQuery(date_from=pd.Timestamp("2026-01-06").date())
        assert len(search_plates(s, dated)) == 2
        page = plate_page(s, int(everything.plate_id.iloc[0]))
        assert page["plate"]["barcode"] == "REG-003" and page["runs"][0]["status"] == "completed"
        assert sorted(page["plate"]["cell_lines"]) == ["HeLa", "MCF7"]
        assert page["layout_roles"]["negative_control"] > 0 and page["qc"][0]["z_prime"] is not None
        trend = readout_over_time(s, "nuclei_count", role="negative_control")
        assert list(trend.barcode) == ["REG-001", "REG-002", "REG-003"] and (trend.n > 0).all()
        from hcs_studio.model.layout import load_plate_layout
        from hcs_studio.model.plate import well_name

        summary = pd.read_csv(runs[0] / "tables" / "per_well_summary.csv")
        layout = load_plate_layout(runs[0] / "plate_layout.json")
        controls = {well_name(*k) for k, a in layout.wells.items() if a.role == "negative_control"}
        vehicle = summary[summary.well.isin(controls)]["nuclei_count"]
        assert trend["mean"].iloc[0] == pytest.approx(vehicle.mean())
        assert trend["n"].iloc[0] == len(vehicle)
        treated = readout_over_time(s, "nuclei_count", compound="Drug D", cell_line="HeLa")
        assert (treated["mean"] < trend["mean"]).all()
        drug = compound_page(s, "Drug A")
        assert drug["compound"]["target"] == "EGFR" and len(drug["plates"]) == 3
        assert len(drug["fits"]) == 3 * 2 and not drug["combinations"].empty  # runs × lines
        line = cell_line_page(s, "MCF7")
        assert line["cell_line"]["rrid"] == "CVCL_0031" and len(line["plates"]) == 3
        table = potency_table(s)
    save_search(context(), "MCF7 viability", PlateQuery(cell_line="MCF7", assay="viability"))
    with session_scope() as s:
        assert saved_searches(s)["MCF7 viability"].assay == "viability"
    xlsx = export_table(table, tmp_path / "potency.xlsx")
    assert pd.read_excel(xlsx).shape == table.shape
    csv = export_table(table, tmp_path / "potency.csv")
    assert list(pd.read_csv(csv).columns) == list(table.columns)
    types = {e["type"] for e in recent_events(20)}
    assert {"registry.experiment", "registry.compound", "registry.cell_line"} <= types


def test_raw_plate_folders_are_registered_once(registry: Path, tmp_path: Path) -> None:
    from hcs_studio.synthetic import Nuclei, SyntheticPlate

    plate, _ = SyntheticPlate(
        [(0, 0), (0, 1)], [Nuclei(count=lambda r, c, f: 3)], plate_size=(8, 12), name="RAW-PLATE"
    ).write(tmp_path / "raw")
    report = scan_folders(context(), [plate], raw=True)
    assert len(report.raw_acquisitions_added) == 1 and not report.errors
    again = scan_folders(context(), [plate], raw=True)
    assert again.raw_acquisitions_added == [] and again.plates_added == []
    with session_scope() as s:
        page = search_plates(s, PlateQuery(text="RAW-PLATE"))
        assert len(page) == 1 and page.acquisitions.iloc[0] == 1 and page.runs.iloc[0] == 0
