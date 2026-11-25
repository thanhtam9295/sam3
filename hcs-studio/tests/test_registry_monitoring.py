"""Phase 17c through the registry: control charts of scanned runs, MSR history, bridging."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from hcs_studio.db import monitoring as dbm
from hcs_studio.db import session_scope
from hcs_studio.monitoring.msr import msr_from_runs
from hcs_studio.services import context, recent_events
from hcs_studio.services.registry import add_event_annotation, scan_folders, write_bridging_report
from hcs_studio.synthetic import registry_runs, screening_plate, write_values_run


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_z_prime_chart_of_registered_runs_flags_a_drift_and_shows_events(registry: Path) -> None:
    start = datetime(2026, 3, 2, 9, tzinfo=UTC)
    for i in range(26):  # 20 good runs, then the controls get noisier (Z′ falls)
        noise = 3.0 if i < 20 else 3.0 + 1.2 * (i - 19)
        values, wells, _ = screening_plate(noise=noise, hits=0, seed=100 + i)
        write_values_run(
            registry / "runs" / f"QC{i:02d}__run",
            values,
            wells,
            barcode=f"QC{i:02d}",
            started=start + timedelta(days=i),
            instrument="Operetta CLS",
        )
    scan_folders(context(), [registry / "runs"])
    add_event_annotation(
        context(),
        "reagent_lot",
        date(2026, 3, 22),
        instrument="Operetta CLS",
        description="new Hoechst lot",
    )
    add_event_annotation(context(), "instrument_service", date(2026, 3, 10), instrument="Nikon Ji")
    with session_scope() as s:
        metrics = dbm.run_metrics(s, "nuclei_count", instrument="Operetta CLS")
        assert len(metrics) == 26 and metrics["n_vehicle"].iloc[0] == 32
        z0 = metrics["z_prime"].iloc[0]
        assert 0.6 < z0 < 0.95  # 100 vs 10 with SD 3: Z′ ≈ 1 − 18/90
        chart = dbm.chart(s, "z_prime", readout="nuclei_count", instrument="Operetta CLS")
    assert chart.flagged and min(chart.flagged) >= 20
    assert list(chart.events["kind"]) == ["reagent_lot"]  # the Nikon service is not this chart's
    with session_scope() as s:
        cv = dbm.chart(s, "vehicle_cv_pct", readout="nuclei_count")
    assert cv.flagged and min(cv.flagged) >= 20


def test_msr_history_of_a_reference_compound(registry: Path) -> None:
    runs = registry_runs(registry / "drc", n_runs=7)
    scan_folders(context(), [runs[0].parent])
    with session_scope() as s:
        history = dbm.msr_history(s, "Drug A", cell_line="MCF7", readout="nuclei_count", window=6)
        potencies = dbm.reference_potency(s, "Drug A", cell_line="MCF7", readout="nuclei_count")
        chart = dbm.chart(
            s, "log_ec50", compound="Drug A", cell_line="MCF7", readout="nuclei_count"
        )
        with pytest.raises(ValueError, match="several cell lines"):
            dbm.reference_potency(s, "Drug A")
    assert history["n_runs"] == 7
    assert history["msr"] == pytest.approx(msr_from_runs(potencies["ec50_m"].tolist()))
    assert 1 < history["msr"] < 1.5  # noise 8 % on a reproducible synthetic assay
    rolling = history["rolling"]["rolling_msr"]
    assert rolling.isna().sum() == 5 and np.isfinite(rolling.iloc[-1])
    assert chart.notes and not chart.flagged  # 7 runs: limits from the runs themselves


def test_bridging_report_of_plates_imaged_on_two_instruments(registry: Path) -> None:
    a = registry_runs(registry / "operetta", n_runs=3, instrument="Operetta CLS")
    b = registry_runs(registry / "nikon", n_runs=3, instrument="Nikon Ji", seed_offset=50)
    scan_folders(context(), [a[0].parent, b[0].parent])
    files = write_bridging_report(
        context("instrument change"),
        "Operetta CLS",
        "Nikon Ji",
        "nuclei_count",
        registry / "bridge",
        msr=2.0,
    )
    names = {p.name for p in files}
    assert names == {"bridging_report.html", "bridging.json", "agreement.csv", "potency_ratios.csv"}
    summary = json.loads((registry / "bridge" / "bridging.json").read_text(encoding="utf-8"))
    assert summary["plates"] == 3 and summary["potency_pairs"] == 3 * 6
    assert summary["equivalent"] and 0.8 < summary["mean_ratio"] < 1.25
    page = (registry / "bridge" / "bridging_report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in page and "Equivalent" in page
    event = recent_events(1, "registry.bridging_report")[0]
    assert event["reason"] == "instrument change" and event["after"]["equivalent"] is True
    with session_scope() as s:
        strict = dbm.bridging(s, "Operetta CLS", "Nikon Ji", "nuclei_count", msr=1.01)
    assert not strict.equivalent
