"""Phase 17b through the registry: normalize runs, uniformity maps, day effects, replicates."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from hcs_studio.db import normalization as dbn
from hcs_studio.db import session_scope
from hcs_studio.model.plate import parse_well_name, well_name
from hcs_studio.normalization.plate import PositionalCorrectionBlocked, WellInfo
from hcs_studio.services import context, recent_events
from hcs_studio.services.registry import save_uniformity_map, scan_folders
from hcs_studio.synthetic import registry_runs, screening_plate, write_values_run


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_b_score_of_a_registered_screen_and_the_dose_response_guard(registry: Path) -> None:
    values, wells, truth = screening_plate(row_effect=-40, col_effect=25, hits=16, seed=3)
    write_values_run(registry / "runs" / "SCREEN-1__run", values, wells, barcode="SCREEN-1")
    registry_runs(registry / "drc", n_runs=1, pharmacology=False)
    scan_folders(context(), [registry / "runs", registry / "drc"])
    with session_scope() as s:
        data, result = dbn.normalize_run(s, "SCREEN-1", "nuclei_count", "b_score")
        assert data.rows == 16 and len(data.values) == 384
        scores = result.frame.set_index("well")["normalized"]
        called = {w for w, z in scores.items() if z <= -3}
        assert len(called & set(truth["hits"])) >= 0.95 * len(truth["hits"])
        with pytest.raises(PositionalCorrectionBlocked, match="uniformity-plate correction"):
            dbn.normalize_run(s, "REG-001", "nuclei_count", "b_score")
        _, percent = dbn.normalize_run(s, "REG-001", "nuclei_count", "percent_of_control")
        vehicle = percent.frame[percent.frame.role == "negative_control"]["normalized"]
        assert vehicle.median() == pytest.approx(100)


def test_uniformity_map_from_a_vehicle_plate_corrects_a_later_plate(registry: Path) -> None:
    rows, cols = 16, 24
    r, c = np.mgrid[0:rows, 0:cols]
    shading = 1 - 0.12 * ((r - 7.5) / 7.5) ** 2 - 0.08 * ((c - 11.5) / 11.5) ** 2  # dim edges
    rng = np.random.default_rng(1)
    vehicle = {
        well_name(i, j): float(1000 * shading[i, j] * (1 + rng.normal(0, 0.01)))
        for i in range(rows)
        for j in range(cols)
    }
    vehicle_wells = [WellInfo(w, "negative_control") for w in vehicle]
    write_values_run(
        registry / "runs" / "UNIF-1__run",
        vehicle,
        vehicle_wells,
        barcode="UNIF-1",
        instrument="Operetta CLS",
    )
    values, wells, _ = screening_plate(noise=1, hits=0, seed=2)
    shaded = {w: v * float(shading[parse_well_name(w)]) for w, v in values.items()}
    write_values_run(registry / "runs" / "SCREEN-2__run", shaded, wells, barcode="SCREEN-2")
    scan_folders(context(), [registry / "runs"])
    stats = save_uniformity_map(
        context("edge evaporation"), "Operetta 384 viability", "UNIF-1", "nuclei_count", degree=2
    )
    assert stats["before_edge_effect_pct"] < -5 and abs(stats["after_edge_effect_pct"]) < 1
    with pytest.raises(ValueError, match="exists"):
        save_uniformity_map(context(), "Operetta 384 viability", "UNIF-1", "nuclei_count")
    with pytest.raises(ValueError, match="not a uniformity plate"):
        save_uniformity_map(context(), "bad", "SCREEN-2", "nuclei_count")
    with session_scope() as s:
        maps = dbn.uniformity_maps(s)
        assert (
            list(maps.name) == ["Operetta 384 viability"]
            and maps.instrument.iloc[0] == "Operetta CLS"
        )
        _, corrected = dbn.normalize_run(
            s, "SCREEN-2", "nuclei_count", "uniformity", "Operetta 384 viability"
        )
    samples = corrected.frame[corrected.frame.role == "sample"]
    raw_cv = samples.raw.std() / samples.raw.mean()
    new_cv = samples.normalized.std() / samples.normalized.mean()
    assert new_cv < 0.4 * raw_cv
    events = recent_events(5, "registry.uniformity_map")  # refused attempts are recorded too
    assert [e["status"] for e in events] == ["failed", "failed", "ok"]
    assert events[-1]["reason"] == "edge evaporation"


def test_day_effects_and_replicate_concordance_across_registered_runs(registry: Path) -> None:
    rng = np.random.default_rng(4)
    day0 = datetime(2026, 2, 2, 9, tzinfo=UTC)
    for d, level in enumerate((1000.0, 1250.0, 850.0)):
        for p in range(2):
            values, wells, _ = screening_plate(
                baseline=level, noise=0.02 * level, hits=0, seed=10 * d + p
            )
            values = {w: v * (1 + rng.normal(0, 0.01)) for w, v in values.items()}
            write_values_run(
                registry / "runs" / f"D{d}P{p}__run",
                values,
                wells,
                barcode=f"D{d}P{p}",
                started=day0 + timedelta(days=d),
            )
    runs = registry_runs(registry / "drc", n_runs=2)
    scan_folders(context(), [registry / "runs", runs[0].parent])
    with session_scope() as s:
        effects = dbn.day_effects(
            s, "nuclei_count", runs=[f"D{d}P{p}" for d in range(3) for p in range(2)]
        )
        summary = effects.summary.set_index("normalization").to_dict("index")
        assert summary["raw"]["r2_day"] > 0.8 and summary["same-day controls"]["r2_day"] < 0.05
        a, _, result = dbn.concordance(s, "REG-001", "REG-002", "nuclei_count")
        assert result.n == len([w for w in a.wells if w.role == "sample"])
        assert result.pearson > 0.95 and not result.discordant
        assert len(result.ic50) == 6 and result.ic50["fold"].max() < 1.5
        _, _, strict = dbn.concordance(s, "REG-001", "REG-002", "nuclei_count", fold_max=1.0001)
        assert strict.discordant and "IC50 differs" in strict.flags[0]
