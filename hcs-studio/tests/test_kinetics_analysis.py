"""Phase 22b: growth fits, doubling time, IC50 and GR over time on data with known answers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.kinetics.analysis import KineticsParams, analyse, write_kinetics
from hcs_studio.kinetics.growth import curve_metrics, fit_growth
from hcs_studio.synthetic.kinetics import response_course

EC50, EMAX, DOUBLING = 1e-7, 1.5, 24.0
CONCENTRATIONS = [1e-9 * 3**i for i in range(9)]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


# ------------------------------------------------------------------ growth curves


T = np.arange(0, 73, 6.0)


@pytest.mark.parametrize(
    ("name", "curve", "model", "doubling"),
    [
        ("exponential", 2000 * 2 ** (T / 20), "exponential", 20.0),
        ("lag", 2000 * 2 ** (np.clip(T - 12, 0, None) / 18), "lag_exponential", 18.0),
        ("logistic", 2000 * 3e4 / (2000 + (3e4 - 2000) * np.exp(-math.log(2) / 16 * T)),
         "logistic", 16.0),
    ],
)  # fmt: skip
def test_growth_models_recover_doubling_time(
    name: str, curve: np.ndarray, model: str, doubling: float
) -> None:
    """Done-when (22b): doubling time within 5 % (Poisson counts, 13 time points)."""
    counts = np.random.default_rng(1).poisson(curve).astype(float)
    fit = fit_growth(T, counts)
    assert fit.model == model, (name, fit.candidates)
    assert fit.doubling_h == pytest.approx(doubling, rel=0.05)
    if model == "lag_exponential":
        assert fit.lag_h == pytest.approx(12, abs=2.5)
    if model == "logistic":
        assert fit.capacity == pytest.approx(3e4, rel=0.1)


def test_flat_and_dying_wells_have_no_doubling_time() -> None:
    rng = np.random.default_rng(2)
    flat = fit_growth(T, rng.poisson(np.full(T.shape, 800.0)).astype(float))
    assert flat.model == "constant" and "not growing" in flat.flags
    assert math.isnan(flat.doubling_h)
    dying = fit_growth(T, rng.poisson(800 * np.exp(-0.03 * T)).astype(float))
    assert "shrinking" in dying.flags and math.isnan(dying.doubling_h)
    assert dying.rate_per_h == pytest.approx(-0.03, rel=0.1)
    short = fit_growth([0, 6], [100, 120])
    assert "too few time points" in short.flags


def test_curve_metrics() -> None:
    t = np.array([0.0, 10, 20, 30])
    y = np.array([100.0, 150, 200, 400])
    metrics = curve_metrics(t, y, fold=2.0, level=300)
    assert metrics["auc"] == pytest.approx(1250 + 1750 + 3000)  # trapezoids
    assert metrics["fold_change"] == 4.0
    assert metrics["time_to_fold_h"] == pytest.approx(20.0)
    assert metrics["time_to_level_h"] == pytest.approx(25.0)
    assert curve_metrics(t, y, fold=10)["time_to_fold_h"] is None


def test_doubling_time_from_images(home: Path) -> None:
    """Done-when (22b): doubling times of 18, 24 and 36 h within 5 % from segmented nuclei
    of a time-lapse plate, through the kinetics table step."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import GrowingCells, exponential

    doubling = {(0, 0): 18.0, (0, 1): 24.0, (0, 2): 36.0}
    cells = GrowingCells(count=lambda r, c, h: exponential(30, doubling[(r, c)])(r, c, h),
                         body_channel=None)  # fmt: skip
    plate = SyntheticPlate(wells=list(doubling), features=[cells, ShotNoise(sd=8.0)],
                           channels=("DAPI",), size=640, timepoints=9, name="KB",
                           time_points_s=tuple(6 * 3600.0 * i for i in range(9)))  # fmt: skip
    path, _ = plate.write(home / "raw")
    steps = [
        {"module": "segment_classical",
         "params": {"channel": 1, "method": "otsu", "min_distance_um": 2.0}},
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "qc"},
        {"module": "kinetics"},
    ]  # fmt: skip
    pipeline = Pipeline.model_validate({"name": "kinetics", "steps": steps})
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False).folder  # fmt: skip
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    for (row, col), expected in doubling.items():
        well = f"{'ABCDEFGH'[row]}{col + 1:02d}"
        assert summary.loc[well, "kn_doubling_h"] == pytest.approx(expected, rel=0.05)
        assert summary.loc[well, "kn_model"] == "exponential"
    fits = pd.read_csv(folder / "kinetics" / "growth_fits.csv")
    assert len(fits) == 3
    info = json.loads((folder / "kinetics" / "kinetics.json").read_text())
    assert info["readout"] == "nuclei_objects_measured" and info["time"] == "time_h"
    checks = {c["check"]: c["status"] for c in info["qc"]["checks"]}
    assert checks["time points"] == "pass" and checks["focus drift"] == "pass"
    assert (folder / "kinetics" / "report.html").exists()
    course = pd.read_csv(folder / "tables" / "time_course.csv")
    assert (course["fields_imaged"] == 1).all() and course["focus_score"].notna().all()
    index = (folder / "index.html").read_text()
    assert "kinetics/report.html" in index


# ---------------------------------------------------------- dose–response over time


def _plate(vehicles: int = 6, replicates: int = 3) -> tuple[dict[str, tuple[str, float]], Any]:
    from dataclasses import replace

    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.model.plate import parse_well_name, well_name

    wells: dict[str, tuple[str, float]] = {}
    for v in range(vehicles):
        wells[well_name(v, 0)] = ("negative_control", 0.0)
    for r in range(replicates):
        for i, conc in enumerate(CONCENTRATIONS):
            wells[well_name(r, i + 1)] = ("sample", conc)
    layout = PlateLayout.for_plate_type(96)
    for well, (role, conc) in wells.items():
        key = parse_well_name(well)
        layout = set_role(layout, [key], role, cell_line="HeLa")
        if conc > 0:
            layout = apply_fields(
                layout, [key], treatments=(Treatment.of("Drug", conc * 1e9, "nM"),)
            )
    return wells, replace(layout, treated_at="2026-01-05T10:00:00")


def _expected_ic50(hours: float) -> float:
    """% of control = exp(−μ₀ · EMAX · occupancy · t) = 50 % → occupancy = ln 2 / (μ₀ EMAX t)."""
    occupancy = math.log(2) / (math.log(2) / DOUBLING * EMAX * hours)
    return EC50 * occupancy / (1 - occupancy)


#: GR = 2^(μ(c)/μ₀) − 1 = 0.5 → 1 − EMAX · occupancy = log2(1.5)
GR50 = EC50 * ((1 - math.log2(1.5)) / EMAX) / (1 - (1 - math.log2(1.5)) / EMAX)
GR_INF = 2 ** (1 - EMAX) - 1


def test_ic50_and_gr_over_time_known_answers() -> None:
    """IC50(t) falls with time as the analytical answer (within 10 %); GR50 and GRinf of an
    exponentially growing population do not change with time (within 15 % / 0.05)."""
    wells, layout = _plate()
    times = [-6.0, 0, 12, 24, 36, 48, 60, 72]
    course = response_course(wells, [t + 6 for t in times], n0=2000, doubling_h=DOUBLING,
                             emax=EMAX, ec50=EC50, treated_h=6)  # fmt: skip
    result = analyse(course, layout, KineticsParams())
    assert result.time_column == "hours_since_treatment"
    ic50 = result.ic50.set_index("time_h")["ic50"]
    for hours in (24, 48, 72):
        assert ic50[hours] == pytest.approx(_expected_ic50(hours), rel=0.10), hours
    assert ic50[72] < ic50[48] < ic50[24]  # potency grows with exposure time
    metrics = result.gr_metrics.set_index("time_h")
    for hours in (24, 48, 72):
        assert metrics.loc[hours, "GR50"] == pytest.approx(GR50, rel=0.15), hours
        assert metrics.loc[hours, "GRinf"] == pytest.approx(GR_INF, abs=0.05), hours
    assert (result.gr_values["time_h"] > 0).all()  # from treatment on
    gr = result.gr_values
    vehicle = gr[(gr["role"] == "negative_control") & (gr["time_h"] >= 24)]["gr"]
    assert (vehicle - 1).abs().max() < 0.1  # controls grow as the controls: GR ≈ 1
    per_well = result.per_well()
    assert per_well["A02"]["kn_doubling_h"] == pytest.approx(DOUBLING, rel=0.05)
    assert per_well["A10"]["kn_auc_vs_vehicle_pct"] < 50


def test_gr_over_intervals_shows_a_late_effect(tmp_path: Path) -> None:
    """A drug acting only 36 h after treatment: GR over intervals ≈ 1 before (growth as in
    the controls) and 2^(μ(c)/μ₀) − 1 after; GR from treatment still mixes the two at 48 h."""
    wells, layout = _plate()
    times = [0.0, 12, 24, 36, 48, 60, 72, 84]
    course = response_course(wells, times, n0=2000, doubling_h=DOUBLING, emax=EMAX,
                             ec50=EC50, onset_h=36)  # fmt: skip
    result = analyse(course, layout, KineticsParams(interval_h=24))
    values = result.gr_interval_values
    assert sorted(values["time_h"].unique()) == [12.0, 36.0, 60.0]  # [0, 24], [24, 48], [48, 72]
    top = values[values["concentration"] == CONCENTRATIONS[-1]].groupby("start_h")["gr"].mean()
    occupancy = CONCENTRATIONS[-1] / (CONCENTRATIONS[-1] + EC50)
    assert top[0.0] == pytest.approx(1.0, abs=0.1)
    assert top[48.0] == pytest.approx(2 ** (1 - EMAX * occupancy) - 1, abs=0.06)
    gr50 = result.gr_interval_metrics.set_index("time_h")["GR50"].to_dict()
    assert math.isnan(gr50[12.0])  # no effect yet
    assert gr50[60.0] == pytest.approx(GR50, rel=0.2)
    files = write_kinetics(result, Path(tmp_path), {})
    assert {"growth_curves.png", "potency_over_time.png", "gr_interval_metrics.csv"} <= set(files)
    from_treatment = result.gr_metrics.set_index("time_h")["GRinf"].to_dict()
    assert from_treatment[48.0] > GR_INF + 0.2  # diluted by the first 36 h


def _constant(n: int, row: int, col: int, hours: float) -> int:
    return n


def test_kinetics_cli_on_separate_acquisitions(home: Path) -> None:
    """Three single-time-point runs of one plate, joined on their times, then analysed."""
    from hcs_studio.app import main
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import GrowingCells

    steps = [
        {"module": "segment_classical",
         "params": {"channel": 1, "method": "otsu", "min_distance_um": 2.0}},
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
    ]  # fmt: skip
    pipeline = Pipeline.model_validate({"name": "count", "steps": steps})
    folders = []
    for hours in (0, 24, 48):
        count = round(25 * 2 ** (hours / 24))
        cells = GrowingCells(count=partial(_constant, count), body_channel=None)
        plate = SyntheticPlate(wells=[(0, 0)], features=[cells, ShotNoise()], size=448,
                               channels=("DAPI",), name="ACQ",
                               acquired_at=f"2026-01-{5 + hours // 24:02d}T10:00:00")  # fmt: skip
        path, _ = plate.write(home / f"raw{hours}")
        folders.append(run_pipeline(open_plate(path), pipeline, output_root=home / f"runs{hours}",
                                    save_images=False).folder)  # fmt: skip
    out = home / "kinetics"
    args = ["kinetics", "analyse", *map(str, folders), "--output", str(out),
            "--models", "exponential"]  # fmt: skip
    assert main(args) == 0
    fits = pd.read_csv(out / "growth_fits.csv")
    assert fits.loc[0, "doubling_h"] == pytest.approx(24.0, rel=0.05)
    assert json.loads((out / "kinetics.json").read_text())["runs"] == [f.name for f in folders]
