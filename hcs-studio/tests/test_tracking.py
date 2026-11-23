"""Phase 22d: cell tracking (frame-to-frame assignment, divisions, gap closing) on moving,
dividing synthetic cells with known lineages."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcs_studio.kinetics.tracking import (
    TrackingOptions,
    track_objects,
    track_table,
    well_summary,
)
from hcs_studio.synthetic.kinetics import simulate_lineages


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _points(times: list[float], seed: int = 1, **kw: float) -> pd.DataFrame:
    sim = simulate_lineages(times, (300, 300), n0=10, speed_um_per_h=kw.get("speed", 10.0),
                            cycle_h=20, radius_um=4, seed=seed)  # fmt: skip
    rows = [{"timepoint": t, "centroid_y_um": c["y"], "centroid_x_um": c["x"],
             "area_um2": math.pi * c["radius"] ** 2, "truth": c["id"], "truth_parent": c["parent"]}
            for t, cells in enumerate(sim) for c in cells]  # fmt: skip
    return pd.DataFrame(rows)


def score(tracked: pd.DataFrame) -> tuple[float, float, int]:
    """Fraction of true links kept in one track, of true divisions found (daughter's parent =
    the mother's last track), and the number of divisions."""
    kept = total = 0
    for _, group in tracked.groupby("truth"):
        ids = group.sort_values("timepoint")["trk_id"].to_numpy()
        total += len(ids) - 1
        kept += int((ids[1:] == ids[:-1]).sum())
    first = tracked.sort_values("timepoint").groupby("truth").first()
    daughters = first[first["truth_parent"] > 0]
    found = 0
    for _, row in daughters.iterrows():
        mother = tracked[tracked["truth"] == row["truth_parent"]].sort_values("timepoint")
        found += int(not mother.empty and row["trk_parent"] == mother["trk_id"].iloc[-1])
    return kept / max(total, 1), found / max(len(daughters), 1), len(daughters)


def test_tracks_and_divisions_of_known_lineages() -> None:
    """10 cells → ~50 over 48 h (cycle 20 h ± 10 %), 10 µm/h persistent random walk, hourly
    frames: ≥ 97 % of links and ≥ 90 % of divisions recovered; speed within 5 %, cell cycle
    within 10 %."""
    times = [float(t) for t in range(49)]
    points = _points(times)
    tracked = pd.concat([points, track_objects(points, times)], axis=1)
    links, divisions, n = score(tracked)
    assert n > 30
    assert links >= 0.97 and divisions >= 0.90, (links, divisions)
    summary = well_summary(track_table(tracked, times))
    assert summary["trk_speed_um_per_h"] == pytest.approx(10.0, rel=0.05)
    assert summary["trk_cycle_h"] == pytest.approx(20.0, rel=0.10)
    assert summary["trk_divisions"] == pytest.approx(n / 2, abs=4)
    assert summary["trk_lost_fraction"] < 0.05


def test_prediction_helps_fast_cells() -> None:
    times = [float(t) for t in range(49)]
    points = _points(times, speed=20.0)
    with_prediction = score(pd.concat([points, track_objects(points, times)], axis=1))
    plain = track_objects(points, times, TrackingOptions(predict=False))
    without = score(pd.concat([points, plain], axis=1))
    assert with_prediction[0] > without[0] + 0.05


def test_gap_closing_and_gate() -> None:
    """A cell missing from one frame keeps its track (gap closing); a jump beyond the gate
    starts a new track; tracks end as 'lost' or at the end of the movie."""
    times = [0.0, 1, 2, 3, 4, 5]
    rows = []
    for t in range(6):
        rows.append({"timepoint": t, "centroid_y_um": 50.0, "centroid_x_um": 50 + 5.0 * t})
        if t != 3:  # cell 2 is not detected at t = 3
            rows.append({"timepoint": t, "centroid_y_um": 150.0, "centroid_x_um": 50 + 5.0 * t})
        if t >= 4:  # an object 200 µm away appears: not a continuation of anything
            rows.append({"timepoint": t, "centroid_y_um": 250.0, "centroid_x_um": 250.0})
    objects = pd.DataFrame(rows)
    objects["area_um2"] = 50.0
    ids = track_objects(objects, times, TrackingOptions(divisions=False))
    tracked = pd.concat([objects, ids], axis=1)
    by_row = tracked.groupby("centroid_y_um")["trk_id"].nunique()
    assert by_row[50.0] == 1 and by_row[150.0] == 1  # the gap is closed
    assert tracked.loc[tracked.centroid_y_um == 250, "trk_id"].iloc[0] not in set(
        tracked.loc[tracked.centroid_y_um < 250, "trk_id"]
    )
    table = track_table(tracked, times)
    assert set(table["ends"]) == {"end"}
    assert table.loc[table["frames"] == 6, "speed_um_per_h"].tolist() == [pytest.approx(5.0)]
    closed = TrackingOptions(divisions=False, max_gap=0)
    assert pd.concat([objects, track_objects(objects, times, closed)], axis=1).groupby(
        "centroid_y_um")["trk_id"].nunique()[150.0] == 2  # fmt: skip


def test_tracking_from_images(home: Path) -> None:
    """Segmented nuclei of moving, dividing cells (hourly frames, 24 h) through the pipeline:
    ≥ 90 % of links and ≥ 75 % of divisions recovered; tracks.csv and per-well values."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import MovingCells

    times = [float(t) for t in range(25)]
    cells = MovingCells(times_h=times, n0=8, speed_um_per_h=8.0, cycle_h=14.0, seed=3)
    plate = SyntheticPlate(wells=[(0, 0)], features=[cells, ShotNoise(sd=6.0)], channels=("DAPI",),
                           size=384, timepoints=len(times), name="TRK",
                           time_points_s=tuple(3600.0 * t for t in times))  # fmt: skip
    path, truth = plate.write(home / "raw")
    steps = [
        {"module": "segment_classical",
         "params": {"channel": 1, "method": "otsu", "min_distance_um": 3.0}},
        {"module": "measure_objects", "params": {"texture": False, "granularity": False}},
        {"module": "tracking"},
    ]  # fmt: skip
    pipeline = Pipeline.model_validate({"name": "tracking", "steps": steps})
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False).folder  # fmt: skip
    from hcs_studio.analysis.table_steps import read_well_objects

    objects = read_well_objects(folder / "wells" / "A01")
    objects = objects[objects["object_type"] == "nuclei"].copy()
    # match each segmented nucleus to the true cell at that time point (within 3 µm)
    per_t = truth["wells"]["A01"]["fields"]["1"]["timepoints"]
    labels, parents = [], []
    for row in objects.itertuples():
        cells = per_t[int(str(row.timepoint))]["moving"]["cells"]
        d = [math.hypot(c["y"] - row.centroid_y_um, c["x"] - row.centroid_x_um) for c in cells]
        k = int(np.argmin(d))
        labels.append(cells[k]["id"] if d[k] < 3 else -1)
        parents.append(cells[k]["parent"] if d[k] < 3 else -1)
    objects["truth"], objects["truth_parent"] = labels, parents
    matched = objects[objects["truth"] > 0]
    assert len(matched) >= 0.95 * len(objects)
    links, divisions, n = score(matched)
    assert n >= 6
    assert links >= 0.90 and divisions >= 0.75, (links, divisions, n)
    tracks = pd.read_csv(folder / "tracking" / "tracks.csv")
    assert {"track", "parent", "speed_um_per_h", "ends"} <= set(tracks.columns)
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    values = summary.to_dict("index")["A01"]
    assert values["trk_speed_um_per_h"] == pytest.approx(8.0, rel=0.25)
    assert values["trk_divisions"] >= 0.75 * n / 2
    assert values["trk_cycle_h"] == pytest.approx(14.0, rel=0.15)
    assert "nuclei_trk_id_mean" not in summary.columns  # ids are not features
