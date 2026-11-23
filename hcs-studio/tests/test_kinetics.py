"""Phase 22a: time-course data — real times, treatment time, drift, confluence, time courses."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.kinetics.drift import common_area, drift_shifts, shift_image


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _pipeline(*steps: dict[str, Any]) -> Any:
    from hcs_studio.pipeline.schema import Pipeline

    return Pipeline.model_validate({"name": "kinetics", "steps": list(steps)})


SEGMENT = {"module": "segment_classical",
           "params": {"channel": 1, "method": "otsu", "min_distance_um": 2.0}}  # fmt: skip
MEASURE = {"module": "measure_objects", "params": {"texture": False, "granularity": False}}


def test_time_lapse_run_counts_drift_and_time_course(home: Path) -> None:
    """Counts per time point exact, drift recovered within 0.3 px, times from the instrument
    and hours since treatment from the layout's treatment time."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import GrowingCells, exponential

    doubling = {(0, 0): 20.0, (0, 1): 30.0, (0, 2): 1e9}
    cells = GrowingCells(
        count=lambda r, c, h: exponential(12, doubling[(r, c)])(r, c, h),
        drift=lambda r, c, h: (0.4 * h, -0.25 * h),  # µm
        margin_px=20,
    )
    times = (0.0, 6 * 3600.0, 12 * 3600.0, 18.5 * 3600.0, 24 * 3600.0)  # uneven, as acquired
    plate = SyntheticPlate(wells=list(doubling), features=[cells, ShotNoise(sd=8.0)],
                           channels=("DAPI", "Phase"), size=384, timepoints=5, name="KIN",
                           time_points_s=times, acquired_at="2026-01-05T10:00:00")  # fmt: skip
    path, truth = plate.write(home / "raw")
    layout = replace(PlateLayout.for_plate_type(96), treated_at="2026-01-05T08:30:00")
    register = {"module": "register_timepoints", "params": {"channel": 2}}
    confluence = {"module": "confluence", "params": {"channel": 2}}
    pipeline = _pipeline(register, SEGMENT, MEASURE, confluence)
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs", layout=layout,
                          save_images=False).folder  # fmt: skip

    manifest = json.loads((folder / "run_manifest.json").read_text())["plate"]
    assert manifest["time_source"] == "instrument" and manifest["time_points_s"] == list(times)
    course = pd.read_parquet(folder / "tables" / "time_course.parquet")
    assert len(course) == 15
    for well, data in truth["wells"].items():
        rows = course[course.well == well].sort_values("timepoint")
        expected = [t["cells"]["count"] for t in data["fields"]["1"]["timepoints"]]
        assert rows["nuclei_objects_measured"].tolist() == expected, well
        assert rows["time_h"].tolist() == pytest.approx([t / 3600 for t in times])
        assert rows["hours_since_treatment"].tolist() == pytest.approx(
            [1.5 + t / 3600 for t in times]
        )
        assert rows["confluence_pct"].notna().all()
        record = json.loads((folder / "wells" / well / "well_summary.json").read_text())
        drift = record["fields"][0]["drift"]
        for got, data_t in zip(drift["shifts_px"], data["fields"]["1"]["timepoints"], strict=True):
            # the correction moves each time point back: minus the drift drawn
            assert got == pytest.approx([-v for v in data_t["cells"]["drift_px"]], abs=0.3)
        assert drift["crop_px"][1] >= 15 and drift["crop_px"][2] >= 9  # bottom, left
    growing = course[course.well == "A01"].sort_values("timepoint")["confluence_pct"].to_numpy()
    assert np.all(np.diff(growing) > 0)


def test_confluence_accuracy_from_texture(home: Path) -> None:
    """Texture confluence within 12 % (≥ 1 pp) of the truth from empty to near-confluent
    fields, calibrated on the pooled fields; an empty well reads ~0."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import GrowingCells

    counts = {(0, c): n for c, n in enumerate((0, 5, 20, 50, 80, 120))}
    cells = GrowingCells(count=lambda r, c, h: counts[(r, c)], body_channel=0, nucleus_channel=1)
    plate = SyntheticPlate(wells=list(counts), features=[cells, ShotNoise(sd=8.0)],
                           channels=("Phase", "DAPI"), size=448, name="CONF")  # fmt: skip
    path, truth = plate.write(home / "raw")
    pipeline = _pipeline({"module": "confluence", "params": {"channel": 1}})
    folder = run_pipeline(open_plate(path), pipeline, output_root=home / "runs",
                          save_images=False).folder  # fmt: skip
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    for well, data in truth["wells"].items():
        expected = data["fields"]["1"]["cells"]["confluence_pct"]
        got = summary.loc[well, "confluence_pct"]
        assert abs(got - expected) <= max(1.0, 0.12 * expected), (well, expected, got)
    fitted = json.loads((folder / "run_manifest.json").read_text())["fitted_state"][0]
    assert fitted["module"] == "confluence" and fitted["texture"] > 2 * fitted["noise"]


def test_confluence_on_an_empty_plate_is_not_noise() -> None:
    from hcs_studio.kinetics.confluence import signal, texture_level

    noise = np.random.default_rng(0).normal(1000, 8, (2, 256, 256))
    values = np.concatenate([signal(plane, "texture", 1.5).ravel() for plane in noise])
    state = texture_level(values)
    assert "warning" in state
    assert float(np.mean(signal(noise[0], "texture", 1.5) > state["level"])) < 0.01


def _texture(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    from scipy import ndimage

    noise = np.random.default_rng(seed).normal(0, 1, shape)
    return np.asarray(ndimage.gaussian_filter(noise, 3.0) * 1000 + 500)


def test_drift_shifts_recover_subpixel_drift_and_skip_jumps() -> None:
    base = _texture((200, 200))
    truth = [(0.0, 0.0), (1.3, -0.7), (2.9, -1.2), (4.1, -2.6)]
    planes = [shift_image(base[None, None], s)[0, 0][20:-20, 20:-20] for s in truth]
    shifts = drift_shifts(planes, upsample=20)
    for got, (dy, dx) in zip(shifts, truth, strict=True):
        assert got == pytest.approx((-dy, -dx), abs=0.1)
    assert common_area([(0.0, 0.0), (-1.2, 2.5), (3.0, -0.4)], (100, 100)) == (3, 2, 3, 1)


def test_register_timepoints_reports_stage_jumps() -> None:
    from hcs_studio.kinetics.drift import RegisterTimepoints, RegisterTimepointsParams
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData

    metadata = ImageMetadata(channels=(ChannelInfo("Phase"),), pixel_size_um=1.0)
    base = _texture((240, 240))
    moves = [(0.0, 0.0), (1.0, 0.5), (31.0, 0.5), (32.0, 1.0)]  # a 30 µm jump at t = 3
    frames = [shift_image(base[None, None], s)[..., 40:-40, 40:-40] for s in moves]
    module = RegisterTimepoints(RegisterTimepointsParams(channel=1, max_shift_um=20))
    info = module.shifts(lambda t: ImageData.from_array(frames[t], metadata), 4)
    assert len(info["warnings"]) == 1 and "time point 3" in info["warnings"][0]
    shifts = info["shifts_px"]
    assert shifts[1] == pytest.approx([-1.0, -0.5], abs=0.1)
    assert shifts[2] == pytest.approx(shifts[1], abs=1e-9)  # the jump is not applied
    assert shifts[3] == pytest.approx([-2.0, -1.0], abs=0.15)  # the drift continues after it
    moved = module.correct(ImageData.from_array(frames[1], metadata), shifts[1], info["crop_px"])
    assert moved.data.shape[-2:] == (160 - 2, 160 - 1)


def test_stitching_and_drift_correction_are_refused(home: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import PipelineRunner
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.synthetic import SyntheticPlate

    path, _ = SyntheticPlate(wells=[(0, 0)], timepoints=2).write(home / "raw")
    pipeline = _pipeline({"module": "register_timepoints"}, {"module": "stitch_well"})
    with pytest.raises(ModuleError, match="stitching"):
        PipelineRunner(open_plate(path), pipeline)


# ---------------------------------------------------------------- times from readers


def test_imagexpress_time_points_from_image_timestamps(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.plate import plate_times
    from hcs_studio.utils.synthetic import write_imagexpress_plate

    def image(row: int, col: int, s: int, w: int, z: int, t: int = 0) -> np.ndarray:
        return np.full((8, 8), 100 + t, dtype=np.uint16)

    root = write_imagexpress_plate(tmp_path / "ix", [(0, 0)], image, n_sites=1, timepoints=3,
                                   time_points_s=(0.0, 5400.0, 14400.0))  # fmt: skip
    plate = open_plate(root)
    assert plate.shape_tczyx is not None and plate.shape_tczyx[0] == 3
    assert plate_times(plate) == ((0.0, 5400.0, 14400.0), "instrument")
    assert plate.acquired_at == "2026-01-05T09:00:00"
    assert int(np.asarray(plate.wells[(0, 0)].fields[0].load()[2, 0, 0, 0, 0])) == 102


def test_ome_zarr_times_and_the_metadata_interval(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.plate import plate_times
    from hcs_studio.synthetic import SyntheticPlate

    timed, _ = SyntheticPlate(wells=[(0, 0)], timepoints=3, time_points_s=(0.0, 600.0, 1500.0),
                              acquired_at="2026-02-01T08:00:00").write(tmp_path / "a")  # fmt: skip
    plate = open_plate(timed)
    assert plate_times(plate) == ((0.0, 600.0, 1500.0), "instrument")
    assert plate.acquired_at == "2026-02-01T08:00:00"

    untimed = replace(plate, time_points_s=None)
    times, source = plate_times(untimed)
    assert times is None and source.startswith("unknown")
    metadata = untimed.metadata.with_overrides({"time_interval_s": 1800})
    assert plate_times(replace(untimed, metadata=metadata)) == (
        (0.0, 1800.0, 3600.0),
        "time interval (metadata)",
    )


def test_nd2_time_points_from_frame_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hcs_studio.io import nikon
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.plate import plate_times
    from tests.builders import FakeND2

    sizes = {"T": 3, "P": 2, "C": 1, "Y": 6, "X": 8}
    data = np.zeros((3, 2, 1, 6, 8), dtype=np.uint16)
    path = tmp_path / "plate.nd2"
    path.write_bytes(b"")
    fake = FakeND2(sizes, ["B02", "C05"], data, frame_times_s=[12.0, 912.0, 1812.0])
    monkeypatch.setattr(nikon, "_open_nd2", lambda p: fake)
    plate = open_plate(path)
    assert plate_times(plate) == ((0.0, 900.0, 1800.0), "instrument")


# ---------------------------------------------------------------- treatment time and acquisitions


def test_treatment_time_sources(home: Path) -> None:
    from lab_events import make_event

    from hcs_studio.automation.intake import receive
    from hcs_studio.kinetics.timecourse import treatment_time
    from hcs_studio.model.annotation import PlateLayout

    manifest = {"plate": {"barcode": "P0042"}}
    assert treatment_time(manifest) == (None, "")
    event = make_event("plate.dispensed",
                       {"barcode": "P0042", "dispensed_at": "2026-01-05T08:00:00+01:00"},
                       source={"system": "dispenser", "instance": "d1"})  # fmt: skip
    assert receive(event, "test").status == "accepted"
    when, source = treatment_time(manifest)
    local = datetime.fromisoformat("2026-01-05T08:00:00+01:00").astimezone().replace(tzinfo=None)
    assert (when, source) == (local, "plate.dispensed event")  # this computer's local time
    layout = replace(PlateLayout.for_plate_type(96), treated_at="2026-01-05T07:45:00")
    assert treatment_time(manifest, layout)[1] == "plate layout"
    assert treatment_time(manifest, layout, "2026-01-05T07:00")[1] == "given"


def test_layout_keeps_the_treatment_time(tmp_path: Path) -> None:
    from hcs_studio.io.layout_io import load_layout, save_layout
    from hcs_studio.model.annotation import PlateLayout

    layout = replace(PlateLayout.for_plate_type(96), treated_at="2026-01-05T08:30:00")
    assert PlateLayout.from_dict(layout.to_dict()).treated_at == "2026-01-05T08:30:00"
    assert "treated_at" not in PlateLayout.for_plate_type(96).to_dict()
    for name in ("layout.xlsx", "layout.csv"):
        save_layout(layout, tmp_path / name)
        assert load_layout(tmp_path / name).layout.treated_at == "2026-01-05T08:30:00", name


def _single_run(home: Path, name: str, started: str, count: int, plate_name: str = "REP") -> Path:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import GrowingCells

    cells = GrowingCells(count=lambda r, c, h: count, body_channel=None)
    plate = SyntheticPlate(wells=[(0, 0)], features=[cells, ShotNoise()], size=256,
                           acquired_at=started, name=plate_name)  # fmt: skip
    path, _ = plate.write(home / name)
    return run_pipeline(open_plate(path), _pipeline(SEGMENT, MEASURE),
                        output_root=home / f"runs_{name}", save_images=False).folder  # fmt: skip


def test_separate_acquisitions_are_joined_on_their_times(home: Path) -> None:
    from hcs_studio.kinetics.timecourse import assemble, readout_matrix

    second = _single_run(home, "b", "2026-01-05T16:00:00", 20)
    first = _single_run(home, "a", "2026-01-05T10:00:00", 10)
    table = assemble([second, first], treated_at="2026-01-05T09:00:00")
    assert table["time_h"].tolist() == [0.0, 6.0]
    assert table["hours_since_treatment"].tolist() == [1.0, 7.0]
    assert table["nuclei_objects_measured"].tolist() == [10, 20]
    assert list(table["acquisition"]) == [0, 1]
    matrix = readout_matrix(table, "nuclei_objects_measured")
    assert list(matrix.loc["A01"]) == [10.0, 20.0]
    assert [float(c) for c in matrix.columns] == [1.0, 7.0]
    other = _single_run(home, "c", "2026-01-05T20:00:00", 5, plate_name="OTHER")
    with pytest.raises(ValueError, match="different plates"):
        assemble([first, other])

    from hcs_studio.app import main

    out = home / "joined.csv"
    assert main(["kinetics", "table", str(first), str(second), "--output", str(out)]) == 0
    assert pd.read_csv(out)["time_h"].tolist() == [0.0, 6.0]
    assert main(["kinetics", "table", str(first), str(second)]) == 2  # needs --output
