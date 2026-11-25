"""Phase 16d: ilastik Multicut and Tracking against the fake ilastik — boundary probabilities
→ Multicut objects (known grid), and tracking of moving, dividing cells: ilastik's links become
the built-in track table and ``trk_`` readouts (speed, divisions, cycle time)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests import test_ilastik_steps
from tests.test_ilastik_steps import calls, make_project

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)


@dataclass
class GridWalls:
    """Bright 2-px walls splitting each 96 × 96 field into 3 × 3 cells (a boundary stain)."""

    name: str = "walls"

    def draw(self, ctx: Any) -> dict[str, Any]:
        for k in (31, 63):
            ctx.image[0, :, k : k + 2, :] += 2000
            ctx.image[0, :, :, k : k + 2] += 2000
        return {"cells": 9}


def test_multicut_from_boundary_probabilities(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1)],
        channels=("membrane",),
        size=96,
        n_fields=2,
        name="MC01",
        features=[GridWalls(), ShotNoise(sd=3)],
    )
    path, _ = plate.write(tmp_path / "raw")
    pixel = make_project(tmp_path / "boundary.ilp", names=("inside", "boundary"))
    multicut = make_project(tmp_path / "multicut.ilp", workflow="EdgeTrainingWithMulticutWorkflow")
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="ilastik_pixel", params={"project": str(pixel)}),
            PipelineStep(module="ilastik_multicut", params={"project": str(multicut)}),
            PipelineStep(
                module="measure_objects",
                params={"labels": ["cells"], "channels": [1], "texture": False},
            ),
        ]
    )
    result = run_pipeline(
        open_plate(path), pipeline, output_root=tmp_path / "runs", workers=1, checksums=False
    )
    assert not result.failed_wells, result.failed_wells
    wells = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    assert wells["cells_objects_measured"].to_dict() == {"A01": 18, "A02": 18}  # 9 per field
    exports = [
        next(a for a in c["argv"] if a.startswith("--export_source")) for c in calls(ilastik["log"])
    ]
    assert exports.count("--export_source=Multicut Segmentation") == 2  # one call per well
    multicut_call = next(c for c in calls(ilastik["log"]) if "Multicut" in json.dumps(c))
    assert "--probabilities" in multicut_call["argv"]


def test_tracking_through_the_pipeline(ilastik: dict[str, Any], tmp_path: Path) -> None:
    """Moving, dividing nuclei (hourly frames, 24 h): ilastik's tracks give the track table and
    per-well speed, divisions and cell-cycle time of the built-in tracker."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.kinetics import MovingCells

    times = [float(t) for t in range(25)]
    cells = MovingCells(times_h=times, n0=8, speed_um_per_h=8.0, cycle_h=14.0, seed=3)
    plate = SyntheticPlate(
        wells=[(0, 0)],
        features=[cells, ShotNoise(sd=6.0)],
        channels=("DAPI",),
        size=384,
        timepoints=len(times),
        name="TRK",
        time_points_s=tuple(3600.0 * t for t in times),
    )
    path, truth = plate.write(tmp_path / "raw")
    project = make_project(
        tmp_path / "tracking.ilp", workflow="ConservationTrackingWorkflowFromBinary"
    )
    steps = [
        {
            "module": "segment_classical",
            "params": {"channel": 1, "method": "otsu", "min_distance_um": 3.0},
        },
        {"module": "ilastik_tracking", "params": {"project": str(project), "labels": "nuclei"}},
        {
            "module": "measure_objects",
            "params": {"labels": ["tracked"], "texture": False, "granularity": False},
        },
        {"module": "ilastik_tracks"},
    ]
    result = run_pipeline(
        open_plate(path),
        Pipeline.model_validate({"steps": steps}),
        output_root=tmp_path / "runs",
        save_images=False,
        workers=1,
        checksums=False,
    )
    assert not result.failed_wells, result.failed_wells
    folder = result.folder
    sources = [
        next(a for a in c["argv"] if a.startswith("--export_source")) for c in calls(ilastik["log"])
    ]
    assert sorted(sources) == ["--export_source=Object-Identities", "--export_source=Plugin"]
    plugin = next(c for c in calls(ilastik["log"]) if "--export_source=Plugin" in c["argv"])
    assert "--export_plugin=CSV-Table" in plugin["argv"]
    links = pd.read_csv(folder / "wells" / "A01" / "ilastik" / "tracks_tracked.csv")
    assert set(links["timepoint"]) == set(range(25))
    born = links.sort_values("timepoint").groupby("track_id").first()
    mothers = born.loc[born["parent_track_id"] > 0, "parent_track_id"].nunique()
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    values = summary.to_dict("index")["A01"]
    assert values["trk_divisions"] == mothers > 0  # ilastik's divisions, counted as ours
    assert values["trk_speed_um_per_h"] == pytest.approx(8.0, rel=0.3)
    true_divisions = sum(
        1
        for t in truth["wells"]["A01"]["fields"]["1"]["timepoints"]
        for c in t["moving"]["cells"]
        if c["parent"]
    )
    assert true_divisions > 0
    tracks = pd.read_csv(folder / "tracking" / "tracks.csv")
    assert {"track", "parent", "speed_um_per_h", "ends"} <= set(tracks.columns)
    assert "tracked_trk_id_mean" not in summary.columns


def test_tracking_csv_spellings(tmp_path: Path) -> None:
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.segmentation.ilastik_backend.multicut_tracking import read_tracking_csv

    path = tmp_path / "t.csv"
    path.write_text(
        "Frame,labelimage_oid,trackId,parentTrackId\n0,1,1,-1\n1,1,2,1\n1,2,3,1\n1,0,0,0\n"
    )
    table = read_tracking_csv(path)
    assert table.to_dict("list") == {
        "timepoint": [0, 1, 1],
        "object_id": [1, 1, 2],
        "track_id": [1, 2, 3],
        "parent_track_id": [0, 1, 1],
        "lineage_id": [0, 0, 0],
    }
    (tmp_path / "bad.csv").write_text("frame,something\n0,1\n")
    with pytest.raises(ModuleError, match="no object_id column"):
        read_tracking_csv(tmp_path / "bad.csv")


def test_step_checks(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData, ModuleError, create_module

    meta = ImageMetadata(channels=(ChannelInfo("DAPI"),), pixel_size_um=0.65)
    multicut = make_project(tmp_path / "m.ilp", workflow="EdgeTrainingWithMulticutWorkflow")
    with pytest.raises(ModuleError, match="boundary probabilities"):
        create_module("ilastik_multicut", {"project": str(multicut)}).validate(meta)
    with pytest.raises(ModuleError, match="needs an ilastik Tracking project"):
        create_module("ilastik_tracking", {"project": str(multicut)}).validate(meta)
    tracking = make_project(tmp_path / "t.ilp", workflow="ConservationTrackingWorkflowFromBinary")
    step = create_module("ilastik_tracking", {"project": str(tracking)})
    image = ImageData(np.zeros((1, 1, 16, 16), dtype=np.float32), meta)
    with pytest.raises(ModuleError, match="time-lapse"):
        step.apply(image, None)


def test_presets(tmp_path: Path) -> None:
    import yaml

    from hcs_studio.app import main
    from hcs_studio.pipeline.schema import Pipeline

    boundary = make_project(tmp_path / "b.ilp", names=("inside", "boundary"))
    multicut = make_project(tmp_path / "m.ilp", workflow="EdgeTrainingWithMulticutWorkflow")
    tracking = make_project(tmp_path / "t.ilp", workflow="ConservationTrackingWorkflowFromBinary")
    out = tmp_path / "p.yaml"
    assert main(["ilastik", "--preset", str(multicut), "--output", str(out)]) == 2
    assert (
        main(
            [
                "ilastik",
                "--preset",
                str(multicut),
                "--boundary",
                str(boundary),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    steps = [s.module for s in Pipeline.model_validate(yaml.safe_load(out.read_text())).steps]
    assert steps == ["ilastik_pixel", "ilastik_multicut", "measure_objects", "qc"]
    assert main(["ilastik", "--preset", str(tracking), "--output", str(out)]) == 0
    steps = [s.module for s in Pipeline.model_validate(yaml.safe_load(out.read_text())).steps]
    assert steps == ["segment_cellpose", "ilastik_tracking", "measure_objects", "ilastik_tracks"]
