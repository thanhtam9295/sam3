"""Phase 16g: ilastik as one option of other phases — spheroid detection (21), confluence (22)
and validation against ground truth (25) — with the fake ilastik."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests import test_ilastik_steps
from tests.test_ilastik_steps import calls, make_project

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)


def test_probability_spheroids_known_answer() -> None:
    from hcs_studio.assays.spheroids import probability_spheroids

    yy, xx = np.mgrid[0:120, 0:120]
    big = (yy - 40) ** 2 + (xx - 40) ** 2 <= 30**2
    core = (yy - 40) ** 2 + (xx - 40) ** 2 <= 10**2  # a necrotic core read as another class
    small = (yy - 95) ** 2 + (xx - 95) ** 2 <= 12**2
    speck = (yy - 100) ** 2 + (xx - 20) ** 2 <= 2**2
    probability = np.where(big & ~core, 0.9, 0.05)
    probability[small | speck] = 0.8
    labels = probability_spheroids(probability, 0.5, min_area_px=50)
    assert int(np.max(labels)) == 2
    assert np.array_equal(labels == 1, big)  # largest first, core filled
    assert np.array_equal(labels == 2, small)  # the speck is dropped


def test_spheroids_detected_by_ilastik(ilastik: dict[str, Any], tmp_path: Path) -> None:
    """Brightfield spheroids of known radius: ilastik's spheroid class → detect_spheroids
    (probability) → the spheroid assay gives volumes within 2 % of the truth."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.segmentation.ilastik_backend.presets import spheroid_preset
    from hcs_studio.segmentation.ilastik_backend.project import read_project
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.spheroids import BrightfieldSpheroid

    specs = {(0, 0): {"radius_um": 200}, (0, 1): {"radius_um": 120}, (0, 2): {"radius_um": 70}}
    plate = SyntheticPlate(
        wells=list(specs),
        features=[
            BrightfieldSpheroid(spec=lambda r, c: specs[(r, c)], meniscus=0.0),
            ShotNoise(sd=8.0),
        ],
        channels=("Brightfield",),
        size=320,
        pixel_size_um=2.0,
        name="SPHI",
    )
    raw, truth = plate.write(tmp_path / "raw")
    # the stand-in's "spheroid" probability: 0 at the background (1000), 1 at the body (420)
    project = make_project(
        tmp_path / "spheroids.ilp", lo=1000, hi=420, names=("background", "spheroid")
    )
    pipeline = spheroid_preset(project, read_project(project))
    modules = [s.module for s in pipeline.steps]
    assert modules == ["ilastik_pixel", "detect_spheroids", "measure_spheroids", "spheroid"]
    result = run_pipeline(
        open_plate(raw), pipeline, output_root=tmp_path / "runs", workers=1,
        checksums=False, save_images=False,
    )  # fmt: skip
    assert not result.failed_wells, result.failed_wells
    summary = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    for well, data in truth["wells"].items():
        expected = data["fields"]["1"]["spheroid"]["volume_um3"]
        assert summary.loc[well, "sp_volume_um3"] == pytest.approx(expected, rel=0.02), well
        assert summary.loc[well, "sp_qc_ok"] == 1, well
    assert len(calls(ilastik["log"])) == 3  # one ilastik call per well


def test_confluence_from_an_ilastik_class(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.kinetics.confluence import cover
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.segmentation.ilastik_backend.presets import confluence_preset
    from hcs_studio.segmentation.ilastik_backend.project import read_project

    plate, _ = test_ilastik_steps._plate(tmp_path)
    project = ilastik["project"]  # ramp 200 → 1200 on channel 1: p ≥ 0.5 where raw ≥ 700
    pipeline = confluence_preset(project, read_project(project), channels=[1])
    pipeline.steps[1].params["min_area_um2"] = 2.0  # small synthetic nuclei
    result = run_pipeline(
        plate, pipeline, output_root=tmp_path / "runs", workers=1, checksums=False,
        save_images=False,
    )  # fmt: skip
    assert not result.failed_wells, result.failed_wells
    fields = pd.read_csv(result.folder / "wells" / "A02" / "fields.csv").set_index("field")
    pixel_area = float(plate.metadata.pixel_size_um or 1.0) ** 2
    for item in plate.wells[(0, 1)].fields:
        data = np.asarray(item.load()[0, 0, 0], dtype=np.float64)
        expected = cover(data >= 700, 2.0 / pixel_area)
        got = fields.loc[item.index, "confluence_pct"]
        assert got == pytest.approx(100 * expected.mean(), abs=1e-6), item.index
    assert 0 < fields["confluence_pct"].max() < 30


def test_probability_methods_need_class_channels() -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ModuleError, create_module

    meta = ImageMetadata(channels=(ChannelInfo("Brightfield"),), pixel_size_um=2.0)
    with pytest.raises(ModuleError, match="ilastik pixel classification step first"):
        create_module("detect_spheroids", {"channel": 1, "method": "probability"}).validate(meta)
    with pytest.raises(ModuleError, match="ilastik pixel classification step first"):
        create_module("confluence", {"method": "probability"}).validate(meta)
    classes = ImageMetadata(
        channels=(
            ChannelInfo("BF"),
            ChannelInfo("ilastik_background"),
            ChannelInfo("ilastik_cells"),
        ),
        pixel_size_um=2.0,
    )
    step = create_module("confluence", {"method": "probability", "class_channel": "ilastik_x"})
    with pytest.raises(ModuleError, match="probabilities_to_objects"):
        step.validate(classes)
    create_module("confluence", {"method": "probability"}).validate(classes)
    assert not create_module("confluence", {"method": "probability"}).needs_fit


def test_ilastik_pipelines_validate_against_ground_truth(
    ilastik: dict[str, Any], tmp_path: Path
) -> None:
    """Phase 25: an ilastik pixel project + objects is validated like any segmentation; ilastik
    is found once for the whole set, then runs field by field."""
    from hcs_studio.pipeline.schema import PipelineStep
    from hcs_studio.synthetic.groundtruth import ground_truth_items
    from hcs_studio.training import groundtruth as gt
    from hcs_studio.training.validate import validate_segmentation

    items = ground_truth_items("easy", plates=1, wells=2, fields=2)
    values = np.concatenate([image.ravel() for _, image, _ in items])
    lo, hi = float(np.percentile(values, 50)), float(np.percentile(values, 99.5))
    project = make_project(tmp_path / "gt.ilp", lo=lo, hi=hi)
    gt.create_set("nuclei-gt", items)
    steps = [
        PipelineStep(module="ilastik_pixel", params={"project": str(project), "channels": [1]}),
        PipelineStep(
            module="probabilities_to_objects",
            params={"channel": "nuclei", "threshold": 0.3, "min_size_px": 10, "name": "nuclei"},
        ),
    ]
    result = validate_segmentation("nuclei-gt", steps, split=None)
    assert len(result.fields) == 4
    assert result.report["overall"]["ap50"] > 0.7
    assert len(calls(ilastik["log"])) == 4  # one call per field (no plate, no well batches)
    used = json.loads(json.dumps(result.report))  # report is JSON-ready
    assert used["overall"]["ap50"] == result.report["overall"]["ap50"]


def test_cli_preset_for_spheroids_and_confluence(tmp_path: Path) -> None:
    import yaml

    from hcs_studio.app import main
    from hcs_studio.pipeline.schema import Pipeline

    project = make_project(tmp_path / "p.ilp", names=("background", "spheroid"))
    out = tmp_path / "p.yaml"
    for use, expected in (
        ("spheroids", ["ilastik_pixel", "detect_spheroids", "measure_spheroids", "spheroid"]),
        ("confluence", ["ilastik_pixel", "confluence", "ilastik_class_areas"]),
    ):
        assert main(["ilastik", "--preset", str(project), "--for", use, "--output", str(out)]) == 0
        pipeline = Pipeline.model_validate(yaml.safe_load(out.read_text(encoding="utf-8")))
        assert [s.module for s in pipeline.steps] == expected
    params = pipeline.steps[1].params
    assert params["method"] == "probability" and params["class_channel"] == "ilastik_spheroid"
    organoids = ["ilastik", "--preset", str(project), "--for", "organoids", "--output", str(out)]
    assert main(organoids) == 0
    steps = Pipeline.model_validate(yaml.safe_load(out.read_text(encoding="utf-8"))).steps
    assert steps[-1].params.get("mode") == "organoids"
    objects = make_project(tmp_path / "o.ilp", workflow="ObjectClassificationWorkflowPrediction")
    assert (
        main(["ilastik", "--preset", str(objects), "--for", "spheroids", "--output", str(out)]) == 2
    )
