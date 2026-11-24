"""Phase 16b: the ilastik project reader, the headless runner and the pipeline steps, against a
fake ilastik that accepts the real command line (``tests/fake_ilastik_headless.py``). One
ilastik call per well; outputs, logs and the manifest as PROMPT_ILASTIK.md §8; resume after a
failed well; known-answer probabilities → objects."""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob
from hcs_studio.segmentation.ilastik_backend.project import check_compatible, read_project
from hcs_studio.segmentation.ilastik_backend.runners import HeadlessRunner
from hcs_studio.segmentation.ilastik_backend.steps import probabilities_to_labels

FAKE = Path(__file__).parent / "fake_ilastik_headless.py"


def make_project(
    path: Path,
    workflow: str = "PixelClassificationWorkflow",
    axes: str = "yxc",
    channels: int = 1,
    lo: float = 200,
    hi: float = 1200,
    names: tuple[str, ...] = ("background", "nuclei"),
) -> Path:
    """A minimal .ilp with the datasets HCS Studio reads (and the fake's ramp)."""
    import h5py

    from hcs_studio.segmentation.ilastik_backend.h5io import axistags_json

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["workflowName"] = workflow.encode()
        f["ilastikVersion"] = b"1.4.2"
        f.create_dataset("PixelClassification/LabelNames", data=[n.encode() for n in names])
        lane = f.create_group("Input Data/infos/lane0000/Raw Data")
        lane["axistags"] = axistags_json(axes).encode()
        lane["shape"] = np.array([64] * (len(axes) - 1) + [channels])
        fake = f.create_group("fake")
        fake.attrs["lo"], fake.attrs["hi"] = lo, hi
    return path


def fake_program(folder: Path) -> Path:
    program = folder / "run_ilastik.sh"
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    return program


@pytest.fixture
def ilastik(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    program = fake_program(tmp_path / "ilastik-1.4.2-Linux")
    monkeypatch.setenv("HCS_ILASTIK", str(program))
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_ILASTIK_LOG", str(log))
    yield {"program": program, "log": log, "project": make_project(tmp_path / "nuclei.ilp")}
    dispose_engines()


def calls(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


# --- project reader


def test_read_project(tmp_path: Path) -> None:
    info = read_project(make_project(tmp_path / "p.ilp", axes="zyxc", channels=2))
    assert info.workflow == "pixel" and info.ilastik_version == "1.4.2"
    assert info.label_names == ["background", "nuclei"] and info.axes == "zyxc"
    assert info.channels == 2 and info.dims == "3D" and len(info.sha256) == 64
    assert check_compatible(info, "zyxc", 2) == []
    problems = check_compatible(info, "yxc", 3)
    assert any("3D data but the input is 2D" in p for p in problems)
    assert any("2 channel(s); 3 are sent" in p for p in problems)
    auto = read_project(make_project(tmp_path / "a.ilp", workflow="NewAutocontextWorkflowBase"))
    assert auto.workflow == "autocontext"
    (tmp_path / "bad.ilp").write_text("not hdf5")
    with pytest.raises(ValueError):
        read_project(tmp_path / "bad.ilp")


# --- runner


def _inputs(folder: Path, n: int) -> list[str]:
    from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, write_h5

    specs = []
    for i in range(n):
        image = np.full((16, 16, 1), 200 + 400 * i, dtype=np.float32)
        specs.append(input_spec(write_h5(folder / f"f{i:03d}_t0.h5", image, "yxc")))
    return specs


def test_headless_runner_one_call_for_many_inputs(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.h5io import read_h5

    job = IlastikJob(
        project=str(ilastik["project"]),
        workflow="pixel",
        raw_data=_inputs(tmp_path / "in", 3),
        export_source="Probabilities",
        output_folder=str(tmp_path / "out"),
        output_suffix="Probabilities",
        threads=2,
        ram_mb=1000,
    )
    runner = HeadlessRunner(ilastik["program"])
    result = runner.run(job, log_path=tmp_path / "logs" / "A01.log")
    assert len(calls(ilastik["log"])) == 1 and len(result.outputs) == 3
    call = calls(ilastik["log"])[0]
    assert call["threads"] == "2" and call["ram"] == "1000"
    assert "--headless" in call["argv"] and "--export_source=Probabilities" in call["argv"]
    probs, axes = read_h5(Path(result.outputs[2]))
    assert axes == "yxc" and probs.shape == (16, 16, 2)
    assert probs[0, 0, 1] == pytest.approx(0.8)  # (1000 - lo 200) / (hi 1200 - lo 200)
    assert result.project_sha256_before == result.project_sha256_after
    assert "--raw_data" in (tmp_path / "logs" / "A01.log").read_text()


def test_headless_runner_errors(
    ilastik: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = IlastikJob(
        project=str(ilastik["project"]),
        workflow="pixel",
        raw_data=_inputs(tmp_path / "in", 1),
        export_source="Probabilities",
        output_folder=str(tmp_path / "out"),
        output_suffix="Probabilities",
    )
    runner = HeadlessRunner(ilastik["program"])
    monkeypatch.setenv("FAKE_ILASTIK_FAIL_ON", "f000")
    with pytest.raises(IlastikError, match="simulated ilastik failure"):
        runner.run(job)
    monkeypatch.delenv("FAKE_ILASTIK_FAIL_ON")
    monkeypatch.setenv("FAKE_ILASTIK_TOUCH_PROJECT", "1")
    with pytest.raises(IlastikError, match="changed the project file"):
        runner.run(job)
    monkeypatch.delenv("FAKE_ILASTIK_TOUCH_PROJECT")
    bad = job.model_copy(update={"export_source": "Object Predictions"})
    with pytest.raises(IlastikError):
        runner.run(bad)
    with pytest.raises(IlastikError, match="could not be started"):
        HeadlessRunner(tmp_path / "missing" / "ilastik").run(job)


# --- probabilities → objects


def test_probabilities_to_objects_known_answer() -> None:
    yy, xx = np.mgrid[:60, :80]
    prob = np.zeros((1, 60, 80), dtype=np.float32)
    for cy, cx in ((30, 25), (30, 39)):  # two touching discs
        prob[0][(yy - cy) ** 2 + (xx - cx) ** 2 <= 64] = 0.9
    prob[0][(yy - 10) ** 2 + (xx - 70) ** 2 <= 4] = 0.9  # a speck (13 px)
    joined = probabilities_to_labels(prob, split="none", smoothing=0, min_size=20)
    assert joined.max() == 1  # touching discs merge; the speck is removed
    split = probabilities_to_labels(
        prob, split="distance", smoothing=0, min_size=20, min_distance=5
    )
    assert split.max() == 2
    seeds = np.zeros((1, 60, 80), dtype=np.int32)
    seeds[0, 30, 25], seeds[0, 30, 39] = 7, 9
    seeded = probabilities_to_labels(prob, split="seeds", seeds=seeds, smoothing=0, min_size=20)
    assert seeded.max() == 2 and seeded[0, 30, 25] != seeded[0, 30, 39]
    # hysteresis keeps a weak rim attached to a strong core
    weak = np.zeros((1, 40, 40), dtype=np.float32)
    weak[0, 10:30, 10:30] = 0.4
    weak[0, 18:22, 18:22] = 0.9
    assert (probabilities_to_labels(weak, 0.5, 0.3, 0, 1, split="none") > 0).sum() == 400
    assert (probabilities_to_labels(weak, 0.5, None, 0, 1, split="none") > 0).sum() == 16


# --- the pipeline: one call per well, outputs, manifest, resume


def _plate(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    from hcs_studio.io.base import open_plate
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1), (0, 2)],
        channels=("DAPI", "GFP"),
        size=96,
        n_fields=2,
        name="ILA01",
        features=[Nuclei(count=lambda r, c, f: 5 + c, intensity=2000), ShotNoise(sd=3)],
    )
    path, truth = plate.write(tmp_path / "raw")
    return open_plate(path), truth


def _pipeline(project: Path, **pixel: Any) -> Any:
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    return Pipeline(
        steps=[
            PipelineStep(
                module="ilastik_pixel", params={"project": str(project), "channels": [1], **pixel}
            ),
            PipelineStep(
                module="probabilities_to_objects",
                params={
                    "channel": "nuclei",
                    "threshold": 0.5,
                    "smoothing_px": 0,
                    "min_size_px": 10,
                    "split": "distance",
                    "min_distance_px": 4,
                    "name": "nuclei",
                },
            ),
            PipelineStep(
                module="measure_objects",
                params={"labels": ["nuclei"], "channels": [1, 4], "texture": False},
            ),
            PipelineStep(module="ilastik_class_areas"),
        ]
    )


def test_pipeline_runs_ilastik_once_per_well(ilastik: dict[str, Any], tmp_path: Path) -> None:
    import pandas as pd

    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.readouts import resolve

    plate, truth = _plate(tmp_path)
    result = run_pipeline(
        plate,
        _pipeline(ilastik["project"]),
        output_root=tmp_path / "runs",
        workers=1,
        checksums=False,
    )
    folder = result.folder
    assert not result.failed_wells
    made = calls(ilastik["log"])
    assert len(made) == 3  # one ilastik call per well, with both fields
    assert all(sum(a.endswith(".h5/data") for a in c["argv"]) == 2 for c in made)
    # run-folder structure (PROMPT_ILASTIK.md §8)
    used = json.loads((folder / "ilastik" / "projects_used.json").read_text())
    assert used[0]["sha256"] == read_project(ilastik["project"]).sha256
    assert used[0]["runner"] == "headless" and used[0]["ilastik_version"] == "1.4.2"
    assert "--raw_data" in (folder / "ilastik" / "logs" / "A01.log").read_text()
    assert not any((folder / "ilastik" / "inputs").rglob("*.h5"))  # cache deleted
    well = folder / "wells" / "A02" / "ilastik"
    assert len(list(well.glob("*_Probabilities.h5"))) == 2
    summary = json.loads((well / "class_summary.json").read_text())
    assert summary["classes"] == ["background", "nuclei"]
    # the manifest traces results to the project and ilastik
    manifest = json.loads((folder / "run_manifest.json").read_text())
    fitted = next(s for s in manifest["fitted_state"] if s["module"] == "ilastik_pixel")
    assert fitted["sha256"] == used[0]["sha256"] and fitted["ilastik_version"] == "1.4.2"
    # objects from ilastik's probabilities match the synthetic truth
    wells = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    for name, expected in truth["wells"].items():
        assert wells.loc[name, "nuclei_objects_measured"] == expected["counts"]["nuclei"], name
    assert 0 < wells["ilastik_area_pct_nuclei"].to_dict()["A01"] < 20
    assert resolve("ilastik_area_pct_nuclei") is not None
    objects = pd.read_parquet(folder / "wells" / "A01" / "objects.parquet")
    probability = [c for c in objects.columns if "ilastik_nuclei" in c and c.endswith("_mean")]
    assert probability and objects[probability[0]].min() > 0.5  # mean class probability


def test_resume_after_a_failed_well(
    ilastik: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hcs_studio.pipeline.executor import run_pipeline

    plate, _ = _plate(tmp_path)
    monkeypatch.setenv("FAKE_ILASTIK_FAIL_ON", "/A02/")
    first = run_pipeline(
        plate,
        _pipeline(ilastik["project"]),
        output_root=tmp_path / "runs",
        workers=1,
        checksums=False,
    )
    assert set(first.failed_wells) == {"A02"}
    assert "simulated ilastik failure" in first.failed_wells["A02"]
    monkeypatch.delenv("FAKE_ILASTIK_FAIL_ON")
    before = len(calls(ilastik["log"]))
    again = run_pipeline(
        plate, _pipeline(ilastik["project"]), workers=1, checksums=False, resume=first.folder
    )
    assert not again.failed_wells and set(again.skipped_wells) == {"A01", "A03"}
    assert len(calls(ilastik["log"])) == before + 1  # only A02 again


def test_step_checks(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.pipeline.executor import PipelineRunner
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep
    from hcs_studio.processing.base import ImageData, ModuleError, create_module

    plate, _ = _plate(tmp_path)
    wrong = make_project(tmp_path / "obj.ilp", workflow="ObjectClassificationWorkflowPixel")
    with pytest.raises(ModuleError, match="needs ilastik Pixel Classification"):
        PipelineRunner(
            plate,
            Pipeline(steps=[PipelineStep(module="ilastik_pixel", params={"project": str(wrong)})]),
        )
    early = Pipeline(
        steps=[
            PipelineStep(module="measure_objects"),
            PipelineStep(module="ilastik_pixel", params={"project": str(ilastik["project"])}),
        ]
    )
    with pytest.raises(ModuleError, match="must come after it"):
        PipelineRunner(plate, early)
    step = create_module("ilastik_pixel", {"project": str(ilastik["project"])})
    meta = ImageMetadata(channels=(ChannelInfo("DAPI"), ChannelInfo("GFP")), pixel_size_um=0.65)
    stack = np.zeros((2, 3, 32, 32), dtype=np.float32)
    with pytest.raises(ModuleError, match="trained on 2D data but the input is 3D"):
        step.apply(ImageData(stack, meta), None)
    flat = ImageData(np.full((2, 1, 32, 32), 1200, dtype=np.float32), meta)
    with pytest.raises(ModuleError, match="1 channel"):
        step.apply(flat, None)  # both channels sent to a 1-channel project
    one = create_module("ilastik_pixel", {"project": str(ilastik["project"]), "channels": [1]})
    preview = one.apply(flat, None)  # a single field (preview): temporary folder, removed
    assert [c.name for c in preview.metadata.channels][-2:] == [
        "ilastik_background",
        "ilastik_nuclei",
    ]
    assert np.allclose(preview.data[3], 1.0) and preview.linear
    assert one.output_metadata(meta).channels[-1].name == "ilastik_nuclei"
    seg = create_module(
        "ilastik_pixel",
        {"project": str(ilastik["project"]), "channels": [1], "export": "segmentation"},
    )
    assert np.array_equal(seg.apply(flat, None).data[3], np.ones((1, 32, 32)))
    autocontext = make_project(tmp_path / "auto.ilp", workflow="NewAutocontextWorkflowBase")
    auto = create_module("ilastik_autocontext", {"project": str(autocontext), "channels": [1]})
    assert np.allclose(auto.apply(flat, None).data[3], 1.0)
    assert "Probabilities Stage 2" in json.dumps(calls(ilastik["log"])[-1])
    os.environ.pop("FAKE_ILASTIK_FAIL_ON", None)


def test_cli_info_and_preset(tmp_path: Path, capsys: Any) -> None:
    import yaml

    from hcs_studio.app import main
    from hcs_studio.pipeline.schema import Pipeline

    project = make_project(tmp_path / "cells.ilp", names=("background", "cell body"))
    assert main(["ilastik", "--info", str(project)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["workflow"] == "pixel" and info["label_names"] == ["background", "cell body"]
    out = tmp_path / "p.yaml"
    assert main(["ilastik", "--preset", str(project), "--channels", "1", "--output", str(out)]) == 0
    pipeline = Pipeline.model_validate(yaml.safe_load(out.read_text()))
    modules = [s.module for s in pipeline.steps]
    assert modules == [
        "ilastik_pixel",
        "probabilities_to_objects",
        "measure_objects",
        "ilastik_class_areas",
        "qc",
    ]
    assert pipeline.steps[1].params["channel"] == "ilastik_cell_body"
    assert (
        main(["ilastik", "--preset", str(project), "--class", "nuclei", "--output", str(out)]) == 2
    )
