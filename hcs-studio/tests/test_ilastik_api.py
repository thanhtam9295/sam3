"""Phase 16h: the Python-API runner (ilastik's ``ilastik.experimental.api`` in its own
environment) against a generated stand-in environment: a ``python3`` wrapper whose path holds a
fake ``ilastik`` package (the API with the same known-answer ramp as the fake headless program,
and ``python -m ilastik`` = the fake headless program) and a minimal ``xarray``."""

from __future__ import annotations

import json
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_ilastik_steps import FAKE, _pipeline, _plate, calls, make_project

API = """
import json, os
import h5py
import numpy as np
import xarray


def _log(entry):
    path = os.environ.get("FAKE_ILASTIK_API_LOG")
    if path:
        with open(path, "a") as handle:
            handle.write(json.dumps(entry) + "\\n")


class PixelClassificationPipeline:
    @classmethod
    def from_ilp_file(cls, path):
        _log({"load": str(path), "threads": os.environ.get("LAZYFLOW_THREADS")})
        return cls(path)

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.lo = float(f["fake"].attrs.get("lo", 0))
            self.hi = float(f["fake"].attrs.get("hi", 1))

    def get_probabilities(self, raw):
        dims = list(raw.dims)
        data = np.asarray(raw.values)
        first = np.take(data, 0, axis=dims.index("c")).astype(np.float64)
        fg = np.clip((first - self.lo) / (self.hi - self.lo), 0, 1)
        _log({"predict": list(data.shape)})
        out = np.stack([1 - fg, fg], axis=-1).astype(np.float32)
        return xarray.DataArray(out, dims=tuple(d for d in dims if d != "c") + ("c",))


class AutocontextPipeline(PixelClassificationPipeline):
    def get_probabilities_stage_2(self, raw):
        return self.get_probabilities(raw)
"""
XARRAY = """
import numpy as np


class DataArray:
    def __init__(self, data, dims):
        self.values = np.asarray(data)
        self.dims = tuple(dims)
"""


def fake_environment(root: Path, api: bool = True) -> Path:
    """``<root>/bin/python3``: runs this interpreter with the fake ilastik package on its path."""
    if sys.platform.startswith("win"):
        pytest.skip("the fake is a shebang script")
    package = root / "site"
    (package / "ilastik").mkdir(parents=True)
    (package / "ilastik" / "__init__.py").write_text("")
    (package / "ilastik" / "__main__.py").write_text(
        f"import runpy\nrunpy.run_path({str(FAKE)!r}, run_name='__main__')\n"
    )
    if api:
        (package / "ilastik" / "experimental" / "api").mkdir(parents=True)
        (package / "ilastik" / "experimental" / "__init__.py").write_text("")
        (package / "ilastik" / "experimental" / "api" / "__init__.py").write_text(API)
    (package / "xarray.py").write_text(XARRAY)
    python = root / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nPYTHONPATH="{package}" exec "{sys.executable}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    return python


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.segmentation.ilastik_backend.runners import python_api

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    monkeypatch.delenv("HCS_FIJI", raising=False)
    monkeypatch.setattr(python_api, "_AVAILABLE", {})
    dispose_engines()
    python = fake_environment(tmp_path / "ilastik-env")
    monkeypatch.setenv("HCS_ILASTIK", str(python))
    monkeypatch.setenv("FAKE_ILASTIK_LOG", str(tmp_path / "headless.jsonl"))
    monkeypatch.setenv("FAKE_ILASTIK_API_LOG", str(tmp_path / "api.jsonl"))
    yield {
        "python": python,
        "headless_log": tmp_path / "headless.jsonl",
        "api_log": tmp_path / "api.jsonl",
        "project": make_project(tmp_path / "nuclei.ilp"),
    }
    dispose_engines()


def use_runner(runner: str) -> None:
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings

    IlastikSettings(runner=runner).save()  # type: ignore[arg-type]


def _job(project: Path, folder: Path, specs: list[str], **extra: Any) -> Any:
    from hcs_studio.segmentation.ilastik_backend.job import IlastikJob

    return IlastikJob(
        project=str(project),
        workflow=extra.pop("workflow", "pixel"),
        raw_data=specs,
        export_source=extra.pop("export_source", "Probabilities"),
        output_folder=str(folder),
        output_suffix="Probabilities",
        **extra,
    )


def test_api_and_headless_agree(env: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.runners import HeadlessRunner, PythonApiRunner
    from tests.test_ilastik_fiji import field_inputs, tczyx

    specs = field_inputs(tmp_path / "in", [(1, 1, 1, 40, 48), (1, 2, 3, 24, 24)])
    api = PythonApiRunner(env["python"]).run(_job(env["project"], tmp_path / "api", specs))
    headless = HeadlessRunner(env["python"]).run(_job(env["project"], tmp_path / "h", specs))
    assert api.runner == "python_api" and len(api.outputs) == 2
    for a, h in zip(api.outputs, headless.outputs, strict=True):
        np.testing.assert_allclose(tczyx(a), tczyx(h), atol=1e-6)
    loads = [c for c in calls(env["api_log"]) if "load" in c]
    assert len(loads) == 1  # the project is loaded once for all inputs
    auto = make_project(tmp_path / "auto.ilp", workflow="NewAutocontextWorkflowBase")
    job = _job(auto, tmp_path / "auto", specs, workflow="autocontext",
               export_source="Probabilities Stage 2", threads=3)  # fmt: skip
    assert PythonApiRunner(env["python"]).run(job).runner == "python_api"
    assert calls(env["api_log"])[-3]["threads"] == "3"


def test_what_the_api_runner_refuses(env: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.runners import PythonApiRunner, make_runner

    api = PythonApiRunner(env["python"])
    seg = _job(env["project"], tmp_path, ["x.h5/data"], export_source="Simple Segmentation")
    assert "probabilities only" in (api.unsupported(seg) or "")
    masked = _job(env["project"], tmp_path, ["x.h5/data"], roles={"prediction_mask": ["m.h5/data"]})
    assert "second input" in (api.unsupported(masked) or "")
    objects = _job(env["project"], tmp_path, ["x.h5/data"], workflow="objects",
                   export_source="Object Predictions")  # fmt: skip
    state = {"runner": "python_api", "ilastik": str(env["python"])}
    runner, note = make_runner(state, objects)
    assert runner.name == "headless" and note.startswith("Python API not used")
    program = PythonApiRunner("/opt/run_ilastik.sh")
    reason = program.unsupported(_job(env["project"], tmp_path, ["x.h5/data"]))
    assert "Python of an ilastik environment" in (reason or "")


def test_pipeline_with_the_api_runner_matches_headless(env: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.pipeline.executor import run_pipeline

    plate, _ = _plate(tmp_path)
    tables = {}
    for runner in ("headless", "python_api"):
        use_runner(runner)
        result = run_pipeline(plate, _pipeline(env["project"]), output_root=tmp_path / runner,
                              workers=1, checksums=False)  # fmt: skip
        assert not result.failed_wells, result.failed_wells
        used = json.loads((result.folder / "ilastik" / "projects_used.json").read_text())
        assert used[0]["runner"] == runner
        tables[runner] = pd.read_csv(result.folder / "tables" / "per_well_summary.csv")
    columns = [c for c in tables["headless"].columns if c.startswith(("nuclei_", "ilastik_"))]
    pd.testing.assert_frame_equal(tables["headless"][columns], tables["python_api"][columns])
    assert len([c for c in calls(env["api_log"]) if "load" in c]) == 3  # once per well


def test_falls_back_to_headless_without_the_api(
    env: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.segmentation.ilastik_backend.job import IlastikError
    from hcs_studio.segmentation.ilastik_backend.runners import PythonApiRunner, runner_state

    old = fake_environment(tmp_path / "old-ilastik", api=False)
    monkeypatch.setenv("HCS_ILASTIK", str(old))
    use_runner("python_api")
    state = runner_state()
    assert state["runner"] == "headless" and "ilastik.experimental.api" in state["runner_note"]
    plate, _ = _plate(tmp_path)
    result = run_pipeline(plate, _pipeline(env["project"]), output_root=tmp_path / "runs",
                          workers=1, checksums=False, wells=[(0, 0)])  # fmt: skip
    assert not result.failed_wells
    used = json.loads((result.folder / "ilastik" / "projects_used.json").read_text())
    assert used[0]["runner"] == "headless"
    assert (
        "ilastik.experimental.api" in (result.folder / "ilastik" / "logs" / "A01.log").read_text()
    )
    from tests.test_ilastik_fiji import field_inputs

    specs = field_inputs(tmp_path / "in", [(1, 1, 1, 16, 16)])
    with pytest.raises(IlastikError, match="no usable Python API"):
        PythonApiRunner(old).run(_job(env["project"], tmp_path / "o", specs))
    script = tmp_path / "run_ilastik.sh"
    script.write_text(f'#!/bin/sh\nexec "{old}" -m ilastik "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HCS_ILASTIK", str(script))
    assert "Python of an ilastik environment" in runner_state()["runner_note"]


def test_installation_check_through_every_runner(
    env: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from hcs_studio.app import main
    from hcs_studio.segmentation.ilastik_backend.selftest import check_installation

    result = check_installation(env["project"])
    assert result.ok and result.version == "1.4.2", result.text()
    assert [c.runner for c in result.checks] == ["headless", "python_api"]
    assert result.checks[1].max_difference == 0.0
    assert "PASS python_api" in result.text()
    assert main(["ilastik", "--test", str(env["project"])]) == 0
    assert "PASS headless" in capsys.readouterr().out
    # without a project: ilastik starts, nothing to run
    bare = check_installation(None)
    assert not bare.checks and "Choose a trained" in bare.text()
    # a runner that fails is reported, with ilastik's message
    monkeypatch.setenv("FAKE_ILASTIK_FAIL_ON", "test")
    failed = check_installation(env["project"], runners=["headless"])
    assert not failed.ok and failed.checks[0].message
    objects = make_project(tmp_path / "o.ilp", workflow="ObjectClassificationWorkflowPrediction")
    assert "Pixel Classification or Autocontext" in check_installation(objects).text()
