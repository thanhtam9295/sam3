"""Phase 16e: the Fiji runner (ilastik4ij plugin), Fiji macro export and the headless-vs-Fiji
parity test.

CI runs against a fake Fiji (``tests/fake_fiji.py``) that interprets the generated macro and
starts the fake ilastik with the plugin's own flags. The real parity test runs only with
``HCS_ILASTIK`` (the ilastik program), ``HCS_FIJI`` (a Fiji with the ilastik update site) and
``HCS_ILASTIK_PIXEL_PROJECT`` (a trained Pixel Classification ``.ilp``).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.segmentation.ilastik_backend.axes import from_ilastik, ilastik_axes, to_ilastik
from hcs_studio.segmentation.ilastik_backend.h5io import input_spec, read_h5, write_h5
from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob
from hcs_studio.segmentation.ilastik_backend.runners import FijiRunner, HeadlessRunner
from hcs_studio.segmentation.ilastik_backend.runners import fiji as fiji_module
from tests import test_ilastik_steps
from tests.test_ilastik_steps import _pipeline, _plate, calls, make_project

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)
FAKE_FIJI = Path(__file__).parent / "fake_fiji.py"


def fake_fiji_app(root: Path, plugin: bool = True) -> Path:
    """A Fiji folder whose launcher runs the fake; with the ilastik4ij jar when ``plugin``."""
    if sys.platform.startswith("win"):
        pytest.skip("the fake is a shebang script")
    launcher = root / "Fiji.app" / "ImageJ-linux64"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_FIJI}" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    if plugin:
        (root / "Fiji.app" / "jars").mkdir(exist_ok=True)
        (root / "Fiji.app" / "jars" / "ilastik4ij-2.1.0.jar").write_bytes(b"")
    return launcher


@pytest.fixture
def fiji(
    ilastik: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    launcher = fake_fiji_app(tmp_path / "apps")
    monkeypatch.setenv("HCS_FIJI", str(launcher.parent))
    log = tmp_path / "fiji_calls.jsonl"
    monkeypatch.setenv("FAKE_FIJI_LOG", str(log))
    yield {**ilastik, "launcher": launcher, "fiji_log": log}


def use_runner(runner: str, **extra: Any) -> None:
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings

    IlastikSettings(runner=runner, **extra).save()  # type: ignore[arg-type]


def field_inputs(
    folder: Path, shapes: list[tuple[int, int, int, int, int]], seed: int = 0
) -> list[str]:
    """ilastik-ready inputs from (T, C, Z, Y, X) shapes; values span the fake's 200–1200 ramp."""
    rng = np.random.default_rng(seed)
    specs = []
    for i, shape in enumerate(shapes):
        data = rng.uniform(0, 1400, size=shape).astype(np.float32)
        axes = ilastik_axes(shape)
        specs.append(input_spec(write_h5(folder / f"f{i:03d}.h5", to_ilastik(data, axes), axes)))
    return specs


def tczyx(path: str) -> np.ndarray:
    array, axes = read_h5(Path(path))
    return np.asarray(from_ilastik(array, axes))


# --------------------------------------------------------------------------- detection


def test_detection_finds_fiji_and_the_plugin(fiji: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings, detect

    found = detect(IlastikSettings(), query_version=False)
    assert found.fiji_ok and found.ilastik4ij_version == "2.1.0"
    assert found.runners() == ["headless", "fiji"]
    bare = fake_fiji_app(tmp_path / "bare", plugin=False)
    found = detect(IlastikSettings(fiji=str(bare.parent)), query_version=False)
    assert not found.fiji_ok and "ilastik4ij: not installed" in found.text()


def test_runner_state_checks_the_fiji_setup(
    fiji: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hcs_studio.segmentation.ilastik_backend.runners import runner_state

    assert runner_state()["runner"] == "headless"
    use_runner("fiji", ram_mb=2048)
    state = runner_state()
    assert state["runner"] == "fiji" and state["fiji"] == str(fiji["launcher"])
    assert state["ilastik4ij"] == "ilastik4ij-2.1.0.jar" and state["ram_mb"] == 2048
    bare = fake_fiji_app(tmp_path / "bare", plugin=False)
    monkeypatch.setenv("HCS_FIJI", str(bare.parent))
    with pytest.raises(IlastikError, match="no ilastik4ij plugin"):
        runner_state()
    # the plugin starts ilastik itself: a Python interpreter is not enough
    assert FijiRunner.check_ilastik(sys.executable) is not None
    assert FijiRunner.check_ilastik(fiji["program"]) is None


# --------------------------------------------------------------------------- macro text


def test_job_macro_uses_the_plugin_keys(ilastik: dict[str, Any], tmp_path: Path) -> None:
    specs = field_inputs(tmp_path / "in", [(1, 1, 1, 32, 32)])
    job = IlastikJob(
        project=str(ilastik["project"]),
        workflow="pixel",
        raw_data=specs,
        export_source="Probabilities",
        output_folder=str(tmp_path / "out"),
        output_suffix="Probabilities",
    )
    text = fiji_module.job_macro(job, "/opt/ilastik-1.4.2-Linux/run_ilastik.sh")
    assert (
        'run("Configure ilastik executable location", "executablefile=[/opt/ilastik-1.4.2-Linux/'
        'run_ilastik.sh] numthreads=-1 maxrammb=4096");'
    ) in text  # no RAM limit set → the plugin's own default
    assert "datasetname=[/data] axisorder=[yxc]" in text
    assert "inputimage=[hcs_raw_0]" in text and "pixelclassificationtype=[Probabilities]" in text
    assert "datasetname=[exported_data]" in text and "f000_Probabilities.h5]" in text
    assert "Configure" not in fiji_module.job_macro(job, "x", configure=False)
    objects = job.model_copy(
        update={
            "workflow": "objects",
            "export_source": "Object Predictions",
            "roles": {"segmentation_image": specs},
        }
    )
    text = fiji_module.job_macro(objects, "x")
    assert "inputproborsegimage=[hcs_second_0] secondinputtype=[Segmentation]" in text
    assert "objectexportsource=[Object Predictions]" in text
    # what the plugin cannot do
    labels = objects.model_copy(update={"roles": {"label_image": specs}})
    assert "label_image" in (fiji_module.unsupported(labels) or "")
    tracking = job.model_copy(
        update={
            "workflow": "tracking",
            "export_source": "Plugin",
            "export_plugin": "CSV-Table",
            "roles": {"segmentation_image": specs},
        }
    )
    assert fiji_module.unsupported(tracking)
    with pytest.raises(IlastikError, match="cannot use paths"):
        fiji_module.macro_path("/data/plate[1]/x.h5")


# --------------------------------------------------------------------------- parity (fake)


@pytest.mark.parametrize(
    ("workflow", "source", "role"),
    [
        ("pixel", "Probabilities", None),
        ("pixel", "Simple Segmentation", None),
        ("objects", "Object Predictions", "prediction_maps"),
        ("objects", "Object Identities", "segmentation_image"),
        ("multicut", "Multicut Segmentation", "probabilities"),
    ],
)
def test_headless_and_fiji_agree(
    fiji: dict[str, Any], tmp_path: Path, workflow: str, source: str, role: str | None
) -> None:
    """The same job through both runners gives the same arrays (2D and 3D fields)."""
    workflows = {
        "pixel": "PixelClassificationWorkflow",
        "objects": "ObjectClassificationWorkflowPrediction",
        "multicut": "EdgeTrainingWithMulticutWorkflow",
    }
    project = make_project(tmp_path / f"{workflow}.ilp", workflow=workflows[workflow])
    shapes = [(1, 1, 1, 40, 48), (1, 1, 3, 24, 24)]
    raw = field_inputs(tmp_path / "in", shapes)
    roles = {}
    if role:
        roles = {role: field_inputs(tmp_path / "second", [(1, 2, *s[2:]) for s in shapes], 1)}
    results = {}
    for runner in (HeadlessRunner(fiji["program"]), FijiRunner(fiji["launcher"], fiji["program"])):
        job = IlastikJob(
            project=str(project),
            workflow=workflow,
            raw_data=raw,
            roles=roles,
            export_source=source,
            output_folder=str(tmp_path / runner.name),
            output_suffix=source.replace(" ", "_"),
            ram_mb=1024,
        )
        result = runner.run(job, log_path=tmp_path / f"{runner.name}.log")
        assert result.runner == runner.name and len(result.outputs) == 2
        results[runner.name] = [tczyx(p) for p in result.outputs]
    for headless, viafiji in zip(results["headless"], results["fiji"], strict=True):
        assert headless.shape == viafiji.shape
        np.testing.assert_allclose(headless, viafiji, rtol=0, atol=1e-6)
    plugin_calls = calls(fiji["fiji_log"])
    assert len(plugin_calls) == 2  # the plugin starts ilastik once per image
    argv = plugin_calls[0]["argv"]
    assert "--readonly=1" in argv and "--input_axes=tzcyx" in argv
    assert any(a.startswith("--raw_data=") for a in argv)
    macro = (tmp_path / "fiji" / f"fiji_{source.replace(' ', '_')}.ijm").read_text()
    assert "maxrammb=1024" in macro
    assert "-macro" in (tmp_path / "fiji.log").read_text()


def test_fiji_failures_are_reported(
    fiji: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = field_inputs(tmp_path / "in", [(1, 1, 1, 16, 16)])
    job = IlastikJob(
        project=str(fiji["project"]),
        workflow="pixel",
        raw_data=raw,
        export_source="Probabilities",
        output_folder=str(tmp_path / "out"),
        output_suffix="Probabilities",
    )
    runner = FijiRunner(fiji["launcher"], fiji["program"])
    # ilastik fails inside the plugin; Fiji itself still exits 0 (as the real one does)
    monkeypatch.setenv("FAKE_ILASTIK_FAIL_ON", "raw_data")
    with pytest.raises(IlastikError, match="macro did not finish") as info:
        runner.run(job)
    assert "Subprocess failed" in str(info.value)
    monkeypatch.delenv("FAKE_ILASTIK_FAIL_ON")
    with pytest.raises(IlastikError, match="could not be started"):
        FijiRunner(tmp_path / "no-fiji", fiji["program"]).run(job)
    with pytest.raises(IlastikError, match="cannot run this step"):
        FijiRunner(fiji["launcher"], sys.executable).run(job)
    assert "update site" in fiji_module._explain(0, False, 'Unrecognized command: "Run Multicut"')
    assert runner.run(job).outputs  # works again


# --------------------------------------------------------------------------- pipelines


def test_pipeline_with_the_fiji_runner_matches_headless(
    fiji: dict[str, Any], tmp_path: Path
) -> None:
    from hcs_studio.pipeline.executor import run_pipeline

    plate, _ = _plate(tmp_path)
    tables = {}
    for runner in ("headless", "fiji"):
        use_runner(runner)
        result = run_pipeline(
            plate,
            _pipeline(fiji["project"]),
            output_root=tmp_path / f"runs_{runner}",
            workers=1,
            checksums=False,
        )
        assert not result.failed_wells, result.failed_wells
        used = json.loads((result.folder / "ilastik" / "projects_used.json").read_text())
        assert used[0]["runner"] == runner
        tables[runner] = pd.read_csv(result.folder / "tables" / "per_well_summary.csv")
        manifest = json.loads((result.folder / "run_manifest.json").read_text())
        fitted = next(s for s in manifest["fitted_state"] if s["module"] == "ilastik_pixel")
        assert fitted["runner"] == runner
    columns = [c for c in tables["headless"].columns if c.startswith(("nuclei_", "ilastik_"))]
    pd.testing.assert_frame_equal(tables["headless"][columns], tables["fiji"][columns])
    assert len(calls(fiji["fiji_log"])) == 6  # 3 wells × 2 fields, one ilastik start each
    log = (tmp_path / "runs_fiji").rglob("A01.log")
    assert "Run Pixel Classification Prediction" in next(log).read_text()


def test_fiji_runner_falls_back_for_label_images(fiji: dict[str, Any], tmp_path: Path) -> None:
    """Object Classification on Cellpose-style labels needs --label_image, which the plugin does
    not offer: the step runs headless and says so."""
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    use_runner("fiji")
    plate, _ = _plate(tmp_path)
    project = make_project(tmp_path / "objects.ilp", workflow="ObjectClassificationWorkflowLabels")
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                module="segment_classical",
                params={"channel": 1, "split_touching": False, "min_size_px": 10},
            ),
            PipelineStep(
                module="ilastik_objects",
                params={"project": str(project), "labels": "nuclei", "channels": [1]},
            ),
        ]
    )
    result = run_pipeline(
        plate, pipeline, output_root=tmp_path / "runs", workers=1, checksums=False, wells=[(0, 0)]
    )
    assert not result.failed_wells, result.failed_wells
    used = json.loads((result.folder / "ilastik" / "projects_used.json").read_text())
    assert used[0]["runner"] == "headless"
    assert "Fiji runner not used" in (result.folder / "ilastik" / "logs" / "A01.log").read_text()
    assert not calls(fiji["fiji_log"])


# --------------------------------------------------------------------------- files for people


def test_cli_macro_and_hdf5_export(fiji: dict[str, Any], tmp_path: Path, capsys: Any) -> None:
    from hcs_studio.app import main
    from hcs_studio.segmentation.ilastik_backend.h5io import is_ilastik_ready

    plate, _ = _plate(tmp_path)
    out = tmp_path / "h5"
    assert main(["ilastik", "--export-h5", str(plate.path), "--wells", "A01-A02",
                 "--channels", "1", "--output", str(out)]) == 0  # fmt: skip
    files = sorted(out.glob("*.h5"))
    assert [f.name for f in files][:2] == ["A01_f001_t0.h5", "A01_f002_t0.h5"]
    assert len(files) == 4 and all(is_ilastik_ready(f) for f in files)
    assert main(["ilastik", "--export-h5", str(plate.path), "--output",
                 str(plate.path / "h5")]) == 2  # fmt: skip
    assert "outside the raw data" in capsys.readouterr().out

    macro = tmp_path / "pixel.ijm"
    assert main(["ilastik", "--macro", str(fiji["project"]), "--output", str(macro)]) == 0
    text = macro.read_text()
    assert '#@ File (label = "Folder of .h5 inputs", style = "directory") inputFolder' in text
    assert 'axisOrder = "yxc";' in text and 'suffix = "_Probabilities";' in text
    assert "pixelclassificationtype=[Probabilities]" in text
    assert all(line.count('"') % 2 == 0 for line in text.splitlines())  # balanced strings
    objects = make_project(tmp_path / "o.ilp", workflow="ObjectClassificationWorkflowPrediction")
    options = ["--second", "segmentation", "--export", "identities", "--axes", "zyxc"]
    assert main(["ilastik", "--macro", str(objects), *options, "--output", str(macro)]) == 0
    text = macro.read_text()
    assert 'secondSuffix = "_segmentation";' in text and "secondinputtype=[Segmentation]" in text
    assert "objectexportsource=[Object Identities]" in text and 'axisOrder = "zyxc";' in text
    assert main(["ilastik", "--macro", str(objects), "--export", "nonsense",
                 "--output", str(macro)]) == 2  # fmt: skip


# --------------------------------------------------------------------------- real parity


@pytest.mark.skipif(
    not (
        os.environ.get("HCS_ILASTIK")
        and os.environ.get("HCS_FIJI")
        and os.environ.get("HCS_ILASTIK_PIXEL_PROJECT")
    ),
    reason="set HCS_ILASTIK, HCS_FIJI and HCS_ILASTIK_PIXEL_PROJECT for the real parity test",
)
def test_real_headless_vs_fiji_parity(tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.project import read_project
    from hcs_studio.segmentation.ilastik_backend.settings import (
        IlastikSettings,
        detect,
        fiji_launcher,
        fiji_root,
    )

    project = Path(os.environ["HCS_ILASTIK_PIXEL_PROJECT"])
    info = read_project(project)
    found = detect(IlastikSettings(), query_version=False)
    launcher = fiji_launcher(fiji_root(os.environ["HCS_FIJI"]))
    assert launcher is not None and found.ilastik4ij, "Fiji with the ilastik4ij plugin needed"
    channels = info.channels or 1
    shape = (
        (1, channels, 1, 256, 256)
        if (info.axes or "yxc").count("z") == 0
        else (1, channels, 8, 128, 128)
    )
    image = os.environ.get("HCS_ILASTIK_PIXEL_IMAGE")
    spec = image or field_inputs(tmp_path / "in", [shape])[0]
    outputs = {}
    for runner in (
        HeadlessRunner(os.environ["HCS_ILASTIK"]),
        FijiRunner(launcher, os.environ["HCS_ILASTIK"]),
    ):
        job = IlastikJob(
            project=str(project),
            workflow=info.workflow,
            raw_data=[spec],
            export_source="Probabilities" if info.workflow == "pixel" else "Probabilities Stage 2",
            output_folder=str(tmp_path / runner.name),
            output_suffix="Probabilities",
            timeout_s=1800,
        )
        outputs[runner.name] = tczyx(runner.run(job).outputs[0])
    assert outputs["headless"].shape == outputs["fiji"].shape
    np.testing.assert_allclose(outputs["headless"], outputs["fiji"], atol=1e-4)
