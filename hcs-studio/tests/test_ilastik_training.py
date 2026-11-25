"""Phase 16f: the "Train in ilastik" round trip (representative fields, a stage-matched training
set, ilastik's window, registration) and the one-line CLI run, against the fake ilastik."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcs_studio.model.annotation import Treatment, WellAnnotation
from hcs_studio.segmentation.ilastik_backend import training_set as ts
from hcs_studio.segmentation.ilastik_backend.h5io import is_ilastik_ready, read_h5
from tests import test_ilastik_steps
from tests.test_ilastik_steps import _pipeline, _plate, calls

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)


def test_representative_fields_cover_lines_controls_and_top_doses() -> None:
    def drug(conc: float) -> WellAnnotation:
        return WellAnnotation(cell_line="HeLa", treatments=(Treatment("taxol", conc),))

    layout = {
        (0, 0): WellAnnotation(cell_line="HeLa", role="negative_control"),
        (0, 1): WellAnnotation(cell_line="HeLa", role="negative_control"),
        (0, 2): drug(1e-9),
        (0, 3): drug(1e-6),  # top dose
        (1, 0): WellAnnotation(cell_line="A549", role="positive_control"),
        (1, 1): WellAnnotation(cell_line="A549", role="empty"),
        (1, 2): WellAnnotation(cell_line="A549", treatments=(Treatment("taxol", 1e-7),)),
    }
    wells = [*sorted(layout), (2, 0)]
    fields = {w: [1, 2] for w in wells}
    chosen = ts.representative_fields(wells, fields, layout)
    reasons = {f.well: f.reason for f in chosen}
    assert reasons == {
        "A01": "HeLa: negative control",
        "A04": "HeLa: taxol top dose",
        "B01": "A549: positive control",
        "B03": "A549: taxol top dose",
    }
    assert all(f.field == 1 for f in chosen)
    spread = ts.representative_fields(wells, fields, None, max_fields=3)
    assert [f.well for f in spread] == ["A01", "B01", "C01"]
    assert ts.representative_fields([], {}, None) == []


def test_training_set_from_a_plate(tmp_path: Path) -> None:
    plate, _ = _plate(tmp_path)
    fields = ts.representative_fields(sorted(plate.wells), ts.fields_per_well(plate), None)
    target = ts.export_training_set(plate.path, fields, tmp_path / "train", channels=[1])
    data = json.loads(target.read_text())
    assert data["format"] == ts.FORMAT and data["stage"] == "raw"
    assert data["channels"] == ["DAPI"] and len(data["fields"]) == 3
    first = tmp_path / "train" / data["fields"][0]["file"]
    assert is_ilastik_ready(first)
    array, axes = read_h5(first)
    assert axes == "yxc" and array.shape[-1] == 1
    raw = np.asarray(plate.wells[(0, 0)].fields[0].load()[0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(array[..., 0], raw)
    with pytest.raises(ValueError, match="outside the raw data"):
        ts.export_training_set(plate.path, fields, plate.path / "train")
    with pytest.raises(ValueError, match="needs a run folder"):
        ts.export_training_set(plate.path, fields, tmp_path / "t2", stage="step:1")


def test_training_set_at_the_stage_the_ilastik_step_sees(
    ilastik: dict[str, Any], tmp_path: Path
) -> None:
    """With a processing step before ilastik (here CLAHE), the training images are processed the
    same way (recomputed from the run, with its fitted estimates)."""
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    plate, _ = _plate(tmp_path)
    steps = _pipeline(ilastik["project"]).steps
    background = PipelineStep(module="clahe")
    result = run_pipeline(
        plate,
        Pipeline(steps=[background, steps[0]]),
        output_root=tmp_path / "runs",
        workers=1,
        checksums=False,
        wells=[(0, 0)],
    )
    assert not result.failed_wells
    assert ts.ilastik_stage(result.folder) == ("step:1", [1])
    fields = [ts.TrainingField("A01", 2, reason="test")]
    target = ts.export_training_set(result.folder, fields, tmp_path / "train", stage="ilastik")
    data = json.loads(target.read_text())
    assert data["stage"] == "step:1" and data["channel_numbers"] == [1]
    array, _ = read_h5(tmp_path / "train" / "A01_f002_t0.h5")
    raw = np.asarray(plate.wells[(0, 0)].fields[1].load()[0, 0, 0], dtype=np.float32)
    assert not np.allclose(array[..., 0], raw)  # processed, not raw
    from hcs_studio.compare.sources import _recompute

    expected = _recompute(result.folder, "A01", 2, "step:1", 0).image.data[0, 0]
    np.testing.assert_array_equal(array[..., 0], expected)
    meta = ts.project_metadata(ilastik["project"], target)
    assert meta["workflow"] == "pixel" and meta["classes"] == ["background", "nuclei"]
    assert meta["training_set"]["stage"] == "step:1" and meta["training_set"]["fields"] == 1
    plain = Pipeline(steps=steps)
    assert plain.image_steps()[0].module == "ilastik_pixel"


def test_ilastik_window_command(tmp_path: Path) -> None:
    existing = tmp_path / "p.ilp"
    existing.write_bytes(b"")
    assert ts.ilastik_gui_command("/opt/ilastik/run_ilastik.sh", existing) == [
        "/opt/ilastik/run_ilastik.sh",
        f"--project={existing}",
    ]
    new = ts.ilastik_gui_command("/env/bin/python", tmp_path / "new.ilp", "objects")
    assert new[:3] == ["/env/bin/python", "-m", "ilastik"]
    assert new[3] == f"--new_project={tmp_path / 'new.ilp'}"
    assert new[4] == "--workflow=ObjectClassificationWorkflowPrediction"
    with pytest.raises(ValueError, match="Unknown workflow"):
        ts.ilastik_gui_command("x", tmp_path / "n.ilp", "nonsense")


def test_register_project_in_the_model_registry(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.services import context
    from hcs_studio.services.training import register_model
    from hcs_studio.training.registry import list_models, resolve_path

    record = register_model(
        context(),
        "nuclei-ilastik",
        "ilastik",
        ilastik["project"],
        hyperparameters=ts.project_metadata(ilastik["project"]),
        notes="HeLa, DAPI",
    )
    assert record.kind == "ilastik"
    stored = list_models(kind="ilastik")[0]
    assert stored.hyperparameters["classes"] == ["background", "nuclei"]
    assert Path(resolve_path(record.ref)).name == "nuclei.ilp"


def test_cli_training_set_and_run(
    ilastik: dict[str, Any], tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from hcs_studio.app import main

    plate, _ = _plate(tmp_path)
    out = tmp_path / "train"
    assert main(["ilastik", "--training-set", str(plate.path), "--channels", "1",
                 "--output", str(out)]) == 0  # fmt: skip
    assert "spread over the plate" in capsys.readouterr().out
    assert len(json.loads((out / "training_set.json").read_text())["fields"]) == 3

    run = ["--run", str(ilastik["project"]), "--input", str(plate.path), "--channels", "1"]
    code = main(["ilastik", *run, "--wells", "A01-A02", "--runs", str(tmp_path / "runs")])
    text = capsys.readouterr().out
    assert code == 0, text
    folder = Path(text.strip().splitlines()[-1].split("Run folder: ")[1])
    wells = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert set(wells.index) == {"A01", "A02"}
    assert len(calls(ilastik["log"])) == 2
    # --runner fiji without a Fiji: a clear error, settings untouched
    monkeypatch.delenv("HCS_FIJI", raising=False)
    code = main(["ilastik", "--run", str(ilastik["project"]), "--input", str(plate.path),
                 "--wells", "A01", "--runner", "fiji", "--runs", str(tmp_path / "r2")])  # fmt: skip
    assert "Fiji runner is selected" in capsys.readouterr().out
    import os

    assert "HCS_ILASTIK_RUNNER" not in os.environ
