"""Phase 25b: model registry, pinned models in pipelines and manifests, training."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.synthetic.groundtruth import ground_truth_items
from hcs_studio.training import groundtruth as gt
from hcs_studio.training import registry
from hcs_studio.training.train import TrainingSpec, train
from hcs_studio.training.validate import validate_segmentation


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _steps_model(tmp: Path, params: dict[str, object], name: str = "seg") -> registry.ModelRecord:
    path = tmp / "model.json"
    step = PipelineStep(module="segment_classical", params=params)
    path.write_text(json.dumps({"steps": [step.model_dump(mode="json")]}), encoding="utf-8")
    return registry.register_model(name, "steps", path, author="tester")


def test_registry_versions_statuses_and_integrity(home: Path, tmp_path: Path) -> None:
    first = _steps_model(tmp_path, {"channel": 1})
    second = _steps_model(tmp_path, {"channel": 1, "method": "li"})
    assert (first.version, second.version, first.status) == (1, 2, "candidate")
    assert first.path != second.path and first.path.read_text(
        encoding="utf-8"
    ) != second.path.read_text(encoding="utf-8")
    assert (
        registry.get_model("seg").version == 2
        and registry.get_model("seg", 1).sha256 == first.sha256
    )
    with pytest.raises(ValueError, match="cannot become production"):
        registry.set_status("seg", 1, "production", approved=True)
    registry.set_status("seg", 1, "validated")
    with pytest.raises(PermissionError, match="approval"):
        registry.set_status("seg", 1, "production")
    live, demoted = registry.set_status("seg", 1, "production", approved=True)
    assert live.status == "production" and demoted == []
    registry.set_status("seg", 2, "validated")
    live, demoted = registry.set_status("seg", 2, "production", approved=True)
    assert [d.version for d in demoted] == [1] and registry.get_model(
        "seg", 1
    ).status == "validated"
    assert registry.get_model("seg", "production").version == 2
    assert registry.resolve("registry:seg@production").version == 2
    registry.set_status("seg", 1, "retired")
    with pytest.raises(ValueError, match="retired"):
        registry.resolve("registry:seg@1")
    os.chmod(second.path, 0o644)  # tamper with the stored file
    second.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        registry.resolve("registry:seg@2")
    with pytest.raises(ValueError, match="steps model, not cellpose"):
        registry.resolve("registry:seg@2", "cellpose")
    with pytest.raises(ValueError, match="kind"):
        registry.register_model("x", "unknown", second.path)


def test_tuning_improves_held_out_ap(home: Path) -> None:
    """Done-when (synthetic): tuning on ~20 annotated fields of a hard cell line improves
    AP@0.5 on held-out fields over the base model."""
    gt.create_set("hard", ground_truth_items("hard", plates=2, wells=6, fields=2))
    split = gt.split_set("hard", by="well", fractions=(0.65, 0.15, 0.2), seed=0)
    assert split.counts()["train"] >= 14 and gt.check_leakage(split.items, "well") == {}
    grid = {"method": ["otsu", "local"], "min_distance_um": [2.0, 5.0], "smoothing_um": [0.5, 1.0]}
    result = train(
        TrainingSpec(kind="steps", name="hard-nuclei", set="hard", hyperparameters={"grid": grid})
    )
    model = result.model
    assert model.kind == "steps" and model.training_set == "hard@2"
    assert model.training_set_sha256 == split.content_sha256
    assert result.baseline is not None
    assert result.validation["overall"]["ap50"] > result.baseline["overall"]["ap50"] + 0.15
    assert model.hyperparameters["selected"]["method"] == "local"
    assert model.feature_summary["features"]["area_um2"]["q50"] > 0
    tuned = [PipelineStep(module="segment_model", params={"model": model.ref})]
    base = [PipelineStep(module="segment_classical", params={"channel": 1})]
    held_out = {
        name: validate_segmentation("hard", steps, split="test").overall["ap50"]
        for name, steps in (("tuned", tuned), ("base", base))
    }
    assert held_out["tuned"] > held_out["base"] + 0.15


def _plate(tmp: Path) -> Path:
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1)],
        channels=("DAPI",),
        size=96,
        features=[Nuclei(count=lambda r, c, f: 12 + 6 * c), ShotNoise(sd=5)],
    )
    path, _ = plate.write(tmp / "raw")
    return path


def test_pinned_model_reproduces_masks_and_manifest_records_it(home: Path, tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline

    first = _steps_model(tmp_path, {"channel": 1})
    second = _steps_model(tmp_path, {"channel": 1, "min_size_px": 400})  # removes every nucleus
    raw = _plate(tmp_path)

    def run(ref: str, out: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        pipeline = Pipeline(
            steps=[
                PipelineStep(module="segment_model", params={"model": ref}),
                PipelineStep(module="measure_objects", params={"labels": ["nuclei"]}),
            ]
        )
        folder = run_pipeline(
            open_plate(raw), pipeline, output_root=tmp_path / out, save_images=False
        ).folder
        objects = pd.concat(
            pd.read_parquet(p) for p in sorted(folder.glob("wells/*/objects.parquet"))
        )
        return objects.reset_index(drop=True), json.loads(
            (folder / "run_manifest.json").read_text(encoding="utf-8")
        )

    a, manifest = run(first.ref, "a")
    b, _ = run(first.ref, "b")
    assert len(a) == 12 + 18 and a.equals(b)  # same version → identical objects (masks)
    [entry] = manifest["models"]
    assert entry["registry"]["version"] == 1 and entry["sha256"] == first.sha256
    assert entry["value"] == "registry:seg@1" and entry["module"] == "segment_model"
    c, other = run(second.ref, "c")
    assert other["models"][0]["registry"]["version"] == 2
    assert other["models"][0]["sha256"] == second.sha256 != first.sha256
    assert len(c) == 0
    with pytest.raises(Exception, match="no version 9"):
        run("registry:seg@9", "d")
    pinned, changes = registry.pin_pipeline(
        Pipeline(steps=[PipelineStep(module="segment_model", params={"model": "registry:seg"})])
    )
    assert changes == ["registry:seg → registry:seg@2"]
    assert pinned.steps[0].params["model"] == "registry:seg@2"


def test_cellpose_and_classifier_accept_registry_references(home: Path, tmp_path: Path) -> None:
    from hcs_studio.segmentation.cellpose_backend import model_file

    weights = tmp_path / "custom_cp"
    weights.write_bytes(b"not really weights")
    model = registry.register_model("cp", "cellpose", weights, base_model="cpsam_v2")
    assert model_file("registry:cp@1") == str(model.path) and model_file("cpsam_v2") == "cpsam_v2"
    entry = registry.describe_model(1, "segment_cellpose", "model", "registry:cp@1")
    assert entry["sha256"] == model.sha256 and entry["registry"]["kind"] == "cellpose"
    builtin = registry.describe_model(1, "segment_cellpose", "model", "cpsam_v2")
    assert builtin["builtin"] is True and ("sha256" in builtin)
    # an object classifier trained on a classification set, used by reference
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "item": [f"i{k % 6}" for k in range(240)],
            "nuclei_area_um2": np.r_[rng.normal(80, 8, 120), rng.normal(140, 10, 120)],
            "dna_mean": np.r_[rng.normal(500, 40, 120), rng.normal(900, 60, 120)],
            "label": ["G1"] * 120 + ["G2"] * 120,
        }
    )
    items = [(gt.Item(id=f"i{k}", well=f"A0{k + 1}"), np.zeros((4, 4)), None) for k in range(6)]
    gt.create_set("phase", items, kind="classification", objects=objects)
    gt.split_set("phase", by="well", fractions=(0.5, 0.5, 0))
    result = train(TrainingSpec(kind="classifier", name="phase-clf", set="phase"))
    metrics = result.validation
    assert metrics["balanced_accuracy"] > 0.95 and metrics["calibration"]["ece"] < 0.2
    from hcs_studio.analysis.classifier import ClassifyObjects, ClassifyParams

    step = ClassifyObjects(ClassifyParams(model_path=result.model.ref, object_type="nuclei"))
    state = step.fit_table(objects.drop(columns=["label", "item"]))
    assert state["sha256"] == result.model.sha256


def test_training_job_in_the_queue(home: Path) -> None:
    from hcs_studio.pipeline.queue import LocalJobQueue
    from hcs_studio.services import context
    from hcs_studio.services.training import submit_training

    gt.create_set("q", ground_truth_items("easy", plates=1, wells=4, fields=1, size=64))
    gt.split_set("q", by="well", fractions=(0.5, 0.5, 0))
    queue = LocalJobQueue(persist=False)
    spec = TrainingSpec(
        kind="steps", name="queued", set="q", hyperparameters={"grid": {"method": ["otsu", "li"]}}
    )
    job_id = submit_training(context("nightly retrain"), queue, spec)
    deadline = time.time() + 120
    while queue.status(job_id).status in ("queued", "running", "paused") and time.time() < deadline:
        queue.poll()
        time.sleep(0.2)
    state = queue.status(job_id)
    assert state.status == "completed", state.message
    assert state.spec.kind == "train" and not state.spec.resources.gpu
    assert registry.get_model("queued").author


def test_models_commands(home: Path, tmp_path: Path) -> None:
    from hcs_studio.app import main

    gt.create_set("cli", ground_truth_items("easy", plates=1, wells=4, fields=1, size=64))
    assert main(["gt", "split", "cli", "--fractions", "0.5,0.5,0"]) == 0
    assert (
        main(
            [
                "models",
                "train",
                "--kind",
                "steps",
                "--name",
                "cli-seg",
                "--set",
                "cli",
                "--grid",
                '{"method": ["otsu", "li"]}',
                "--reason",
                "first model",
            ]
        )
        == 0
    )
    assert main(["models", "list"]) == 0 and main(["models", "show", "cli-seg@1"]) == 0
    assert main(["models", "status", "cli-seg@1", "validated", "--reason", "looked good"]) == 0
    assert main(["models", "status", "cli-seg@1", "production", "--reason", "go"]) == 2  # no --yes
    assert main(["models", "status", "cli-seg@1", "production", "--reason", "go", "--yes"]) == 0
    assert registry.get_model("cli-seg", "production").version == 1
    with pytest.raises(SystemExit):  # --reason is required
        main(["models", "status", "cli-seg@1", "retired"])
    weights = tmp_path / "w.joblib"
    weights.write_bytes(b"x")
    assert main(["models", "register", str(weights), "--name", "clf", "--kind", "classifier"]) == 0
    pipeline = tmp_path / "p.yaml"
    Pipeline(
        steps=[
            PipelineStep(module="segment_model", params={"model": "registry:cli-seg@production"})
        ]
    ).save(pipeline)
    assert main(["models", "pin", str(pipeline)]) == 0
    assert Pipeline.load(pipeline).steps[0].params["model"] == "registry:cli-seg@1"
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    with session_scope() as session:
        kinds = {e.type for e in session.scalars(select(EventRecord))}
    assert {"models.train", "models.status", "models.register"} <= kinds


@pytest.mark.skipif(not os.environ.get("HCS_REAL_CELLPOSE"), reason="set HCS_REAL_CELLPOSE=1")
def test_real_cellpose_fine_tune_registers(home: Path) -> None:
    """Real Cellpose (cpsam_v2 weights must be downloaded): fine-tune for 1 epoch on 4 small
    fields, validate against the base model, register. On a CPU this checks the integration,
    not the accuracy gain (that needs a GPU, ~100 epochs and real fields)."""
    pytest.importorskip("cellpose")
    gt.create_set("cp", ground_truth_items("easy", plates=1, wells=6, fields=1, size=96))
    gt.split_set("cp", by="well", fractions=(4, 2, 0))
    result = train(
        TrainingSpec(
            kind="cellpose",
            name="cp-ft",
            set="cp",
            device="cpu",
            hyperparameters={"epochs": 1, "min_train_masks": 1, "diameter_um": 6.5},
        )
    )
    assert result.model.kind == "cellpose" and result.model.base_model == "cpsam_v2"
    assert result.baseline is not None and result.baseline["overall"]["ap50"] > 0.5
    assert "ap50" in result.validation["overall"]
