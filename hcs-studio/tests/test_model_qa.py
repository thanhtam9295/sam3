"""Phase 25c: model comparison report, downstream impact (MSR), spot checks, drift."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.synthetic.groundtruth import ground_truth_items, nuclei_field
from hcs_studio.training import groundtruth as gt
from hcs_studio.training import qa, registry
from hcs_studio.training.train import TrainingSpec, train

GRID = {"method": ["otsu", "local"], "min_distance_um": [2.0, 5.0], "smoothing_um": [0.5, 1.0]}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _hard_model(name: str = "hard-nuclei") -> registry.ModelRecord:
    if not gt.versions("hard"):
        gt.create_set("hard", ground_truth_items("hard", plates=2, wells=6, fields=2))
        gt.split_set("hard", by="well", fractions=(0.65, 0.15, 0.2))
    return train(
        TrainingSpec(kind="steps", name=name, set="hard", hyperparameters={"grid": GRID})
    ).model


def test_comparison_report_is_saved(home: Path) -> None:
    """Done-when: the candidate beats the base model on held-out fields and the comparison
    report is saved; a later candidate is compared with the production version."""
    from hcs_studio.training.compare import compare_candidate

    first = _hard_model()
    comparison, report = compare_candidate(first.ref)
    folder = report.parent
    assert report.name == "comparison.html" and folder.is_relative_to(registry.models_root())
    for name in (
        "comparison.json",
        "comparison.csv",
        "ap_by_iou.png",
        "count_error.png",
        "examples.png",
    ):
        assert (folder / name).exists(), name
    table = comparison.table().set_index("model")
    candidate, base = table.index
    assert base.startswith("base:") and comparison.split == "test"
    assert table.loc[candidate, "AP@0.5"] > table.loc[base, "AP@0.5"] + 0.15
    assert "hard@2" in report.read_text() and 'src="examples.png"' in report.read_text()
    metrics = registry.get_model("hard-nuclei", 1).metrics["test"]
    assert metrics["report"] == str(report) and metrics["ap50"] == pytest.approx(
        table.loc[candidate, "AP@0.5"]
    )
    registry.set_status("hard-nuclei", 1, "validated")
    registry.set_status("hard-nuclei", 1, "production", approved=True)
    second = _hard_model()
    comparison, _ = compare_candidate(second.ref)
    assert list(comparison.results)[1] == "hard-nuclei@1 (production)"


@dataclass
class HardNuclei:
    """Fields of the synthetic 'hard' cell line drawn into a plate (channel 1)."""

    cell_line: str = "hard"
    name: str = "hard_nuclei"

    def draw(self, ctx: Any) -> dict[str, Any]:
        image, masks = nuclei_field(ctx.rng, self.cell_line, ctx.image.shape[-1])
        ctx.image[0, 0] += image[0] - 100
        return {"count": int(np.max(masks))}


def _run(
    tmp: Path, ref: str, features: list[Any], out: str, steps: list[PipelineStep] | None = None
) -> Path:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(1, 1), (1, 2)],
        channels=("DAPI",),
        size=128,
        n_fields=3,
        features=[*features, ShotNoise(sd=4)],
        name=out,
    )
    raw, _ = plate.write(tmp / f"raw_{out}")
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="segment_model", params={"model": ref}),
            PipelineStep(module="measure_objects", params={"labels": ["nuclei"]}),
            *(steps or []),
        ]
    )
    return run_pipeline(open_plate(raw), pipeline, output_root=tmp / out).folder


def test_drift_and_spot_checks(home: Path, tmp_path: Path) -> None:
    from hcs_studio.synthetic import Nuclei

    model = _hard_model()
    qa_step = PipelineStep(module="model_qa", params={"spot_checks": 2, "seed": 1})
    same = _run(tmp_path, model.ref, [HardNuclei()], "same", [qa_step])
    record = json.loads((same / "qa" / "model_qa.json").read_text())
    assert len(record["spot_checks"]) == 2 and record["drift"]["model"] == model.ref
    assert record["drift"]["alerts"] == [], record["drift"]
    assert set(record["drift"]["features"]) >= {"area_um2", "intensity_mean", "objects_per_field"}
    manifest = json.loads((same / "run_manifest.json").read_text())
    assert manifest["run_steps"][0]["module"] == "model_qa"
    # small, dim nuclei of another "cell line": the model's training data no longer fits
    other = _run(tmp_path, model.ref, [Nuclei(count=lambda r, c, f: 20, intensity=300)], "other")
    report = qa.drift_report(other)
    assert any(a.startswith("area_um2") for a in report["alerts"]) and "suggestion" in report
    assert report["features"]["area_um2"]["shift_iqr"] < -1
    # spot checks: queued once per run, reviewed with reasons, summarised per model
    assert qa.queue_spot_checks(same, 5) == record["spot_checks"]  # idempotent
    ids = qa.queue_spot_checks(other, 3, seed=0)
    assert len(ids) == 3
    pending = qa.spot_checks("pending")
    assert len(pending) == 5 and set(pending["model"]) == {"hard-nuclei@1"}
    files = qa.field_files(same, pending.iloc[0]["well"], int(pending.iloc[0]["field"]), "nuclei")
    assert {"preview", "outline"} <= set(files)
    with pytest.raises(ValueError, match="Say what is wrong"):
        qa.review_spot_check(ids[0], "flagged", "anna")
    qa.review_spot_check(ids[0], "flagged", "anna", "nuclei split in two")
    for check in [*record["spot_checks"], *ids[1:]]:
        qa.review_spot_check(check, "accepted", "anna")
    summary = qa.spot_check_summary().set_index("model")
    assert summary.loc["hard-nuclei@1", "checks"] == 5
    assert summary.loc["hard-nuclei@1", "accept_rate"] == pytest.approx(4 / 5)


def test_drift_maths() -> None:
    train = {"features": {"area_um2": {"q05": 5, "q25": 9, "q50": 10, "q75": 11, "q95": 15}}}
    rng = np.random.default_rng(0)
    ok = qa.drift(train, {"area_um2": rng.normal(10, 1.48, 400)})
    assert ok["alerts"] == [] and abs(ok["features"]["area_um2"]["shift_iqr"]) < 0.3
    moved = qa.drift(train, {"area_um2": rng.normal(13, 1.48, 400)})
    assert moved["features"]["area_um2"]["shift_iqr"] == pytest.approx(1.5, abs=0.2)
    wide = qa.drift(train, {"area_um2": rng.normal(10, 6, 400)})
    assert wide["features"]["area_um2"]["spread_ratio"] > 2 and wide["alerts"]


def test_downstream_impact_against_msr(home: Path, tmp_path: Path) -> None:
    """Re-running the dose-response reference plate with an equivalent candidate keeps counts
    and IC50s (ratio within the MSR); a broken candidate is flagged."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.layout import load_plate_layout
    from hcs_studio.pharmacology.synthetic import write_image_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.training.impact import downstream_impact, write_impact

    def steps_model(params: dict[str, Any]) -> registry.ModelRecord:
        path = tmp_path / "m.json"
        step = PipelineStep(module="segment_classical", params={"channel": 1, **params})
        path.write_text(json.dumps({"steps": [step.model_dump(mode="json")]}))
        return registry.register_model("ref-seg", "steps", path)

    reference_model = steps_model({})
    equivalent = steps_model({"min_size_px": 10})
    broken = steps_model({"min_size_px": 400})
    plate_path, layout_path = write_image_plate(tmp_path / "plate")
    wells = [(r, c) for r in (14, 15) for c in [0, *range(1, 11), 23]]  # Drug D on MCF7 + controls
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="segment_model", params={"model": reference_model.ref}),
            PipelineStep(module="measure_objects", params={"texture": False, "granularity": False}),
            PipelineStep(module="qc"),
        ]
    )
    reference = run_pipeline(
        open_plate(plate_path),
        pipeline,
        output_root=tmp_path / "ref",
        wells=wells,
        layout=load_plate_layout(layout_path),
        layout_path=layout_path,
        checksums=False,
    ).folder
    same = downstream_impact(reference, equivalent.ref, tmp_path / "impact_same")
    assert "segment_model registry:ref-seg@1" in same.replaced and same.readout == "nuclei_count"
    assert (
        same.counts_summary["wells"] == 24 and abs(same.counts_summary["mean_difference_pct"]) < 1
    )
    [row] = same.potency.to_dict("records")
    assert row["compound"] == "Drug D" and row["fold_change"] < 1.2 and not row["significant"]
    assert row["msr"] == 3.0 and row["msr_source"] == "default"
    assert same.verdict["equivalent"]
    report = write_impact(same, tmp_path / "impact_same")
    assert (
        report.parent / "counts_bland_altman.png"
    ).exists() and "equivalent" in report.read_text()
    bad = downstream_impact(reference, broken.ref, tmp_path / "impact_bad")
    assert not bad.verdict["equivalent"] and bad.counts_summary["mean_difference_pct"] < -100
    assert any("count difference" in r for r in bad.verdict["reasons"])
    with pytest.raises(ValueError, match="No step uses"):
        downstream_impact(reference, _named_model(tmp_path).ref, tmp_path / "x")


def _named_model(tmp: Path) -> registry.ModelRecord:
    path = tmp / "o.json"
    step = PipelineStep(module="segment_classical", params={"channel": 1})
    path.write_text(json.dumps({"steps": [step.model_dump(mode="json")]}))
    return registry.register_model("other", "steps", path)


def test_qa_commands(home: Path, tmp_path: Path) -> None:
    from hcs_studio.app import main

    model = _hard_model()
    run = _run(tmp_path, model.ref, [HardNuclei()], "cli")
    assert main(["models", "compare", "hard-nuclei@1", "--output", str(tmp_path / "cmp")]) == 0
    assert (tmp_path / "cmp" / "comparison.html").exists()
    assert main(["models", "drift", str(run)]) == 0
    assert main(["models", "spot-check", "queue", str(run), "-n", "2", "--seed", "3"]) == 0
    assert main(["models", "spot-check", "list"]) == 0
    first = int(qa.spot_checks("pending").iloc[0]["id"])
    assert main(["models", "spot-check", "review", str(first), "accepted", "--reason", "fine"]) == 0
    assert main(["models", "spot-check", "review", str(first), "flagged"]) == 2  # no reason
    assert main(["models", "spot-check", "summary"]) == 0
    assert qa.spot_checks("accepted")["reviewer"].iloc[0]


@pytest.mark.gui
def test_models_tab(
    home: Path, tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    from hcs_studio.gui.models_panel import ModelsPanel
    from hcs_studio.services import training as services

    def quick_compare(ctx: Any, candidate: str, **kwargs: Any) -> Any:
        """The real comparison (figures) is tested above; building matplotlib figures in the
        worker thread while qtbot.waitUntil spins the event loop can starve it for minutes."""
        report = tmp_path / "report.html"
        report.write_text("report")
        name, version = registry.parse_ref(candidate)
        registry.update_metrics(
            name, int(version or 1), {"test": {"ap50": 0.5, "report": str(report)}}
        )
        return None, report

    monkeypatch.setattr(services, "compare_models", quick_compare)

    model = _hard_model()
    run = _run(tmp_path, model.ref, [HardNuclei()], "gui")

    def cell(row: int, column: int) -> str:
        item = panel.model_table.item(row, column)
        return item.text() if item is not None else ""

    qa.queue_spot_checks(run, 2, seed=0)
    answers: list[bool] = [False, True]
    panel = ModelsPanel(confirm=lambda question: answers.pop(0))
    qtbot.addWidget(panel)
    panel.refresh()
    qtbot.waitUntil(lambda: panel.model_table.rowCount() == 1, timeout=20000)
    assert panel.check_table.rowCount() == 2 and "hard-nuclei@1" in cell(0, 0)
    panel.model_table.selectRow(0)
    panel.status_combo.setCurrentText("validated")
    panel.set_status()  # no reason
    assert "reason" in panel.status.text()
    panel.reason.setText("good on test plates")
    panel.set_status()
    qtbot.waitUntil(lambda: cell(0, 2) == "validated", timeout=20000)
    panel.model_table.selectRow(0)
    panel.status_combo.setCurrentText("production")
    panel.reason.setText("replaces the default segmentation")
    panel.set_status()
    assert "cancelled" in panel.status.text()  # first confirmation: no
    panel.set_status()
    qtbot.waitUntil(lambda: cell(0, 2) == "production", timeout=20000)
    panel.model_table.selectRow(0)
    panel.compare()
    qtbot.waitUntil(lambda: "Report saved" in panel.status.text(), timeout=60000)
    qtbot.waitUntil(lambda: cell(0, 4) != "–", timeout=20000)
    panel.check_table.selectRow(0)
    assert panel.preview.pixmap() is not None and not panel.preview.pixmap().isNull()
    panel.review("flagged")
    assert "wrong" in panel.status.text()
    panel.review("accepted")
    qtbot.waitUntil(lambda: panel.check_table.rowCount() == 1, timeout=20000)
    assert "1 accepted" in panel.rates.text()
