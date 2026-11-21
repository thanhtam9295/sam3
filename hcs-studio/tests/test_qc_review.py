"""Phase 18c: reviewing QC decisions, the QC report, learned QC and the QC tab."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.qc.rules import default_rules

STEPS = """
name: qc-review
steps:
  - {module: segment_classical, params: {channel: 1, min_distance_um: 2}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
"""


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def make_run(root: Path) -> tuple[Path, dict[str, Any]]:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic.artifacts import qc_plate

    path, truth = qc_plate(root / "raw", per_kind=3, clean=9, kinds=("blur", "bubble", "empty"))
    rules = default_rules(nucleus_diameter_um=truth["nucleus_diameter_um"])
    pipeline = Pipeline.model_validate(
        {**Pipeline.from_text(STEPS).to_dict(), "qc": rules.to_dict()}
    )
    result = run_pipeline(open_plate(path), pipeline, output_root=root / "runs")
    return result.folder, truth


def _label(truth: dict[str, Any], well: str, field: int) -> Any:
    return truth["labels"][f"{well}/{field}"]


def test_reviews_change_exclusions_everywhere_and_are_recorded(home: Path) -> None:
    from hcs_studio.qc.review import load_overrides
    from hcs_studio.services import ReasonRequired, context, recent_events
    from hcs_studio.services.operations import review_qc

    folder, truth = make_run(home)
    fields = pd.read_parquet(folder / "qc" / "field_qc.parquet")
    blur = fields[
        [_label(truth, w, f) == "blur" for w, f in zip(fields.well, fields.field, strict=True)]
    ]
    well, field = str(blur.well.iloc[0]), int(blur.field.iloc[0])
    before = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    with pytest.raises(ReasonRequired):
        review_qc(context(""), folder, "field", well, "include", field=field)

    result = review_qc(context("cells are sharp on inspection"), folder, "field", well,
                       "include", field=field)  # fmt: skip
    row = result.fields[(result.fields.well == well) & (result.fields.field == field)].iloc[0]
    assert not row.excluded and row.rule_excluded and row.review == "include"
    flags = pd.read_csv(folder / "tables" / "qc_flags.csv")
    hit = flags[(flags.well == well) & (flags.field == field)].iloc[0]
    assert not hit.image_qc_excluded and not hit.excluded  # the tables were rebuilt
    after = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    excluded_before = int(before["fields_excluded"].to_dict()[well])
    assert int(after["fields_excluded"].to_dict()[well]) == excluded_before - 1

    review_qc(context("plate edge was wet"), folder, "well", well, "exclude")
    wells = pd.read_csv(folder / "qc" / "well_qc.csv").set_index("well").to_dict("index")
    assert wells[well]["failed"]
    assert "failed on review: plate edge was wet" in str(wells[well]["reasons"])
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert summary.loc[well, "qc_failed"]

    history = load_overrides(folder)
    assert [o.action for o in history] == ["include", "exclude"] and history[0].actor
    event = recent_events(1, "qc.review")[0]
    assert event["reason"] == "plate edge was wet" and event["after"]["well_failed"] is True
    plate = json.loads((folder / "qc" / "plate_qc.json").read_text())
    assert plate["reviews"] == 2 and "seconds" in plate
    page = (folder / "qc" / "qc_report.html").read_text()
    assert "plate edge was wet" in page and page.count("data:image/png;base64,") > 5
    with pytest.raises(KeyError, match="No field 99"):
        review_qc(context("x"), folder, "field", well, "include", field=99)


def test_command_line_review_and_report(home: Path) -> None:
    from hcs_studio.app import main

    folder, _ = make_run(home)
    wells = pd.read_csv(folder / "qc" / "well_qc.csv")
    well = str(wells.well.iloc[0])
    assert main(["qc-review", str(folder), "--well", well, "--action", "exclude",
                 "--reason", "contaminated"]) == 0  # fmt: skip
    assert pd.read_csv(folder / "qc" / "well_qc.csv").set_index("well").loc[well, "failed"]
    assert main(["qc-review", str(folder), "--well", "Z99", "--action", "exclude",
                 "--reason", "x"]) == 2  # fmt: skip


def test_learned_qc_catches_an_artifact_the_rules_miss(home: Path) -> None:
    """Reviewers confirm blurred fields on one plate; a classifier trained on those reviews
    finds the blurred fields of another plate whose rules have no blur rule."""
    from hcs_studio.analysis.classifier import file_sha256, save_classifier
    from hcs_studio.io.base import open_plate
    from hcs_studio.qc.learned import train_field_qc
    from hcs_studio.qc.review import Override, apply_override
    from hcs_studio.qc.rules import LearnedQc
    from hcs_studio.qc.run import run_qc
    from hcs_studio.synthetic.artifacts import qc_plate

    rules = default_rules(nucleus_diameter_um=4.55)
    no_blur = rules.model_copy(update={"field": [r for r in rules.field if r.flag != "blur"]})
    path, _ = qc_plate(home / "a" / "raw", per_kind=12, clean=60, kinds=("blur",), seed=1)
    trained_on = home / "a" / "qc"
    result = run_qc(open_plate(path), rules, trained_on, workers=4)
    blurred_a = result.fields[result.fields.flag_blur]
    for well, field in zip(blurred_a.well, blurred_a.field, strict=True):
        review = Override(
            level="field", well=str(well), field=int(field), action="confirm", reason="blurred"
        )
        apply_override(trained_on, review)
    classifier = train_field_qc([trained_on])
    accuracy = classifier.cv_balanced_accuracy
    assert classifier.n_examples["bad"] == 12 and accuracy is not None and accuracy > 0.9
    model = home / "learned.joblib"
    save_classifier(classifier, model)

    learned = no_blur.model_copy(
        update={"learned": LearnedQc(model=str(model), sha256=file_sha256(model), exclude=True)}
    )
    path_b, truth_b = qc_plate(home / "b" / "raw", per_kind=12, clean=60, kinds=("blur",), seed=2)
    fields = run_qc(open_plate(path_b), learned, home / "b" / "qc", workers=4).fields
    fields["truth"] = [
        truth_b["labels"][f"{w}/{f}"] for w, f in zip(fields.well, fields.field, strict=True)
    ]
    blurred = fields[fields.truth == "blur"]
    clean = fields[fields.truth.isna()]
    assert blurred.flag_learned.mean() >= 0.9 and blurred.excluded.mean() >= 0.9
    assert clean.flag_learned.mean() <= 0.05
    assert "learned (p bad" in blurred.reasons.iloc[0]
    model.write_bytes(model.read_bytes() + b"x")  # a changed model is refused
    with pytest.raises(ValueError, match="changed since"):
        run_qc(open_plate(path_b), learned, home / "c", workers=2)


def test_learned_qc_needs_enough_reviews(home: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.qc.learned import train_field_qc
    from hcs_studio.qc.run import run_qc
    from hcs_studio.synthetic.artifacts import qc_plate

    path, _ = qc_plate(home / "raw", per_kind=1, clean=11, kinds=("blur",))
    run_qc(open_plate(path), default_rules(nucleus_diameter_um=4.55), home / "qc")
    with pytest.raises(ValueError, match="at least 3 bad"):
        train_field_qc([home / "qc"])


pytest.importorskip("PySide6")


@pytest.mark.gui
def test_qc_tab_shows_flags_and_records_reviews(qtbot: Any, home: Path) -> None:
    from hcs_studio.gui.qc_panel import QcPanel
    from hcs_studio.model.plate import parse_well_name

    folder, _ = make_run(home)
    panel = QcPanel()
    qtbot.addWidget(panel)
    panel.open(folder)
    qtbot.waitUntil(lambda: panel.gallery.count() > 0, timeout=20000)
    severities = [
        float(panel.fields.loc[panel.gallery.item(i).data(256), "severity"])
        for i in range(panel.gallery.count())
    ]
    assert severities == sorted(severities, reverse=True)
    assert "plate" in panel.summary.text() and panel.metric_combo.findText("severity") >= 0
    failed = set(panel.wells.loc[panel.wells.failed.astype(bool), "well"])
    assert {(r, c) for r, c in panel.map.hatched} == {parse_well_name(w) for w in failed}
    panel.metric_combo.setCurrentText("excluded fields")
    panel.review("field", "include")
    assert "reason" in panel.status.text()
    well, field = panel.current_field()  # type: ignore[misc]
    panel.reason.setText("looks fine")
    with qtbot.waitSignal(panel.reviewed, timeout=60000):
        panel.review("field", "include")

    def reviewed() -> bool:
        return "review" in panel.fields and bool(panel.fields["review"].astype(bool).any())

    qtbot.waitUntil(reviewed, timeout=20000)
    row = panel.fields[(panel.fields.well == well) & (panel.fields.field == field)].iloc[0]
    assert row.review == "include" and not row.excluded
    panel.open(home)  # a folder without QC
    assert "no image QC" in panel.summary.text() and panel.gallery.count() == 0
