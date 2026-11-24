"""Phase 26b: run states, signatures, locking, linked new versions, reports; and the scripted
end-to-end audit scenario."""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.security import accounts, audit, signoff
from hcs_studio.services import Actor, context
from hcs_studio.services import operations as ops
from hcs_studio.services import security as sec


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


PIPELINE = Pipeline(
    steps=[
        PipelineStep(module="segment_classical", params={"channel": 1}),
        PipelineStep(module="measure_objects", params={"labels": ["nuclei"], "texture": False}),
        PipelineStep(module="qc"),
    ]
)
GATE = Pipeline(
    steps=[
        PipelineStep(
            module="gate_objects",
            params={
                "object_type": "nuclei",
                "name": "big",
                "rules": [{"feature": "area_um2", "min": 20}],
            },
        )
    ]
)


def _layout() -> Any:
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role

    layout = set_role(
        PlateLayout.for_plate_type(96), [(0, 0)], "negative_control", cell_line="U2OS"
    )
    return replace(layout, barcode="SIGN01")


def _run(tmp: Path, reason: str = "first run") -> Path:
    from hcs_studio.io.base import open_plate
    from hcs_studio.synthetic import Nuclei, ShotNoise, SyntheticPlate

    plate = SyntheticPlate(
        wells=[(0, 0), (0, 1)],
        channels=("DAPI",),
        size=96,
        name="SIGN01",
        features=[Nuclei(count=lambda r, c, f: 10 + 5 * c), ShotNoise(sd=5)],
    )
    raw, _ = plate.write(tmp / "raw")
    result = ops.run_pipeline(
        context(reason), open_plate(raw), PIPELINE, output_root=tmp / "runs", layout=_layout()
    )
    return Path(result.folder)


def test_scripted_scenario_is_fully_audited(home: Path) -> None:
    """Done-when: every action of an end-to-end scenario is in the audit log with the user and
    the reason; a locked run cannot be changed; re-analysis creates a linked new version."""
    from hcs_studio.io.layout_store import LayoutStore

    me = accounts.os_username()
    accounts.create_account(me, "admin", "Me Myself")
    accounts.create_account("rita", "reviewer", "Rita Reviewer")
    rita = Actor("rita", "Rita Reviewer", "account")
    saved = ops.save_layout_version(
        context("plate map from the dispenser"), LayoutStore("SIGN01"), _layout()
    )
    folder = _run(home, "first run of the plate")
    assert signoff.status(folder).state == "completed"
    report = (folder / "index.html").read_text()
    assert 'class="draftmark"' in report and ">completed" in report
    ops.annotate_run(context("cell line corrected"), folder, replace(saved, assay="viability"))
    ops.analyze_run(context("gate big nuclei"), folder, GATE, include_run_steps=False)
    with pytest.raises(PermissionError, match="reviewer"):  # an analyst may not sign
        accounts.update_account(me, role="analyst")
        sec.sign_run(context("looks fine"), folder, "reviewed")
    accounts.update_account(me, role="admin")
    sec.sign_run(context("checked gating and QC", rita), folder, "reviewed")
    with pytest.raises(ValueError, match="Only a completed run"):
        sec.sign_run(context("again"), folder, "reviewed")  # cannot sign reviewed twice
    approved = sec.sign_run(context("release", rita), folder, "approved")
    assert approved.state == "approved" and approved.locked
    assert [s["meaning"] for s in approved.signatures] == ["reviewed", "approved"]
    assert approved.signatures[1]["display_name"] == "Rita Reviewer"
    # locked: refused, and the files are read-only
    with pytest.raises(signoff.RunLocked, match="locked"):
        ops.analyze_run(context("try again"), folder, GATE, include_run_steps=False)
    with pytest.raises(signoff.RunLocked):
        ops.annotate_run(context("sneaky"), folder, saved)
    with pytest.raises(signoff.RunLocked):
        ops.write_report(context(""), folder)
    mode = (folder / "run_manifest.json").stat().st_mode
    assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    report = (folder / "index.html").read_text()
    assert 'class="draftmark"' not in report and "Rita Reviewer" in report and "release" in report
    # re-analysis makes a linked new version
    with pytest.raises(Exception, match="reason"):
        ops.analyze_run(context(""), folder, GATE, include_run_steps=False, new_version=True)
    record = ops.analyze_run(
        context("new gate threshold"), folder, GATE, include_run_steps=False, new_version=True
    )
    new = Path(record["folder"])
    assert new.name == f"{folder.name}_v2" and new.exists()
    state = signoff.status(new)
    assert state.version == 2 and state.parent_run == signoff.run_key(folder) and not state.locked
    assert json.loads((new / "run_manifest.json").read_text())["parent_run"] == signoff.run_key(
        folder
    )
    assert signoff.status(folder).locked  # the original stays approved
    assert [v["version"] for v in signoff.versions_of(new)] == [1, 2]
    assert 'class="draftmark"' in (new / "index.html").read_text()
    # the audit log has every action, with user and reason
    events = audit.query()
    actions = [(e["type"], e["actor_id"], e["reason"], e["status"]) for e in events]
    expected = [
        ("layout.save", me, "plate map from the dispenser", "ok"),
        ("run.start", me, "first run of the plate", "ok"),
        ("run.annotate", me, "cell line corrected", "ok"),
        ("run.analyze", me, "gate big nuclei", "ok"),
        ("run.sign", me, "looks fine", "denied"),
        ("run.sign", "rita", "checked gating and QC", "ok"),
        ("run.sign", me, "again", "failed"),
        ("run.sign", "rita", "release", "ok"),
        ("run.analyze", me, "try again", "failed"),
        ("run.annotate", me, "sneaky", "failed"),
        ("run.report", me, "", "failed"),
        ("run.analyze", me, "new gate threshold", "ok"),
    ]
    for item in expected:
        assert item in actions, item
    assert all(e["actor_role"] for e in events)
    assert audit.verify_chain().ok
    run_rows = audit.run_audit(signoff.run_key(folder))
    assert {"run.start", "run.sign"} <= {r["type"] for r in run_rows}
    assert "Audit trail of this run" in (folder / "index.html").read_text()


def test_unlock_and_cli(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hcs_studio.app import main

    folder = _run(home)
    assert main(["run-state", str(folder)]) == 0
    assert main(["sign", str(folder), "--meaning", "approved"]) == 2  # not reviewed yet
    assert main(["sign", str(folder), "--meaning", "reviewed", "--comment", "ok"]) == 0
    assert main(["sign", str(folder), "--meaning", "approved"]) == 0
    assert signoff.status(folder).locked
    assert (
        main(
            ["analyze", str(folder), "--pipeline", str(_gate_file(home)), "--only", "--reason", "x"]
        )
        == 2
    )
    assert (
        main(
            [
                "analyze",
                str(folder),
                "--pipeline",
                str(_gate_file(home)),
                "--only",
                "--new-version",
                "--reason",
                "re-gate",
            ]
        )
        == 0
    )
    assert (folder.parent / f"{folder.name}_v2").exists()
    assert main(["unlock", str(folder), "--reason", "wrong plate map approved"]) == 0
    state = signoff.status(folder)
    assert state.state == "reviewed" and not state.locked
    assert (folder / "run_manifest.json").stat().st_mode & stat.S_IWUSR
    accounts.create_account(accounts.os_username(), "reviewer")
    assert main(["sign", str(folder), "--meaning", "approved"]) == 0
    assert main(["unlock", str(folder), "--reason", "again"]) == 2  # reviewers cannot unlock


def _gate_file(tmp: Path) -> Path:
    path = tmp / "gate.yaml"
    GATE.save(path)
    return path


def test_regulated_mode_requires_password_and_two_people(home: Path) -> None:
    from hcs_studio.config import load_settings, save_settings

    settings = load_settings()
    settings.security.regulated_mode = True
    save_settings(settings)
    assert load_settings().security.reauth_required
    accounts.create_account(accounts.os_username(), "analyst")  # runs the plate
    accounts.create_account("rita", "reviewer", "Rita", password="rita password")
    accounts.create_account("anna", "reviewer", "Anna", password="anna password")
    folder = _run(home)
    rita, anna = Actor("rita", kind="account"), Actor("anna", kind="account")
    with pytest.raises(PermissionError, match="password again"):
        sec.sign_run(context("", rita), folder, "reviewed")
    with pytest.raises(PermissionError, match="Re-authentication failed"):
        sec.sign_run(context("", rita), folder, "reviewed", password="wrong")
    sec.sign_run(context("", rita), folder, "reviewed", password="rita password")
    with pytest.raises(PermissionError, match="differ from the reviewer"):
        sec.sign_run(context("", rita), folder, "approved", password="rita password")
    result = sec.sign_run(context("", anna), folder, "approved", password="anna password")
    assert result.locked and all(s["reauthenticated"] for s in result.signatures)
    assert "password confirmed" in (folder / "index.html").read_text()


def test_states_of_unfinished_runs(home: Path, tmp_path: Path) -> None:
    run = tmp_path / "draft"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"status": "cancelled"}))
    assert signoff.status(run).state == "draft"
    with pytest.raises(ValueError, match="completed run"):
        signoff.sign(run, "reviewed", username="x")
    with pytest.raises(ValueError, match="not a run folder"):
        signoff.status(tmp_path)


def test_registry_backup(home: Path, tmp_path: Path) -> None:
    import sqlite3

    from hcs_studio.app import main

    _run(home)
    assert main(["db", "backup", "--output", str(tmp_path / "backups")]) == 0
    [folder] = list((tmp_path / "backups").iterdir())
    copy = next(folder.glob("*.sqlite"))
    with sqlite3.connect(copy) as connection:
        events = connection.execute("select count(*) from event").fetchone()[0]
    assert events >= 1 and next(folder.glob("audit_chain_head.jsonl")).exists()


@pytest.mark.gui
def test_sign_run_dialog(home: Path, qtbot: Any, inline_tasks: None) -> None:
    pytest.importorskip("PySide6")
    from hcs_studio.gui.security_dialogs import SignRunDialog

    folder = _run(home)
    dialog = SignRunDialog(folder)
    qtbot.addWidget(dialog)
    assert dialog.meaning.currentText() == "reviewed" and dialog.sign_button.isEnabled()
    assert "Reviewed:" in dialog.statement.text()
    dialog.comment.setText("all wells checked")
    dialog.sign()
    assert "reviewed" in dialog.status.text() and not dialog.sign_button.isEnabled()
    second = SignRunDialog(folder)
    qtbot.addWidget(second)
    assert second.meaning.currentText() == "approved"
    second.sign()
    assert "locked" in second.status.text()
    third = SignRunDialog(folder)
    qtbot.addWidget(third)
    assert not third.sign_button.isEnabled() and "locked" in third.summary.text()
