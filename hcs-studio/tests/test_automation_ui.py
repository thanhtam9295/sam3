"""Phase 19c: the ``hcs-studio auto`` command line, the retry service (audited, reason
required) and the Automation tab with the event-log viewer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _event(barcode: str, location: str) -> dict[str, Any]:
    from lab_events import make_event

    return make_event(
        "plate.imaged",
        {
            "barcode": barcode,
            "instrument": "ImageXpress",
            "acquisition_id": "a1",
            "acquisition_start": "2026-11-24T09:00:00+00:00",
            "raw_data": {"location": location},
        },
        source={"system": "scheduler", "host": "lab", "version": "1"},
    )


def _configure(home: Path) -> None:
    from hcs_studio.automation.settings import AutomationSettings, WatchFolder
    from hcs_studio.config import load_settings, save_settings

    (home / "p.yaml").write_text(yaml.safe_dump({"steps": [{"module": "qc"}]}), encoding="utf-8")
    (home / "routing.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {"name": "all", "default": True, "pipeline": "p.yaml"},
                    {"name": "late", "barcode_prefix": "X", "pipeline": "p.yaml"},
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings()
    settings.automation = AutomationSettings(
        enabled=True,
        routing_file=str(home / "routing.yaml"),
        watch=[WatchFolder(path=str(home / "missing"))],
    )
    save_settings(settings)


def test_cli(home: Path, capsys: Any) -> None:
    from hcs_studio.app import main

    _configure(home)
    event = home / "event.json"
    event.write_text(json.dumps(_event("CLI-1", str(home / "raw" / "CLI-1"))), encoding="utf-8")
    assert main(["auto", "send", str(event)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"
    assert main(["auto", "send", str(event)]) == 0  # the same event again: a duplicate
    assert json.loads(capsys.readouterr().out)["status"] == "duplicate"
    (home / "bad.json").write_text('{"type": "plate.imaged"}', encoding="utf-8")
    assert main(["auto", "send", str(home / "bad.json")]) == 1
    capsys.readouterr()
    assert main(["auto", "events", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["status"] for r in rows] == ["rejected", "duplicate", "accepted"]
    assert main(["auto", "check"]) == 1
    out = capsys.readouterr().out
    assert "late: never used" in out and "folder not found" in out
    assert main(["auto", "serve", "--once"]) == 0  # the data is not there: it waits
    capsys.readouterr()
    assert main(["auto", "status"]) == 0
    assert "waiting_data" in capsys.readouterr().out


def test_retry_needs_a_reason_and_is_audited(home: Path) -> None:
    from sqlalchemy import select

    from hcs_studio.automation.intake import receive
    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import AutoRun
    from hcs_studio.services import ReasonRequired, context, recent_events
    from hcs_studio.services.automation import retry_auto_run

    receipt = receive(_event("R-1", "/nowhere/R-1"), "http")
    with session_scope() as session:
        run = session.scalars(select(AutoRun)).one()
        run.status = "failed"
    with pytest.raises(ReasonRequired):
        retry_auto_run(context(), receipt.key)
    retry_auto_run(context("data copied again"), receipt.key)
    with session_scope() as session:
        assert session.scalars(select(AutoRun)).one().status == "pending"
    event = recent_events(5, "automation.retry")[0]
    assert event["reason"] == "data copied again" and event["before"]["status"] == "failed"


@pytest.mark.gui
def test_automation_tab(home: Path, qtbot: Any, inline_tasks: Any) -> None:
    pytest.importorskip("PySide6")
    from hcs_studio.automation.intake import receive, send
    from hcs_studio.gui.automation_panel import AutomationPanel

    receive(_event("TAB-1", "/nowhere/TAB-1"), "http")
    receive({"type": "nonsense"}, "inbox")
    send(
        "analysis.failed",
        {
            "barcode": "TAB-1",
            "stage": "qc",
            "error": "blurred",
            "finished_at": "2026-11-24T10:00:00+00:00",
        },
    )
    panel = AutomationPanel()
    qtbot.addWidget(panel)
    opened: list[Path] = []
    panel.open_run.connect(opened.append)
    panel.refresh()
    assert panel.run_table.rowCount() == 1 and "1 pending" in panel.summary.text()
    assert panel.event_table.rowCount() == 3
    panel.status_filter.setCurrentText("rejected")
    assert panel.event_table.rowCount() == 1
    panel.event_table.selectRow(0)
    assert "nonsense" in panel.message.toPlainText()
    panel.status_filter.setCurrentText("all")
    panel.direction_filter.setCurrentText("out")
    assert [e["type"] for e in panel.visible_events()] == ["analysis.failed"]
    panel.run_table.selectRow(0)
    panel.reason = lambda key: None  # cancelled: nothing happens
    panel.retry()
    panel.reason = lambda key: "after fixing the share"
    panel.retry()
    from hcs_studio.services import recent_events

    assert recent_events(5, "automation.retry")[0]["reason"] == "after fixing the share"
