"""Phase 19 done-when: a plate export dropped into a watched folder (or a scheduler event)
triggers QC → analysis → report → email with no clicks; a duplicate event does not start a
second run; a plate that fails QC produces a schema-valid ``reimage.request``. Real job
processes, a local SMTP server, the outbox read back through ``lab_events``."""

from __future__ import annotations

import json
import shutil
import socketserver
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hcs_studio.automation.notify import Notice, Notifier
from hcs_studio.automation.routing import Routing, Rule, check_routing, load_routing
from hcs_studio.automation.runner import AutoRunner, is_night
from hcs_studio.automation.settings import AutomationSettings, Channel, WatchFolder
from hcs_studio.pipeline.schema import Pipeline, PipelineStep

PIPELINE = Pipeline(
    steps=[
        PipelineStep(module="segment_classical", params={"channel": 1}),
        PipelineStep(module="measure_objects", params={"labels": ["nuclei"], "texture": False}),
        PipelineStep(module="qc"),
    ]
).to_dict()
WELLS = [(0, c) for c in range(6)]  # A01, A02 vehicle; A03, A04 samples; A05, A06 positive


def _decoded(raw: str) -> str:
    """Headers + decoded text body: quoted-printable wraps long lines (e.g. Windows paths)."""
    import email
    from email import policy

    message = email.message_from_string(raw, policy=policy.default)
    headers = "".join(f"{key}: {value}\n" for key, value in message.items())
    parts = [part for part in message.walk() if part.get_content_maintype() == "text"]
    return headers + "\n" + "\n".join(str(part.get_content()) for part in parts)


class Mailbox(socketserver.ThreadingTCPServer):
    """A tiny SMTP server that keeps the messages it receives."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        super().__init__(("127.0.0.1", 0), _SmtpHandler)


class _SmtpHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        def say(line: str) -> None:
            self.wfile.write((line + "\r\n").encode())

        say("220 test SMTP")
        recipients: list[str] = []
        while True:
            line = self.rfile.readline().decode("utf-8", "replace").strip()
            if not line:
                return
            verb = line.split(" ", 1)[0].upper()
            if verb in ("EHLO", "HELO"):
                say("250 ok")
            elif verb == "MAIL":
                recipients = []
                say("250 ok")
            elif verb == "RCPT":
                recipients.append(line.split(":", 1)[1].strip(" <>"))
                say("250 ok")
            elif verb == "DATA":
                say("354 end with .")
                body = []
                while True:
                    data = self.rfile.readline().decode("utf-8", "replace")
                    if data.rstrip("\r\n") == ".":
                        break
                    body.append(data)
                assert isinstance(self.server, Mailbox)
                self.server.messages.append({"to": recipients, "data": _decoded("".join(body))})
                say("250 queued")
            elif verb == "QUIT":
                say("221 bye")
                return
            else:
                say("250 ok")


def _nuclei(row: int, col: int, site: int, wave: int, z: int) -> Any:
    """64 × 64 field with discs: 12 in vehicle/sample wells, 3 in positive controls."""
    image = np.full((64, 64), 100, dtype=np.uint16)
    n = 3 if col >= 4 else 12
    yy, xx = np.mgrid[:64, :64]
    for i in range(n):
        cy, cx = 8 + 16 * (i // 4), 8 + 16 * (i % 4)
        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16] = 2000 if wave == 0 else 300
    return image


def _export(folder: Path, barcode: str) -> Path:
    from hcs_studio.utils.synthetic import write_imagexpress_plate

    plate = write_imagexpress_plate(folder, WELLS, _nuclei, n_sites=1)
    htd = plate / "Plate1.HTD"
    htd.write_text(
        htd.read_text(encoding="latin-1").replace("Screen 7", barcode), encoding="latin-1"
    )
    return plate


def _save_layout(barcode: str) -> None:
    from hcs_studio.io.layout_store import LayoutStore, plate_key
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role

    layout = PlateLayout.for_plate_type(96, barcode=barcode)
    layout = set_role(
        layout, [(0, 0), (0, 1)], "negative_control", vehicle="DMSO", cell_line="HeLa"
    )
    layout = set_role(layout, [(0, 4), (0, 5)], "positive_control", cell_line="HeLa")
    LayoutStore(plate_key(barcode=barcode)).save(layout, note="test")


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    from hcs_studio.db.engine import dispose_engines
    from hcs_studio.qc.rules import FieldRule, PlateDecision, QcRules, save_rules

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    config = tmp_path / "config"
    config.mkdir()
    (config / "basic.yaml").write_text(yaml.safe_dump(PIPELINE), encoding="utf-8")
    failing = QcRules(
        name="always fails",
        field=[FieldRule(flag="too_bright", metric="p50", above=0, description="test")],
        plate=PlateDecision(stop_run=True, min_z_prime=None),
    )
    save_rules(failing, config / "failing_qc.yaml")
    routing = {
        "rules": [
            {
                "name": "qc test",
                "barcode_prefix": "AUTO-QC",
                "pipeline": "basic.yaml",
                "qc_rules": "failing_qc.yaml",
                "notify": ["owner@lab.org"],
            },
            {
                "name": "viability",
                "barcode_regex": "^AUTO-E2E",
                "instrument": "ImageXpress",
                "pipeline": "basic.yaml",
                "notify": ["owner@lab.org"],
            },
        ]
    }
    (config / "routing.yaml").write_text(yaml.safe_dump(routing), encoding="utf-8")
    mailbox = Mailbox()
    threading.Thread(target=mailbox.serve_forever, daemon=True).start()
    for name in ("exports", "inbox", "outbox", "results"):
        (tmp_path / name).mkdir()
    settings = AutomationSettings(
        enabled=True,
        inbox=str(tmp_path / "inbox"),
        outbox=str(tmp_path / "outbox"),
        watch=[
            WatchFolder(path=str(tmp_path / "exports"), instrument="ImageXpress", quiet_seconds=0)
        ],
        routing_file=str(config / "routing.yaml"),
        output_root=str(tmp_path / "results"),
        workers=1,
        channels=[
            Channel(
                name="lab mail",
                kind="email",
                to=["lab@lab.org"],
                smtp_host="127.0.0.1",
                smtp_port=mailbox.server_address[1],
                smtp_tls=False,
            )
        ],
        admins=["admin@lab.org"],
    )
    yield {"root": tmp_path, "settings": settings, "mailbox": mailbox, "config": config}
    mailbox.shutdown()
    mailbox.server_close()
    dispose_engines()


def _run_until(runner: AutoRunner, key_prefix: str, timeout: float = 240) -> Any:
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import AutoRun

    deadline = time.time() + timeout
    while time.time() < deadline:
        runner.step()
        with session_scope() as session:
            runs = session.scalars(select(AutoRun).where(AutoRun.barcode.startswith(key_prefix)))
            done = [r for r in runs if r.status not in ("pending", "waiting_data", "running")]
            if done:
                session.expunge_all()
                return done[0]
        time.sleep(0.5)
    raise AssertionError(f"no automatic run of {key_prefix} finished")


def _outbox(root: Path) -> list[dict[str, Any]]:
    from lab_events import validate

    events = []
    for path in sorted((root / "outbox").glob("*.json")):
        events.append(
            validate(json.loads(path.read_text(encoding="utf-8")))
        )  # every outgoing event is valid
    return events


def test_dropped_export_runs_without_clicks(lab: dict[str, Any]) -> None:
    from hcs_studio.automation.intake import receive
    from hcs_studio.pipeline.queue import LocalJobQueue

    root, settings, mailbox = lab["root"], lab["settings"], lab["mailbox"]
    _save_layout("AUTO-E2E-1")
    staged = _export(root / "staging" / "AUTO-E2E-1", "AUTO-E2E-1")
    queue = LocalJobQueue(persist=False)
    runner = AutoRunner(settings, queue)
    shutil.copytree(staged, root / "exports" / "AUTO-E2E-1")  # the instrument's export
    try:
        run = _run_until(runner, "AUTO-E2E")
    finally:
        queue.shutdown()
    assert run.status == "completed", run.detail
    folder = Path(run.run_folder)
    assert (folder / "index.html").exists() and folder.is_relative_to(root / "results")
    assert (folder / "plate_layout.json").exists()  # the layout was attached by barcode
    events = _outbox(root)
    assert [e["type"] for e in events] == ["analysis.started", "analysis.completed"]
    completed = events[1]["payload"]
    assert completed["barcode"] == "AUTO-E2E-1" and completed["qc_verdict"] in ("pass", "warn")
    assert completed["key_numbers"]["mean nuclei_count"] == pytest.approx(9.0, abs=0.01)
    assert completed["key_numbers"]["Z′ nuclei_count"] is not None
    # the email reached the lab list and the rule's owner
    assert len(mailbox.messages) == 1
    mail = mailbox.messages[0]
    assert set(mail["to"]) == {"lab@lab.org", "owner@lab.org"}
    assert "AUTO-E2E-1" in mail["data"] and "analysis completed" in mail["data"]
    assert "index.html" in mail["data"] and "mean nuclei_count: 9" in mail["data"]
    # the scheduler reports the same plate afterwards: a duplicate, no second run
    from lab_events import make_event

    raw = str(root / "exports" / "AUTO-E2E-1")
    late = make_event(
        "plate.imaged",
        {
            "barcode": "AUTO-E2E-1",
            "instrument": "ImageXpress",
            "acquisition_id": "sched-77",
            "acquisition_start": "2026-11-24T09:00:00+00:00",
            "raw_data": {"location": raw},
        },
        source={"system": "scheduler", "host": "lab", "version": "1"},
    )
    assert receive(late, "http").status == "duplicate"
    runner.step()
    assert len(queue.states) == 1


def test_failing_qc_requests_reimaging(lab: dict[str, Any]) -> None:
    from lab_events import make_event

    from hcs_studio.pipeline.queue import LocalJobQueue

    root, settings, mailbox = lab["root"], lab["settings"], lab["mailbox"]
    raw = _export(root / "raw" / "AUTO-QC-1", "AUTO-QC-1")
    event = make_event(
        "plate.imaged",
        {
            "barcode": "AUTO-QC-1",
            "instrument": "ImageXpress",
            "acquisition_id": "q1",
            "acquisition_start": "2026-11-24T10:00:00+00:00",
            "raw_data": {"location": str(raw)},
        },
        source={"system": "scheduler", "host": "lab", "version": "1"},
    )
    import os

    inbox = root / "inbox" / "event.json"
    inbox.write_text(json.dumps(event), encoding="utf-8")
    os.utime(inbox, (time.time() - 10, time.time() - 10))
    queue = LocalJobQueue(persist=False)
    try:
        run = _run_until(AutoRunner(settings, queue), "AUTO-QC")
    finally:
        queue.shutdown()
    assert run.status == "qc_failed", run.detail
    events = _outbox(root)
    types = [e["type"] for e in events]
    assert types == ["analysis.started", "reimage.request", "analysis.failed"]
    reimage = events[1]["payload"]
    assert reimage["barcode"] == "AUTO-QC-1" and reimage["acquisition_id"] == "q1"
    assert len(reimage["wells"]) == 6 and len(reimage["fields"]) == 6
    assert events[1]["correlation_id"] == event["event_id"]
    assert events[2]["payload"]["stage"] == "qc"
    # no layout for this barcode: owners were also asked to annotate it
    kinds = [m["data"] for m in mailbox.messages]
    assert any("please annotate this plate" in m for m in kinds)
    failed = next(m for m in mailbox.messages if "failed image QC" in m["data"])
    assert set(failed["to"]) == {"lab@lab.org", "owner@lab.org", "admin@lab.org"}
    assert "re-imaging request" in failed["data"]


def test_unrouted_and_waiting_data(lab: dict[str, Any]) -> None:
    from lab_events import make_event
    from sqlalchemy import select

    from hcs_studio.automation.intake import receive
    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import AutoRun
    from hcs_studio.pipeline.queue import LocalJobQueue

    root, settings = lab["root"], lab["settings"]
    source = {"system": "scheduler", "host": "lab", "version": "1"}
    unknown = _export(root / "raw" / "OTHER-1", "OTHER-1")
    receive(
        make_event(
            "plate.imaged",
            {
                "barcode": "OTHER-1",
                "instrument": "ImageXpress",
                "acquisition_id": "o1",
                "acquisition_start": "2026-11-24T10:00:00+00:00",
                "raw_data": {"location": str(unknown)},
            },
            source=source,
        ),
        "http",
    )
    # announced before the data arrived: waits for it
    receive(
        make_event(
            "plate.imaged",
            {
                "barcode": "AUTO-E2E-9",
                "instrument": "ImageXpress",
                "acquisition_id": "w1",
                "acquisition_start": "2026-11-24T10:00:00+00:00",
                "raw_data": {"location": str(root / "raw" / "AUTO-E2E-9")},
            },
            source=source,
        ),
        "http",
    )
    runner = AutoRunner(settings.model_copy(update={"watch": []}), LocalJobQueue(persist=False))
    runner.step()
    with session_scope() as session:
        status = {r.barcode: r.status for r in session.scalars(select(AutoRun))}
    assert status == {"OTHER-1": "unrouted", "AUTO-E2E-9": "waiting_data"}
    assert any("no routing rule matches" in m["data"] for m in lab["mailbox"].messages)


def test_routing_rules(tmp_path: Path) -> None:
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(PIPELINE), encoding="utf-8")
    routing = Routing(
        base=str(tmp_path),
        rules=[
            Rule(name="screen", barcode_prefix="SCR", protocol="20x*", pipeline="p.yaml"),
            Rule(name="by assay", assay="viability", pipeline="p.yaml"),
            Rule(name="nikon", instrument="nikon ji", pipeline="missing.yaml"),
            Rule(name="default", default=True, pipeline="p.yaml"),
            Rule(name="after default", barcode_prefix="X", pipeline="p.yaml"),
        ],
    )
    route = routing.route
    assert route({"barcode": "scr-001", "protocol": "20X_4ch"}).name == "screen"  # type: ignore[union-attr]
    assert route({"barcode": "SCR-001", "protocol": "10x"}).name == "default"  # type: ignore[union-attr]
    assert route({"barcode": "P1", "assay": "Viability"}).name == "by assay"  # type: ignore[union-attr]
    assert route({"barcode": "P1", "instrument": "Nikon Ji"}).name == "nikon"  # type: ignore[union-attr]
    problems = check_routing(routing)
    assert any("missing.yaml does not exist" in p for p in problems)
    assert any("after default: never used" in p for p in problems)
    (tmp_path / "r.yaml").write_text(
        yaml.safe_dump({"rules": [{"name": "d", "default": True, "pipeline": "p.yaml"}]}),
        encoding="utf-8",
    )
    assert check_routing(load_routing(tmp_path / "r.yaml")) == []
    with pytest.raises(ValueError):
        Rule(name="bad", barcode_regex="(", pipeline="p.yaml")


def test_night_hours_and_secret_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import datetime

    assert is_night("20-7", datetime(2026, 1, 1, 23)) and is_night("20-7", datetime(2026, 1, 1, 3))
    assert not is_night("20-7", datetime(2026, 1, 1, 12)) and not is_night("", None)
    assert is_night("1-5", datetime(2026, 1, 1, 2)) and not is_night("1-5", datetime(2026, 1, 1, 6))
    # a webhook must be a keychain reference; a missing secret is reported, not sent in clear
    channel = Channel(name="teams", kind="teams", webhook="https://example.org/hook")
    notifier = Notifier([channel])
    notifier.notify(Notice("failed", "P1", "x"))
    assert notifier.sent[0][2] == "failed" and "secret:" in notifier.sent[0][3]
    monkeypatch.delenv("HCS_SECRET_TEAMS_HOOK", raising=False)
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda *a: None)
    notifier = Notifier([Channel(name="teams", kind="teams", webhook="secret:teams/hook")])
    notifier.notify(Notice("failed", "P1", "x"))
    assert notifier.sent[0][2] == "failed" and "no webhook stored" in notifier.sent[0][3]
