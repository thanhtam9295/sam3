"""Phase 19a: completeness checks per export format, the quiet-period folder watcher (a slow copy
is not started early), event intake over HTTP and the inbox folder, duplicates."""

from __future__ import annotations

import json
import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.builders import make_imagexpress, make_operetta
from hcs_studio.automation.completeness import check
from hcs_studio.automation.settings import WatchFolder
from hcs_studio.automation.watcher import FolderWatcher


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def _imaged(barcode: str = "AUTO-001", **payload: Any) -> dict[str, Any]:
    from lab_events import make_event

    body = {
        "barcode": barcode,
        "instrument": "ImageXpress",
        "acquisition_id": "acq-1",
        "acquisition_start": "2026-11-24T09:00:00+00:00",
        "raw_data": {"location": "/data/raw/AUTO-001"},
        **payload,
    }
    return make_event("plate.imaged", body, source={"system": "scheduler", "host": "lab", "version": "1"})


# --- completeness ---------------------------------------------------------------------------------


def test_imagexpress_completeness(tmp_path: Path) -> None:
    plate = make_imagexpress(tmp_path / "ix", [(0, 0), (0, 1)], n_sites=2)
    result = check(plate)
    assert result.complete and result.reader == "imagexpress"
    assert result.expected == result.found == 2 * 2 * 2  # wells × sites × wavelengths
    tiffs = sorted(p for p in plate.rglob("*.tif") if "_thumb" not in p.name.lower())
    tiffs[-1].unlink()
    missing = check(plate)
    assert not missing.complete and "7 of 8 images" in missing.reason
    # a half-written TIFF (pixel data cut off) is not complete either
    data = tiffs[0].read_bytes()
    tiffs[-1].write_bytes(data[: len(data) // 2])
    assert not check(plate).complete


def test_operetta_completeness(tmp_path: Path) -> None:
    plate = make_operetta(tmp_path / "op", [(1, 1)], n_fields=2, n_planes=1)
    assert check(plate).complete
    images = sorted((plate / "Images").glob("*.tiff")) or sorted(plate.rglob("*.tif*"))
    images[0].unlink()
    result = check(plate)
    assert not result.complete and "missing" in result.reason


def test_nd2_completeness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from hcs_studio.io import nikon

    file = tmp_path / "nd2" / "WellB02_ChannelDAPI_Seq0000.nd2"
    file.parent.mkdir()
    file.write_bytes(b"ND2 placeholder")
    state = {"frames": 4, "readable": 2}

    class Handle:
        attributes = SimpleNamespace(sequenceCount=state["frames"])

        def read_frame(self, index: int) -> None:
            if index >= state["readable"]:
                raise OSError("frame not written yet")

        def close(self) -> None:
            pass

    monkeypatch.setattr(nikon, "_open_nd2", lambda path: Handle())
    from hcs_studio.automation.completeness import nikon as nikon_check

    assert not nikon_check(file.parent).complete
    state["readable"] = 4
    assert nikon_check(file.parent).complete


# --- the watcher: a slow copy ---------------------------------------------------------------------


def test_slow_copy_is_not_started_early(home: Path) -> None:
    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import AutoRun

    source = make_imagexpress(home / "staging" / "AUTO-002", [(0, 0), (0, 1)], n_sites=2)
    watched = home / "exports"
    watched.mkdir()
    target = watched / "AUTO-002"
    watcher = FolderWatcher(WatchFolder(path=str(watched), instrument="ImageXpress",
                                        quiet_seconds=30))  # fmt: skip
    files = sorted(p for p in source.rglob("*") if p.is_file())
    htd = next(p for p in files if p.suffix.upper() == ".HTD")
    order = [htd] + [p for p in files if p != htd]  # MetaXpress writes the HTD first
    clock = 1000.0
    reported = []
    for i, file in enumerate(order):
        destination = target / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if i == len(order) // 2:  # a file appears half-written for a while
            data = file.read_bytes()
            destination.write_bytes(data[: len(data) // 3])
            for _ in range(3):
                clock += 40  # longer than the quiet period: the checker must catch it
                reported += watcher.scan(now=clock)
        shutil.copy2(file, destination)
        clock += 5  # copying: never quiet for 30 s
        reported += watcher.scan(now=clock)
    states = {c["state"] for c in watcher.status()}
    assert reported == [] and states <= {"copying", "waiting", "incomplete"}
    watcher.scan(now=clock + 10)  # still within the quiet period
    receipts = watcher.scan(now=clock + 45)
    assert receipts, watcher.status()
    assert [r.status for r in receipts] == ["accepted"]
    assert watcher.scan(now=clock + 100) == []  # reported once
    with session_scope() as session:
        runs = session.query(AutoRun).all()
        assert len(runs) == 1 and runs[0].instrument == "ImageXpress"
        assert runs[0].raw.endswith("exports/AUTO-002")
    # a watcher started later (after a restart) stays quiet about it
    fresh = FolderWatcher(WatchFolder(path=str(watched), quiet_seconds=0))
    assert fresh.scan(now=0) == [] and fresh.scan(now=10) == []


# --- intake ---------------------------------------------------------------------------------------


def test_duplicates_and_reimaging(home: Path) -> None:
    from hcs_studio.automation.intake import receive

    event = _imaged()
    assert receive(event, "http").status == "accepted"
    assert receive(event, "http").status == "duplicate"  # the same event resent
    again = _imaged()  # a new event for the same acquisition
    assert receive(again, "inbox").status == "duplicate"
    # the watcher reports the same raw data with its own acquisition ID
    watcher_event = _imaged(acquisition_id="AUTO-001@1764000000")
    assert receive(watcher_event, "watch").status == "duplicate"
    reimaged = _imaged(acquisition_id="acq-2", raw_data={"location": "/data/raw/AUTO-001_2"})
    assert receive(reimaged, "http").status == "accepted"
    time_point = _imaged(time_point_index=1, acquisition_id="acq-3")
    assert receive(time_point, "http").status == "accepted"
    bad = _imaged()
    bad["schema_version"] = "9.0"
    rejected = receive(bad, "http")
    assert rejected.status == "rejected" and rejected.errors


def test_http_and_inbox_intake(home: Path) -> None:
    from hcs_studio.automation.intake import EventServer, poll_inbox
    from hcs_studio.db import session_scope
    from hcs_studio.db.automation import LabEventRecord

    server = EventServer(port=0).start()
    try:
        request = urllib.request.Request(
            server.url, data=json.dumps(_imaged("HTTP-1")).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )  # fmt: skip
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 202
            assert json.loads(response.read())["status"] == "accepted"
        bad = urllib.request.Request(server.url, data=b"{not json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad, timeout=10)
        assert error.value.code == 400
    finally:
        server.stop()
    inbox = home / "inbox"
    inbox.mkdir()
    (inbox / "a.json").write_text(json.dumps(_imaged("FILE-1")))
    (inbox / "b.json").write_text('{"type": "plate.imaged"}')
    import os
    import time

    old = time.time() - 10
    for f in inbox.glob("*.json"):
        os.utime(f, (old, old))
    receipts = poll_inbox(inbox)
    assert sorted(r.status for r in receipts) == ["accepted", "rejected"]
    assert (inbox / "processed" / "a.json").exists()
    assert (inbox / "rejected" / "b.json.error.txt").read_text()
    with session_scope() as session:
        statuses = {(r.channel, r.status) for r in session.query(LabEventRecord)}
    assert {("http", "accepted"), ("http", "rejected"), ("inbox", "accepted"),
            ("inbox", "rejected")} <= statuses  # fmt: skip
