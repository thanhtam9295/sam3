"""Watch instrument export folders for finished plates (Phase 19.1, fallback trigger).

Each watched folder holds plate exports as sub-folders (ImageXpress ``.HTD`` + TIFFs,
Harmony exports with ``Images/Index.idx.xml``, folders of ``.nd2`` files) or single ``.nd2``
files. A candidate is reported as a ``plate.imaged`` event only when

1. nothing in it changed (file count, sizes, modification times) for ``quiet_seconds``, and
2. the format's completeness checker (:mod:`~hcs_studio.automation.completeness`) agrees,

so a plate that is still being copied is never started. Polling (no file-system
notifications) works the same on local disks and network shares, where change notifications
are unreliable. The event goes through :func:`~hcs_studio.automation.intake.receive` like a
scheduler event, so a plate reported by both the scheduler and the watcher runs once.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hcs_studio.automation.completeness import Completeness, check
from hcs_studio.automation.settings import WatchFolder

log = logging.getLogger("hcs_studio.automation")

Signature = tuple[int, int, float]


def signature(path: Path) -> Signature:
    """(files, bytes, newest modification time) of a file or folder tree."""
    if path.is_file():
        stat = path.stat()
        return (1, stat.st_size, stat.st_mtime)
    files = size = 0
    newest = 0.0
    for folder, _, names in os.walk(path):
        for name in names:
            try:
                stat = os.stat(os.path.join(folder, name))
            except OSError:  # removed while scanning: try again next time
                continue
            files += 1
            size += stat.st_size
            newest = max(newest, stat.st_mtime)
    return (files, size, newest)


@dataclass
class Candidate:
    path: Path
    first_seen: float
    last_change: float
    signature: Signature
    reported: bool = False
    state: str = "copying"  # copying | waiting | incomplete | reported | gave_up
    reason: str = ""
    checked: Completeness | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "state": self.state, "reason": self.reason,
                "files": self.signature[0], "bytes": self.signature[1]}  # fmt: skip


def plate_imaged_event(path: Path, watch: WatchFolder, newest: float) -> dict[str, Any]:
    """A ``plate.imaged`` event for a complete export (barcode and time from the reader)."""
    from lab_events import make_event

    from hcs_studio.automation.intake import SOURCE
    from hcs_studio.io.base import open_plate
    from hcs_studio.paths import to_rooted

    plate = open_plate(path)
    barcode = plate.barcode or plate.name or path.stem
    started = _aware(plate.acquired_at) or datetime.fromtimestamp(newest, UTC).isoformat()
    rooted = to_rooted(path)
    raw: dict[str, Any] = (
        {"location": str(path)} if rooted.root == "ABSOLUTE" else
        {"root": rooted.root, "path": rooted.rel}
    )  # fmt: skip
    payload: dict[str, Any] = {
        "barcode": barcode,
        "instrument": watch.instrument or plate.vendor or "unknown",
        "acquisition_id": f"{path.name}@{int(newest)}",
        "acquisition_start": started,
        "raw_data": raw,
    }
    if watch.protocol:
        payload["protocol"] = watch.protocol
    if plate.rows * plate.cols in (6, 24, 96, 384, 1536):
        payload["plate_type"] = plate.rows * plate.cols
    import socket

    return make_event("plate.imaged", payload, source={**SOURCE, "host": socket.gethostname()})


def _aware(text: str | None) -> str | None:
    """An ISO time with a time zone (instrument times without one are local time)."""
    if not text:
        return None
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return (value if value.tzinfo else value.astimezone()).isoformat()


@dataclass
class FolderWatcher:
    """Polls one export folder; :meth:`scan` reports finished plates."""

    watch: WatchFolder
    #: give up on an export that is still incomplete after this long (minutes)
    data_wait_minutes: float = 120.0
    candidates: dict[Path, Candidate] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        from hcs_studio.paths import resolve

        text = self.watch.path
        try:
            return resolve(text) if ":" in text.split("/", 1)[0] and len(text.split(":")[0]) > 1 else Path(text)
        except (KeyError, ValueError):
            return Path(text)

    def _children(self) -> list[Path]:
        root = self.root
        if not root.is_dir():
            return []
        out = []
        for child in sorted(root.iterdir()):
            if child.name.startswith((".", "~")) or child.name in ("processed", "rejected"):
                continue
            if child.is_dir() or child.suffix.lower() == ".nd2":
                out.append(child)
        return out

    def scan(self, now: float | None = None) -> list[Any]:
        """Check every candidate; send events for the finished ones. Returns the receipts."""
        from hcs_studio.automation.intake import already_registered, receive

        now = time.time() if now is None else now
        receipts = []
        present = set()
        for path in self._children():
            present.add(path)
            try:
                current = signature(path)
            except OSError:
                continue
            candidate = self.candidates.get(path)
            if candidate is None:
                candidate = Candidate(path, now, now, current)
                self.candidates[path] = candidate
                if already_registered(path):  # handled before a restart: stay quiet
                    candidate.reported, candidate.state = True, "reported"
                continue
            if current != candidate.signature:
                candidate.signature, candidate.last_change = current, now
                if candidate.reported:  # written again after being reported: a new export
                    candidate.reported, candidate.state = False, "copying"
                candidate.state = "copying"
                continue
            if candidate.reported or candidate.state == "gave_up":
                continue
            if now - candidate.last_change < self.watch.quiet_seconds:
                candidate.state = "waiting"
                continue
            completeness = check(path)
            candidate.checked = completeness
            if not completeness.complete:
                candidate.state, candidate.reason = "incomplete", completeness.reason
                if now - candidate.first_seen > self.data_wait_minutes * 60:
                    candidate.state = "gave_up"
                    log.warning("Gave up waiting for %s: %s", path, completeness.reason)
                continue
            try:
                event = plate_imaged_event(path, self.watch, candidate.signature[2])
            except Exception as exc:
                candidate.state, candidate.reason = "incomplete", f"does not open: {exc}"
                continue
            receipts.append(receive(event, "watch"))
            candidate.reported, candidate.state, candidate.reason = True, "reported", ""
        for gone in set(self.candidates) - present:
            del self.candidates[gone]
        return receipts

    def status(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.candidates.values()]
