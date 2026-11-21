"""Lab events in and out (Phase 19): validation, the event log, duplicates, HTTP and folders.

Every message is validated against the shared ``lab_events`` JSON Schemas; an unknown major
version or a malformed message is **rejected with the reasons** (HTTP 400, or the file moves
to ``rejected/`` with a ``.error.txt``). Every message — accepted, duplicate, rejected — is
written to the event log (table ``lab_event``).

A ``plate.imaged`` event registers an acquisition, keyed by barcode + acquisition ID (or the
acquisition start time when there is no ID): the same acquisition arriving twice (the same
event resent, or a watcher and the scheduler both reporting it) is logged as ``duplicate`` and
not run again; a re-imaged plate has a new acquisition ID / start time and runs again.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sqlalchemy import select

from hcs_studio import __version__
from hcs_studio.db import session_scope
from hcs_studio.db.automation import AutoRun, LabEventRecord

log = logging.getLogger("hcs_studio.automation")

SOURCE = {"system": "hcs-studio", "host": "", "version": __version__}


@dataclass
class Receipt:
    status: str  # accepted | duplicate | rejected | ignored
    detail: str = ""
    event_id: str = ""
    key: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "event_id": self.event_id,
                "acquisition_key": self.key, "errors": self.errors}  # fmt: skip


def acquisition_key(payload: dict[str, Any]) -> str:
    """barcode + acquisition ID (or start time) + time point: one automatic run per key."""
    identity = payload.get("acquisition_id") or payload.get("acquisition_start", "")
    key = f"{payload['barcode']}|{identity}"
    if payload.get("time_point_index") is not None:
        key += f"|t{int(payload['time_point_index'])}"
    return key


def raw_location(payload: dict[str, Any]) -> str:
    raw = payload["raw_data"]
    if raw.get("root") and raw.get("path"):
        return f"{str(raw['root']).upper()}:{raw['path']}"
    return str(raw.get("location", ""))


def _log(
    direction: str,
    channel: str,
    status: str,
    message: dict[str, Any] | None,
    detail: str = "",
    key: str = "",
) -> None:
    message = message or {}
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    with session_scope() as session:
        session.add(
            LabEventRecord(
                event_id=str(message.get("event_id", ""))[:64],
                direction=direction,
                type=str(message.get("type", ""))[:64],
                channel=channel,
                status=status,
                detail=detail,
                acquisition_key=key,
                barcode=str((payload or {}).get("barcode", ""))[:256],
                message=message,
            )
        )


def receive(message: dict[str, Any] | str | bytes, channel: str) -> Receipt:
    """Validate, log and (for ``plate.imaged``) register an incoming event."""
    from lab_events import InvalidEvent, validate

    try:
        event = validate(message)
    except InvalidEvent as exc:
        errors = list(exc.args[0]) if exc.args and isinstance(exc.args[0], list) else [str(exc)]
        raw: dict[str, Any] = {}
        if isinstance(message, dict):
            raw = message
        else:
            try:
                parsed = json.loads(message)
                raw = parsed if isinstance(parsed, dict) else {"raw": parsed}
            except ValueError:
                raw = {
                    "raw": message.decode("utf-8", "replace")
                    if isinstance(message, bytes)
                    else message
                }
        _log("in", channel, "rejected", raw, "; ".join(errors))
        log.warning("Rejected event from %s: %s", channel, "; ".join(errors))
        return Receipt("rejected", "the event does not match the lab_events schema",
                       str(raw.get("event_id", "")), errors=errors)  # fmt: skip
    event_id = str(event["event_id"])
    with session_scope() as session:
        seen = session.scalars(
            select(LabEventRecord).where(
                (LabEventRecord.event_id == event_id)
                & (LabEventRecord.direction == "in")
                & (LabEventRecord.status.in_(("accepted", "ignored")))
            )
        ).first()
    if seen is not None:
        _log("in", channel, "duplicate", event, "this event was received before")
        return Receipt("duplicate", "this event was received before", event_id)
    if event["type"] != "plate.imaged":
        _log("in", channel, "ignored" if event["type"] != "plate.dispensed" else "accepted", event)
        return Receipt("accepted" if event["type"] == "plate.dispensed" else "ignored",
                       f"{event['type']} recorded", event_id)  # fmt: skip
    payload = event["payload"]
    key = acquisition_key(payload)
    with session_scope() as session:
        existing = session.scalars(select(AutoRun).where(AutoRun.key == key)).first()
        if existing is None:
            session.add(
                AutoRun(
                    key=key,
                    barcode=payload["barcode"],
                    acquisition_id=str(payload.get("acquisition_id", "")),
                    instrument=payload["instrument"],
                    protocol=str(payload.get("protocol", "")),
                    time_point=payload.get("time_point_index"),
                    raw=raw_location(payload),
                    event_id=event_id,
                )
            )
    if existing is not None:
        detail = f"acquisition {key} was already received (run {existing.status})"
        _log("in", channel, "duplicate", event, detail, key)
        log.info("Duplicate acquisition ignored: %s", key)
        return Receipt("duplicate", detail, event_id, key)
    _log("in", channel, "accepted", event, "", key)
    log.info("Plate imaged: %s (%s)", payload["barcode"], key)
    return Receipt("accepted", "queued for automatic analysis", event_id, key)


# ------------------------------------------------------------------ outgoing


def send(
    event_type: str,
    payload: dict[str, Any],
    *,
    outbox: str = "",
    scheduler_url: str = "",
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build, validate, log and deliver an outgoing event (outbox file and/or HTTP POST)."""
    import socket

    from lab_events import make_event

    event = make_event(
        event_type, payload, source={**SOURCE, "host": socket.gethostname()},
        correlation_id=correlation_id,
    )  # fmt: skip
    status, detail = "sent", ""
    if outbox:
        folder = Path(outbox)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        target = folder / f"{stamp}_{event_type}_{event['event_id']}.json"
        part = target.with_name(target.name + ".part")
        part.write_text(json.dumps(event, indent=1), encoding="utf-8")
        part.replace(target)
        detail = f"written to {target.name}"
    if scheduler_url:
        request = urllib.request.Request(
            scheduler_url,
            data=json.dumps(event).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                detail = (detail + "; " if detail else "") + f"POST {response.status}"
        except (urllib.error.URLError, OSError) as exc:
            status, detail = (
                "send_failed",
                (detail + "; " if detail else "") + f"POST failed: {exc}",
            )
            log.warning("Could not send %s to the scheduler: %s", event_type, exc)
    _log("out", "outbox" if outbox else "http", status, event, detail)
    return event


# ------------------------------------------------------------------- folders


def poll_inbox(folder: Path, settle_seconds: float = 1.0) -> list[Receipt]:
    """Receive every ``*.json`` in ``folder`` (files older than ``settle_seconds``, so a file
    being written is not read half-way); move them to ``processed/`` or ``rejected/``."""
    import time

    folder = Path(folder)
    if not folder.is_dir():
        return []
    receipts = []
    now = time.time()
    for path in sorted(folder.glob("*.json")):
        try:
            if now - path.stat().st_mtime < settle_seconds:
                continue
            text = path.read_bytes()
        except OSError:
            continue
        receipt = receive(text, "inbox")
        target = folder / ("rejected" if receipt.status == "rejected" else "processed")
        target.mkdir(exist_ok=True)
        destination = target / path.name
        if destination.exists():
            destination = target / f"{path.stem}_{int(now)}{path.suffix}"
        path.replace(destination)
        if receipt.status == "rejected":
            destination.with_name(destination.name + ".error.txt").write_text(
                "\n".join(receipt.errors), encoding="utf-8"
            )
        receipts.append(receipt)
    return receipts


# ---------------------------------------------------------------------- HTTP


class _Handler(BaseHTTPRequestHandler):
    server_version = f"hcs-studio/{__version__}"

    def _reply(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._reply(200, {"status": "ok", "version": __version__})
        else:
            self._reply(404, {"error": "POST lab events to /events"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/events":
            self._reply(404, {"error": "POST lab events to /events"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            self._reply(413, {"error": "event too large"})
            return
        receipt = receive(self.rfile.read(length), "http")
        code = {"accepted": 202, "ignored": 202, "duplicate": 200, "rejected": 400}[receipt.status]
        self._reply(code, receipt.to_dict())

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("HTTP %s", format % args)


class EventServer:
    """The local HTTP API (``POST /events``, ``GET /health``) in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else str(host)
        return f"http://{name}:{port}/events"

    def start(self) -> EventServer:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
