"""The event contract between the lab scheduler and HCS Studio.

Messages are JSON objects with an envelope (``schema_version``, ``event_id``, ``type``,
``emitted_at``, ``source``, optional ``correlation_id``) and a ``payload`` defined per type
by the JSON Schemas in ``schemas/v<major>/``. Both sides validate every message:

>>> event = make_event("reimage.request", {"barcode": "P1", "reason": "blurred"},
...                    source={"system": "hcs-studio"})
>>> validate(event)["type"]
'reimage.request'

Versioning: the major version selects the schema folder; a message with an unknown major
version is rejected (:class:`UnsupportedVersion`). Minor versions only add optional fields,
so receivers accept fields they do not know.

The schema files are the language-neutral contract; this module is the Python helper.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from typing import Any

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJORS = (1,)
EVENT_TYPES = (
    "plate.imaged",
    "plate.dispensed",
    "analysis.started",
    "analysis.completed",
    "analysis.failed",
    "reimage.request",
    "cherrypick.request",
)
#: Payload fields holding times (checked as RFC 3339 with a time zone).
TIME_FIELDS = {
    "plate.imaged": ("acquisition_start", "acquisition_end"),
    "plate.dispensed": ("dispensed_at",),
    "analysis.started": ("started_at",),
    "analysis.completed": ("finished_at",),
    "analysis.failed": ("finished_at",),
}
__version__ = "1.0.0"
__all__ = [
    "EVENT_TYPES",
    "SCHEMA_VERSION",
    "InvalidEvent",
    "UnsupportedVersion",
    "make_event",
    "schema",
    "validate",
]


class InvalidEvent(ValueError):
    """The message does not follow the schema; ``errors`` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("Invalid lab event: " + "; ".join(errors))
        self.errors = errors


class UnsupportedVersion(InvalidEvent):
    pass


@cache
def schema(name: str, major: int = 1) -> dict[str, Any]:
    """A schema of version ``major``: ``"envelope"`` or an event type."""
    folder = resources.files("lab_events") / "schemas" / f"v{major}"
    data: dict[str, Any] = json.loads((folder / f"{name}.schema.json").read_text("utf-8"))
    return data


def _errors(instance: Any, schema_: dict[str, Any], where: str) -> list[str]:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema_)
    out = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path)):
        path = "/".join(str(p) for p in error.absolute_path)
        out.append(f"{where}{('/' + path) if path else ''}: {error.message}")
    return out


def _check_time(value: Any, where: str) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return [f"{where}: {value!r} is not an RFC 3339 date-time"]
    return [] if parsed.tzinfo is not None else [f"{where}: {value!r} has no time zone"]


def validate(message: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """Validate a message (dict or JSON text) and return it as a dict."""
    if isinstance(message, (str, bytes)):
        try:
            message = json.loads(message)
        except ValueError as exc:
            raise InvalidEvent([f"not JSON: {exc}"]) from exc
    if not isinstance(message, dict):
        raise InvalidEvent(["a lab event is a JSON object"])
    version = str(message.get("schema_version", ""))
    try:
        major = int(version.split(".")[0])
    except ValueError:
        raise InvalidEvent([f"schema_version {version!r} is not MAJOR.MINOR"]) from None
    if major not in SUPPORTED_MAJORS:
        raise UnsupportedVersion(
            [
                f"schema_version {version} is not supported (this side understands "
                f"{', '.join(f'{m}.x' for m in SUPPORTED_MAJORS)}); update lab_events"
            ]
        )
    errors = _errors(message, schema("envelope", major), "envelope")
    if errors:
        raise InvalidEvent(errors)
    try:
        uuid.UUID(str(message["event_id"]))
    except ValueError:
        errors.append(f"event_id: {message['event_id']!r} is not a UUID")
    errors += _check_time(message["emitted_at"], "emitted_at")
    kind = message["type"]
    payload = message["payload"]
    errors += _errors(payload, schema(kind, major), f"payload ({kind})")
    for name in TIME_FIELDS.get(kind, ()):
        if name in payload:
            errors += _check_time(payload[name], f"payload/{name}")
    if errors:
        raise InvalidEvent(errors)
    return message


def make_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    source: dict[str, str],
    correlation_id: str | None = None,
    emitted_at: datetime | None = None,
) -> dict[str, Any]:
    """Build and validate an event (a new UUID; the current UTC time by default)."""
    if event_type not in EVENT_TYPES:
        raise InvalidEvent([f"unknown event type {event_type!r} ({', '.join(EVENT_TYPES)})"])
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "type": event_type,
        "emitted_at": (emitted_at or datetime.now(UTC)).isoformat(),
        "source": source,
        "payload": payload,
    }
    if correlation_id:
        event["correlation_id"] = correlation_id
    return validate(event)
