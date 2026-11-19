"""The lab_events contract (PROMPT_PLATFORM.md §2.7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from lab_events import EVENT_TYPES, InvalidEvent, UnsupportedVersion, make_event, schema, validate

SOURCE = {"system": "scheduler", "host": "robot-pc", "version": "3.2"}
EXAMPLES: dict[str, dict[str, Any]] = {
    "plate.imaged": {
        "barcode": "P00123",
        "instrument": "Operetta CLS",
        "protocol": "20x confocal 3ch",
        "acquisition_start": "2026-11-19T13:10:00+00:00",
        "acquisition_end": "2026-11-19T13:52:00Z",
        "time_point_index": 0,
        "plate_type": 384,
        "raw_data": {"root": "SCREENING", "path": "raw/P00123"},
    },
    "plate.dispensed": {"barcode": "P00123", "dispensed_at": "2026-11-18T09:00:00+01:00"},
    "analysis.started": {"barcode": "P00123", "run_id": "r1", "started_at": "2026-11-19T14:00:00Z"},
    "analysis.completed": {
        "barcode": "P00123",
        "run_id": "r1",
        "qc_verdict": "pass",
        "report_path": "RESULTS:P00123/index.html",
        "key_numbers": {"z_prime": 0.71, "mean_cell_count": 412.0},
        "finished_at": "2026-11-19T14:20:00Z",
    },
    "analysis.failed": {
        "barcode": "P00123",
        "stage": "qc",
        "error": "plate failed QC",
        "finished_at": "2026-11-19T14:05:00Z",
    },
    "reimage.request": {
        "barcode": "P00123",
        "reason": "out of focus",
        "wells": ["B03", "AF48"],
        "fields": [{"well": "B03", "field": 2}],
    },
    "cherrypick.request": {
        "destination_plate": "CONF-001",
        "picks": [
            {
                "source_plate": "LIB-7",
                "source_well": "C05",
                "destination_well": "A01",
                "volume_nl": 25,
            }
        ],
    },
}


def test_every_event_type_has_a_schema_and_a_valid_example() -> None:
    assert (
        set(EXAMPLES) == set(EVENT_TYPES) == set(schema("envelope")["properties"]["type"]["enum"])
    )
    for kind, payload in EXAMPLES.items():
        event = make_event(kind, payload, source=SOURCE, correlation_id="c-1")
        assert validate(json.dumps(event)) == event  # round trip through JSON text
        assert event["schema_version"] == "1.0" and event["correlation_id"] == "c-1"


def test_invalid_messages_report_every_problem() -> None:
    good = make_event("plate.imaged", EXAMPLES["plate.imaged"], source=SOURCE)
    bad = json.loads(json.dumps(good))
    del bad["payload"]["barcode"]
    bad["payload"]["time_point_index"] = -1
    bad["payload"]["raw_data"] = {}
    with pytest.raises(InvalidEvent) as caught:
        validate(bad)
    text = " | ".join(caught.value.errors)
    assert "'barcode' is a required property" in text and "-1 is less than" in text
    assert len(caught.value.errors) == 3
    for change, message in (
        ({"event_id": "not-a-uuid"}, "not a UUID"),
        ({"emitted_at": "2026-11-19 14:00"}, "no time zone"),
        ({"emitted_at": "yesterday"}, "not an RFC 3339"),
        ({"type": "plate.washed"}, "is not one of"),
        ({"source": {}}, "'system' is a required property"),
    ):
        with pytest.raises(InvalidEvent, match=message):
            validate({**good, **change})
    timed = json.loads(json.dumps(good))
    timed["payload"]["acquisition_start"] = "13:10"
    with pytest.raises(InvalidEvent, match="payload/acquisition_start"):
        validate(timed)
    with pytest.raises(InvalidEvent, match="not JSON"):
        validate("{not json")
    with pytest.raises(InvalidEvent, match="unknown event type"):
        make_event("plate.washed", {}, source=SOURCE)


def test_versions_unknown_majors_rejected_minors_and_extra_fields_accepted() -> None:
    event = make_event(
        "plate.dispensed",
        EXAMPLES["plate.dispensed"],
        source=SOURCE,
        emitted_at=datetime(2026, 11, 19, tzinfo=UTC),
    )
    newer_minor = {**event, "schema_version": "1.4", "priority": "high"}
    newer_minor["payload"] = {**event["payload"], "dispense_volume_nl": 50}
    assert validate(newer_minor)["schema_version"] == "1.4"  # additive changes are fine
    with pytest.raises(UnsupportedVersion, match="update lab_events"):
        validate({**event, "schema_version": "2.0"})
    with pytest.raises(InvalidEvent, match=r"MAJOR\.MINOR"):
        validate({**event, "schema_version": "one"})
