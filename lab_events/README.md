# lab_events

The event contract between the lab scheduler (PF400 plate handler, washer, imagers) and
HCS Studio. The JSON Schema files in `src/lab_events/schemas/v1/` are the contract; any
language can validate against them. This package is the Python helper both projects import.

| Type | Direction | Meaning |
|---|---|---|
| `plate.imaged` | scheduler → HCS Studio | a plate finished imaging (barcode, instrument, protocol, times, raw data location, time point) |
| `plate.dispensed` | scheduler → HCS Studio | compounds were dispensed (treatment time) |
| `analysis.started` | HCS Studio → scheduler | analysis of an acquisition started |
| `analysis.completed` | HCS Studio → scheduler | analysis finished: QC verdict, report, key numbers |
| `analysis.failed` | HCS Studio → scheduler | analysis failed, at which stage, and why |
| `reimage.request` | HCS Studio → scheduler | image wells or fields of a plate again, with the reason |
| `cherrypick.request` | HCS Studio → scheduler | run a cherry-pick (source → destination wells, volumes) |

Every message:

```json
{
  "schema_version": "1.0",
  "event_id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
  "type": "plate.imaged",
  "emitted_at": "2026-11-19T14:02:11+00:00",
  "source": {"system": "scheduler", "host": "robot-pc", "version": "3.2"},
  "correlation_id": "optional",
  "payload": {"barcode": "P00123", "instrument": "Operetta CLS",
              "acquisition_start": "2026-11-19T13:10:00+00:00",
              "raw_data": {"root": "SCREENING", "path": "raw/P00123"}}
}
```

```python
from lab_events import make_event, validate

event = make_event(
    "reimage.request",
    {"barcode": "P00123", "reason": "out of focus", "wells": ["B03"]},
    source={"system": "hcs-studio"},
)
validate(event)  # raises InvalidEvent (with every problem) or UnsupportedVersion
```

Versions: the major version selects the schema folder; unknown majors are rejected with a
clear error. Minor versions only add optional fields: receivers accept unknown fields.
Times are RFC 3339 with a time zone. `event_id` is a UUID; receivers ignore duplicates.
