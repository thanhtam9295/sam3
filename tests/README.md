# Tests

| Directory | Marker | Contents |
|---|---|---|
| `tests/unit/` | `unit` | box ops, mask IoU/IoM, NMS, matching, association, RLE, overlap removal |
| `tests/integration/` | `integration` | multi-module pipelines; CLI/REST front ends; GPU model smoke test (`gpu`) |
| `tests/scientific/` | `scientific` | reference agreement, golden baselines, toy SA-Co/VEval end-to-end (`slow`) |
| `tests/tooling/` | `unit` | claims, scanner, validation guard, skill sync |
| `test/` | — | original upstream tests (unchanged) |
| `sam3/perflib/tests/` | — | upstream perflib tests (unchanged) |

```bash
python -m pytest -m unit -q          # seconds
python -m pytest -m "not slow" -q    # pre-commit
python -m pytest -q                  # everything runnable on this machine
```

Golden baselines: `tests/scientific/baselines/`, produced by
`tools/validation/regenerate_baselines.py` from `tests/scientific/cases.py`.
Do not edit by hand; see `docs/validation/SCIENTIFIC_VALIDATION.md`.
Strict `xfail` tests pin known defects (KI-004, KI-005, KI-007).
