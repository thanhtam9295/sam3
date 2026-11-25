---
name: sam3-testing
description: How to set up a reproducible environment and run the SAM 3 test tiers (unit, integration, scientific, gpu, slow), interpret failures and add new tests. Use before committing any code change, when CI fails, or when asked to add or run tests.
---

# Running and writing SAM 3 tests

## Environment (once per machine)

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip sync requirements/lock/py312-dev.txt        # exact, reproducible versions
uv pip install -e . --no-deps
```

No `uv`? `python -m pip install -r requirements/lock/py312-dev.txt && pip install -e . --no-deps`.
See `docs/DEPENDENCIES.md`.

## Tiers

| Command | When |
|---|---|
| `python -m pytest -m unit -q` | every edit (~10 s) |
| `python -m pytest -m "not slow" -q` | before every commit |
| `python -m pytest -q` | before opening a PR (includes ~25 s toy video eval) |
| `python -m pytest -m scientific -q` | any change near algorithms; see `scientific-validation` skill |
| `python -m pytest -m gpu -q` | on a CUDA machine; auto-skipped elsewhere |

`test/` holds the original upstream tests; `tests/` holds the tiered suite;
`sam3/perflib/tests/` holds upstream perflib tests. All run via `pyproject.toml`.

## Reading results

* `xfailed` = a documented known defect (strict). Leave it alone.
* `XPASS(strict)` failure = you changed behaviour of a known defect. That is an
  algorithm change: follow `scientific-validation`.
* A golden-baseline failure means a number changed. Never regenerate baselines to
  make it pass.

## Writing tests

* Put tests in `tests/<tier>/test_*.py` and set `pytestmark = pytest.mark.<tier>`.
* Deterministic only: seeded RNG (the autouse fixture seeds numpy/torch/random).
* No network, no checkpoints, no real or proprietary data — synthesize inputs.
  Model tests needing weights must be `gpu` + `slow` and must skip cleanly.
* Prefer comparing to an independent reference (torchvision, pycocotools,
  scipy, brute force) over asserting a magic number.
