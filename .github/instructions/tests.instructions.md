---
applyTo: "tests/**,test/**"
---

* Mark every test module with a tier: `pytestmark = pytest.mark.unit` (or
  `integration`, `scientific`; add `gpu`/`slow` as needed).
* Tests must be deterministic, offline and checkpoint-free; synthesize inputs.
  Never add real or proprietary data, even small samples.
* Prefer comparing against an independent reference implementation over magic numbers.
* Do not remove or relax `xfail(strict=True)` markers: they document known
  defects (see `docs/KNOWN_ISSUES.md`).
* `test/` contains the original upstream tests: leave them in place.
