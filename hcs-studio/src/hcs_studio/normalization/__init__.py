"""Normalization across plates and days, uniformity correction and replicate checks
(Phase 17b). Headless: functions take per-well values and annotations and return plain
data; ``db/queries`` feeds them from the registry, the GUI and the command line show them.

- :mod:`.plate` — per-plate methods (control-based, robust Z, B-score) and the guard rail
  that blocks row/column corrections on dose–response and combination layouts.
- :mod:`.uniformity` — spatial correction maps from vehicle-only (uniformity) plates.
- :mod:`.batch` — normalization to same-day controls; day and plate effects before/after.
- :mod:`.concordance` — replicate-plate agreement (correlation, Bland–Altman, CV, IC50 fold).
"""
