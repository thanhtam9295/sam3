"""Assay monitoring over time (Phase 17c). Headless.

- :mod:`.charts` — Levey–Jennings control charts: limits from a baseline period,
  configurable rules (a point beyond k SD, n of m points beyond k SD on one side, n points in
  a row on one side of the mean, n points steadily rising or falling) and event annotations.
- :mod:`.msr` — minimum significant ratio of potencies (single compound across runs,
  test–retest of a compound set, rolling MSR).
- :mod:`.bridging` — the same plates or reference compounds measured on two instruments.
"""
