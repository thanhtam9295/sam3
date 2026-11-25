---
name: scientific-validation
description: Required procedure before changing any quantitative analysis algorithm in SAM 3 (IoU/IoM, NMS, mask/box conversion, matching, association, RLE, connected components, EDT, cgF1/HOTA/TETA/AP metrics, post-processing) or any scientific baseline. Use whenever a change touches a path in tools/validation/protected_paths.txt, when a scientific test fails, or when asked to "fix", "optimize" or "speed up" a metric or kernel.
---

# Scientific validation

Validated algorithms are the scientific record of this project. You may not
change their numerical behaviour without evidence and a human's approval.

## 0. Decide whether this applies

```bash
python tools/validation/check_protected_changes.py --files <paths you plan to edit>
```

If it prints "No protected ... paths changed", this skill does not apply.

## 1. Before editing — stop and ask

* State the intended change, why, and the expected numerical effect.
* Get explicit human approval to proceed. Do **not** start editing a protected
  file on your own initiative, even to fix an obvious bug listed in
  `docs/KNOWN_ISSUES.md`. Propose it instead.
* Claim the files: `python tools/agents/claims.py claim <paths> --task T-xxx`.

## 2. Record the "before" state

```bash
python -m pytest -m scientific -q            # must pass before you start
python tools/validation/regenerate_baselines.py --diff   # must show all [same]
```

## 3. Make the change, then measure

1. Run the reference-agreement tests: `python -m pytest tests/scientific/test_reference_agreement.py -q`.
   These compare against torchvision / pycocotools / scipy / brute force and
   must keep passing (a change that breaks agreement is almost certainly wrong).
2. Run `regenerate_baselines.py --diff` and copy every `[CHANGED]` line into the report.
3. Run the toy end-to-end evaluation: `python -m pytest tests/scientific/test_toy_video_eval_regression.py -q`.
4. If a strict `xfail` test now passes (XPASS → failure), you fixed a known
   issue: remove the `xfail` marker in the same change and cite the KI id.
5. Speed claims need numbers: `python scripts/measure_speed.py` or a micro-benchmark,
   same hardware, before vs after.

## 4. Write the report

Copy `docs/validation/reports/TEMPLATE.md` to
`docs/validation/reports/YYYY-MM-DD-<slug>.md` and fill every section. Leave
`Status: PROPOSED` and `Approved-by: <pending>` — **only a human sets
`Status: APPROVED` and their name.** An AI agent must never approve its own
(or another agent's) report.

## 5. Baselines

Only after approval, and only by the human or at their explicit instruction:

```bash
python tools/validation/regenerate_baselines.py --case <name> \
  --approved-by "<human name>" --report docs/validation/reports/<file>.md
```

Never hand-edit `tests/scientific/baselines/*.json`. Never loosen tolerances in
`tests/scientific/compare.py` without a report.

## 6. Log it

Add a DEVLOG entry and update KNOWN_ISSUES / TASKS (see the `project-logs` skill).

Full policy: `docs/validation/SCIENTIFIC_VALIDATION.md`. Algorithm reference: `docs/ALGORITHMS.md`.
