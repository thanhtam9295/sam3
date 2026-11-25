---
name: project-logs
description: How to maintain the shared project memory - docs/DEVLOG.md (what changed and why), docs/KNOWN_ISSUES.md (defects and caveats) and docs/TASKS.md (outstanding work and ownership). Use at the end of every task, when you discover a bug or limitation, and when you start, pause or finish a task.
---

# Project logs

These three files are how agents on different platforms share context. They
are append-friendly and excluded from the PR-overlap check, so concurrent edits
merge cleanly if you follow the format.

## docs/DEVLOG.md — newest entry first, under the `<!-- new entries below -->` marker

```markdown
### 2026-11-25 · T-012 · codex · agent/codex/T-012-cgf1-docs
- What: one line per change.
- Why: the reason / the task.
- Validation: tests run and result (e.g. `pytest -m "not slow"`: 92 passed).
- Follow-ups: new T-### / KI-### ids, or "none".
```

Never rewrite or delete old entries. Corrections go in a new entry.

## docs/KNOWN_ISSUES.md

* Add a new `KI-###` (next free number) with: Severity, Area, Status, Evidence
  (file:line, reproduction or test name), Impact, Proposed action.
* Algorithm defects: describe; do **not** fix without the `scientific-validation` skill.
* Closing: set `Status: Resolved in <commit/PR>`; keep the entry.

## docs/TASKS.md

* Add tasks as table rows: `T-###`, title, status (`todo|in-progress|blocked|review|done`),
  owner (agent id or human), files/areas, notes.
* Set `in-progress` + owner when you start; `review` when a PR is open; `done` after merge.
* Keep the `Files` column accurate — other agents use it to avoid overlap.
