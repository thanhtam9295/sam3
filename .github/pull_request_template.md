## Task

T-### — <!-- title from docs/TASKS.md -->
Agent / author: <!-- claude | codex | antigravity | copilot | human -->

## What and why

## Checklist (AGENTS.md §7)

- [ ] In scope; only files claimed for this task were edited
- [ ] `python -m pytest -m "not slow" -q` passes locally (full suite in CI)
- [ ] No protected algorithm path changed **or** an approved report is included:
      `docs/validation/reports/....md`
- [ ] No data, checkpoints, `.env`, tokens or API keys (`scan_sensitive.py --staged`)
- [ ] Lock files regenerated only if dependencies changed (`tools/deps/lock.sh`)
- [ ] `docs/DEVLOG.md` entry added; `docs/TASKS.md` / `docs/KNOWN_ISSUES.md` updated
- [ ] Skills edited only in `.agents/skills/` and mirrors re-synced

## Scientific impact

<!-- "None" or: which metrics/outputs change and by how much (link the report). -->
