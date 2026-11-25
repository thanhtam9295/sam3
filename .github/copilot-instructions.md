# GitHub Copilot instructions

Read and follow [`AGENTS.md`](../AGENTS.md) at the repository root. It is the
single source of truth for this repository; this file only adds Copilot
mechanics and repeats the safety-critical rules.

## Copilot specifics

* **Identity:** agent id `copilot`. Copilot coding agent branches should be
  renamed or created as `agent/copilot/T-###-<slug>` where possible; reference
  the `T-###` task id from `docs/TASKS.md` in the PR title.
* **Skills:** `.github/skills/` (generated from `.agents/skills/`; edit the
  canonical copy and run `python tools/agents/sync_skills.py`).
* **Path-specific rules:** `.github/instructions/*.instructions.md`.
* **Environment for the coding agent:** `.github/workflows/copilot-setup-steps.yml`.

## Safety-critical rules (from AGENTS.md)

1. Quantitative algorithms (`tools/validation/protected_paths.txt`) need a
   human-approved validation report; never change them as a side effect.
2. No data, checkpoints, `.env`, tokens or API keys in commits, PR text or comments.
3. Run `python -m pytest -m "not slow" -q` and
   `python tools/security/scan_sensitive.py --all` before requesting review.
4. Do not edit files claimed by another agent or touched by another open PR
   (the `pr-overlap` check will flag it); keep PRs small and single-purpose.
5. Add a `docs/DEVLOG.md` entry and update `docs/TASKS.md` in every PR.
