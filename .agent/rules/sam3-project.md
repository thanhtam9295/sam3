---
trigger: always_on
---

# SAM 3 workspace rules (Antigravity)

Follow `AGENTS.md` at the repository root; it is the single source of truth.
Key rules, repeated here because they are safety-critical:

1. Do not change files listed in `tools/validation/protected_paths.txt`
   without the `scientific-validation` skill and human approval.
2. Never commit experimental data, checkpoints, `.env` files, tokens or keys.
3. Do not push, publish, open PRs, or upload anything without explicit approval.
4. Work in your own worktree on `agent/antigravity/T-###-<slug>` and claim files
   with `python tools/agents/claims.py claim <paths> --task T-###` before editing.
5. Finish with tests (`python -m pytest -m "not slow" -q`) and a `docs/DEVLOG.md` entry.
