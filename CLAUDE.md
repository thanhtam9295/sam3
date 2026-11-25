# CLAUDE.md

@AGENTS.md

## Claude Code specifics

The shared instructions above (imported from `AGENTS.md`) are authoritative.
This section only adds Claude Code mechanics.

* **Identity:** your agent id is `claude`. Branches: `agent/claude/T-###-<slug>`.
  Set `export AGENT_ID=claude` in each shell before using `tools/agents/claims.py`.
* **Skills:** project skills are in `.claude/skills/` (generated from
  `.agents/skills/`; edit the canonical copy, then run `python tools/agents/sync_skills.py`).
* **Parallel sessions:** use `python tools/agents/worktree.py new ...` (or
  Claude Code's own worktree support) so each session has its own worktree;
  claims are shared across worktrees of the clone.
* **Protected algorithms:** before editing any path in
  `tools/validation/protected_paths.txt`, use plan mode, present the plan and
  the expected numerical effect, and wait for explicit approval.
* **Permissions:** `.claude/settings.json` asks before `git push`, PR/release
  creation and anything touching `.env`/credentials. Do not try to bypass it.
* Personal, uncommitted preferences go in `CLAUDE.local.md` (git-ignored).
