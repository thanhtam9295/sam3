# GEMINI.md — Google Antigravity / Gemini

@AGENTS.md

## Antigravity specifics

`AGENTS.md` (imported above) is authoritative; this file only adds
Antigravity mechanics. If your Antigravity build does not expand `@` imports,
open and follow [`AGENTS.md`](AGENTS.md) before doing anything else.

* **Identity:** agent id `antigravity`; branches `agent/antigravity/T-###-<slug>`;
  `export AGENT_ID=antigravity` (PowerShell: `$env:AGENT_ID='antigravity'`).
* **Rules:** always-on workspace rules are in `.agent/rules/`.
* **Skills:** workspace skills are in `.agent/skills/` (generated mirror of
  `.agents/skills/` — edit the canonical copy and run `python tools/agents/sync_skills.py`).
* **Agent Manager / parallel agents:** give every agent its own worktree
  (`python tools/agents/worktree.py new --agent antigravity ...`) and its own
  task id. Never let two Antigravity agents share a worktree.
* **Browser / terminal autonomy:** do not upload files, sign in to services, or
  run `git push` from the browser or terminal without explicit human approval.
  Keep terminal auto-execution on "request review" for `git push`, `gh`, `curl`
  uploads and package publishing.
