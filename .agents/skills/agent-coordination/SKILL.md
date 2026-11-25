---
name: agent-coordination
description: Branch, worktree and file-claim protocol that lets Claude Code, Codex, Antigravity, Copilot and humans work on this repository in parallel without conflicting edits. Use at the start of every task, before editing any file, before committing, and when finishing or abandoning a task.
---

# Multi-agent coordination

One agent = one task = one branch = one worktree. Never edit files another
agent has claimed. Full rules: `docs/agents/COORDINATION.md`.

## Start a task

1. Pick or create a task in `docs/TASKS.md` (id `T-###`). Check its `Owner` is
   empty or you; check `Files` for overlap with in-progress tasks.
2. Set your identity: `export AGENT_ID=<claude|codex|antigravity|copilot>`
   (PowerShell: `$env:AGENT_ID='codex'`).
3. Create an isolated worktree and claim the files you will touch:

```bash
python tools/agents/worktree.py new --agent $AGENT_ID --task T-012 --slug short-name \
  --claim path/to/file.py --claim tests/unit/
cd ../sam3-worktrees/$AGENT_ID-T-012-short-name
```

   Cloud/sandboxed agents that already run in their own clone (Codex cloud,
   Copilot coding agent, remote Claude Code): just work on a branch named
   `agent/<agent>/T-###-slug` and claim files (claims are local to the clone;
   the PR-overlap CI check covers cross-machine conflicts).

4. If a claim reports `CONFLICT`, do **not** edit that file. Choose other work,
   or ask the human to sequence the tasks.

## While working

* Need another file? Claim it first: `python tools/agents/claims.py claim <path> --task T-012`.
* Stay inside the task's scope; note unrelated findings in `docs/TASKS.md` instead of fixing them.
* Enable hooks once per clone: `git config core.hooksPath .githooks`.

## Finish

1. Tests pass (`sam3-testing` skill); logs updated (`project-logs` skill).
2. Commit on your branch. Push / open a PR **only if the human asked you to.**
3. Release claims: `python tools/agents/claims.py release --all`.
4. After merge: `python tools/agents/worktree.py remove --agent $AGENT_ID --task T-012 --slug short-name`.

## Never

* Commit to `main` directly, force-push shared branches, or rewrite others' history.
* Override a claim (`SAM3_CLAIMS_OVERRIDE=1`) without human approval.
* Edit `AGENTS.md`, the skills, CI workflows or protected algorithm paths as a
  side effect of another task.
