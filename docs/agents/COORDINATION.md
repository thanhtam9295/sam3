# Branch, worktree and file-claim strategy for independent AI agents

Goal: several agents (Claude Code, OpenAI Codex, Google Antigravity, GitHub
Copilot) and humans work **in parallel** without two of them changing the same
source file at the same time, and without anyone bypassing review.

## 1. Model: one task → one branch → one worktree → one PR

```
main ─────●─────────────●──────────────●────────►   protected; merge via PR only
           \             \              \
            agent/codex/T-012-cgf1-docs  agent/claude/T-001-importlib  feature/<human>
            (worktree ../sam3-worktrees/codex-T-012-cgf1-docs)
```

| Rule | Why |
|---|---|
| Branch `agent/<agent>/T-###-<slug>` (agents) / `feature/<slug>` (humans) | Owner and task visible in every tool; `claims.py` derives the agent id from it |
| Branch from up-to-date `origin/main`; keep it short-lived (≤ a few days) | Small diffs merge cleanly |
| One worktree per active branch, outside the repo (`../sam3-worktrees/`) | Agents never share a working tree, index or untracked files |
| No direct commits to `main`; no force-push to shared branches | History integrity |
| Rebase *your own* branch on `main` before review; never rewrite others' | |
| PRs small and single-purpose; title starts with `T-###` | Review and overlap detection stay tractable |

Agent ids: `claude`, `codex`, `antigravity`, `copilot`, `human`.

## 2. Preventing simultaneous conflicting edits — three layers

### Layer 1 — plan: `docs/TASKS.md`
Each task lists the files/areas it will touch. Do not start a task whose
**Files** overlap an `in-progress` task. Assign an owner before work begins.

### Layer 2 — local, real-time: file claims (`tools/agents/claims.py`)
* Claims are JSON records in `<git-common-dir>/agent-claims/`. All worktrees of
  a clone share that directory, so every local agent sees every claim
  instantly. Nothing is committed.
* Creation is atomic (`O_CREAT|O_EXCL` mutex) — two agents cannot win the same
  file. A directory claim (`sam3/eval/`) covers everything below it.
* Claims expire (default 8 h; `--ttl-hours`) so a crashed agent cannot block
  work forever; `prune` deletes expired ones.
* `.githooks/pre-commit` runs `claims.py check --staged`: a commit containing a
  file claimed by another agent is rejected. Override (`SAM3_CLAIMS_OVERRIDE=1`)
  only with human approval.

```bash
export AGENT_ID=codex
python tools/agents/claims.py claim sam3/eval/cgf1_eval.py tests/unit/ --task T-012
python tools/agents/claims.py status
python tools/agents/claims.py release --all
```

### Layer 3 — remote: PR overlap check (`tools/agents/pr_overlap_check.py`)
Cloud agents (Codex cloud tasks, Copilot coding agent, remote Claude Code
sessions) run in their own clones and cannot see local claims. The `pr-overlap`
CI job compares the PR's files with every other open PR:
* overlap on a **protected** path → job fails;
* other overlaps → warning annotation (append-only logs `DEVLOG/TASKS/KNOWN_ISSUES` are ignored).
Resolution: sequence (merge one, rebase the other) or split the work.

Branch protection on `main` should require: `ci / tests`, `ci / guards`,
`ci / pr-overlap`, `ufmt_check`, one approving review, Code Owner review
(`.github/CODEOWNERS`), linear history, no force pushes.

## 3. Lifecycle

```bash
# 1. plan: add/assign T-012 in docs/TASKS.md (status in-progress, owner codex)
# 2. isolate + claim
python tools/agents/worktree.py new --agent codex --task T-012 --slug cgf1-docs \
       --claim docs/ALGORITHMS.md
cd ../sam3-worktrees/codex-T-012-cgf1-docs && export AGENT_ID=codex
git config core.hooksPath .githooks         # once per clone
# 3. work, test (python -m pytest -m "not slow" -q), log (docs/DEVLOG.md)
# 4. commit; push / PR only with human approval
# 5. release claims; after merge remove the worktree
python tools/agents/claims.py release --all
python tools/agents/worktree.py remove --agent codex --task T-012 --slug cgf1-docs --delete-branch
```

## 4. Platform notes

| Platform | How it isolates | What to configure |
|---|---|---|
| Claude Code (CLI/desktop/web) | separate sessions; use `worktree.py` or Claude Code's worktree support | `AGENT_ID=claude`; `.claude/settings.json` asks before push |
| OpenAI Codex (CLI / IDE) | local: `worktree.py`; cloud tasks: own container | `AGENT_ID=codex`; cloud tasks rely on Layer 3 |
| Google Antigravity | Agent Manager can run several agents — give each its own worktree | `AGENT_ID=antigravity`; keep terminal auto-run on review for push/upload |
| GitHub Copilot coding agent | own branch `copilot/*` + Actions runner | rename/describe with `T-###`; setup via `copilot-setup-steps.yml`; Layer 3 |
| GitHub Copilot (IDE agent mode) | the developer's checkout | use the developer's worktree and claims as `human` or `copilot` |

## 5. Shared files that are hot spots

* `docs/DEVLOG.md`, `docs/TASKS.md`, `docs/KNOWN_ISSUES.md`: append-only by
  convention, so parallel edits merge. Add new entries at the marked place.
* Governance (`AGENTS.md`, platform instruction files, `.agents/skills/`,
  `.github/workflows/`, `tools/`): change only in a dedicated task.
* Lock files: only one dependency task at a time (claim `requirements/`).

## 6. Conflict playbook

1. `CONFLICT` on claim → don't edit; choose another task or ask the human.
2. Merge conflict on rebase → resolve only hunks from your task; if the other
   side is another agent's in-flight work, stop and ask.
3. Two PRs overlap on a protected path → the later one waits.
