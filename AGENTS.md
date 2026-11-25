# AGENTS.md — shared instructions for every AI agent and human contributor

This file is the **single source of truth** for how to work in this repository.
Claude Code (`CLAUDE.md`), Google Antigravity (`GEMINI.md`, `.agent/rules/`),
GitHub Copilot (`.github/copilot-instructions.md`) and OpenAI Codex (reads this
file natively) all defer to it. If a platform file and this file disagree,
**this file wins**; fix the platform file.

## 1. Project in one paragraph

`sam3` is a fork of Meta's **SAM 3 / SAM 3.1** (Segment Anything with
Concepts): a PyTorch foundation model that detects, segments and tracks every
instance of an open-vocabulary concept (text or exemplar prompt) in images and
video, plus the evaluation stack for the SA-Co benchmarks (cgF1, HOTA, TETA,
phrase AP, mAP). It is used here for **scientific image/video analysis**, so
numerical results must be reproducible and quantitative algorithms are
treated as validated scientific instruments.

* Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* Quantitative algorithms and metrics: [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md)
* Upstream usage docs: [`README.md`](README.md), [`README_TRAIN.md`](README_TRAIN.md), [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md)

## 2. Non-negotiable rules

1. **Preserve behaviour.** Do not change public APIs, model outputs or metric
   values unless the task explicitly requires it.
2. **Validated algorithms are protected.** Any change to a path listed in
   [`tools/validation/protected_paths.txt`](tools/validation/protected_paths.txt)
   requires the [scientific validation procedure](docs/validation/SCIENTIFIC_VALIDATION.md)
   and a report **approved by a human**. Ask before you start. Agents never
   approve reports, never regenerate baselines on their own initiative and
   never loosen test tolerances. Known algorithm defects are recorded in
   [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — report, don't silently fix.
3. **No data, weights or secrets in git.** No experimental/instrument data,
   checkpoints, `.env` files, tokens or API keys — ever. Read secrets from the
   environment at runtime. See [`docs/SECURITY_AND_DATA.md`](docs/SECURITY_AND_DATA.md).
4. **No outward actions without approval.** Do not push, open/merge PRs,
   publish packages or releases, change repository visibility, or send any
   repository content or data to an external service unless the human has
   approved that specific action.
5. **One task, one branch, one worktree; claim before you edit.** Never edit a
   file another agent has claimed. See §5.
6. **Reproducible environments.** Install from the lock files; change
   dependencies only via [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
7. **Leave a trail.** Every task ends with a `docs/DEVLOG.md` entry and updated
   `docs/TASKS.md` / `docs/KNOWN_ISSUES.md`.

## 3. Setup and commands

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip sync requirements/lock/py312-dev.txt         # pinned, reproducible
uv pip install -e . --no-deps
git config core.hooksPath .githooks                      # secrets / claims / validation hook
```

| Purpose | Command |
|---|---|
| Fast tests (each edit) | `python -m pytest -m unit -q` |
| Pre-commit tests | `python -m pytest -m "not slow" -q` |
| Full CPU suite | `python -m pytest -q` |
| Scientific tier | `python -m pytest -m scientific -q` |
| GPU tests (CUDA host) | `python -m pytest -m gpu -q` |
| Format check (CI) | `ufmt check sam3 scripts` (black 24.2.0 + usort 1.0.2 + ruff-api) |
| Secret/data scan | `python tools/security/scan_sensitive.py --staged` |
| Algorithm guard | `python tools/validation/check_protected_changes.py --base origin/main` |
| Skill mirrors | `python tools/agents/sync_skills.py --check` |

GPU is required to build/run the model (CUDA is hard-coded at construction,
KI-009). All tests except `gpu`-marked ones run on CPU with no network and no
checkpoints.

## 4. Code conventions

* Python 3.12 is the reference interpreter (CI and lock files). Keep
  compatibility with the versions in `pyproject.toml` where practical.
* Formatting: `ufmt` (black + usort) exactly as `.github/workflows/format.yml`.
  Line length 88. Keep the Meta copyright header on upstream files.
* Keep upstream files' structure so future upstream merges stay easy: prefer
  new modules over large edits to upstream modules.
* Tests: `tests/<unit|integration|scientific|tooling>/`, deterministic,
  synthetic inputs, no network. See the `sam3-testing` skill.
* Notebooks: commit with outputs cleared.

## 5. Parallel work by multiple agents

Full protocol: [`docs/agents/COORDINATION.md`](docs/agents/COORDINATION.md).

* Branch name: `agent/<agent-id>/T-###-<slug>` where agent-id ∈
  `claude | codex | antigravity | copilot | human`. Humans may use
  `feature/...`. `main` is protected; integrate only via reviewed PRs.
* Local agents: `python tools/agents/worktree.py new --agent <id> --task T-### --slug <slug> --claim <path>...`
* Claims: `python tools/agents/claims.py claim|status|check|release`. Claims
  live in the shared git directory, so they are visible to every worktree.
* Cross-machine (cloud agents): the `pr-overlap` CI job fails a PR that edits
  protected files also edited in another open PR, and warns for other overlaps.
* Before starting, read `docs/TASKS.md`; never take a task with an owner other
  than you; keep the `Files` column accurate.
* Do not edit shared governance files (`AGENTS.md`, platform instruction files,
  `.agents/skills/`, `.github/workflows/`, `tools/validation/`) unless that is
  the task.

## 6. Agent Skills

Reusable procedures follow the open **Agent Skills** standard (`SKILL.md` with
YAML front matter). Canonical source: [`.agents/skills/`](.agents/skills/) —
edit only there, then run `python tools/agents/sync_skills.py`.

| Skill | Use it when |
|---|---|
| `scientific-validation` | touching any quantitative algorithm, metric or baseline |
| `sam3-testing` | running/adding tests, CI failures |
| `agent-coordination` | starting, committing or finishing any task |
| `project-logs` | updating DEVLOG / KNOWN_ISSUES / TASKS |
| `data-and-secrets-hygiene` | any commit; any data, token or upload question |
| `sam3-segmentation-analysis` | running SAM 3 on images/videos, evaluating results |
| `windows-portable-build` | packaging the Windows executable |

Discovery paths (generated mirrors, do not edit): `.claude/skills/` (Claude
Code), `.github/skills/` (Copilot), `.agent/skills/` (Antigravity). Codex reads
`.agents/skills/` directly.

## 7. Definition of done

- [ ] Change is within the task scope; claimed files only.
- [ ] `python -m pytest -m "not slow" -q` passes (full suite before a PR).
- [ ] Protected paths untouched, or an approved validation report is included.
- [ ] `scan_sensitive.py --staged` passes; no data/weights/secrets.
- [ ] Lock files updated if (and only if) dependencies changed.
- [ ] DEVLOG entry added; TASKS / KNOWN_ISSUES updated.
- [ ] Commit messages explain *why*; no model or agent identifiers needed in code.
- [ ] Claims released.

## 8. Where things are

| Path | Contents |
|---|---|
| `sam3/model/` | model architecture, predictors, video tracking |
| `sam3/sam/` | SAM-1/2 style prompt encoder & mask decoder |
| `sam3/perflib/` | fast kernels (IoU, NMS, CC, mask→box, Triton) |
| `sam3/eval/` | metrics: cgF1, HOTA, TETA, YT-VIS AP, COCO |
| `sam3/train/` | training loop, losses, matcher, data |
| `sam3/agent/` | SAM 3 Agent (LLM + SAM 3 tool use) |
| `scripts/` | benchmarking, evaluation data prep |
| `tests/`, `test/` | tiered test suite / original upstream tests |
| `tools/` | agent coordination, validation and security tooling |
| `docs/` | architecture, algorithms, logs, policies, deployment |
| `packaging/windows/`, `deploy/` | Windows executable, M365 Copilot / service deployment |
