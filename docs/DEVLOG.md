# Development log

Newest first. One entry per task/PR. Never edit old entries; correct them in a
new entry. Format: see the `project-logs` skill.

<!-- new entries below -->

### 2026-11-25 · T-000 · claude · claude/beautiful-einstein-h50l6a
- What: set up the repository for collaborative development by Claude Code,
  OpenAI Codex, Google Antigravity and GitHub Copilot.
  * `AGENTS.md` as the canonical instruction file; thin adapters `CLAUDE.md`,
    `GEMINI.md` + `.agent/rules/`, `.github/copilot-instructions.md` +
    `.github/instructions/`; `.claude/settings.json` asks before push/PR/release.
  * Seven Agent Skills in `.agents/skills/`, mirrored to `.claude/skills/`,
    `.github/skills/`, `.agent/skills/` by `tools/agents/sync_skills.py`.
  * Coordination: `tools/agents/worktree.py`, `tools/agents/claims.py` (claims in
    the shared git dir), `tools/agents/pr_overlap_check.py` (CI), `.githooks/pre-commit`.
  * Scientific safeguards: `tools/validation/protected_paths.txt`,
    `check_protected_changes.py`, approval-gated `regenerate_baselines.py`,
    `docs/validation/`.
  * Tests: `tests/unit`, `tests/integration`, `tests/scientific` (reference
    agreement, 11 golden baselines, toy SA-Co/VEval end-to-end), `tests/tooling`.
  * Reproducibility: `requirements/*.in` + hash-pinned universal lock files (`uv`).
  * Security: `tools/security/scan_sensitive.py`, hardened `.gitignore`.
  * Deployment: `packaging/windows/` (PyInstaller CLI), `deploy/service/`
    (FastAPI), `deploy/m365-copilot/` (declarative agent + API plugin).
  * Docs: ARCHITECTURE, ALGORITHMS, KNOWN_ISSUES, TASKS, COORDINATION,
    SCIENTIFIC_VALIDATION, DEPENDENCIES, SECURITY_AND_DATA, deployment guides,
    ENTERPRISE_MIGRATION.
  * Config only in existing files: `pyproject.toml` pytest section (testpaths,
    markers, importlib mode); `.gitignore` additions; `.github/` additions.
    No file under `sam3/` was modified.
- Why: requested multi-platform agent collaboration with scientific safeguards.
- Validation (Python 3.12.3, torch 2.10.0, CPU, env built from
  `requirements/lock/py312-dev.txt` with `--require-hashes`):
  * Baseline before any change: original 12 upstream tests passed (after
    installing undeclared `psutil` and `setuptools<81`).
  * After: `pytest -q` → 99 passed, 1 skipped (GPU smoke test, no CUDA),
    3 xfailed (strict; document KI-004, KI-005, KI-007). Tiers: unit 55 + 3 xfail,
    integration 10 + 1 skip, scientific 23.
  * Reference agreement: box IoU/GIoU, mask IoU, NMS, masks→boxes, RLE,
    connected components, Hungarian and cgF1 all agree with torchvision /
    pycocotools / scipy / brute force / hand derivation.
  * End-to-end worktree + claim + pre-commit test in a scratch clone: a commit
    touching another agent's claimed file was blocked.
  * Toy VEval: all 54 shared reference metrics reproduced exactly.
  * `scan_sensitive.py --all` and `--history` (50 local commits) clean, after
    allow-listing the public `assets/dog.gif` (7.1 MB).
- Findings: KI-001 … KI-017 in KNOWN_ISSUES.md (notably KI-004, high).
- Follow-ups: T-001 … T-014.
