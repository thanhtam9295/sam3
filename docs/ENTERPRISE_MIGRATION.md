# Migrating to an approved enterprise GitHub environment

Target: a repository in your organisation's **GitHub Enterprise Cloud**
(optionally with data residency, `*.ghe.com`) or **GitHub Enterprise Server**.
This is a plan; **perform no step that transfers code or data without the
owner's and the organisation's approval.**

## 0. Decide first (owner + IT/security)

| Question | Notes |
|---|---|
| Target org / repo name / visibility | Default **private** or **internal**; never public without approval |
| History: full history or a fresh squashed import? | Full history keeps provenance (and the upstream Meta lineage); a fresh import avoids carrying any unknown history risk |
| Keep the upstream link? | Add `facebookresearch/sam3` as an `upstream` remote for future merges (read-only fetch) |
| Licence review | Code is under the SAM License (`LICENSE`); checkpoints have separate gated terms |
| Which AI agents are approved? | Enterprise plans of Copilot / Claude / Codex / Antigravity, data-retention settings, which repos they may access |

## 1. Pre-migration audit (on the current repository)

```bash
git clone --mirror https://github.com/<current-owner>/sam3.git sam3-mirror.git
cd sam3-mirror.git && git worktree add ../sam3-audit HEAD && cd ../sam3-audit
python tools/security/scan_sensitive.py --all
python tools/security/scan_sensitive.py --history       # full history (KI-014)
```
Optional second opinion: `gitleaks detect --source . --log-opts="--all"`.
Any finding → stop; rotate credentials; decide on `git filter-repo` *before*
the import (never import a history that contains secrets or data).

Also inventory: open PRs/issues to carry over, Actions secrets (names only),
branch protection, webhooks, deploy keys, GitHub Apps (Claude, Copilot,
Codex connectors).

## 2. Import

Pick one (all preserve commits; none copy secrets):

* **GitHub Enterprise Importer (GEI)** — `gh gei migrate-repo` for
  GitHub.com → GHEC; also migrates PRs, issues, releases. Recommended for GHEC.
* **Mirror push** — works for GHES and GHEC:
  ```bash
  git clone --mirror <source-url> sam3.git
  cd sam3.git
  git push --mirror https://<enterprise-host>/<org>/sam3.git    # repo created empty & private first
  ```
  Then remove refs you do not want (e.g. `refs/pull/*` are not pushable; stale `agent/*` branches).
* **Fresh import** — new repo, single initial commit of the audited tree,
  README note pointing to the source commit SHA.

## 3. Reconfigure in the enterprise repo

| Item | Action |
|---|---|
| `.github/CODEOWNERS` | replace `@thanhtam9295` with an approved team, e.g. `@<org>/sam3-scientific-reviewers` |
| Branch protection / rulesets on `main` | require PR, 1+ review, Code Owners, status checks `tests`, `guards`, `pr-overlap`, `ufmt_check`; block force-push & deletion; linear history |
| Rulesets for `agent/**` | allow agents to push only these branches; restrict who can push to `main` |
| Secret scanning + push protection, Dependabot alerts, code scanning (CodeQL) | enable (GitHub Advanced Security on Enterprise) |
| Actions | allow only approved actions (pin third-party actions by SHA: `astral-sh/setup-uv`); default `GITHUB_TOKEN` read-only |
| Runners | GitHub-hosted, or self-hosted inside the network; add a GPU runner for T-005 |
| GHES / data residency | `pr_overlap_check.py` honours `GITHUB_API_URL` automatically in Actions; update any hard-coded `github.com` URLs in docs |
| Package index | if PyPI is blocked, mirror via Artifactory/Nexus and set `UV_INDEX_URL`; hashes in the lock still verify integrity |
| Hugging Face access | allow `huggingface.co` egress for checkpoint download, or host checkpoints in an internal artifact store and use `--checkpoint` |
| `pyproject.toml` URLs | update Homepage/Bug Tracker if desired (metadata only) |

## 4. AI agents in the enterprise

| Agent | Enterprise setup |
|---|---|
| GitHub Copilot (Business/Enterprise) | enable coding agent for the repo; policies for models/public-code matching; it reads `AGENTS.md`, `.github/copilot-instructions.md`, `.github/instructions/`, `.github/skills/`; environment from `copilot-setup-steps.yml` |
| Claude Code (Team/Enterprise) | install the Claude GitHub App on the enterprise org only for this repo; managed settings can enforce `.claude/settings.json`-style deny rules org-wide |
| OpenAI Codex (Business/Enterprise) | connect the enterprise GitHub org; restrict to this repo; reads `AGENTS.md` and `.agents/skills/` |
| Google Antigravity | use with an organisation-approved Google account; workspace = enterprise clone; reads `GEMINI.md`, `.agent/rules/`, `.agent/skills/` |

For each: confirm data retention / training opt-out, region, audit logging,
and that tokens are scoped to this repository.

## 5. Cut-over

1. Freeze the source repo (archive or read-only) after the final sync.
2. Every collaborator re-clones from the enterprise URL; agents' worktrees are
   recreated (`tools/agents/worktree.py`); local claims do not migrate (they are not in git).
3. Run CI once on `main`; confirm all checks pass.
4. DEVLOG entry recording source URL, source SHA, target URL, method, date, approver.

## 6. Rollback

Keep the source repository archived (not deleted) for an agreed period; the
mirror clone from step 1 is a complete backup.
