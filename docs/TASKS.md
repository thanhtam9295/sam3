# Outstanding tasks

Shared work queue for humans and AI agents. Rules (`agent-coordination` and
`project-logs` skills): take only tasks with no owner; set `in-progress` + your
agent id before starting; keep **Files** accurate so others can avoid overlap;
tasks marked **🔒** touch protected algorithm paths and need human approval
*before* work starts. Next free id: **T-015**.

Status: `todo` · `in-progress` · `blocked` · `review` · `done`

| ID | Title | Status | Owner | Files / areas | Notes |
|---|---|---|---|---|---|
| T-001 | Replace `pkg_resources` with `importlib.resources` for the BPE path | todo | | `sam3/model_builder.py` | KI-001. Behaviour-neutral; verify tokenizer path resolves in wheel + editable installs. |
| T-002 | Declare real runtime deps in `pyproject.toml` (psutil, pycocotools, scipy, torch, torchvision) | todo | | `pyproject.toml`, `requirements/*.in`, lock files | KI-002. Needs human decision: diverges from upstream metadata. |
| T-003 | 🔒 Quantify KI-004 (perflib IoM) on SA-Co/VEval and decide fix vs keep | blocked | | `sam3/perflib/iou.py`, `sam3/model/sam3_multiplex_detector_utils.py` | Needs GPU + evaluation data + human approval. Compare `USE_PERFLIB=0` vs `1`. |
| T-004 | Make model construction device-aware (CPU build) | todo | | `sam3/model/position_encoding.py`, `sam3/model_builder.py` | KI-009. Precompute must be numerically identical; add CPU model smoke test. 🔒-adjacent: run scientific tier on GPU before/after. |
| T-005 | GPU CI job: Triton/CUDA kernels vs CPU references | todo | | `tests/scientific/`, `.github/workflows/` | KI-011. Needs a self-hosted or enterprise GPU runner. |
| T-006 | Run the full-history secret scan on an unshallowed mirror | todo | | — | KI-014. `git fetch --unshallow`, then `scan_sensitive.py --history`. |
| T-007 | Add `gitleaks` (pinned) as a second secret scanner in CI | todo | | `.github/workflows/ci.yml` | Defence in depth; pin action by commit SHA. |
| T-008 | Hash-locked Windows CUDA lock file | todo | | `requirements/lock/`, `packaging/windows/build.ps1` | KI-015. Generate on a host that can reach download.pytorch.org. |
| T-009 | Build and smoke-test the Windows bundle on a GPU workstation | todo | | `packaging/windows/` | Record size, CUDA driver, timings in DEVLOG. |
| T-010 | M365: by-reference image operation (SharePoint/OneDrive URL) with host allow-list | todo | | `deploy/service/`, `deploy/m365-copilot/` | KI-016. Must prevent SSRF; use Graph with delegated auth. |
| T-011 | Validate M365 package with Microsoft 365 Agents Toolkit in an approved tenant | blocked | | `deploy/m365-copilot/` | Needs tenant admin; no publishing without approval. |
| T-012 | Domain validation report for the user's imaging modality | blocked | | `docs/validation/reports/` | Needs approved, non-committed expert annotations. Defines when SAM 3 counts may be used scientifically. |
| T-014 | Resolve KI-017: upstream code vs pinned `ufmt` toolchain | todo | | `sam3/`, `scripts/` (whitespace only) | Needs human decision; large mechanical diff; verify scientific tier unchanged. |
| T-013 | Decide on fixing KI-005 / KI-007 / KI-012 (low-severity algorithm defects) | todo | | see KNOWN_ISSUES | 🔒 Each needs its own validation report; propose first. |

## Done

| ID | Title | Done | Notes |
|---|---|---|---|
| T-000 | Multi-agent repository setup | 2026-11-25 | See DEVLOG 2026-11-25. |
