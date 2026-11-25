# Reproducible dependencies

## Files

| File | Purpose | Edit? |
|---|---|---|
| `pyproject.toml` | upstream package metadata and extras (loose ranges) | only via a dedicated task (T-002) |
| `requirements/runtime.in` | inference: core deps + undeclared imports (KI-002) + `setuptools<81` (KI-001) | yes |
| `requirements/service.in` | REST service for M365 Copilot (FastAPI) | yes |
| `requirements/dev.in` | CI/test superset (`[dev,train]` extras, pytest) | yes |
| `requirements/lock/py312-*.txt` | **exact pins with SHA-256 hashes**, universal (Linux, macOS, Windows), Python 3.12 | never by hand |
| `tools/deps/lock.sh` | regenerates all lock files with `uv pip compile --universal --generate-hashes` | |

Locked headline versions (2026-11-25): Python 3.12, torch 2.10.0, torchvision
0.25.0, numpy 1.26.4, timm 1.0.31, scipy 1.17.1, pycocotools 2.0.11,
setuptools 80.10.2.

## Install exactly

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip sync --require-hashes requirements/lock/py312-dev.txt      # or py312-runtime.txt
uv pip install -e . --no-deps
```

Without `uv`: `pip install --require-hashes -r requirements/lock/py312-dev.txt && pip install -e . --no-deps`.

`sync` removes anything not in the lock, so the environment is exactly the
tested one. On Linux the PyPI `torch` wheel is the CUDA 12.8 build (also runs
on CPU); on Windows it is CPU-only — see deployment/WINDOWS_PORTABLE.md (KI-015).

## Changing a dependency

1. Claim `requirements/` (only one dependency task at a time).
2. Edit the `.in` file (pin with `==` anything numerically relevant: torch,
   torchvision, numpy, scipy, pycocotools, opencv, scikit-image, timm).
3. `bash tools/deps/lock.sh` (on Windows: run it in Git Bash or WSL).
4. Review the lock diff: list every changed version in the PR.
5. Run the **full** suite including `-m scientific`. A numerically relevant
   upgrade that changes any baseline is an algorithm change → validation report.
6. DEVLOG entry with old → new versions.

Security updates: same flow; Dependabot/Renovate PRs must regenerate the lock
via `lock.sh` rather than editing it.
