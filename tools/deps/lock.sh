#!/usr/bin/env bash
# Regenerate the pinned, hash-checked lock files. Run from the repo root.
# Review the diff like code: every version change can change numerical output.
set -euo pipefail
PY=3.12
common=(--universal --python-version "$PY" --generate-hashes --no-emit-package sam3 \
        --index-url https://pypi.org/simple --quiet)
uv pip compile requirements/runtime.in "${common[@]}" -o requirements/lock/py312-runtime.txt
uv pip compile requirements/service.in "${common[@]}" -c requirements/lock/py312-runtime.txt -o requirements/lock/py312-service.txt
uv pip compile requirements/dev.in     "${common[@]}" -o requirements/lock/py312-dev.txt
echo "Locked. Now: uv pip sync requirements/lock/py312-dev.txt && pytest -q"
