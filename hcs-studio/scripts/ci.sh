#!/usr/bin/env bash
# Lint, type-check and test. Run from the hcs-studio folder with the environment active.
set -euo pipefail
cd "$(dirname "$0")/.."

ruff check . ../lab_events
ruff format --check . ../lab_events
mypy
mypy --platform win32  # Windows-only modules (resource, killpg, windll) differ

# napari needs a real OpenGL context; on a headless Linux machine use a virtual display.
if [[ "$(uname)" == "Linux" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]] && command -v xvfb-run >/dev/null; then
    xvfb-run -a -s "-screen 0 1600x1000x24" pytest "$@"
else
    pytest "$@"
fi
