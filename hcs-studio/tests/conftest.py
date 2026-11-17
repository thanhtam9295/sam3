"""Shared test configuration."""

from __future__ import annotations

import os
import sys

import pytest


def _has_display() -> bool:
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip GUI tests when there is no display (napari needs a real OpenGL context)."""
    if _has_display():
        return
    skip = pytest.mark.skip(reason="no display; run under xvfb-run (see scripts/ci.sh)")
    for item in items:
        if "gui" in item.keywords:
            item.add_marker(skip)
