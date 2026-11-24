"""Shared test configuration."""

from __future__ import annotations

import os
import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep tests away from the real settings folder (settings, registry, event log)."""
    if not os.environ.get("HCS_STUDIO_HOME"):
        import tempfile

        os.environ["HCS_STUDIO_HOME"] = tempfile.mkdtemp(prefix="hcs-test-home-")


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


@pytest.fixture
def inline_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``BackgroundTask`` work synchronously in the GUI thread.

    Heavy pure-Python work (numba compiles, matplotlib figures) in the worker thread while
    ``qtbot.waitUntil`` spins the event loop can be starved of the GIL for minutes late in a
    long GUI test session (seen as intermittent faulthandler timeouts). Tests of panel logic
    use this; the threading itself is covered by the lighter GUI tests."""
    from hcs_studio.gui.workers import BackgroundTask

    def request(self: BackgroundTask, work: object) -> None:
        self._generation += 1
        try:
            result = work()  # type: ignore[operator]
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(result)

    monkeypatch.setattr(BackgroundTask, "request", request)
