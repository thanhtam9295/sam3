"""Phase 16 (a) GUI: Tools → ilastik settings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.ilastik_settings import IlastikSettingsDialog  # noqa: E402
from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings  # noqa: E402
from tests.test_ilastik import fake_ilastik  # noqa: E402


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shebang scripts")
def test_ilastik_settings_dialog_detects_validates_and_saves(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "settings"))
    monkeypatch.delenv("HCS_FIJI", raising=False)
    program = fake_ilastik(tmp_path / "ilastik-1.4.2-Linux" / "run_ilastik.sh", "1.4.2")
    monkeypatch.setenv("HCS_ILASTIK", str(program))
    dialog = IlastikSettingsDialog()
    qtbot.addWidget(dialog)
    assert dialog.ilastik.text() == "" and dialog.threads.text() == "no limit"
    with qtbot.waitSignal(dialog.task.finished, timeout=30000):
        dialog.run_detection()
    assert dialog.ilastik.text() == str(program)  # filled from the detection
    assert "ilastik 1.4.2" in dialog.version.text() and "headless" in dialog.status.text()
    assert dialog.detect_button.isEnabled()

    dialog.ram.setValue(300)  # below ilastik's minimum: refused, nothing saved
    dialog.save()
    assert "at least 500" in dialog.status.text() and not IlastikSettings.path().exists()
    dialog.ram.setValue(16000)
    dialog.threads.setValue(8)
    dialog.save()
    saved = IlastikSettings.load()
    assert saved.ilastik == str(program) and saved.ram_mb == 16000 and saved.threads == 8

    blocked = fake_ilastik(tmp_path / "ilastik-1.3.3-Linux" / "run_ilastik.sh", "1.3.3")
    dialog.ilastik.setText(str(blocked))
    with qtbot.waitSignal(dialog.task.finished, timeout=30000):
        dialog.run_detection()
    assert "not compatible" in dialog.version.text() and "No runner" in dialog.status.text()
