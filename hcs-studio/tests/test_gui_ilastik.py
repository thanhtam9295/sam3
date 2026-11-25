"""Phase 16 (a) GUI: Tools → ilastik settings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.ilastik_settings import IlastikSettingsDialog  # noqa: E402
from hcs_studio.segmentation.ilastik_backend.settings import IlastikSettings  # noqa: E402
from tests import test_ilastik_steps  # noqa: E402
from tests.test_ilastik import fake_ilastik  # noqa: E402

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)


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


class _Viewer:
    def __init__(self) -> None:
        self.shown: list[tuple[Any, Any]] = []

    def show_field(self, data: Any, metadata: Any) -> None:
        assert data.ndim == 5, "ViewerPanel.show_field takes (T, C, Z, Y, X)"
        self.shown.append((data, metadata))


class _Process:
    def __init__(self) -> None:
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shebang scripts")
def test_ilastik_panel_preview_preset_macro_and_training_round_trip(
    qtbot: QtBot, tmp_path: Path, ilastik: dict[str, Any]
) -> None:
    import json
    import shutil

    import numpy as np

    from hcs_studio.gui.ilastik_panel import IlastikPanel
    from hcs_studio.processing.base import ImageData
    from tests.test_ilastik_steps import _plate, make_project

    plate, _ = _plate(tmp_path)
    field = plate.wells[(0, 0)].fields[0]
    image = ImageData(np.asarray(field.load()[0], dtype=np.float32), plate.metadata)
    viewer = _Viewer()
    logs: list[str] = []
    panel = IlastikPanel(
        viewer,  # type: ignore[arg-type]
        lambda: image,
        logs.append,
        plate=lambda: plate,
        selected_wells=lambda: [(0, 0), (0, 1)],
    )
    qtbot.addWidget(panel)

    # project → details and compatibility badge
    panel.project.setText(str(ilastik["project"]))
    panel.load_project()
    assert "classes: background, nuclei" in panel.details.text()
    assert panel.badge.text().startswith("✓") and panel.channels.text() == "1"
    panel.channels.setText("1 5")
    assert not panel.check_fit() and "do not exist" in panel.badge.text()
    panel.channels.setText("1")

    # preview on the current field: class channels shown in the viewer
    with qtbot.waitSignal(panel.task.finished, timeout=60000):
        panel.preview()
    names = [c.name for c in viewer.shown[-1][1].channels]
    assert names[-2:] == ["ilastik_background", "ilastik_nuclei"]
    assert panel.last_preview is not None and panel.last_preview.data.shape[0] == 4
    assert viewer.shown[-1][0].shape[:2] == (1, 4)

    # preset into the pipeline builder
    with qtbot.waitSignal(panel.preset_ready) as signal:
        panel.use_preset()
    assert signal.args[0].steps[0].module == "ilastik_pixel"

    # Fiji macro
    panel.choose_save = lambda title, pattern: str(tmp_path / "m.ijm")
    assert panel.export_macro() == tmp_path / "m.ijm"
    assert "pixelclassificationtype=[Probabilities]" in (tmp_path / "m.ijm").read_text()

    # library (model registry)
    record = panel.register_project()
    assert record is not None and panel.project.text() == record.ref
    assert panel.library.count() == 2 and panel.badge.text().startswith("✓")

    # Train in ilastik: training set of the selected wells, then ilastik's window
    panel.project.setText("")
    panel.load_project()
    launched: list[list[str]] = []
    process = _Process()

    def launch(command: list[str]) -> _Process:
        launched.append(command)
        return process

    panel.launch = launch
    panel.choose_folder = lambda title: str(tmp_path / "training")
    with qtbot.waitSignal(panel.task.finished, timeout=60000):
        panel.train_in_ilastik()
    data = json.loads((tmp_path / "training" / "training_set.json").read_text())
    assert [f["well"] for f in data["fields"]] == ["A01", "A02"]
    assert launched[0][-2] == f"--new_project={tmp_path / 'training' / 'project.ilp'}"
    assert launched[0][-1] == "--workflow=PixelClassificationWorkflow"
    # the user trains and saves in ilastik, then closes it
    shutil.copy(make_project(tmp_path / "trained.ilp"), tmp_path / "training" / "project.ilp")
    panel._check_ilastik()  # still open: nothing happens
    assert panel.ilastik_process is process
    process.code = 0
    panel.confirm = lambda title, text: True
    panel._check_ilastik()
    assert panel.project.text().startswith("registry:project@1")
    from hcs_studio.training.registry import list_models

    stored = {m.name: m for m in list_models(kind="ilastik")}
    assert stored["project"].hyperparameters["training_set"]["fields"] == 2
