"""Phase 6 GUI: pipeline builder, plate selection, run dialog, job queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("napari")
pytestmark = pytest.mark.gui

from PySide6.QtCore import QModelIndex, QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from hcs_studio.gui.jobs_panel import JobsPanel, job_for_folder  # noqa: E402
from hcs_studio.gui.main_window import MainWindow  # noqa: E402
from hcs_studio.gui.pipeline_panel import PipelinePanel  # noqa: E402
from hcs_studio.gui.plate_map import PlateMapView  # noqa: E402
from hcs_studio.gui.run_dialog import RunDialog  # noqa: E402
from hcs_studio.io.base import open_plate  # noqa: E402
from hcs_studio.pipeline.jobs import Job  # noqa: E402
from hcs_studio.pipeline.schema import Pipeline  # noqa: E402
from hcs_studio.utils.gpu import GpuStatus  # noqa: E402
from tests.builders import make_imagexpress  # noqa: E402
from tests.test_segmentation import CENTRES, nuclei_image  # noqa: E402

SEG = Pipeline.from_text("name: seg\nsteps: [{module: segment_classical, params: {channel: 1}}]")
SLOW = Pipeline.from_text(
    "name: slow\nsteps: [{module: denoise, params: {method: nlm}}, "
    "{module: segment_classical, params: {channel: 1}}]"
)


def _image(row: int, col: int, site: int, wave: int) -> Any:
    return np.clip(nuclei_image(CENTRES, seed=site + col).data[wave, 0], 0, None).astype(np.uint16)


@pytest.fixture
def plate_dir(tmp_path: Path) -> Path:
    return make_imagexpress(
        tmp_path / "raw" / "ix", [(0, 0), (0, 1), (1, 0), (1, 1)], n_sites=2, image_fn=_image
    )


def test_pipeline_builder_reorder_and_preview_all(qtbot: QtBot) -> None:
    panel = PipelinePanel()
    qtbot.addWidget(panel)
    for module in ("denoise", "background", "segment_classical"):
        panel.add_step(module, {"channel": 1} if module == "segment_classical" else None)
    panel.steps.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)  # drag the last step to the top
    assert [s.module for s in panel.pipeline().steps] == [
        "segment_classical",
        "denoise",
        "background",
    ]
    assert panel.steps.item(0).text().startswith("1. Segmentation")
    panel.steps.setCurrentRow(0)
    with qtbot.waitSignal(panel.preview_requested) as preview:
        panel.request_preview_all()
    assert len(preview.args[0].steps) == 3
    with qtbot.waitSignal(panel.run_requested) as run:
        panel.request_run()
    assert run.args[0].name == "pipeline"
    for row in range(3):
        panel.steps.item(row).setCheckState(Qt.CheckState.Unchecked)
    panel.request_run()
    assert "no enabled steps" in panel.error.text()


def test_plate_map_multi_selection(qtbot: QtBot, plate_dir: Path) -> None:
    view = PlateMapView()
    qtbot.addWidget(view)
    view.resize(600, 420)
    view.show()
    view.set_plate(open_plate(plate_dir))
    changes: list[list[tuple[int, int]]] = []
    view.selection_changed.connect(changes.append)
    view.select_well(0, 0)
    assert view.selected_wells() == [(0, 0)]
    view.select_range(1, 1)  # Shift+click from A01 to B02
    assert view.selected_wells() == [(0, 0), (0, 1), (1, 0), (1, 1)]
    view.set_selection([(0, 1), (5, 5)])  # wells without data are ignored
    assert view.selected_wells() == [(0, 1)] and changes[-1] == [(0, 1)]

    view.set_selection([])
    start = view.mapFromScene(view._items[(0, 0)].sceneBoundingRect().topLeft()) - QPoint(4, 4)
    end = view.mapFromScene(view._items[(1, 1)].sceneBoundingRect().center())
    no_keys = Qt.KeyboardModifier.NoModifier
    left = Qt.MouseButton.LeftButton
    QTest.mousePress(view.viewport(), left, no_keys, QPoint(start.x(), max(1, start.y())))
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), left, no_keys, end)
    assert view.selected_wells() == [(0, 0), (0, 1), (1, 0), (1, 1)]  # rubber band


def test_run_dialog_scopes(qtbot: QtBot, plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    dialog = RunDialog(plate, SEG, current_field=(0, 1, 1), selected_wells=[(0, 0), (1, 1)])
    qtbot.addWidget(dialog)
    assert dialog.scope() == "selected"
    dialog.buttons["field"].setChecked(True)
    (job,) = dialog.build_jobs()
    assert job.wells == [(0, 1)] and job.fields == {(0, 1): [2]} and "A02 field 2" in job.label
    dialog.buttons["well"].setChecked(True)
    assert dialog.build_jobs()[0].fields is None
    dialog.buttons["selected"].setChecked(True)
    assert dialog.build_jobs()[0].wells == [(0, 0), (1, 1)]
    dialog.buttons["plate"].setChecked(True)
    dialog.workers.setValue(3)
    dialog.output.setText(str(tmp_path / "out"))
    (whole,) = dialog.build_jobs()
    assert whole.wells is None and whole.workers == 3 and whole.output_root == str(tmp_path / "out")
    assert whole.metadata == plate.metadata and whole.pipeline == SEG.to_dict()
    dialog.buttons["plates"].setChecked(True)
    with pytest.raises(ValueError, match="at least one other plate"):
        dialog.build_jobs()
    dialog.add_other_plate(tmp_path / "plate2")
    jobs = dialog.build_jobs()
    assert len(jobs) == 2 and jobs[1].plate_source == {"path": str(tmp_path / "plate2")}
    dialog.workers.setValue(0)
    assert dialog.build_jobs()[0].workers is None  # auto

    no_field = RunDialog(plate, SEG, current_field=None, selected_wells=[])
    qtbot.addWidget(no_field)
    assert not no_field.buttons["field"].isEnabled() and no_field.scope() == "plate"


def _job(plate_dir: Path, tmp_path: Path, pipeline: Pipeline = SEG, **kwargs: Any) -> Job:
    plate = open_plate(plate_dir)
    return Job(
        plate.source,
        pipeline.to_dict(),
        plate.metadata,
        output_root=str(tmp_path / "runs"),
        checksums=False,
        label="test",
        **kwargs,
    )


def test_job_queue_runs_retries_and_resumes(qtbot: QtBot, plate_dir: Path, tmp_path: Path) -> None:
    panel = JobsPanel()
    qtbot.addWidget(panel)
    victim = next(p for p in plate_dir.glob("Plate1_B02_s1_w1*.tif") if "_thumb" not in p.name)
    good = victim.read_bytes()
    victim.write_bytes(b"broken")
    with qtbot.waitSignal(panel.job_finished, timeout=120000) as done:
        panel.add_jobs([_job(plate_dir, tmp_path)])
    state = done.args[0]
    assert state.status == "completed_with_failures" and list(state.failed) == ["B02"]
    assert state.done == 6 and panel.progress.value() == 6  # B02 stops at its unreadable field 1
    assert panel.wells_table.rowCount() == 4
    model = panel.wells_table.model()
    statuses = {model.index(r, 0).data(): model.index(r, 1).data() for r in range(4)}
    assert statuses == {"A01": "done", "A02": "done", "B01": "done", "B02": "failed"}
    assert panel.retry_button.isEnabled()

    victim.write_bytes(good)
    with qtbot.waitSignal(panel.job_finished, timeout=120000) as retried:
        panel.retry_failed()
    assert retried.args[0].status == "completed" and retried.args[0].folder == state.folder
    manifest = json.loads((Path(state.folder) / "run_manifest.json").read_text())
    assert manifest["failed_wells"] == {} and len(manifest["history"]) == 2

    resume_job = job_for_folder(Path(state.folder))
    assert resume_job.resume == state.folder and len(resume_job.wells or []) == 4
    with qtbot.waitSignal(panel.job_finished, timeout=120000) as nothing_left:
        panel.add_jobs([resume_job])
    assert nothing_left.args[0].status == "completed" and nothing_left.args[0].done == 0


def test_pause_cancel_resume_and_interrupted(qtbot: QtBot, plate_dir: Path, tmp_path: Path) -> None:
    panel = JobsPanel()
    qtbot.addWidget(panel)
    panel.add_jobs([_job(plate_dir, tmp_path, SLOW)])
    # Wait for a whole well: a well interrupted part-way is discarded and redone on resume.
    qtbot.waitUntil(
        lambda: any(w.get("status") == "done" for w in panel.states[0].wells.values()),
        timeout=120000,
    )
    panel.toggle_pause()
    assert panel.states[0].status == "paused" and panel.pause_button.text() == "Continue"
    panel.toggle_pause()
    assert panel.states[0].status == "running"
    with qtbot.waitSignal(panel.job_finished, timeout=120000) as cancelled:
        panel.cancel()
    state = cancelled.args[0]
    assert state.status == "cancelled" and 0 < state.done < 8
    panel.jobs_table.setCurrentCell(0, 0)
    with qtbot.waitSignal(panel.job_finished, timeout=120000) as resumed:
        panel.resume_selected()
    assert resumed.args[0].status == "completed" and resumed.args[0].folder == state.folder
    skipped = [w for w, info in resumed.args[0].wells.items() if info["status"] == "done (earlier)"]
    assert skipped, "wells finished before the cancel should not be redone"

    panel.add_jobs([_job(plate_dir, tmp_path, SLOW)])
    qtbot.waitUntil(lambda: panel.states[-1].done >= 1, timeout=120000)
    with qtbot.waitSignal(panel.job_finished, timeout=60000) as killed:
        assert panel.current is not None
        panel.current.kill()
    assert killed.args[0].status == "interrupted" and "resume" in killed.args[0].message


def test_main_window_runs_selected_wells(
    qtbot: QtBot, plate_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    window = MainWindow(gpu_status=GpuStatus(backend="cpu"))
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.loader.loaded, timeout=10000):
        window.open_path(plate_dir)
    window.plate_panel.plate_map.set_selection([(0, 0), (1, 1)])
    window.pipeline_panel.set_pipeline(SEG)
    dialog = window.open_run_dialog(window.pipeline_panel.pipeline())
    assert dialog is not None and dialog.scope() == "selected"
    dialog.output.setText(str(tmp_path / "gui_runs"))
    dialog.checksums.setChecked(False)
    with qtbot.waitSignal(window.jobs_panel.job_finished, timeout=120000) as done:
        dialog._accept()
    assert done.args[0].status == "completed"
    assert sorted(p.name for p in (Path(done.args[0].folder) / "wells").iterdir()) == ["A01", "B02"]
    assert "Job 'Plate1 (2 wells)' completed" in window.log_view.toPlainText()
    window.close()
