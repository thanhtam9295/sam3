"""Batch execution: worker processes, resume after cancel or a forced kill, jobs, CLI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tifffile

from hcs_studio.app import main
from hcs_studio.io.base import open_plate, reopen_plate
from hcs_studio.io.generic import TiffFolderReader
from hcs_studio.pipeline.executor import (
    PipelineRunner,
    auto_workers,
    pipeline_hash,
    run_pipeline,
    well_is_complete,
)
from hcs_studio.pipeline.jobs import Job, JobProcess
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import create_module
from tests.builders import make_imagexpress
from tests.test_segmentation import CENTRES, nuclei_image

WELLS = [(r, c) for r in range(2) for c in range(3)]  # 6 wells
PIPELINE = """
name: batch
steps:
  - {module: flatfield, params: {min_fields: 4}}
  - {module: segment_classical, params: {channel: 1}}
  - {module: expand_cells, params: {distance_um: 2}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
"""


def _image(row: int, col: int, site: int, wave: int) -> Any:
    centres = CENTRES[: 2 + (row * 3 + col) % 4]
    return np.clip(nuclei_image(centres, seed=site + 10 * col).data[wave, 0], 0, None).astype(
        np.uint16
    )


@pytest.fixture
def plate_dir(tmp_path: Path) -> Path:
    return make_imagexpress(tmp_path / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)


def _objects(folder: Path) -> pd.DataFrame:
    frame = pd.read_parquet(folder / "tables" / "per_object_all.parquet")
    return frame.sort_values(["well", "field", "object_type", "object_id"]).reset_index(drop=True)


def _summary_times(folder: Path) -> dict[str, int]:
    """When each completed well's outputs were written (its first preview).

    The well summaries themselves are rewritten after every run with plate-level statistics,
    so they cannot show whether a well was processed again.
    """
    times = {}
    for summary in (folder / "wells").glob("*/well_summary.json"):
        previews = sorted((summary.parent / "previews").glob("*.png"))
        times[summary.parent.name] = (previews[0] if previews else summary).stat().st_mtime_ns
    return times


def test_parallel_run_matches_sequential(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    pipeline = Pipeline.from_text(PIPELINE)
    events: list[dict[str, Any]] = []
    one = run_pipeline(plate, pipeline, output_root=tmp_path / "one", checksums=False)
    two = run_pipeline(
        plate,
        pipeline,
        output_root=tmp_path / "two",
        checksums=False,
        workers=2,
        events=events.append,
    )
    assert one.fields_written == two.fields_written == 12 and not two.failed_wells
    pd.testing.assert_frame_equal(_objects(one.folder), _objects(two.folder))
    a = tifffile.imread(one.folder / "wells" / "B03" / "masks" / "f002_cells.ome.tif")
    b = tifffile.imread(two.folder / "wells" / "B03" / "masks" / "f002_cells.ome.tif")
    assert np.array_equal(a, b)
    kinds = [e["type"] for e in events]
    assert (
        kinds[0] == "run_started"
        and kinds.count("well_finished") == 6
        and kinds.count("field_done") == 12
    )
    last = [e for e in events if e["type"] == "field_done"][-1]
    assert last["done"] == last["total"] == 12 and last["eta_s"] == pytest.approx(0)
    assert next(e for e in events if e["type"] == "finished")["status"] == "completed"
    assert "with 2 worker(s)" in (two.folder / "logs" / "run.log").read_text(encoding="utf-8")


def test_cancel_then_resume_skips_completed_wells(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    pipeline = Pipeline.from_text(PIPELINE)
    cancel = threading.Event()

    def stop_after_two(event: dict[str, Any]) -> None:
        if event["type"] == "well_finished" and event["well"] == "A02":
            cancel.set()

    first = run_pipeline(
        plate, pipeline, output_root=tmp_path, checksums=False, events=stop_after_two, cancel=cancel
    )
    folder = first.folder
    assert first.cancelled and first.fields_written == 4
    assert (
        json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))["status"]
        == "cancelled"
    )
    done_before = _summary_times(folder)
    assert sorted(done_before) == ["A01", "A02"]
    fitted = (folder / "state" / "fitted.pkl").stat().st_mtime_ns

    second = run_pipeline(plate, pipeline, checksums=False, resume=folder)
    assert sorted(second.skipped_wells) == ["A01", "A02"] and second.fields_written == 8
    after = _summary_times(folder)
    assert {w: after[w] for w in done_before} == done_before, "completed wells were redone"
    assert (folder / "state" / "fitted.pkl").stat().st_mtime_ns == fitted  # estimates reused
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and [h["status"] for h in manifest["history"]] == [
        "cancelled",
        "completed",
    ]
    assert len(manifest["wells"]) == 6
    full = run_pipeline(plate, pipeline, output_root=tmp_path / "full", checksums=False)
    pd.testing.assert_frame_equal(_objects(folder), _objects(full.folder))


def test_resume_refuses_a_changed_pipeline(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    first = run_pipeline(
        plate, Pipeline.from_text(PIPELINE), output_root=tmp_path, wells=[(0, 0)], checksums=False
    )
    changed = Pipeline.from_text(PIPELINE.replace("distance_um: 2", "distance_um: 3"))
    with pytest.raises(ValueError, match="different pipeline"):
        run_pipeline(plate, changed, resume=first.folder)
    with pytest.raises(ValueError, match="not an HCS Studio run folder"):
        run_pipeline(plate, changed, resume=tmp_path / "nothing")
    assert pipeline_hash(plate, Pipeline.from_text(PIPELINE), {}) != pipeline_hash(
        plate, changed, {}
    )
    assert well_is_complete(
        first.folder, "A01", pipeline_hash(plate, Pipeline.from_text(PIPELINE), {})
    )


def test_forced_kill_then_cli_resume(plate_dir: Path, tmp_path: Path) -> None:
    """Acceptance: an unattended run resumes after a forced kill."""
    slow = PIPELINE.replace("steps:\n", "steps:\n  - {module: denoise, params: {method: nlm}}\n")
    pipeline_file = tmp_path / "slow.yaml"
    pipeline_file.write_text(slow, encoding="utf-8")
    out = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "hcs_studio",
        "run",
        "--pipeline",
        str(pipeline_file),
        "--input",
        str(plate_dir),
        "--output",
        str(out),
        "--workers",
        "2",
        "--no-checksums",
    ]
    options: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined,unused-ignore]
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options
    )
    deadline = time.monotonic() + 120
    folder = None
    while time.monotonic() < deadline:
        found = list(out.glob("*/wells/*/well_summary.json"))
        if found:
            folder = found[0].parents[2]
            break
        time.sleep(0.1)
    if sys.platform == "win32":  # the whole process tree, workers included
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)  # the whole process group, workers included
    process.wait(timeout=30)
    assert folder is not None, "no well finished before the deadline"
    done_before = _summary_times(folder)
    assert 1 <= len(done_before) < 6, "the run should have been interrupted part-way"
    assert (
        json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))["status"]
        == "running"
    )

    assert main(["run", "--resume", str(folder), "--workers", "2", "--no-checksums"]) == 0
    after = _summary_times(folder)
    assert sorted(after) == ["A01", "A02", "A03", "B01", "B02", "B03"]
    assert {w: after[w] for w in done_before} == done_before
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and len(manifest["history"]) == 2
    assert manifest["history"][0].get("status") is None  # the killed attempt never finished
    full = tmp_path / "full"
    assert (
        main(
            [
                "run",
                "--pipeline",
                str(pipeline_file),
                "--input",
                str(plate_dir),
                "--output",
                str(full),
                "--workers",
                "2",
                "--no-checksums",
            ]
        )
        == 0
    )
    pd.testing.assert_frame_equal(_objects(folder), _objects(next(full.iterdir())))
    assert not list(folder.rglob("*.part"))


def test_retry_failed_wells(
    plate_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    victim = next(p for p in plate_dir.glob("Plate1_B02_s1_w1*.tif") if "_thumb" not in p.name)
    good = victim.read_bytes()
    victim.write_bytes(b"broken")
    pipeline_file = tmp_path / "p.yaml"
    pipeline_file.write_text(PIPELINE.replace("min_fields: 4", "min_fields: 2"), encoding="utf-8")
    out = tmp_path / "out"
    assert (
        main(
            [
                "run",
                "--pipeline",
                str(pipeline_file),
                "--input",
                str(plate_dir),
                "--output",
                str(out),
                "--no-checksums",
            ]
        )
        == 1
    )
    folder = next(out.iterdir())
    assert list(
        json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))["failed_wells"]
    ) == ["B02"]
    capsys.readouterr()
    victim.write_bytes(good)
    before = _summary_times(folder)
    assert main(["run", "--retry-failed", str(folder), "--no-checksums"]) == 0
    printed = capsys.readouterr().out
    assert "Fields written: 2" in printed
    after = _summary_times(folder)
    assert after["B02"] != before["B02"] and after["A01"] == before["A01"]
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failed_wells"] == {} and manifest["status"] == "completed"
    assert main(["run", "--retry-failed", str(folder)]) == 0
    assert "No failed wells" in capsys.readouterr().out


def test_cli_multiple_plates_and_workers(
    plate_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = make_imagexpress(tmp_path / "raw" / "ix2", [(0, 0)], n_sites=2, image_fn=_image)
    pipeline_file = tmp_path / "p.yaml"
    pipeline_file.write_text(
        "name: seg\nsteps: [{module: segment_classical, params: {channel: 1}}]", encoding="utf-8"
    )
    code = main(
        [
            "run",
            "--pipeline",
            str(pipeline_file),
            "--input",
            str(plate_dir),
            "--input",
            str(other),
            "--wells",
            "A01",
            "--output",
            str(tmp_path / "out"),
            "--workers",
            "auto",
            "--no-checksums",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and out.count("Output folder:") == 2 and "== Plate1 ==" in out
    assert (
        main(["run", "--pipeline", str(pipeline_file), "--input", str(plate_dir), "--workers", "0"])
        == 2
    )
    assert main(["run", "--input", str(plate_dir)]) == 2
    assert "--pipeline and --input are required" in capsys.readouterr().out


def test_field_filter_and_pause(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    pipeline = Pipeline.from_text(
        "name: seg\nsteps: [{module: segment_classical, params: {channel: 1}}]"
    )
    only = run_pipeline(
        plate, pipeline, output_root=tmp_path, wells=[(0, 0)], fields={(0, 0): [2]}, checksums=False
    )
    assert only.fields_written == 1
    previews = sorted(p.name for p in (only.folder / "wells" / "A01" / "previews").iterdir())
    assert previews and all(name.startswith("f002") for name in previews)  # only field 2

    pause = threading.Event()
    events: list[dict[str, Any]] = []
    pause.set()
    thread = threading.Thread(
        target=lambda: run_pipeline(
            plate,
            pipeline,
            output_root=tmp_path / "p",
            wells=[(0, 0)],
            checksums=False,
            pause=pause,
            events=events.append,
        )
    )
    thread.start()
    time.sleep(1.0)
    assert not any(e["type"] == "field_done" for e in events), "worked while paused"
    pause.clear()
    thread.join(timeout=60)
    assert any(e["type"] == "finished" and e["status"] == "completed" for e in events)


def test_gpu_steps_are_serialised(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Lock:
        def __enter__(self) -> None:
            calls.append("acquire")

        def __exit__(self, *args: object) -> None:
            calls.append("release")

    plate = open_plate(
        make_imagexpress(
            Path(os.environ.get("TMPDIR", "/tmp")) / f"gpu{os.getpid()}",
            [(0, 0)],
            1,
            image_fn=_image,
        )
    )
    runner = PipelineRunner(
        plate, Pipeline.from_text("steps: [{module: denoise}, {module: background}]")
    )
    runner.gpu_lock = Lock()
    monkeypatch.setattr(type(runner.modules[0]), "uses_gpu", lambda self: True)
    runner.process(runner.load(plate.wells[(0, 0)].fields[0]))
    assert calls == ["acquire", "release"]  # only the GPU step took the lock

    sam = create_module("segment_sam3", {"prompt": "x", "channels": [1]})
    assert sam.exclusive and auto_workers(plate, [sam]) == 1
    assert auto_workers(plate, [create_module("denoise")]) >= 1


def test_pattern_plates_reopen_in_workers(tmp_path: Path) -> None:
    for well in ("B02", "B03"):
        for c in (1, 2):
            tifffile.imwrite(
                tmp_path / f"exp_{well}_c{c}.tif", np.full((32, 32), 100 * c, dtype=np.uint16)
            )
    plate = TiffFolderReader.read_with_pattern(
        tmp_path, r"exp_(?P<well>[A-Z]\d\d)_c(?P<channel>\d)", 0.5
    )
    again = reopen_plate(plate.source)
    assert sorted(again.wells) == sorted(plate.wells) and again.metadata.pixel_size_um == 0.5
    assert (
        open_plate(
            tmp_path / "exp_B02_c1.tif" if False else make_imagexpress(tmp_path / "ix", [(0, 0)], 1)
        ).source["reader"]
        == "imagexpress"
    )


def test_job_process_runs_and_cancels(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    job = Job(
        plate_source=plate.source,
        pipeline=Pipeline.from_text(PIPELINE).to_dict(),
        metadata=plate.metadata,
        wells=[(0, 0), (0, 1)],
        output_root=str(tmp_path / "jobs"),
        checksums=False,
        label="test",
    )
    process = JobProcess(job)
    process.start()
    received: list[dict[str, Any]] = []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not any(e["type"] == "finished" for e in received):
        received += process.poll()
        time.sleep(0.1)
    process.process.join(timeout=30)
    finished = [e for e in received if e["type"] == "finished"]
    assert finished and finished[0]["status"] == "completed" and finished[0]["fields"] == 4
    assert sum(e["type"] == "field_done" for e in received) == 4

    slow = Pipeline.from_text(
        PIPELINE.replace("steps:\n", "steps:\n  - {module: denoise, params: {method: nlm}}\n")
    )
    cancelled = JobProcess(
        Job(
            plate.source,
            slow.to_dict(),
            plate.metadata,
            output_root=str(tmp_path / "c"),
            checksums=False,
        )
    )
    cancelled.start()
    events: list[dict[str, Any]] = []
    while time.monotonic() < deadline and not any(e["type"] == "field_done" for e in events):
        events += cancelled.poll()
        time.sleep(0.05)
    cancelled.cancel.set()
    while time.monotonic() < deadline and not any(e["type"] == "finished" for e in events):
        events += cancelled.poll()
        time.sleep(0.1)
    assert next(e for e in events if e["type"] == "finished")["status"] == "cancelled"
    cancelled.process.join(timeout=30)

    broken = JobProcess(Job({"path": str(tmp_path / "missing")}, slow.to_dict()))
    broken.start()
    broken.process.join(timeout=60)
    errors = [e for e in broken.poll() if e["type"] == "finished"]
    assert errors and errors[0]["status"] == "error" and "FileNotFoundError" in errors[0]["error"]
