"""Phase 10: run options, resource limits, error handling and the installation self-test."""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.app import main
from hcs_studio.io.base import open_plate
from hcs_studio.pipeline import executor
from hcs_studio.pipeline.executor import (
    auto_workers,
    estimate_output_bytes,
    run_pipeline,
    worker_memory_bytes,
)
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.utils.resources import free_disk_bytes, peak_rss_bytes
from tests.builders import make_imagexpress
from tests.test_batch import PIPELINE, _image

WELLS = [(r, c) for r in range(2) for c in range(3)]


@pytest.fixture
def plate_dir(tmp_path: Path) -> Path:
    return make_imagexpress(tmp_path / "raw" / "ix", WELLS, n_sites=2, image_fn=_image)


def test_resources_are_reported() -> None:
    peak = peak_rss_bytes()
    assert peak is not None and peak > 10 * 1024**2
    free = free_disk_bytes(Path("/nonexistent/folder/below/root"))
    assert free is not None and free > 0


def test_run_records_timing_and_performance(plate_dir: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        open_plate(plate_dir), Pipeline.from_text(PIPELINE), output_root=tmp_path, checksums=False
    )
    summary = json.loads((result.folder / "wells" / "A01" / "well_summary.json").read_text())
    timing = summary["timing"]
    assert timing["total_s"] > 0 and timing["worker_peak_memory_mb"] > 10
    assert set(timing["steps_s"]) == {
        "1 flatfield",
        "2 segment_classical",
        "3 expand_cells",
        "4 measure_objects",
        "5 qc",
    }
    manifest = json.loads((result.folder / "run_manifest.json").read_text())
    performance = manifest["performance"]
    assert performance["fields"] == 12 and performance["wells"] == 6
    assert performance["summed_s"]["4 measure_objects"] > 0
    assert performance["output_gb"] > 0
    assert manifest["options"] == {
        "save_images": True,
        "image_dtype": "float32",
        "max_memory_gb": None,
        "workers_requested": 1,
    }


def test_runs_without_saving_images(plate_dir: Path, tmp_path: Path) -> None:
    from hcs_studio.output.load import load_field_output

    plate = open_plate(plate_dir)
    full = run_pipeline(plate, Pipeline.from_text(PIPELINE), output_root=tmp_path / "a")
    lean = run_pipeline(
        plate, Pipeline.from_text(PIPELINE), output_root=tmp_path / "b", save_images=False
    )
    assert list(full.folder.rglob("*_corrected.ome.tif"))
    assert not list(lean.folder.rglob("*_corrected.ome.tif"))
    assert list(lean.folder.rglob("masks/*.ome.tif")) and list(lean.folder.rglob("previews/*.png"))
    a = pd.read_parquet(full.folder / "tables" / "per_object_all.parquet")
    b = pd.read_parquet(lean.folder / "tables" / "per_object_all.parquet")
    pd.testing.assert_frame_equal(a, b)  # the measurements do not depend on saving images
    with pytest.raises(ValueError, match="did not save images"):
        load_field_output(lean.folder, "A01", 1)

    def size(folder: Path) -> int:
        return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())

    assert size(lean.folder) < 0.7 * size(full.folder)  # tiny fields: previews dominate


def test_refuses_to_start_without_disk_space(
    plate_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hcs_studio.utils.resources as resources

    plate = open_plate(plate_dir)
    estimate = estimate_output_bytes(plate, 12, plate.metadata, True)
    assert estimate > estimate_output_bytes(plate, 12, plate.metadata, False) > 0
    monkeypatch.setattr(resources, "free_disk_bytes", lambda path: 1024)
    with pytest.raises(ValueError, match=r"only 0\.0 GB is free"):
        run_pipeline(plate, Pipeline.from_text(PIPELINE), output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_disk_full_stops_the_run_and_it_resumes(
    plate_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = executor.write_png  # type: ignore[attr-defined]
    calls = {"n": 0}

    def flaky(path: Path, rgb: Any) -> None:
        calls["n"] += 1
        if calls["n"] > 10:  # the disk fills up during the second well
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        original(path, rgb)

    monkeypatch.setattr(executor, "write_png", flaky)
    plate = open_plate(plate_dir)
    with pytest.raises(OSError, match="output disk is full"):
        run_pipeline(plate, Pipeline.from_text(PIPELINE), output_root=tmp_path / "out")
    folder = next((tmp_path / "out").iterdir())
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    done = [
        w
        for w in ("A01", "A02", "A03", "B01", "B02", "B03")
        if executor.well_is_complete(folder, w, manifest["pipeline_hash"])
    ]
    assert done == ["A01"]  # stopped after the failure instead of failing every well

    monkeypatch.setattr(executor, "write_png", original)  # space was freed
    result = run_pipeline(plate, Pipeline.from_text(PIPELINE), resume=folder)
    assert result.skipped_wells == ["A01"] and not result.failed_wells


def test_memory_limit_caps_workers(plate_dir: Path) -> None:
    plate = open_plate(plate_dir)
    modules = Pipeline.from_text(PIPELINE).image_steps()
    per_worker = worker_memory_bytes(plate)
    created = [s.create() for s in modules]
    assert auto_workers(plate, created, max_memory_bytes=per_worker * 1.5) == 1
    assert auto_workers(plate, created, max_memory_bytes=None) >= 1


def test_memory_limit_reaches_deconvolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from hcs_studio.processing import deconvolution

    backend = deconvolution.get_backend("cpu")
    unlimited = deconvolution.memory_budget(backend, None)
    monkeypatch.setattr(deconvolution, "MEMORY_LIMIT", 1024**3)
    assert deconvolution.memory_budget(backend, None) == min(unlimited, 0.5 * 1024**3)


def test_ctrl_c_stops_cleanly_and_resumes(plate_dir: Path, tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("SIGINT to a child process is a POSIX test")
    slow = PIPELINE.replace("steps:\n", "steps:\n  - {module: denoise, params: {method: nlm}}\n")
    pipeline_file = tmp_path / "slow.yaml"
    pipeline_file.write_text(slow)
    out = tmp_path / "out"
    process = subprocess.Popen(
        [
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
            "--no-checksums",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not list(out.glob("*/wells/*/well_summary.json")):
        time.sleep(0.1)
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=120)
    assert process.returncode == 130, output
    assert "Stopping after the current field" in output and "--resume" in output
    folder = next(out.iterdir())
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    assert main(["run", "--resume", str(folder), "--no-checksums"]) == 0


def test_profiling_handles_integer_features(tmp_path: Path) -> None:
    from hcs_studio.analysis.profiling import ProfileOptions, run_profiling
    from tests.test_profiling import make_profile_run

    run = make_profile_run(tmp_path, "P", seed=0, cells=40)
    for objects_file in run.glob("wells/*/objects.parquet"):
        frame = pd.read_parquet(objects_file)
        frame["n_neighbours"] = np.random.default_rng(0).integers(0, 9, len(frame))
        frame.to_parquet(objects_file, index=False)
    result = run_profiling([run], ProfileOptions(umap=False, figures=False))
    assert "n_neighbours" in result.features


def test_selftest_passes(tmp_path: Path) -> None:
    from hcs_studio.selftest import run_selftest

    checks = run_selftest(workers=2, keep=tmp_path)
    assert [c.name for c in checks if not c.ok] == []
    assert len(checks) == 10


def test_gui_error_handler_logs_and_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    if not (os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin")):
        pytest.skip("needs a display")
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    from hcs_studio.gui.launch import app_log_path, install_error_handler, setup_logging

    setup_logging()
    install_error_handler(None)
    try:
        raise RuntimeError("boom in a slot")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = sys.__excepthook__
    import logging

    for handler in logging.getLogger("hcs_studio").handlers:
        handler.flush()
    assert "boom in a slot" in app_log_path().read_text(encoding="utf-8")


def test_user_guide_builds_as_html(tmp_path: Path) -> None:
    import importlib.util
    import re

    spec = importlib.util.spec_from_file_location(
        "build_docs", Path(__file__).resolve().parent.parent / "scripts" / "build_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pages = module.build(tmp_path / "docs")
    assert {p.name for p in pages} == {
        "index.html",
        "install.html",
        "troubleshooting.html",
        "acceptance.html",
        "processing.html",
        "measurements.html",
        "annotation.html",
        "pharmacology.html",
        "figures.html",
        "ilastik.html",
        "registry.html",
        "image_qc.html",
        "assays.html",
        "spheroids.html",
        "validation.html",
    }
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert '.md"' not in text and ".md#" not in text  # links point at the HTML pages
        for target in re.findall(r'(?:href|src)="(?!https?:|mailto:)([^"#]+)', text):
            assert (page.parent / target).exists(), f"{page.name}: broken link {target}"
