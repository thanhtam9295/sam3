"""Phase 12 in the batch queue: a 384-well, 3-channel, 20-plane plate projected with
resume; saved bit depth; linear well mosaic of projected fields."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from hcs_studio.io.base import open_plate
from hcs_studio.pipeline.executor import run_pipeline, to_storage_dtype
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import ImageData, create_module
from hcs_studio.utils.synthetic import write_ome_zarr_plate

WELLS = [(r, c) for r in range(16) for c in range(24)]
PIPELINE = """
name: project-and-clean
steps:
  - {module: projection}
  - {module: background, params: {method: tophat, radius_um: 5}}
"""


def _field(row: int, col: int, field: int) -> np.ndarray:
    """(C=3, Z=20, 32, 32): noise around 100 and one bright pixel per channel at a plane
    that depends on the well."""
    rng = np.random.default_rng(row * 24 + col)
    data = rng.normal(100, 3, (3, 20, 32, 32))
    for c in range(3):
        data[c, (row + col + 7 * c) % 20, 8 + 6 * c, 16] += 1000 + 100 * c
    return np.clip(data, 0, None).astype(np.uint16)


@pytest.fixture(scope="module")
def plate_384(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p384") / "plate.zarr"
    return write_ome_zarr_plate(root, WELLS, _field)


def _times(folder: Path) -> dict[str, int]:
    return {
        p.parent.parent.name: p.stat().st_mtime_ns
        for p in (folder / "wells").glob("*/previews/f001.png")
    }


def test_384_well_3_channel_20_plane_plate_projects_with_resume(
    plate_384: Path, tmp_path: Path
) -> None:
    """Done when: a full 384-well 3-channel 20-plane plate projects in the batch queue
    with resume."""
    plate = open_plate(plate_384)
    assert len(plate.wells) == 384 and plate.shape_tczyx == (1, 3, 20, 32, 32)
    pipeline = Pipeline.from_text(PIPELINE)
    cancel = threading.Event()
    finished: list[str] = []

    def stop_after_100(event: dict[str, Any]) -> None:
        if event["type"] == "well_finished":
            finished.append(event["well"])
            if len(finished) >= 100:
                cancel.set()

    first = run_pipeline(
        plate,
        pipeline,
        output_root=tmp_path,
        checksums=False,
        workers=2,
        events=stop_after_100,
        cancel=cancel,
    )
    assert first.cancelled and 100 <= first.fields_written < 384
    folder = first.folder
    done = _times(folder)
    assert len(done) == first.fields_written

    second = run_pipeline(plate, pipeline, checksums=False, workers=2, resume=folder)
    assert not second.failed_wells and not second.cancelled
    assert len(second.skipped_wells) == len(done)
    assert second.fields_written == 384 - len(done)
    after = _times(folder)
    assert len(after) == 384
    assert {w: after[w] for w in done} == done, "completed wells were redone"
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"

    # the saved images are the projection + background of the raw stacks
    projection, background = (
        create_module("projection"),
        create_module("background", {"method": "tophat", "radius_um": 5}),
    )
    for row, col in [(0, 0), (7, 13), (15, 23)]:
        name = f"{'ABCDEFGHIJKLMNOP'[row]}{col + 1:02d}"
        saved = tifffile.imread(folder / "wells" / name / "enhanced" / "f001_corrected.ome.tif")
        raw = _field(row, col, 0).astype(np.float32)
        assert saved.squeeze().shape == (3, 32, 32)
        projected = projection.apply(ImageData(raw, plate.metadata), None)
        assert np.array_equal(projected.data[:, 0], raw.max(axis=1))  # MIP: the known maximum
        expected = background.apply(projected, None).data[:, 0]
        np.testing.assert_allclose(saved.squeeze(), expected, rtol=1e-6, atol=1e-4)
        summary = json.loads((folder / "wells" / name / "well_summary.json").read_text())
        reports = summary["fields"][0]["reports"][0]
        assert reports[0]["module"] == "projection"
        assert reports[0]["planes"] == [list(range(1, 21))] * 3
        assert reports[1]["module"] == "background"


def test_saved_bit_depth_option(plate_384: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_384)
    pipeline = Pipeline.from_text("name: p\nsteps:\n  - {module: projection}\n")
    result = run_pipeline(
        plate,
        pipeline,
        output_root=tmp_path,
        wells=[(0, 0)],
        checksums=False,
        image_dtype="original",
    )
    well = result.folder / "wells" / "A01"
    saved = tifffile.imread(well / "enhanced" / "f001_corrected.ome.tif")
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved.squeeze(), _field(0, 0, 0).max(axis=1))
    record = json.loads((well / "well_summary.json").read_text())["fields"][0]
    assert record["corrected_saved_as"]["bit_depth_assumed"]  # OME-Zarr has no bit depth
    manifest = json.loads((result.folder / "run_manifest.json").read_text())
    assert manifest["options"]["image_dtype"] == "original"

    stored, info = to_storage_dtype(np.array([-3.0, 2.6, 300.0]), "original", 8)
    assert stored.tolist() == [0, 3, 255] and info["clipped_pixels"] == 2
    with pytest.raises(ValueError, match="Unknown image type"):
        run_pipeline(plate, pipeline, output_root=tmp_path, image_dtype="int8")


def test_background_warning_reaches_the_well_summary(plate_384: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_384)
    pipeline = Pipeline.from_text(
        "name: p\nsteps:\n  - {module: projection}\n"
        "  - {module: background, params: {method: tophat, radius_um: 1, "
        "object_diameter_um: 5}}\n"
    )
    events: list[dict[str, Any]] = []
    result = run_pipeline(
        plate,
        pipeline,
        output_root=tmp_path,
        wells=[(0, 0)],
        checksums=False,
        events=events.append,
    )
    summary = json.loads((result.folder / "wells" / "A01" / "well_summary.json").read_text())
    assert any("less than 2×" in w for w in summary["warnings"])
    assert any(e["type"] == "log" and "less than 2×" in e["text"] for e in events)


def test_linear_well_mosaic(tmp_path: Path) -> None:
    from tests.test_analysis import PX, SCENE, _scene, _stitch_plate

    root, _ = _stitch_plate(tmp_path)
    pipeline = Pipeline.from_text(
        "name: m\nsteps:\n  - {module: projection}\n"
        "  - {module: stitch_well, params: {save_mosaic: true, mosaic_downsample: 1}}\n"
    )
    result = run_pipeline(open_plate(root), pipeline, output_root=tmp_path / "out")
    assert not result.failed_wells
    well = result.folder / "wells" / "A01"
    summary = json.loads((well / "well_summary.json").read_text())
    assert summary["stitching"]["mosaic_image"] == "mosaic.ome.tif"
    with tifffile.TiffFile(well / "mosaic.ome.tif") as tif:
        mosaic = tif.asarray().squeeze()
        assert f'PhysicalSizeX="{PX}"' in (tif.ome_metadata or "")
    assert mosaic.shape[0] == 2 and abs(mosaic.shape[1] - SCENE) <= 2
    scene = _scene(seed=0)[0].astype(np.uint16).astype(np.float32)
    n = min(SCENE, mosaic.shape[1], mosaic.shape[2]) - 2
    correlation = np.corrcoef(mosaic[0, :n, :n].ravel(), scene[:n, :n].ravel())[0, 1]
    assert correlation > 0.99
    np.testing.assert_allclose(mosaic[1, :n, :n], mosaic[0, :n, :n] * 0.5, atol=1.0)
