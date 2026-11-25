"""Pipeline files, the executor, the run folder and its manifest."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from hcs_studio.app import main
from hcs_studio.io.base import open_plate
from hcs_studio.model.layout import load_plate_layout
from hcs_studio.pipeline.executor import (
    PipelineRunner,
    make_run_folder,
    parse_wells,
    run_pipeline,
)
from hcs_studio.pipeline.schema import Pipeline
from hcs_studio.processing.base import ModuleError
from tests.builders import make_imagexpress

SHAPE = (64, 80)
WELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _vignette() -> Any:
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return 1.0 - 0.3 * (((yy - 32) / 32) ** 2 + ((xx - 40) / 40) ** 2)


def _image(row: int, col: int, site: int, wave: int) -> Any:
    """Uniform sample (brighter in wave 2) seen through a vignette, plus a 100-count offset."""
    rng = np.random.default_rng(row * 1000 + col * 100 + site * 10 + wave)
    signal = 400.0 * (wave + 1) + rng.normal(0, 2, SHAPE)
    return (100 + _vignette() * signal).astype(np.uint16)


@pytest.fixture
def plate_dir(tmp_path: Path) -> Path:
    return make_imagexpress(tmp_path / "raw" / "ix", WELLS, n_sites=3, image_fn=_image)


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))}


PIPELINE_YAML = """
name: correct-and-enhance
steps:
  - module: flatfield
    params: {dark_offset: 100, min_fields: 8}
  - module: background
    enabled: false
  - module: clahe
    params: {tile_um: 10}
"""


def test_pipeline_yaml_round_trip(tmp_path: Path) -> None:
    pipeline = Pipeline.from_text(PIPELINE_YAML)
    assert [s.module for s in pipeline.enabled_steps()] == ["flatfield", "clahe"]
    # Defaults are written out so the saved file is self-describing.
    assert pipeline.steps[0].params["max_fields"] == 200
    assert pipeline.steps[2].params == {"channels": None, "tile_um": 10.0, "clip_limit": 0.01}

    for name in ("p.yaml", "p.json"):
        pipeline.save(tmp_path / name)
        assert Pipeline.load(tmp_path / name) == pipeline
    assert (
        json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))["name"]
        == "correct-and-enhance"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("steps: [{module: nope}]", "Unknown module"),
        ("steps: [{module: denoise, params: {sigma_px: -1}}]", "greater than 0"),
        ("steps: [{module: denoise, params: {typo: 1}}]", "Extra inputs"),
        ("steps: [{module: denoise, extra: 1}]", "Extra inputs"),
        ("name: 'bad/name'", "pattern"),
        ("format_version: 99", "not supported"),
        ("- just a list", "mapping"),
        ("name: [unclosed", "Not valid YAML"),
    ],
)
def test_pipeline_validation(text: str, message: str) -> None:
    with pytest.raises((ValidationError, ValueError), match=message):
        Pipeline.from_text(text)


def test_runner_separates_corrected_from_enhanced(plate_dir: Path) -> None:
    plate = open_plate(plate_dir)
    runner = PipelineRunner(plate, Pipeline.from_text(PIPELINE_YAML))
    field = plate.wells[(0, 0)].fields[0]
    result = runner.process(runner.load(field))

    assert result.corrected.linear and result.corrected.history == ("flatfield",)
    assert not result.final.linear and result.final.history == ("flatfield", "clahe")
    # The profile is normalised to mean 1, so the field's mean intensity is preserved
    # while the shading is removed.
    for wave in range(2):
        plane = result.corrected.data[wave, 0]
        assert plane.mean() == pytest.approx(400 * (wave + 1) * _vignette().mean(), rel=0.03)
        assert plane.std() / plane.mean() < 0.03
    assert runner.fitted_summary()[0]["n_fields"] == 12
    partial = runner.process(runner.load(field), upto=1)
    assert partial.final.linear and np.array_equal(partial.final.data, result.corrected.data)

    preview = PipelineRunner(plate, Pipeline.from_text(PIPELINE_YAML), fit_field_limit=9)
    preview.fit()
    assert preview.fitted_summary()[0]["n_fields"] == 9


def test_runner_reports_which_step_cannot_run(tmp_path: Path) -> None:
    for c in range(2):
        tifffile.imwrite(tmp_path / f"A01_f1_c{c + 1}.tif", np.ones(SHAPE, dtype=np.uint16))
    plate = open_plate(tmp_path)  # plain TIFF folder: no pixel size
    with pytest.raises(ModuleError, match=r"Step 2 \(CLAHE.*pixel size is unknown"):
        PipelineRunner(
            plate, Pipeline.from_text(PIPELINE_YAML.replace("min_fields: 8", "min_fields: 2"))
        )
    runner = PipelineRunner(plate, Pipeline.from_text("steps: [{module: flatfield}]"))
    with pytest.raises(ModuleError, match=r"Step 1 \(Flat-field correction\): .*at least 8"):
        runner.fit()


def test_run_writes_a_self_describing_folder(plate_dir: Path, tmp_path: Path) -> None:
    plate = open_plate(plate_dir)
    before = _snapshot(plate_dir)
    layout = tmp_path / "layout.csv"
    layout.write_text("well,compound\nA01,DMSO\n", encoding="utf-8")
    messages: list[tuple[int, int, str]] = []

    result = run_pipeline(
        plate,
        Pipeline.from_text(PIPELINE_YAML),
        wells=[(0, 0), (1, 1)],
        layout_path=layout,
        progress=lambda done, total, text: messages.append((done, total, text)),
    )
    folder = result.folder
    assert _snapshot(plate_dir) == before, "the run modified the raw data"
    assert folder.parent == plate_dir.parent and plate_dir not in folder.parents
    assert folder.name.startswith("Plate1__correct-and-enhance__")
    assert result.fields_written == 6 and not result.failed_wells
    assert messages[-1] == (6, 6, "Finished") and messages[0][2].startswith("Estimating")

    assert sorted(p.name for p in (folder / "wells").iterdir()) == ["A01", "B02"]
    assert Pipeline.load(folder / "pipeline.yaml") == Pipeline.from_text(PIPELINE_YAML)
    assert (folder / "plate_layout_source.csv").read_text(encoding="utf-8") == layout.read_text(
        encoding="utf-8"
    )
    kept = load_plate_layout(folder / "plate_layout.json")
    assert kept.get((0, 0)).compound == "DMSO"
    assert (
        json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))["layout"]["wells"]
        == 1
    )
    assert "Run started" in (folder / "logs" / "run.log").read_text(encoding="utf-8")

    well = folder / "wells" / "B02"
    with tifffile.TiffFile(well / "enhanced" / "f002_corrected.ome.tif") as tif:
        corrected = tif.asarray()
        assert tif.series[0].axes == "CYX" and corrected.dtype == np.float32
        assert 'PhysicalSizeX="0.65"' in (tif.ome_metadata or "")
        assert 'Name="DAPI"' in (tif.ome_metadata or "")
    assert corrected[1].mean() == pytest.approx(800 * _vignette().mean(), rel=0.03)
    enhanced = tifffile.imread(well / "enhanced" / "f002_enhanced.ome.tif")
    assert enhanced.max() <= 1.0  # CLAHE output: display only
    assert (well / "previews" / "f002.png").read_bytes()[:4] == b"\x89PNG"
    summary = json.loads((well / "well_summary.json").read_text(encoding="utf-8"))
    assert summary["fields"][1]["steps_enhanced"] == ["flatfield", "clahe"]
    assert "corrected files only" in summary["measurement_note"]

    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["wells"] == ["A01", "B02"]
    assert manifest["software"]["version"] and "git_commit" in manifest["software"]
    assert manifest["pipeline"]["steps"][0]["params"]["dark_offset"] == 100
    assert manifest["fitted_state"][0]["module"] == "flatfield"
    assert manifest["plate"]["metadata"]["pixel_size_um"] == 0.65
    assert len(manifest["inputs"]) == 12  # 2 wells × 3 sites × 2 wavelengths
    entry = manifest["inputs"][0]
    assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert not list(folder.rglob("*.part"))


def test_linear_only_pipeline_writes_no_enhanced_file(plate_dir: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        open_plate(plate_dir),
        Pipeline.from_text("name: bg\nsteps: [{module: background, params: {radius_um: 5}}]"),
        output_root=tmp_path / "out",
        wells=[(0, 0)],
        checksums=False,
    )
    names = sorted(p.name for p in (result.folder / "wells" / "A01" / "enhanced").iterdir())
    assert names == ["f001_corrected.ome.tif", "f002_corrected.ome.tif", "f003_corrected.ome.tif"]
    manifest = json.loads((result.folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert "sha256" not in manifest["inputs"][0] and "not computed" in manifest["input_checksums"]


def test_output_inside_raw_data_is_refused(plate_dir: Path) -> None:
    plate = open_plate(plate_dir)
    pipeline = Pipeline.from_text("steps: []")
    for root in (plate_dir, plate_dir / "results"):
        with pytest.raises(ValueError, match="inside the raw data folder"):
            make_run_folder(plate, pipeline, root)
    with pytest.raises(ValueError, match="Wells without images"):
        run_pipeline(plate, pipeline, wells=[(5, 5)])
    assert not list(plate_dir.parent.glob("Plate1__*")), "a refused run left a folder behind"


def test_failed_well_is_recorded_and_the_run_continues(plate_dir: Path) -> None:
    plate = open_plate(plate_dir)
    victim = next(plate_dir.glob("Plate1_A02_s2_w1*.tif"))
    victim.write_bytes(b"not a tiff")
    result = run_pipeline(
        plate, Pipeline.from_text("steps: [{module: denoise}]"), wells=[(0, 1), (1, 0)]
    )
    assert list(result.failed_wells) == ["A02"] and result.fields_written == 4
    manifest = json.loads((result.folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_failures" and "A02" in manifest["failed_wells"]
    summary = json.loads(
        (result.folder / "wells" / "A02" / "well_summary.json").read_text(encoding="utf-8")
    )
    assert summary["error"] and len(summary["fields"]) == 1  # field 1 written before the failure
    assert "Traceback" in (result.folder / "logs" / "run.log").read_text(encoding="utf-8")


def test_cancel(plate_dir: Path) -> None:
    cancel = threading.Event()

    def progress(done: int, total: int, message: str) -> None:
        if done == 2:
            cancel.set()

    result = run_pipeline(
        open_plate(plate_dir),
        Pipeline.from_text("steps: [{module: denoise}]"),
        progress=progress,
        cancel=cancel,
    )
    assert result.cancelled and result.fields_written == 2
    assert (
        json.loads((result.folder / "run_manifest.json").read_text(encoding="utf-8"))["status"]
        == "cancelled"
    )


def test_parse_wells(plate_dir: Path) -> None:
    plate = open_plate(plate_dir)
    assert parse_wells("all", plate) == WELLS
    assert parse_wells("A01-B01", plate) == [(0, 0), (1, 0)]
    assert parse_wells("B02, A01-A12", plate) == [(0, 0), (0, 1), (1, 1)]
    with pytest.raises(ValueError, match="None of the selected wells"):
        parse_wells("H12", plate)
    with pytest.raises(ValueError):
        parse_wells("A01-B02-C03", plate)


def test_cli_run_and_modules(
    plate_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = tmp_path / "p.yaml"
    pipeline.write_text(PIPELINE_YAML, encoding="utf-8")
    out_root = tmp_path / "results"
    code = main(
        [
            "run",
            "--pipeline",
            str(pipeline),
            "--input",
            str(plate_dir),
            "--wells",
            "A01",
            "--output",
            str(out_root),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0 and "Fields written: 3" in output and "[3/3] Finished" in output
    assert len(list(out_root.glob("Plate1__*/wells/A01/enhanced/*_corrected.ome.tif"))) == 3

    pipeline.write_text("steps: [{module: nope}]", encoding="utf-8")
    assert main(["run", "--pipeline", str(pipeline), "--input", str(plate_dir)]) == 2
    assert "Unknown module" in capsys.readouterr().out

    assert main(["modules"]) == 0
    listing = capsys.readouterr().out
    assert "flatfield: Flat-field correction" in listing
    assert "clahe: CLAHE (local contrast)  [display only: not for measurement]" in listing
    assert "radius_um (default 25.0)" in listing and "af_channel (required)" in listing
