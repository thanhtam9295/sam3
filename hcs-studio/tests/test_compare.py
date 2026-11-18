"""Phase 14: comparison montages, scale bars, sidecars, data / movie export, arrangement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tifffile
from PIL import Image

from hcs_studio.compare.arrange import by_annotation, combination_grid, compound_grid, grid
from hcs_studio.compare.export import (
    batch_compound_figures,
    export_data,
    export_figure,
    export_movie,
    list_presets,
    load_images,
    load_preset,
    regenerate,
    save_preset,
)
from hcs_studio.compare.render import (
    auto_contrast,
    compose,
    default_channels,
    nice_length,
    outline,
    render,
)
from hcs_studio.compare.sources import load_panel, plane, representative_field
from hcs_studio.compare.spec import ChannelDisplay, Comparison, ExportSettings, ScaleBar
from tests.compare_builders import PIXEL_UM, field, make_run

WELLS = [f"{r}{c:02d}" for r in "AB" for c in range(1, 7)]
BAR = "#ff00ff"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_run(tmp_path_factory.mktemp("cmp"))


def _two_by_six(run: Path, fmt: Literal["png", "svg"] = "png") -> Comparison:
    base = Comparison(
        settings=ExportSettings(
            format=fmt,
            dpi=300,
            width_mm=180,
            scale_bar=ScaleBar(length_um=10.0, color=BAR, show_label=True, panels="all"),
        )
    )
    comparison = by_annotation(run, WELLS, base)
    images = load_images(comparison)
    channels = auto_contrast(
        [plane(i.image, "max") for i in images.values()], default_channels(images[0].image.metadata)
    )
    return comparison.model_copy(update={"channels": channels})


# ---------------------------------------------------------------- the done-when


def test_2x6_linked_comparison_png_scale_bars(run: Path, tmp_path: Path) -> None:
    """Done when: a 2 × 6 comparison with linked contrast exports to PNG with correct
    scale bars (checked against the known pixel size)."""
    comparison = _two_by_six(run)
    assert (comparison.rows, comparison.cols, len(comparison.panels)) == (2, 6, 12)
    assert comparison.linked and not comparison.unlinked()
    path = export_figure(comparison, tmp_path / "cmp.png")
    sidecar = json.loads((tmp_path / "cmp.png.json").read_text())
    pixels = np.asarray(Image.open(path).convert("RGB")).astype(int)
    magenta = (pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] > 200)
    panels = sidecar["geometry"]["panels"]
    assert len(panels) == 12
    for info in panels:
        x0, y0, x1, y1 = (round(v) for v in info["bbox_px"])
        # 64 px at 0.5 µm → 32 µm field; the panel's output scale follows from its width
        assert info["image_px"] == [64, 64]
        scale = (info["bbox_px"][2] - info["bbox_px"][0]) / 64
        assert info["output_px_per_image_px"] == pytest.approx(scale)
        region = magenta[y0:y1, x0:x1]
        rows = np.flatnonzero(region.any(axis=1))
        assert rows.size, f"no scale bar in panel {info['panel']}"
        widths = [np.ptp(np.flatnonzero(region[r])) + 1 for r in rows]
        expected = 10.0 / PIXEL_UM * scale  # 10 µm = 20 image px
        assert max(widths) == pytest.approx(expected, abs=1.5)
        assert info["scale_bar"]["length_output_px"] == pytest.approx(expected)
    # every panel was drawn with the same (linked) display
    assert len({json.dumps(c) for c in sidecar["comparison"]["channels"]}) == 2


def test_2x6_comparison_svg_scale_bars_and_text(run: Path, tmp_path: Path) -> None:
    comparison = _two_by_six(run, "svg")
    path = export_figure(comparison, tmp_path / "cmp.svg")
    svg = path.read_text()
    sidecar = json.loads((tmp_path / "cmp.svg.json").read_text())
    for info in sidecar["geometry"]["panels"]:
        gid = info["scale_bar"]["gid"]
        group = re.search(rf'<g id="{gid}">(.*?)</g>', svg, re.S)
        assert group, gid
        d = re.search(r'd="([^"]+)"', group.group(1)).group(1)  # type: ignore[union-attr]
        xs = [float(v) for v in re.findall(r"([-\d.]+) [-\d.]+", d)]
        width_pt = max(xs) - min(xs)
        panel_pt = (info["bbox_px"][2] - info["bbox_px"][0]) / 300 * 72
        assert width_pt == pytest.approx(20 / 64 * panel_pt, rel=0.01)  # 10 µm of 32 µm
    # labels are editable text, not outlines
    assert "10 µm" in svg and "MCF7" in svg and "Drug X 10 µM" in svg


def test_sidecar_regenerates_a_pixel_identical_png(run: Path, tmp_path: Path) -> None:
    """Done when: the export sidecar regenerates a pixel-identical PNG."""
    comparison = _two_by_six(run)
    comparison.panels[0].overlays = ["nuclei"]
    first = export_figure(comparison, tmp_path / "a.png")
    sidecar = json.loads((tmp_path / "a.png.json").read_text())
    assert sidecar["format"] == "hcs-studio-figure/1"
    assert all(p["field"] is not None for p in sidecar["comparison"]["panels"])
    assert sidecar["checksums"] and all(len(v) == 64 for v in sidecar["checksums"].values())
    assert sidecar["sources"][0]["steps"]  # processing steps recorded
    again = regenerate(tmp_path / "a.png.json", tmp_path / "b.png")
    a = np.asarray(Image.open(first))
    b = np.asarray(Image.open(again))
    assert a.shape == b.shape and np.array_equal(a, b)


# ------------------------------------------------------------------ rendering


def test_compose_display_and_outline() -> None:
    planes = np.stack([np.linspace(0, 100, 11, dtype=np.float32)[None].repeat(2, 0)] * 2)
    red = ChannelDisplay(index=0, color="#ff0000", low=0, high=100)
    green = ChannelDisplay(index=1, color="#00ff00", low=50, high=100, gamma=2.0)
    rgb = compose(planes, [red, green])
    assert rgb[0, 5].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert rgb[0, 10].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert rgb[0, 8, 1] == pytest.approx(0.6**0.5, rel=1e-5)
    hidden = compose(planes, [red, green.model_copy(update={"visible": False})])
    assert float(np.max(hidden[..., 1])) == 0
    labels = np.zeros((7, 7), int)
    labels[2:5, 2:5] = 1
    edge = outline(labels)
    assert edge.sum() == 8 and not edge[3, 3]
    assert nice_length(32) == 5 and nice_length(500) == 100 and nice_length(9) == 1


def test_unlinked_contrast_is_flagged_and_crop_applies(run: Path, tmp_path: Path) -> None:
    comparison = _two_by_six(run, "svg")
    brighter = [c.model_copy(update={"high": c.high / 2}) for c in comparison.channels]
    comparison.panels[3].display = brighter
    comparison = comparison.model_copy(update={"crop": (4.0, 4.0, 20.0, 24.0)})  # µm
    path = export_figure(comparison, tmp_path / "unlinked.svg")
    assert path.read_text().count("⚠ contrast unlinked") == 1
    info = json.loads((tmp_path / "unlinked.svg.json").read_text())["geometry"]["panels"][0]
    assert info["image_px"] == [40, 32]  # 20 × 16 µm at 0.5 µm per pixel


def test_split_channels_and_headers(run: Path, tmp_path: Path) -> None:
    comparison = _two_by_six(run, "svg")
    split = comparison.model_copy(
        update={"settings": comparison.settings.model_copy(update={"split_channels": True})}
    )
    _fig, rendered = render(split, load_images(split), dpi=100)
    assert len(rendered.panels) == 12 * 3  # DAPI, GFP, merge per well
    assert {p["channel"] for p in rendered.panels} == {0, 1, None}
    path = export_figure(split, tmp_path / "split.svg")
    text = path.read_text()
    assert text.count(">merge<") == 6 and "DAPI" in text


# -------------------------------------------------------------- sources, fields


def test_sources_raw_and_recomputed_steps(run: Path) -> None:
    corrected = load_panel(by_annotation(run, ["A03"], Comparison()).panels[0])
    raw = load_panel(
        by_annotation(run, ["A03"], Comparison()).panels[0].model_copy(update={"source": "raw"})
    )
    expected = field(0, 2, 0).astype(np.float32)
    np.testing.assert_array_equal(raw.image.data, expected)
    np.testing.assert_array_equal(corrected.image.data, expected)  # no correction steps
    assert raw.files and all(Path(f).exists() or "zarr" in f for f in raw.files)
    step = load_panel(
        by_annotation(run, ["A03"], Comparison()).panels[0].model_copy(update={"source": "step:1"})
    )
    assert [ls.name for ls in step.image.labels] == ["nuclei"]
    assert step.steps == ("segment_classical",)
    with pytest.raises(ValueError, match="step 9 does not exist"):
        load_panel(
            by_annotation(run, ["A04"], Comparison())
            .panels[0]
            .model_copy(update={"source": "step:9"})
        )


def test_representative_field(tmp_path: Path) -> None:
    well = tmp_path / "wells" / "A01"
    well.mkdir(parents=True)
    (well / "well_summary.json").write_text(
        json.dumps({"fields": [{"field": f} for f in (1, 2, 3)]})
    )
    (well / "fields.csv").write_text("field,n_nuclei\n1,10\n2,52\n3,90\n")
    assert representative_field(tmp_path, "A01", "n_nuclei") == 2  # closest to the median
    assert representative_field(tmp_path, "A01", None) == 1  # nothing to measure → first


# ------------------------------------------------------------------ arrangement


def test_arrangements(run: Path) -> None:
    by = by_annotation(run, list(reversed(WELLS)), Comparison())
    assert by.row_headers == ["HeLa", "MCF7"]
    assert by.col_headers[0].endswith("0.04115 µM") and by.col_headers[-1].endswith("10 µM")
    assert by.panel_at(1, 5).well == "A06"  # type: ignore[union-attr]
    plain = grid(run, WELLS[:5], Comparison(), cols=3)
    assert (plain.rows, plain.cols) == (2, 3) and plain.panel_at(1, 1).well == "A05"  # type: ignore[union-attr]
    one = compound_grid(run, "Drug X", Comparison())
    assert (one.rows, one.cols, one.title) == (2, 6, "Drug X")

    from hcs_studio.pharmacology.synthetic import write_run

    combo_run = write_run(run.parent / "synth")
    matrix = combination_grid(combo_run, "MCF7", "Drug A", "Drug B", Comparison())
    assert (matrix.rows, matrix.cols) == (6, 6) and len(matrix.panels) == 36
    assert matrix.row_headers[0] == "Drug A 10 µM" and matrix.row_headers[-1] == "Drug A 0 µM"
    assert matrix.col_headers[0] == "Drug B 0 µM"
    corner = matrix.panel_at(5, 0)
    assert corner is not None
    from hcs_studio.compare.arrange import layout_of
    from hcs_studio.model.plate import parse_well_name

    a = layout_of(combo_run).get(parse_well_name(corner.well))
    names = {t.compound for t in a.treatments}
    assert a.role == "negative_control" and {"Drug A", "Drug B"} <= names  # the matrix's own corner


# ---------------------------------------------------------- data, movies, batch


def test_data_export(run: Path, tmp_path: Path) -> None:
    comparison = by_annotation(run, ["A01", "B06"], Comparison(crop=(0, 0, 16, 16)))
    files = export_data(comparison, tmp_path / "data", dtype="uint16")
    tiffs = [f for f in files if f.suffix == ".tif"]
    assert len(tiffs) == 2
    data = tifffile.imread(tiffs[0])
    assert data.dtype == np.uint16 and data.squeeze().shape == (2, 32, 32)
    table = (tmp_path / "data" / "panels_readouts.csv").read_text()
    assert "nuclei_count" in table and "B06" in table
    assert json.loads((tmp_path / "data" / "panels.json").read_text())["kind"] == "data"


def test_movies(run: Path, tmp_path: Path) -> None:
    import imageio_ffmpeg

    comparison = grid(run, ["A01", "A06"], Comparison(settings=ExportSettings(width_mm=85)))
    gif = export_movie(comparison, tmp_path / "z.gif")
    assert getattr(Image.open(gif), "n_frames", 1) == 5
    mp4 = export_movie(comparison, tmp_path / "z.mp4")
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(mp4))
    assert frames == 5
    assert json.loads((tmp_path / "z.mp4.json").read_text())["frames"] == 5


def test_presets_and_batch(run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    settings = ExportSettings(width_mm=85, dpi=600, split_channels=True, font_size_pt=6)
    save_preset("Nature single column", settings)
    assert list_presets() == ["Nature single column"]
    assert load_preset("Nature single column") == settings
    figures = batch_compound_figures(
        run, tmp_path / "batch", Comparison(settings=ExportSettings(dpi=100))
    )
    assert [f.name for f in figures] == ["Drug-X.png"]
    assert (tmp_path / "batch" / "Drug-X.png.json").exists()


def test_command_line(run: Path, tmp_path: Path) -> None:
    from hcs_studio.app import main

    comparison = by_annotation(run, ["A01", "A06"], Comparison(settings=ExportSettings(dpi=100)))
    export_figure(comparison, tmp_path / "f.png")
    assert (
        main(
            [
                "figure",
                "--regenerate",
                str(tmp_path / "f.png.json"),
                "--output",
                str(tmp_path / "g.png"),
            ]
        )
        == 0
    )
    a, b = (np.asarray(Image.open(tmp_path / n)) for n in ("f.png", "g.png"))
    assert np.array_equal(a, b)
    assert (
        main(
            [
                "figure",
                "--run",
                str(run),
                "--per-compound",
                "--dpi",
                "100",
                "--output",
                str(tmp_path / "per"),
            ]
        )
        == 0
    )
    assert (tmp_path / "per" / "Drug-X.png").exists()
    assert main(["figure"]) == 2


# ------------------------------------------------------------------- hardening


def test_panel_cache_is_bounded_by_bytes(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hcs_studio.compare import sources

    sources.clear_cache()
    one = sources._nbytes(load_panel(by_annotation(run, ["A01"], Comparison()).panels[0]))
    monkeypatch.setattr(sources, "CACHE_LIMIT_BYTES", int(one * 2.5))
    for well in ("A01", "A02", "A03", "A04"):
        load_panel(by_annotation(run, [well], Comparison()).panels[0])
    assert len(sources._CACHE) == 2 and sources.cache_bytes() <= one * 2.5
    assert [k[1] for k in sources._CACHE] == ["A03", "A04"]  # the most recent
    sources.clear_cache()
    assert sources.cache_bytes() == 0


def test_oversized_raster_figures_are_refused(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hcs_studio.compare import export

    monkeypatch.setattr(export, "MAX_MEGAPIXELS", 1.0)
    comparison = _two_by_six(run)
    with pytest.raises(ValueError, match="megapixels at 300 dpi"):
        export_figure(comparison, tmp_path / "big.png")
    svg = comparison.model_copy(
        update={"settings": comparison.settings.model_copy(update={"format": "svg"})}
    )
    assert export_figure(svg, tmp_path / "big.svg").exists()  # vector formats are fine


def test_time_lapse_movie_from_raw_data(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.utils.synthetic import write_ome_zarr_plate

    def series(row: int, col: int, f: int) -> np.ndarray:
        data = np.full((4, 1, 1, 32, 32), 100, np.uint16)
        for t in range(4):
            data[t, 0, 0, 4 + 6 * t : 10 + 6 * t, 10:20] = 2000  # a moving object
        return data

    plate = write_ome_zarr_plate(
        tmp_path / "t.zarr",
        [(0, 0), (0, 1)],
        series,
        channels=("GFP",),
        plate_size=(8, 12),
        timepoints=4,
    )
    assert open_plate(plate).shape_tczyx == (4, 1, 1, 32, 32)
    run = run_pipeline(
        open_plate(plate),
        Pipeline.from_text("name: t\nsteps:\n  - {module: qc}\n"),
        output_root=tmp_path / "out",
        checksums=False,
    ).folder
    comparison = grid(run, ["A01", "A02"], Comparison(settings=ExportSettings(width_mm=85)))
    with pytest.raises(ValueError, match="raw data"):
        export_movie(comparison, tmp_path / "t.gif", axis="t")
    raw = comparison.model_copy(
        update={"panels": [p.model_copy(update={"source": "raw"}) for p in comparison.panels]}
    )
    gif = export_movie(raw, tmp_path / "t.gif", axis="t")
    assert getattr(Image.open(gif), "n_frames", 1) == 4
    frames = []
    image = Image.open(gif)
    for k in range(4):
        image.seek(k)
        frames.append(np.asarray(image.convert("L")))
    assert all(not np.array_equal(frames[0], f) for f in frames[1:])  # the object moves
    with pytest.raises(ValueError, match=r"\.gif or \.mp4"):
        export_movie(raw, tmp_path / "t.avi", axis="t")


def test_combinations_in_layout() -> None:
    from hcs_studio.compare.arrange import combinations_in_layout
    from hcs_studio.pharmacology.synthetic import layout as synthetic_layout

    assert combinations_in_layout(synthetic_layout()) == [
        ("MCF7", "Drug A", "Drug B"),
        ("MCF7", "Drug A", "Drug C"),
    ]


def test_review_fixes_layout_cache_unit_order_context_and_folder_fingerprint(
    run: Path, tmp_path: Path
) -> None:
    import shutil

    from hcs_studio.compare import sources
    from hcs_studio.compare.render import annotation_text
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import dilution_series

    # labels follow a rewritten layout file (no stale cache)
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    panel = by_annotation(copy, ["A01"], Comparison()).panels[0]
    assert "MCF7" in annotation_text(panel, "{cell_line}", 1)
    layout = load_plate_layout_of(copy)
    save_layout(
        layout.updated({k: replace_line(a, "U2OS") for k, a in layout.wells.items()}),
        copy / "plate_layout.json",
    )
    assert annotation_text(panel, "{cell_line}", 1) == "U2OS"

    # columns sorted within a unit kind: 10 µM never sorts between µg/mL doses
    mixed = PlateLayout.for_plate_type(96)
    mixed = dilution_series(mixed, (0, 0), "Drug X", 10.0, "µM", factor=10, points=3, cell_line="A")
    mixed = dilution_series(
        mixed, (1, 0), "Antibody", 10.0, "µg/mL", factor=10, points=3, cell_line="A"
    )
    save_layout(mixed, copy / "plate_layout.json")
    wells = [f"{r}{c:02d}" for r in "AB" for c in (1, 2, 3)]
    headers = by_annotation(copy, wells, Comparison()).col_headers
    assert [h.split()[0] for h in headers] == ["Antibody"] * 3 + ["Drug"] * 3

    # raw panels of one run share one reopened plate
    sources.clear_cache()
    for well in ("A01", "A02", "B03"):
        load_panel(
            by_annotation(run, [well], Comparison()).panels[0].model_copy(update={"source": "raw"})
        )
    assert len(sources._CONTEXTS) == 1

    # folder sources (OME-Zarr) are fingerprinted: a changed store blocks regeneration
    raw = by_annotation(copy, ["A01"], Comparison(settings=ExportSettings(dpi=80)))
    raw = raw.model_copy(
        update={"panels": [p.model_copy(update={"source": "raw"}) for p in raw.panels]}
    )
    sources.clear_cache()
    export_figure(raw, tmp_path / "raw.png")
    sidecar = json.loads((tmp_path / "raw.png.json").read_text())
    assert any(v.startswith("tree:") for v in sidecar["checksums"].values())
    store = Path(next(k for k, v in sidecar["checksums"].items() if v.startswith("tree:")))
    chunk = next(p for p in store.rglob("*") if p.is_file() and p.name == "0.0.0.0")
    chunk.write_bytes(chunk.read_bytes())  # rewritten: new modification time
    import os
    import time

    os.utime(chunk, ns=(time.time_ns(), time.time_ns() + 10**9))
    with pytest.raises(ValueError, match="source files changed"):
        regenerate(tmp_path / "raw.png.json")


def load_plate_layout_of(folder: Path):  # type: ignore[no-untyped-def]
    from hcs_studio.model.layout import load_plate_layout

    return load_plate_layout(folder / "plate_layout.json")


def replace_line(annotation, line: str):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    return replace(annotation, cell_line=line)
