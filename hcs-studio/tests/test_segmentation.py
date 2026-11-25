"""Segmentation against synthetic images with a known answer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError
from scipy import ndimage

from hcs_studio.model.plate import ChannelInfo, ImageMetadata
from hcs_studio.processing.base import ImageData, LabelSet, ModuleError, create_module
from hcs_studio.segmentation import cellpose_backend, sam3_backend
from hcs_studio.segmentation.objects import (
    delete_labels,
    merge_labels,
    overlap_parents,
    point_parents,
    relabel_sequential,
    split_label,
    touches_edge,
)

META = ImageMetadata(
    channels=(ChannelInfo("DAPI"), ChannelInfo("Actin"), ChannelInfo("FISH")),
    pixel_size_um=0.5,
)


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> Any:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2


def nuclei_image(
    centres: list[tuple[int, int]],
    shape: tuple[int, int] = (128, 128),
    radius: float = 8,
    seed: int = 0,
) -> Any:
    """3 channels: nuclei discs, a cytoplasm halo (radius × 2), and dim background."""
    rng = np.random.default_rng(seed)
    nuclei = np.zeros(shape)
    halo = np.zeros(shape)
    for c in centres:
        nuclei[disc(shape, c, radius)] = 1000
        halo[disc(shape, c, radius * 2)] = 300
    noise = rng.normal(0, 10, (3, *shape))
    data = np.stack([100 + nuclei, 100 + halo, 100 + 0 * halo]) + noise
    return ImageData(np.asarray(data[:, None], dtype=np.float32), META)


CENTRES = [(30, 30), (30, 90), (90, 30), (90, 90), (60, 60)]


# --------------------------------------------------------------- helpers


def test_relabel_and_parent_lookup() -> None:
    labels = np.array([[[0, 7, 7, 3], [0, 0, 3, 3]]])
    assert relabel_sequential(labels).tolist() == [[[0, 2, 2, 1], [0, 0, 1, 1]]]
    child = np.array([[[1, 1, 2, 2, 2, 0, 3]]])
    parent = np.array([[[5, 5, 6, 6, 5, 6, 0]]])
    assert overlap_parents(child, parent).tolist() == [0, 5, 6, 0]  # largest overlap wins
    assert point_parents(child, parent).tolist() == [0, 5, 6, 0]  # parent under the centroid
    edge = np.zeros((1, 6, 6), dtype=int)
    edge[0, 0, 2] = 4
    edge[0, 2:4, 2:4] = 9
    assert touches_edge(edge) == {4}


def test_merge_split_delete() -> None:
    two = np.zeros((1, 40, 60), dtype=np.int32)
    two[0][disc((40, 60), (20, 20), 9)] = 3
    two[0][disc((40, 60), (20, 38), 9)] = 3  # two touching discs drawn as one object
    two[0][disc((40, 60), (5, 55), 3)] = 8
    parts = split_label(two, 3, np.array([[0, 20, 18], [0, 20, 40]]))
    assert sorted(np.unique(parts)) == [0, 3, 8, 9]
    left = parts[0, 20, 15]
    right = parts[0, 20, 42]
    assert {int(left), int(right)} == {3, 9}
    assert (parts > 0).sum() == (two > 0).sum()
    with pytest.raises(ModuleError, match="at least two seed points"):
        split_label(two, 3, np.array([[0, 20, 18], [0, 0, 0]]))
    with pytest.raises(ModuleError, match="does not exist"):
        split_label(two, 99, np.array([[0, 1, 1], [0, 2, 2]]))
    merged = merge_labels(parts, [9, 3])
    assert sorted(np.unique(merged)) == [0, 3, 8]
    with pytest.raises(ModuleError, match="at least two"):
        merge_labels(parts, [3, 3])
    assert sorted(np.unique(delete_labels(parts, [8]))) == [0, 3, 9]


def test_label_set_validation() -> None:
    with pytest.raises(ValueError, match="Z, Y, X"):
        LabelSet("x", np.zeros((4, 4), dtype=np.int32))
    with pytest.raises(ValueError, match="together"):
        LabelSet("x", np.zeros((1, 4, 4), dtype=np.int32), parent="nuclei")
    image = nuclei_image([])
    with pytest.raises(ValueError, match="has shape"):
        image.with_labels(LabelSet("x", np.zeros((1, 3, 3), dtype=np.int32)), "t")
    with pytest.raises(ModuleError, match="No segmentation named 'cells'"):
        image.label_set("cells")


# --------------------------------------------------------------- classical


def test_classical_finds_every_nucleus() -> None:
    image = nuclei_image(CENTRES)
    out = create_module("segment_classical", {"channel": 1, "min_distance_um": 4}).apply(
        image, None
    )
    nuclei = out.label_set("nuclei")
    assert nuclei.count == 5 and out.linear and out.history == ("segment_classical",)
    for c in CENTRES:
        assert nuclei.data[0, c[0], c[1]] > 0
    assert len({int(nuclei.data[0, c[0], c[1]]) for c in CENTRES}) == 5
    area = (nuclei.data == nuclei.data[0, 30, 30]).sum()
    assert area == pytest.approx(np.pi * 8**2, rel=0.15)
    assert out.reports[-1] == {"module": "segment_classical", "output": "nuclei", "objects": 5}


def test_watershed_separates_touching_nuclei() -> None:
    touching = nuclei_image([(60, 50), (60, 64)])  # 14 px apart, radius 8: they overlap
    split = create_module("segment_classical", {"channel": 1, "min_distance_um": 3}).apply(
        touching, None
    )
    whole = create_module("segment_classical", {"channel": 1, "split_touching": False}).apply(
        touching, None
    )
    assert split.label_set("nuclei").count == 2
    assert whole.label_set("nuclei").count == 1


def test_local_threshold_handles_uneven_background() -> None:
    image = nuclei_image(CENTRES)
    ramp = np.linspace(0, 1500, 128, dtype=np.float32)[None, None, None, :]
    data = image.data.copy()
    data[0:1] += ramp  # background brighter than the left nuclei on the right side
    uneven = ImageData(data, META)
    global_ = create_module("segment_classical", {"channel": 1}).apply(uneven, None)
    local = create_module(
        "segment_classical", {"channel": 1, "method": "local", "local_block_um": 20}
    ).apply(uneven, None)
    assert global_.label_set("nuclei").count != 5
    assert local.label_set("nuclei").count == 5


def test_classical_3d_gives_unique_ids_per_plane() -> None:
    image = nuclei_image(CENTRES)
    stack = ImageData(np.repeat(image.data, 3, axis=1), META)
    out = create_module("segment_classical", {"channel": 1}).apply(stack, None)
    labels = out.label_set("nuclei").data
    assert labels.shape == (3, 128, 128)
    assert len(np.unique(labels)) - 1 == 15  # planes segmented separately, IDs not reused


# ----------------------------------------------------------- hierarchy


def test_cells_by_expansion_keep_nucleus_ids() -> None:
    image = create_module("segment_classical", {"channel": 1}).apply(nuclei_image(CENTRES), None)
    out = create_module("expand_cells", {"distance_um": 4}).apply(image, None)
    nuclei, cells = out.label_set("nuclei"), out.label_set("cells")
    assert cells.parent == "nuclei" and cells.count == 5
    assert np.all(cells.data[nuclei.data > 0] == nuclei.data[nuclei.data > 0])
    assert cells.parent_ids is not None and cells.parent_ids[1:].tolist() == [1, 2, 3, 4, 5]
    grown = (cells.data == 1).sum() - (nuclei.data == 1).sum()
    assert grown > 0
    radius = np.sqrt((cells.data == cells.data[0, 30, 30]).sum() / np.pi)
    assert radius == pytest.approx(8 + 8, abs=1.5)  # 4 µm = 8 px

    masked = create_module("expand_cells", {"distance_um": 20, "mask_channel": 2}).apply(
        image, None
    )
    radius = np.sqrt(
        (masked.label_set("cells").data == masked.label_set("cells").data[0, 30, 30]).sum() / np.pi
    )
    assert radius == pytest.approx(16, abs=1.5)  # limited by the 16 px halo, not 40 px


def test_spots_are_found_and_linked_to_cells() -> None:
    image = nuclei_image(CENTRES)
    data = image.data.copy()
    spots_truth = [(26, 34), (34, 26), (95, 95), (64, 56), (5, 5)]  # last one outside all cells
    for y, x in spots_truth:
        data[2, 0, y, x] += 3000
    data[2, 0] = ndimage.gaussian_filter(data[2, 0], 0.8)
    image = ImageData(data, META)
    image = create_module("segment_classical", {"channel": 1}).apply(image, None)
    image = create_module("expand_cells", {"distance_um": 6}).apply(image, None)
    out = create_module("detect_spots", {"channel": 3, "diameter_um": 1.0}).apply(image, None)
    spots = out.label_set("spots")
    assert spots.count == 4 and spots.parent == "cells"
    cells = out.label_set("cells").data
    assert spots.parent_ids is not None
    for spot_id in range(1, 5):
        centre = ndimage.center_of_mass(spots.data == spot_id)
        cell = cells[tuple(round(v) for v in centre)]
        assert spots.parent_ids[spot_id] == cell > 0
    counts = np.bincount(spots.parent_ids[1:], minlength=6)
    assert counts[int(cells[0, 30, 30])] == 2  # two spots in the top-left cell

    everywhere = create_module(
        "detect_spots", {"channel": 3, "diameter_um": 1.0, "parent": None}
    ).apply(image, None)
    assert everywhere.label_set("spots").count == 5


def test_filters() -> None:
    centres = [(4, 60), (60, 60), (100, 100)]
    image = create_module("segment_classical", {"channel": 1}).apply(nuclei_image(centres), None)
    image = create_module("expand_cells", {"distance_um": 3}).apply(image, None)
    out = create_module("filter_objects", {"labels": "nuclei"}).apply(image, None)
    assert out.label_set("nuclei").count == 2  # the one cut by the top edge is gone
    assert out.label_set("cells").count == 2, "its cell should go too"
    assert out.reports[-1]["removed"]["edge"] == 1

    big = create_module(
        "filter_objects", {"labels": "nuclei", "remove_edge": False, "min_size": 45}
    ).apply(image, None)
    assert big.label_set("nuclei").count == 2  # full nuclei ≈ 50 µm²; the cut one ≈ 40 µm²

    data = image.data.copy()
    data[0][
        :, image.label_set("nuclei").data[0] == image.label_set("nuclei").data[0, 100, 100]
    ] += 2000
    bright = ImageData(data, META, labels=image.labels)
    gated = create_module(
        "filter_objects",
        {
            "labels": "nuclei",
            "remove_edge": False,
            "intensity_channel": 1,
            "min_mean_intensity": 2000,
        },
    ).apply(bright, None)
    assert gated.label_set("nuclei").count == 1

    enhanced = create_module("contrast").apply(bright, None)
    with pytest.raises(ModuleError, match="linear data"):
        create_module(
            "filter_objects", {"labels": "nuclei", "intensity_channel": 1, "min_mean_intensity": 1}
        ).apply(enhanced, None)
    with pytest.raises(ModuleError, match="Size gates"):
        create_module("filter_objects", {"min_size": 1}).validate(
            ImageMetadata(channels=META.channels)
        )


# ----------------------------------------------------------------- Cellpose


class FakeCellpose:
    """Records eval() calls; segments by thresholding the first channel."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def eval(self, x: Any, **kwargs: Any) -> tuple[Any, None, None]:
        self.calls.append({"shape": x.shape, **kwargs})
        first = x[0]
        labelled = np.zeros(first.shape, dtype=np.int32)
        if first.ndim == 3:
            labelled = ndimage.label(first > 500)[0]
        else:
            labelled = ndimage.label(first > 500)[0]
        return labelled, None, None


def test_cellpose_step_calls_the_model_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCellpose()
    monkeypatch.setattr(cellpose_backend, "get_model", lambda model, gpu: fake)
    image = nuclei_image(CENTRES)
    out = create_module(
        "segment_cellpose", {"channel": 1, "second_channel": 2, "diameter_um": 8, "device": "cpu"}
    ).apply(image, None)
    assert (
        out.label_set("nuclei").count == 5 and out.label_set("nuclei").source == "cellpose cpsam_v2"
    )
    call = fake.calls[0]
    assert call["shape"] == (2, 128, 128) and call["diameter"] == 16 and call["channel_axis"] == 0
    assert out.reports[-1]["gpu"] is False

    stack = ImageData(
        np.repeat(image.data, 4, axis=1), META.__class__(**{**META.__dict__, "z_step_um": 1.0})
    )
    fake.calls.clear()
    out3d = create_module("segment_cellpose", {"channel": 1, "do_3d": True, "device": "cpu"}).apply(
        stack, None
    )
    assert fake.calls[0]["do_3D"] is True and fake.calls[0]["anisotropy"] == 2.0
    assert out3d.label_set("nuclei").data.shape == (4, 128, 128)
    fake.calls.clear()
    planes = create_module("segment_cellpose", {"channel": 1, "device": "cpu"}).apply(stack, None)
    assert len(fake.calls) == 4 and len(np.unique(planes.label_set("nuclei").data)) - 1 == 20

    nuclei = create_module("segment_classical", {"channel": 1}).apply(image, None)
    cells = create_module(
        "segment_cellpose", {"channel": 2, "output": "cells", "parent": "nuclei", "device": "cpu"}
    ).apply(nuclei, None)
    assert cells.label_set("cells").parent == "nuclei"


def test_cellpose_errors() -> None:
    with pytest.raises(ModuleError, match="Unknown Cellpose model 'nope'"):
        cellpose_backend.get_model("nope", False)
    with pytest.raises(ModuleError, match="Channel 4 does not exist"):
        create_module("segment_cellpose", {"channel": 4}).validate(META)
    with pytest.raises(ValidationError):
        create_module("segment_cellpose", {"channel": 1, "device": "tpu"})


@pytest.mark.skipif(
    not (Path.home() / ".cellpose" / "models" / "cpsam_v2").exists(),
    reason="Cellpose cpsam_v2 weights not downloaded",
)
def test_real_cellpose_counts_separated_nuclei() -> None:
    centres = [(y, x) for y in range(24, 220, 48) for x in range(24, 220, 48)]  # 5 × 5 nuclei
    rng = np.random.default_rng(5)
    shape = (240, 240)
    nuclei = np.zeros(shape)
    for c in centres:
        nuclei[disc(shape, c, 9)] = 1500
    data = ndimage.gaussian_filter(nuclei, 1.5) + 100 + rng.normal(0, 15, shape)
    image = ImageData(
        np.asarray(data[None, None], dtype=np.float32),
        ImageMetadata(channels=(ChannelInfo("DAPI"),), pixel_size_um=0.5),
    )
    out = create_module("segment_cellpose", {"channel": 1, "device": "cpu"}).apply(image, None)
    labels = out.label_set("nuclei")
    assert labels.count == 25
    for c in centres:
        assert labels.data[0, c[0], c[1]] > 0
    assert len({int(labels.data[0, y, x]) for y, x in centres}) == 25
    areas = np.bincount(labels.data.ravel())[1:]
    assert np.all(np.abs(areas - np.pi * 9**2) < 0.2 * np.pi * 9**2)


def test_fine_tune_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cellpose")
    for i in range(3):
        tifffile.imwrite(tmp_path / f"f{i}_img.tif", np.ones((2, 32, 32), dtype=np.float32))
        masks = np.zeros((32, 32), dtype=np.int32)
        for k in range(6 if i < 2 else 2):
            masks[k * 5 : k * 5 + 4, 2:6] = k + 1
        tifffile.imwrite(tmp_path / f"f{i}_masks.tif", masks)
    seen: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, gpu: bool, pretrained_model: str) -> None:
            seen["base"] = pretrained_model
            self.net = "net"

    def fake_train(net: Any, **kwargs: Any) -> tuple[str, Any, Any]:
        seen.update(kwargs)
        return str(tmp_path / "models" / kwargs["model_name"]), np.array([1.0, 0.5]), np.array([])

    import cellpose.models
    import cellpose.train

    monkeypatch.setattr(cellpose.models, "CellposeModel", FakeModel)
    monkeypatch.setattr(cellpose.train, "train_seg", fake_train)
    path, history = cellpose_backend.fine_tune(tmp_path, epochs=3, model_name="mine", device="cpu")
    assert path == tmp_path / "models" / "mine" and seen["base"] == "cpsam_v2"
    assert len(seen["train_data"]) == 2  # the image with only 2 objects is skipped
    assert seen["channel_axis"] == 0 and seen["n_epochs"] == 3
    assert history["train_loss"] == [1.0, 0.5] and history["images_train"] == 2

    (tmp_path / "f9_img.tif").write_bytes(b"")
    with pytest.raises(ModuleError, match=r"Masks missing for: f9_masks\.tif"):
        cellpose_backend.fine_tune(tmp_path)


# -------------------------------------------------------------------- SAM 3


@pytest.fixture
def fake_sam() -> Any:
    client = sam3_backend.Sam3Client(sys.executable, fake=True, device="cpu")
    yield client
    client.close()


def test_sam_worker_protocol_with_the_stand_in_model(fake_sam: Any) -> None:
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    rgb[disc((64, 80), (20, 20), 6)] = 255
    rgb[disc((64, 80), (40, 60), 8)] = 255
    fake_sam.set_image(rgb)
    everything = fake_sam.text("nucleus")
    assert everything.labels.shape == (64, 80) and everything.labels.dtype == np.int32
    assert everything.labels.max() == 2 and everything.scores == [0.9, 0.9]
    boxed = fake_sam.box([50, 30, 70, 50])
    assert boxed.labels.max() == 1 and boxed.labels[40, 60] == 1 and boxed.labels[20, 20] == 0
    clicked = fake_sam.points([[20, 20]], [1])
    assert clicked.labels[20, 20] == 1 and clicked.labels[40, 60] == 0
    with pytest.raises(ModuleError, match="SAM 3 needs an RGB image"):
        fake_sam.set_image(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ModuleError, match="Unknown command"):
        fake_sam._call({"cmd": "dance"})


def test_sam_worker_failure_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModuleError, match="Could not start the SAM 3 worker"):
        sam3_backend.Sam3Client(str(tmp_path / "no-python"))
    if sys.platform.startswith("win"):
        return  # the broken worker below is a shebang script
    broken = tmp_path / "python"
    broken.write_text("#!/bin/sh\necho 'boom' >&2\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)
    with pytest.raises(ModuleError, match=r"(?s)worker stopped.*boom"):
        sam3_backend.Sam3Client(str(broken))


def test_sam_paint_resolves_overlaps_by_score() -> None:
    from hcs_studio.segmentation.sam3_worker import paint

    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:10, 5:10] = True
    labels, scores = paint(np.stack([a, b]), [0.4, 0.9], (10, 10))
    assert scores == [0.9, 0.4]
    assert labels[6, 6] == 1 and labels[3, 3] == 2 and labels[9, 9] == 1


def test_sam_pipeline_step_and_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    assert not sam3_backend.Sam3Settings.load().configured
    with pytest.raises(ModuleError, match=r"SAM 3 is not set up"):
        create_module("segment_sam3", {"prompt": "nucleus", "channels": [1]})._client()  # type: ignore[attr-defined]
    sam3_backend.Sam3Settings(python=sys.executable, checkpoint="/x/sam3.pt").save()
    assert sam3_backend.Sam3Settings.load() == sam3_backend.Sam3Settings(
        sys.executable, "/x/sam3.pt", "auto"
    )

    module = create_module(
        "segment_sam3", {"prompt": "nucleus", "channels": [1], "output": "nuclei"}
    )
    module.fake_model = True  # type: ignore[attr-defined]
    out = module.apply(nuclei_image(CENTRES), None)
    assert out.label_set("nuclei").count == 5 and out.reports[-1]["prompt"] == "nucleus"
    assert out.label_set("nuclei").source == "sam3 'nucleus'"
    with pytest.raises(ValidationError):
        create_module("segment_sam3", {"prompt": "x", "channels": [1, 2, 3, 1]})
    rgb = sam3_backend.to_rgb(nuclei_image(CENTRES).data[:, 0], [0, 1])
    assert rgb.shape == (128, 128, 3) and rgb[30, 30, 0] == 255 and int(np.max(rgb[..., 2])) == 0


@pytest.mark.skipif(not os.environ.get("HCS_SAM3_PYTHON"), reason="set HCS_SAM3_PYTHON to run")
def test_real_sam3_api_with_random_weights() -> None:
    """Runs the real SAM 3 code (random weights) through the worker: API compatibility only."""
    client = sam3_backend.Sam3Client(
        os.environ["HCS_SAM3_PYTHON"], device="cpu", random_weights=True, confidence=0.0
    )
    try:
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[disc((128, 128), (64, 64), 20)] = 200
        client.set_image(rgb)
        for result in (
            client.text("nucleus"),
            client.box([40, 40, 90, 90]),
            client.points([[64, 64]], [1]),
        ):
            assert result.labels.shape == (128, 128) and result.labels.dtype == np.int32
    finally:
        client.close()


# ------------------------------------------------------------- run folder


def test_run_writes_masks_parent_links_and_outlined_previews(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from tests.builders import make_imagexpress

    def image(row: int, col: int, site: int, wave: int) -> Any:
        data = nuclei_image(CENTRES, shape=(128, 128), seed=site).data[wave, 0]
        return np.clip(data, 0, None).astype(np.uint16)

    root = make_imagexpress(tmp_path / "raw" / "ix", [(0, 0)], n_sites=2, image_fn=image)
    pipeline = Pipeline.from_text(
        """
name: seg
steps:
  - module: segment_classical
    params: {channel: 1}
  - module: expand_cells
    params: {distance_um: 2}
"""
    )
    result = run_pipeline(open_plate(root), pipeline, checksums=False)
    well = result.folder / "wells" / "A01"
    nuclei = tifffile.imread(well / "masks" / "f001_nuclei.ome.tif")
    assert nuclei.dtype == np.uint16 and nuclei.shape == (128, 128) and len(np.unique(nuclei)) == 6
    cells = tifffile.imread(well / "masks" / "f001_cells.ome.tif")
    assert np.all(cells[nuclei > 0] == nuclei[nuclei > 0])
    links = (well / "masks" / "f001_cells_parents.csv").read_text(encoding="utf-8").splitlines()
    assert links[0] == "object_id,nuclei_id" and links[1:] == [f"{i},{i}" for i in range(1, 6)]
    summary = json.loads((well / "well_summary.json").read_text(encoding="utf-8"))
    objects = summary["fields"][0]["objects"]
    assert objects["nuclei"]["count"] == 5 and objects["cells"]["parent"] == "nuclei"
    import imagecodecs

    preview = imagecodecs.png_decode((well / "previews" / "f001.png").read_bytes())
    assert preview.shape[-1] == 3  # composite without outlines burned in
    nuclei_overlay = imagecodecs.png_decode(
        (well / "previews" / "f001_outline_nuclei.png").read_bytes()
    )
    cells_overlay = imagecodecs.png_decode(
        (well / "previews" / "f001_outline_cells.png").read_bytes()
    )
    assert nuclei_overlay.shape == (*preview.shape[:2], 4)
    drawn = nuclei_overlay[..., 3] > 0
    assert drawn.sum() > 50 and np.all(nuclei_overlay[drawn][:, :3] == (255, 255, 0))
    assert (cells_overlay[..., 3] > 0).sum() > 50 and np.all(
        cells_overlay[cells_overlay[..., 3] > 0][:, :3] == (255, 0, 255)
    )
    assert not nuclei_overlay[~drawn].any()  # transparent elsewhere
    assert (well / "previews" / "f001_raw.png").exists() and (
        well / "previews" / "f001_corrected.png"
    ).exists()
    files = summary["fields"][0]["files"]
    assert files["overlays"] == {
        "nuclei": "previews/f001_outline_nuclei.png",
        "cells": "previews/f001_outline_cells.png",
    }


@pytest.mark.skipif(
    not os.environ.get("HCS_SAM3_PYTHON") or not os.environ.get("HCS_SAM3_WEIGHTS"),
    reason="set HCS_SAM3_PYTHON and HCS_SAM3_WEIGHTS=1 (needs the gated SAM 3 weights)",
)
def test_real_sam3_agrees_with_cellpose_on_real_nuclei() -> None:
    """SAM 3 'cell nucleus' vs Cellpose on scikit-image's confocal cells3d (plane 30)."""
    import skimage.data

    plane = skimage.data.cells3d()[30, 1].astype(np.float32)
    meta = ImageMetadata(channels=(ChannelInfo("DNA"),), pixel_size_um=0.5)
    reference = (
        create_module("segment_cellpose", {"channel": 1, "device": "cpu"})
        .apply(ImageData(plane[None, None], meta), None)
        .label_set("nuclei")
        .data[0]
    )
    client = sam3_backend.Sam3Client(os.environ["HCS_SAM3_PYTHON"], device="cpu")
    try:
        client.set_image(sam3_backend.to_rgb(plane[None], [0]))
        found = client.text("cell nucleus", 0.5).labels
    finally:
        client.close()
    matched = 0
    for i in np.unique(found)[1:]:
        mask = found == i
        candidates = [j for j in np.unique(reference[mask]) if j]
        best = max(
            ((mask & (reference == j)).sum() / (mask | (reference == j)).sum() for j in candidates),
            default=0,
        )
        matched += best > 0.5
    n_reference = len(np.unique(reference)) - 1
    assert matched >= 0.9 * n_reference
