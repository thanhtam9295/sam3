"""Phase 25: versioned ground-truth sets, leakage-free splits, validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.synthetic.groundtruth import ground_truth_items
from hcs_studio.training import groundtruth as gt
from hcs_studio.training.validate import segmentation_steps, validate_segmentation


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


def test_versions_hashes_and_round_trip(home: Path) -> None:
    items = ground_truth_items("easy", plates=1, wells=2, fields=1)
    first = gt.create_set("nuclei", items, channels=["Hoechst"], author="me")
    assert first.ref == "nuclei@1" and len(first.content_sha256) == 64
    image, masks = gt.read_item(first, first.items[0])
    assert image.shape == (1, 128, 128) and masks is not None
    assert np.array_equal(masks > 0, items[0][2] > 0) and int(np.max(masks)) == int(np.max(items[0][2]))
    # an edit is a new version; the old one stays as it was
    corrected = items[0][2].copy()
    corrected[corrected == 1] = 0
    second = gt.create_set("nuclei", [(items[0][0], items[0][1], corrected)], based_on="nuclei@1")
    assert second.version == 2 and second.parent_version == 1 and len(second.items) == 2
    assert second.content_sha256 != first.content_sha256
    assert (
        gt.load_set("nuclei").version == 2
        and gt.load_set("nuclei@1").content_sha256 == first.content_sha256
    )
    assert [s.ref for s in gt.list_sets()] == ["nuclei@2"]
    with pytest.raises(ValueError, match="no version 7"):
        gt.load_set("nuclei@7")
    with pytest.raises(ValueError, match="unique"):
        gt.create_set("x", [items[0], items[0]])
    with pytest.raises(ValueError, match="masks"):
        gt.create_set("x", [(items[0][0], items[0][1], np.zeros((5, 5)))])
    assert not (home / "home" / "annotations" / "x").exists() or not gt.versions("x")


def test_splits_by_well_and_plate_never_leak(home: Path) -> None:
    items = ground_truth_items("easy", plates=3, wells=4, fields=3, size=48)
    base = gt.create_set("s", items)
    by_well = gt.split_set("s@1", by="well", fractions=(0.5, 0.25, 0.25), seed=1)
    assert by_well.version == 2 and by_well.content_sha256 == base.content_sha256  # same content
    assert gt.check_leakage(by_well.items, "well") == {}
    counts = by_well.counts()
    assert counts == {"train": 18, "validation": 9, "test": 9, "unassigned": 0}  # 12 wells × 3
    by_plate = gt.split_set("s@1", by="plate", fractions=(1, 1, 1))
    assert {i.plate for i in by_plate.subset("test")}.isdisjoint(
        {i.plate for i in by_plate.subset("train")}
    )
    assert gt.check_leakage(by_plate.items, "plate") == {}
    held = gt.assign_splits(base.items, by="plate", test_groups=["P3"], fractions=(0.5, 0.5, 0))
    assert {i.plate for i in held if i.split == "test"} == {"P3"}
    leaky = [
        i.model_copy(update={"split": s})
        for i, s in zip(base.items[:2], ["train", "test"], strict=True)
    ]
    assert gt.check_leakage(leaky, "well") == {"P1|B02": ["test", "train"]}
    assert (
        json.loads((gt.set_folder("s", 3) / "set.json").read_text())["split_rule"]["by"] == "plate"
    )
    with pytest.raises(ValueError, match="Unknown"):
        gt.assign_splits(base.items, test_groups=["nope"])


def test_import_pairs_with_provenance_and_export(home: Path, tmp_path: Path) -> None:
    import tifffile

    folder = tmp_path / "pairs"
    folder.mkdir()
    for item, image, masks in ground_truth_items("hard", plates=1, wells=2, fields=1, size=48):
        stem = f"PL7_{item.well}_f1"
        tifffile.imwrite(folder / f"{stem}_img.tif", image[0])
        tifffile.imwrite(folder / f"{stem}_masks.tif", masks.astype(np.uint16))
    provenance = pd.DataFrame({"file": ["PL7_B02_f1_img.tif"], "plate": ["PL7"], "field": [1],
                               "annotator": ["anna"]})  # fmt: skip
    ann = gt.import_pairs(folder, "hard", provenance=provenance,
                          defaults={"cell_line": "HCT116", "pixel_size_um": 0.65})  # fmt: skip
    first, second = ann.items
    assert (first.plate, first.well, first.annotator, first.cell_line) == (
        "PL7",
        "B02",
        "anna",
        "HCT116",
    )
    assert (second.plate, second.well, second.annotator) == ("", "B03", "")
    out = tmp_path / "export"
    assert len(gt.export_pairs("hard", None, out)) == 2 and (out / "PL7_B03_f1_masks.tif").exists()
    with pytest.raises(ValueError, match="no items"):
        gt.export_pairs("hard", "test", tmp_path / "none")


def test_validation_per_cell_line(home: Path) -> None:
    items = ground_truth_items("easy", plates=1, wells=2, fields=2) + ground_truth_items(
        "hard", plates=1, wells=2, fields=2
    )
    gt.create_set("mixed", items)
    steps = [PipelineStep(module="segment_classical", params={"channel": 1})]
    result = validate_segmentation("mixed", steps, split=None, keep_predictions=True)
    by_line = result.report["by_cell_line"]
    assert set(by_line) == {"easy", "hard"}
    assert by_line["easy"]["ap50"] > 0.7 > 0.3 > by_line["hard"]["ap50"]
    assert by_line["hard"]["merges"] > 20  # the default watershed merges the clusters
    assert len(result.fields) == 8 and len(result.predictions) == 8
    with pytest.raises(ValueError, match="no items in split"):
        validate_segmentation("mixed", steps, split="test")
    pipeline = Pipeline(
        steps=[*steps, PipelineStep(module="measure_objects", params={"labels": ["nuclei"]})]
    )
    assert [s.module for s in segmentation_steps(pipeline)] == ["segment_classical"]


def test_gt_and_validate_commands(home: Path, tmp_path: Path) -> None:
    import tifffile

    from hcs_studio.app import main

    folder = tmp_path / "pairs"
    folder.mkdir()
    for item, image, masks in ground_truth_items("easy", plates=2, wells=3, fields=1, size=64):
        tifffile.imwrite(folder / f"{item.plate}_{item.well}_img.tif", image[0])
        tifffile.imwrite(folder / f"{item.plate}_{item.well}_masks.tif", masks.astype(np.uint16))
    assert main(["gt", "import", str(folder), "--name", "cli", "--cell-line", "U2OS",
                 "--pixel-size", "0.65", "--reason", "first set"]) == 0  # fmt: skip
    assert main(["gt", "split", "cli", "--by", "well", "--fractions", "0.5,0,0.5"]) == 0
    assert main(["gt", "show", "cli@2"]) == 0 and main(["gt", "list"]) == 0
    assert main(["gt", "export", "cli", "--split", "train", "--output", str(tmp_path / "tr")]) == 0
    pipeline = tmp_path / "seg.yaml"
    Pipeline(steps=[PipelineStep(module="segment_classical", params={"channel": 1})]).save(pipeline)
    out = tmp_path / "val"
    assert (
        main(["validate-seg", "--set", "cli", "--pipeline", str(pipeline), "--output", str(out)])
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert (
        metrics["set"] == "cli@2"
        and metrics["split"] == "test"
        and metrics["overall"]["fields"] == 3
    )
    assert main(["gt", "show", "missing"]) == 2
    from sqlalchemy import select

    from hcs_studio.db import session_scope
    from hcs_studio.db.models import EventRecord

    with session_scope() as session:
        kinds = [e.type for e in session.scalars(select(EventRecord))]
    assert "groundtruth.import" in kinds and "groundtruth.split" in kinds
