"""Phase 16c: ilastik Object Classification against the fake ilastik — classifying HCS Studio's
own objects (Cellpose-like masks sent as a label image), majority vote per object, per-well
class fractions on the plate and as a pharmacology readout (dose–response), and projects
trained on probabilities (ilastik's own objects)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.segmentation.ilastik_backend.objects import majority_classes
from tests import test_ilastik_steps
from tests.test_ilastik_steps import calls, make_project

ilastik = test_ilastik_steps.ilastik  # the fixture (fake ilastik)

EC50_M = 0.3e-6


def _object_project(
    path: Path, min_area: int = 80, workflow: str = "ObjectClassificationWorkflowLabels"
) -> Path:
    import h5py

    make_project(path, workflow=workflow, names=("small", "large"))
    with h5py.File(path, "a") as f:
        f["fake"].attrs["min_area"] = min_area
    return path


@dataclass
class SizedNuclei:
    """12 discs per field on a 4 × 3 grid; ``large(row, col)`` of them have radius 6 px
    (113 px), the others radius 3 px (29 px)."""

    large: Any
    name: str = "nuclei"

    def draw(self, ctx: Any) -> dict[str, Any]:
        n_large = int(self.large(ctx.row, ctx.col))
        yy, xx = np.mgrid[: ctx.image.shape[-2], : ctx.image.shape[-1]]
        for i in range(12):
            cy, cx = 16 + 24 * (i // 4), 12 + 24 * (i % 4)
            radius = 6 if i < n_large else 3
            ctx.image[0, :, (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] += 2000
        return {"count": 12, "large": n_large}


def _dose_plate(tmp_path: Path) -> tuple[Any, Path, dict[str, float]]:
    """Row B: vehicle B01–B02, Drug X 10 µM → 1.5 nM (1:3) B03–B10, positive control B11–B12;
    the share of large nuclei follows c / (c + EC50)."""
    from hcs_studio.io.base import open_plate
    from hcs_studio.io.layout_store import LayoutStore, plate_key
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import dilution_series, set_role
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate

    layout = PlateLayout.for_plate_type(384, barcode="OBJ01")
    layout = set_role(
        layout, [(1, 0), (1, 1)], "negative_control", vehicle="DMSO", cell_line="HeLa"
    )
    layout = set_role(layout, [(1, 10), (1, 11)], "positive_control", cell_line="HeLa")
    layout = dilution_series(
        layout, (1, 2), "Drug X", 10.0, "µM", factor=3, points=8, cell_line="HeLa"
    )
    conc = {}
    for (row, col), well in layout.wells.items():
        if well.treatments:
            conc[(row, col)] = well.treatments[0].concentration

    def large(row: int, col: int) -> int:
        if col >= 10:
            return 12
        c = conc.get((row, col), 0.0)
        return round(12 * c / (c + EC50_M))

    plate = SyntheticPlate(
        wells=[(1, c) for c in range(12)],
        channels=("DAPI",),
        size=96,
        name="OBJ01",
        features=[SizedNuclei(large), ShotNoise(sd=3)],
    )
    path, _ = plate.write(tmp_path / "raw")
    saved = LayoutStore(plate_key(barcode="OBJ01")).save(layout)
    layout_file = LayoutStore(plate_key(barcode="OBJ01")).path(saved.version)
    return open_plate(path), layout_file, {f"B{c + 1:02d}": large(1, c) / 12 for c in range(12)}


def _pipeline(project: Path) -> Any:
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    return Pipeline(
        steps=[
            PipelineStep(
                module="segment_classical",
                params={"channel": 1, "split_touching": False, "min_size_px": 10},
            ),
            PipelineStep(
                module="ilastik_objects",
                params={"project": str(project), "labels": "nuclei", "channels": [1]},
            ),
            PipelineStep(module="measure_objects", params={"labels": ["nuclei"], "texture": False}),
            PipelineStep(module="qc"),
            PipelineStep(module="ilastik_object_classes", params={"object_type": "nuclei"}),
        ]
    )


def test_majority_vote() -> None:
    objects = np.zeros((1, 4, 6), dtype=np.int32)
    objects[0, :, :3], objects[0, :, 3:] = 5, 9
    classes = np.zeros((1, 4, 6), dtype=np.uint8)
    classes[0, :, :3] = 2
    classes[0, 0, 0] = 1  # one pixel disagrees
    classes[0, :, 3:] = 0  # ilastik ignored object 9 (e.g. too small for its filter)
    assert majority_classes(objects, classes) == [(5, 2, 0.9167), (9, 0, 1.0)]
    assert majority_classes(np.zeros((1, 2, 2), dtype=np.int32), classes[:, :2, :2]) == []


def test_classifies_existing_objects_per_well(ilastik: dict[str, Any], tmp_path: Path) -> None:
    """Done-when: Object Classification on existing masks gives per-well class fractions on
    the plate (per-well table, readouts) and in pharmacology dose-response curves."""
    from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.readouts import resolve

    plate, layout_file, truth = _dose_plate(tmp_path)
    project = _object_project(tmp_path / "size.ilp")
    from hcs_studio.model.layout import load_plate_layout

    result = run_pipeline(
        plate,
        _pipeline(project),
        output_root=tmp_path / "runs",
        workers=1,
        checksums=False,
        layout=load_plate_layout(layout_file),
        layout_path=layout_file,
    )
    assert not result.failed_wells
    folder = result.folder
    made = calls(ilastik["log"])
    assert len(made) == 12  # one call per well
    assert all(
        "--label_image" in c["argv"] and "--export_source=Object Predictions" in c["argv"]
        for c in made
    )
    classes = pd.read_parquet(
        folder / "wells" / "B05" / "ilastik" / "object_classes_ilastik.parquet"
    )
    assert len(classes) == 12 and set(classes["class_name"]) <= {"small", "large"}
    assert (classes["agreement"] == 1.0).all()
    wells = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    fractions = wells["class_ilastik_large_fraction"].to_dict()
    for well, expected in truth.items():
        assert fractions[well] == pytest.approx(expected, abs=1e-9), well
    assert wells["class_ilastik_small_count"].to_dict()["B01"] == 12
    assert resolve("class_ilastik_large_fraction") is not None
    objects = pd.read_parquet(folder / "wells" / "B11" / "objects_annotations.parquet")
    assert (objects["class_ilastik"] == "large").all()
    # the class fraction as a dose-response readout (Phase 13)
    pharm = run_pharmacology(
        [folder],
        PharmOptions(
            readouts=["class_ilastik_large_fraction"],
            normalization="percent_inhibition",
            figures=False,
            pdf=False,
        ),
    )
    curve = pharm.readouts["class_ilastik_large_fraction"].curves[0]
    assert curve.compound == "Drug X"
    assert 10**curve.fit.log_ec50 == pytest.approx(EC50_M, rel=0.5)


def test_project_trained_on_probabilities(ilastik: dict[str, Any], tmp_path: Path) -> None:
    """ilastik makes its own objects from the class probabilities: two calls per well
    (identities, predictions) plus the pixel classification; a new label set."""
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.pipeline.schema import Pipeline, PipelineStep

    plate, layout_file, _truth = _dose_plate(tmp_path)
    pixel = make_project(tmp_path / "pixel.ilp", lo=200, hi=1200)
    objects = _object_project(
        tmp_path / "size_pred.ilp", workflow="ObjectClassificationWorkflowPrediction"
    )
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="ilastik_pixel", params={"project": str(pixel), "channels": [1]}),
            PipelineStep(
                module="ilastik_objects",
                params={
                    "project": str(objects),
                    "input": "prediction_maps",
                    "channels": [1],
                    "source_channels": ["ilastik_background", "ilastik_nuclei"],
                },
            ),
            PipelineStep(
                module="measure_objects",
                params={"labels": ["ilastik_objects"], "channels": [1], "texture": False},
            ),
            PipelineStep(
                module="ilastik_object_classes", params={"object_type": "ilastik_objects"}
            ),
        ]
    )
    from hcs_studio.model.layout import load_plate_layout

    result = run_pipeline(
        plate,
        pipeline,
        output_root=tmp_path / "runs",
        workers=1,
        checksums=False,
        wells=[(1, 0), (1, 11)],
        layout=load_plate_layout(layout_file),
    )
    assert not result.failed_wells, result.failed_wells
    sources = [
        next(a for a in c["argv"] if a.startswith("--export_source")) for c in calls(ilastik["log"])
    ]
    assert sources.count("--export_source=Object Identities") == 2
    assert sources.count("--export_source=Object Predictions") == 2
    wells = pd.read_csv(result.folder / "tables" / "per_well_summary.csv").set_index("well")
    assert wells["ilastik_objects_objects_measured"].to_dict() == {"B01": 12, "B12": 12}
    fractions = wells["class_ilastik_large_fraction"].to_dict()
    assert fractions == {"B01": 0.0, "B12": 1.0}
    summary = json.loads(
        (result.folder / "wells" / "B12" / "ilastik" / "object_classes_ilastik.json").read_text()
    )
    assert summary["by_class"] == {"large": 12}


def test_object_step_checks(ilastik: dict[str, Any], tmp_path: Path) -> None:
    from hcs_studio.model.plate import ChannelInfo, ImageMetadata
    from hcs_studio.processing.base import ImageData, ModuleError, create_module

    meta = ImageMetadata(channels=(ChannelInfo("DAPI"),), pixel_size_um=0.65)
    with pytest.raises(ModuleError, match="needs an Object Classification project"):
        create_module("ilastik_objects", {"project": str(ilastik["project"])}).validate(meta)
    project = _object_project(tmp_path / "o.ilp")
    step = create_module("ilastik_objects", {"project": str(project), "input": "prediction_maps"})
    with pytest.raises(ModuleError, match="needs ilastik class channels"):
        step.validate(meta)
    labels_step = create_module("ilastik_objects", {"project": str(project), "labels": "cells"})
    image = ImageData(np.zeros((1, 1, 32, 32), dtype=np.float32), meta)
    with pytest.raises(ModuleError, match="No segmentation named 'cells'"):
        labels_step.apply(image, None)


def test_objects_preset(tmp_path: Path) -> None:
    from hcs_studio.segmentation.ilastik_backend.presets import pixel_preset
    from hcs_studio.segmentation.ilastik_backend.project import read_project

    project = _object_project(tmp_path / "phenotype.ilp")
    pipeline = pixel_preset(project, read_project(project), channels=[1, 2])
    assert [s.module for s in pipeline.steps] == [
        "segment_cellpose",
        "ilastik_objects",
        "measure_objects",
        "qc",
        "ilastik_object_classes",
    ]
    assert pipeline.steps[1].params["channels"] == [1, 2]
