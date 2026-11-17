"""Known-answer tests for Phase 8: gates, classifier, clustering, spatial statistics, and
table steps in a run (with stitching)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.analysis.classifier import (
    load_classifier,
    robust_stats,
    save_classifier,
    train_classifier,
    uncertain_objects,
)
from hcs_studio.analysis.spatial import (
    Window,
    clark_evans,
    colonies,
    csr_envelope,
    ripley,
    voronoi_delaunay,
)
from hcs_studio.analysis.stitching import union_area
from hcs_studio.analysis.table_steps import WellContext, run_table_steps
from hcs_studio.pipeline.schema import Pipeline, PipelineStep
from hcs_studio.processing.base import ModuleError, create_module

# ----------------------------------------------------------------- helpers


def _objects(n: int, well: str = "A01", kind: str = "nuclei", **columns: Any) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "well": well,
            "row": 1,
            "col": 1,
            "field": 1,
            "timepoint": 0,
            "object_type": kind,
            "object_id": np.arange(1, n + 1),
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def _fake_run(tmp_path: Path, wells: dict[str, pd.DataFrame], size: float = 500.0) -> Path:
    folder = tmp_path / "run"
    for well, frame in wells.items():
        d = folder / "wells" / well
        d.mkdir(parents=True)
        frame.to_parquet(d / "objects.parquet", index=False)
        (d / "well_summary.json").write_text(
            json.dumps({"well": well, "field_size_um": [size, size]}), encoding="utf-8"
        )
    return folder


# ------------------------------------------------------------------- gates


def test_gate_counts_objects_inside_thresholds(tmp_path: Path) -> None:
    values = np.arange(10, dtype=float)  # 0..9
    frame = pd.concat(
        [
            _objects(10, intensity=values, area=values * 10),
            _objects(3, kind="cells", intensity=[100.0, 100.0, 100.0], area=[1.0, 1.0, 1.0]),
        ],
        ignore_index=True,
    )
    frame.loc[2, "intensity"] = np.nan  # missing values are outside the gate
    folder = _fake_run(tmp_path, {"A01": frame})
    gate = create_module(
        "gate_objects",
        {"name": "bright", "rules": [{"feature": "intensity", "min": 2, "max": 7}]},
    )
    either = create_module(
        "gate_objects",
        {
            "name": "either",
            "combine": "any",
            "rules": [{"feature": "intensity", "max": 0.5}, {"feature": "area", "min": 85}],
        },
    )
    result = run_table_steps(folder, [gate, either])  # type: ignore[list-item]
    summary = result.per_well["A01"]
    assert summary["gate_bright_count"] == 5  # 3,4,5,6,7 (2 is NaN)
    assert summary["gate_bright_fraction"] == pytest.approx(0.5)
    assert summary["gate_either_count"] == 2  # intensity 0, area 90
    annotations = pd.read_parquet(folder / "wells" / "A01" / "objects_annotations.parquet")
    assert annotations["gate_bright"].isna().sum() == 3  # cells are not gated
    assert annotations.loc[annotations.object_type == "nuclei", "gate_bright"].sum() == 5


def test_gate_reports_unknown_features(tmp_path: Path) -> None:
    folder = _fake_run(tmp_path, {"A01": _objects(3, intensity=[1.0, 2.0, 3.0])})
    gate = create_module("gate_objects", {"rules": [{"feature": "intensty", "min": 1}]})
    with pytest.raises(ModuleError, match="did you mean intensity"):
        run_table_steps(folder, [gate])  # type: ignore[list-item]


# -------------------------------------------------------------- classifier


def _two_classes(n: int, shift: float = 0.0, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0, 5], 1, (n, 3))
    b = rng.normal([4, 4, 5], 1, (n, 3))
    x = np.vstack([a, b]) + shift
    frame = _objects(2 * n, size=x[:, 0], brightness=x[:, 1], noise=x[:, 2])
    return frame, ["small"] * n + ["large"] * n


def test_classifier_cross_validation_and_round_trip(tmp_path: Path) -> None:
    examples, labels = _two_classes(60)
    for algorithm in ("random_forest", "hist_gradient_boosting"):
        model = train_classifier(
            examples,
            labels,
            algorithm=algorithm,
            object_type="nuclei",
            cv_folds=5,
        )
        assert model.classes == ["large", "small"]
        assert model.cv_folds == 5
        assert model.cv_accuracy is not None and model.cv_accuracy > 0.95
        confusion = np.asarray(model.confusion)
        assert confusion.shape == (2, 2) and confusion.sum() == 120
        assert np.trace(confusion) / confusion.sum() == pytest.approx(model.cv_accuracy)
    path = tmp_path / "model.joblib"
    save_classifier(model, path)
    again = load_classifier(path)
    stats = robust_stats(examples, again.features)
    assert np.allclose(again.probabilities(examples, stats), model.probabilities(examples, stats))


def test_plate_normalisation_removes_a_plate_offset() -> None:
    train, labels = _two_classes(80, seed=1)
    model = train_classifier(train, labels, normalisation="plate_robust_z", object_type="nuclei")
    shifted, truth = _two_classes(80, shift=3.0, seed=2)  # e.g. a brighter plate
    probabilities = model.probabilities(shifted, robust_stats(shifted, model.features))
    predicted = np.asarray(model.classes)[probabilities.argmax(axis=1)]
    assert (predicted == np.asarray(truth)).mean() > 0.95
    raw = train_classifier(train, labels, normalisation="none", object_type="nuclei")
    raw_predicted = np.asarray(raw.classes)[raw.probabilities(shifted, None).argmax(axis=1)]
    assert (raw_predicted == np.asarray(truth)).mean() < 0.8  # without it, the offset hurts


def test_active_learning_suggests_boundary_objects() -> None:
    examples, labels = _two_classes(50)
    model = train_classifier(examples, labels, object_type="nuclei")
    grid = np.linspace(-2, 6, 41)
    candidates = _objects(41, size=grid, brightness=grid, noise=np.full(41, 5.0))
    candidates["field"] = np.arange(41) % 9
    stats = robust_stats(examples, model.features)
    suggested = uncertain_objects(model, candidates, stats, n=5)
    assert len(suggested) == 5
    assert np.all(np.abs(suggested["size"] - 2.0) < 1.5)  # the class boundary is at 2
    assert suggested["margin"].is_monotonic_increasing
    again = uncertain_objects(model, candidates, stats, n=5, exclude=suggested)
    assert not set(again.object_id) & set(suggested.object_id)


def test_classify_step_in_a_run(tmp_path: Path) -> None:
    examples, labels = _two_classes(40)
    model = train_classifier(examples, labels, object_type="nuclei")
    path = tmp_path / "model.joblib"
    save_classifier(model, path)
    plate_a, truth_a = _two_classes(30, seed=5)
    plate_b, _ = _two_classes(10, seed=6)
    plate_b = plate_b.iloc[:10].assign(well="A02")  # only "small" objects
    folder = _fake_run(tmp_path, {"A01": plate_a, "A02": plate_b})
    step = create_module("classify_objects", {"name": "size", "model_path": str(path)})
    result = run_table_steps(folder, [step])  # type: ignore[list-item]
    a, b = result.per_well["A01"], result.per_well["A02"]
    assert a["class_size_small_count"] + a["class_size_large_count"] == 60
    assert a["class_size_small_fraction"] == pytest.approx(0.5, abs=0.05)
    assert b["class_size_small_fraction"] >= 0.8
    record = result.steps[0]["state"]
    assert record["model_sha256"] and record["cv_accuracy"] > 0.9
    annotations = pd.read_parquet(folder / "wells" / "A01" / "objects_annotations.parquet")
    assert (annotations["class_size"] == np.asarray(truth_a)).mean() > 0.95
    assert np.allclose(annotations["class_size_p_small"] + annotations["class_size_p_large"], 1)


def test_classifier_needs_the_same_features(tmp_path: Path) -> None:
    examples, labels = _two_classes(20)
    path = tmp_path / "model.joblib"
    save_classifier(train_classifier(examples, labels, object_type="nuclei"), path)
    folder = _fake_run(tmp_path, {"A01": _objects(5, size=np.ones(5))})
    step = create_module("classify_objects", {"model_path": str(path)})
    with pytest.raises(ModuleError, match="lacks 2 feature"):
        run_table_steps(folder, [step])  # type: ignore[list-item]


# -------------------------------------------------------------- clustering


def _blobs(seed: int, per: int = 60, well: str = "A01") -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 5]])
    truth = np.repeat(np.arange(3), per)
    points = centres[truth] + rng.normal(0, 0.7, (len(truth), 3))
    frame = _objects(len(truth), well=well, f1=points[:, 0], f2=points[:, 1], f3=points[:, 2])
    return frame, truth


def _agreement(found: np.ndarray, truth: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, found))


@pytest.mark.parametrize("method", ["kmeans", "hdbscan"])
def test_phenotypic_clusters_recover_blobs(tmp_path: Path, method: str) -> None:
    a, truth_a = _blobs(0)
    b, truth_b = _blobs(1, well="A02")
    folder = _fake_run(tmp_path, {"A01": a, "A02": b})
    step = create_module(
        "phenotypic_clusters",
        {"method": method, "n_clusters": 3, "min_cluster_size": 20, "name": "p"},
    )
    result = run_table_steps(folder, [step])  # type: ignore[list-item]
    for well, truth in (("A01", truth_a), ("A02", truth_b)):
        annotations = pd.read_parquet(folder / "wells" / well / "objects_annotations.parquet")
        found = annotations["cluster_p"].to_numpy(int)
        assert _agreement(found, truth) > 0.95
        summary = result.per_well[well]
        fractions = [summary[f"cluster_p_{k}_fraction"] for k in (1, 2, 3)]
        assert fractions == pytest.approx([1 / 3] * 3, abs=0.02)
    state = result.steps[0]["state"]
    assert set(state["features"]) == {"f1", "f2", "f3"}
    assert len(state["top_features_median_z"]) == 3


def test_leiden_is_optional(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    a, _ = _blobs(0)
    folder = _fake_run(tmp_path, {"A01": a})
    step = create_module("phenotypic_clusters", {"method": "leiden"})
    try:
        import leidenalg  # noqa: F401
    except ImportError:
        with pytest.raises(ModuleError, match="leidenalg"):
            run_table_steps(folder, [step])  # type: ignore[list-item]
    else:
        run_table_steps(folder, [step])  # type: ignore[list-item]


# ----------------------------------------------------------------- spatial


def _csr(n: int, side: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, side, (n, 2))


def _thomas(parents: int, children: int, sigma: float, side: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, side, (parents, 2))
    points = np.repeat(centres, children, axis=0) + rng.normal(0, sigma, (parents * children, 2))
    return points[np.all((points >= 0) & (points < side), axis=1)]


def test_window_area_edges_and_union() -> None:
    rects = [(0.0, 0.0, 100.0, 100.0), (0.0, 80.0, 100.0, 180.0)]
    assert union_area(rects) == pytest.approx(18000)
    window = Window.from_rects(rects, max_pixels=200_000)
    assert window.area_um2 == pytest.approx(18000, rel=0.01)
    assert window.perimeter_um == pytest.approx(2 * (100 + 180), rel=0.02)
    distance = window.border_distance(np.array([[50.0, 90.0], [5.0, 50.0], [50.0, 300.0]]))
    assert distance == pytest.approx([50, 5, 0], abs=1.0)
    points = window.random_points(5000, np.random.default_rng(0))
    assert window.contains(points).all()
    assert np.max(points[:, 1]) > 170


def test_ripley_csr_is_flat_and_inside_the_envelope() -> None:
    side, radii = 1000.0, np.array([20.0, 40.0, 60.0, 80.0])
    window = Window.from_rects([(0.0, 0.0, side, side)])
    points = _csr(1500, side, seed=1)
    stats = ripley(points, window, radii)
    assert np.abs(stats["L"] - radii).max() < 2.5  # L(r) ≈ r under CSR
    assert np.abs(stats["g"][1:] - 1).max() < 0.15
    envelope = csr_envelope(len(points), window, radii, 39, np.random.default_rng(2))
    assert np.all(stats["H"] >= envelope["H_low"] - 1e-9)
    assert np.all(stats["H"] <= envelope["H_high"] + 1e-9)
    assert clark_evans(points, window) == pytest.approx(1.0, abs=0.06)


def test_ripley_detects_clustering_and_regularity() -> None:
    side, radii = 1000.0, np.array([10.0, 25.0, 50.0])
    window = Window.from_rects([(0.0, 0.0, side, side)])
    clustered = _thomas(60, 20, 12.0, side, seed=3)
    stats = ripley(clustered, window, radii)
    envelope = csr_envelope(len(clustered), window, radii, 39, np.random.default_rng(4))
    assert np.all(stats["H"] > envelope["H_high"])
    assert clark_evans(clustered, window) < 0.6
    step = 20.0  # square lattice: R = 2, nearest neighbour = step, 4 Voronoi/Delaunay neighbours
    yy, xx = np.meshgrid(np.arange(10, side, step), np.arange(10, side, step), indexing="ij")
    lattice = np.column_stack([yy.ravel(), xx.ravel()]) + 1e-3 * np.random.default_rng(5).normal(
        size=(yy.size, 2)
    )
    assert clark_evans(lattice, window) == pytest.approx(2.0, rel=0.06)
    regular = ripley(lattice, window, np.array([10.0]))
    assert regular["K"][0] == pytest.approx(0, abs=1e-9)  # no neighbour closer than 20 µm
    areas, _ = voronoi_delaunay(lattice, window)
    interior = np.isfinite(areas)
    assert interior.sum() > 0.9 * len(lattice)
    assert np.median(areas[interior]) == pytest.approx(step**2, rel=0.01)


def test_voronoi_and_delaunay_on_a_hexagonal_lattice() -> None:
    spacing = 10.0
    rows = [
        [(r * spacing * math.sqrt(3) / 2, c * spacing + (r % 2) * spacing / 2) for c in range(20)]
        for r in range(20)
    ]
    points = np.asarray([p for row in rows for p in row]) + 5
    window = Window.from_rects([(0.0, 0.0, 190.0, 215.0)])
    areas, neighbours = voronoi_delaunay(points, window)
    inner = np.isfinite(areas)
    assert np.median(neighbours[inner]) == 6
    assert np.median(areas[inner]) == pytest.approx(spacing**2 * math.sqrt(3) / 2, rel=0.01)
    border = (points[:, 0] < 6) | (points[:, 1] < 6)
    assert not inner[border].any()  # open cells at the edge are not measured


def test_dbscan_colonies() -> None:
    rng = np.random.default_rng(0)
    big = rng.normal([100, 100], 5, (40, 2))
    small = rng.normal([400, 400], 5, (15, 2))
    loners = np.array([[700.0, 100.0], [100.0, 700.0]])
    labels = colonies(np.vstack([big, small, loners]), "dbscan", 15.0, 5, 10)
    assert set(labels[:40]) == {1} and set(labels[40:55]) == {2}
    assert list(labels[55:]) == [-1, -1]
    hdb = colonies(np.vstack([big, small, loners]), "hdbscan", 15.0, 5, 10)
    assert len(set(hdb[hdb > 0])) == 2


def test_spatial_step_per_field_and_whole_well(tmp_path: Path) -> None:
    clustered = _thomas(20, 15, 8.0, 500.0, seed=7)
    frame = _objects(len(clustered), centroid_y_um=clustered[:, 0], centroid_x_um=clustered[:, 1])
    folder = _fake_run(tmp_path, {"A01": frame})
    step = create_module(
        "spatial_clusters", {"scope": "field", "radii_um": [10, 30], "simulations": 19}
    )
    result = run_table_steps(folder, [step])  # type: ignore[list-item]
    summary = result.per_well["A01"]
    assert summary["spatial_objects"] == len(clustered)
    assert summary["spatial_pattern_10um"] == 1.0  # clustered at 10 µm
    assert summary["spatial_clark_evans_R"] < 0.7
    assert summary["spatial_colonies"] >= 10
    annotations = pd.read_parquet(folder / "wells" / "A01" / "objects_annotations.parquet")
    assert annotations["spatial_nn_um"].notna().all()
    details = json.loads((folder / "wells" / "A01" / "analysis.json").read_text())
    curves = details["step1_spatial_clusters"]["groups"][0]
    assert len(curves["r_um"]) == len(curves["H"]) and curves["envelope"]["H_high"]
    with pytest.raises(ModuleError, match="Stitch well"):
        run_table_steps(folder, [create_module("spatial_clusters")])  # type: ignore[list-item]


def test_table_steps_are_separated_from_image_steps() -> None:
    pipeline = Pipeline(
        steps=[
            PipelineStep(module="segment_classical", params={"channel": 1}),
            PipelineStep(module="gate_objects", params={"rules": [{"feature": "a", "min": 1}]}),
            PipelineStep(module="measure_objects"),
        ]
    )
    assert [s.module for s in pipeline.image_steps()] == ["segment_classical", "measure_objects"]
    assert [s.module for s in pipeline.table_steps()] == ["gate_objects"]
    assert WellContext.load(Path("/nonexistent/A01")).field_size_um is None


# ------------------------------------------------- whole run with stitching

PX = 0.29901  # µm per pixel of the Operetta builder
TILE, STEP = 200, 150  # 2 × 2 fields with 50 px overlaps
SCENE = TILE + STEP


def _scene(seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Nuclei (radius 10 px) on a textured background; none near the overlap mid-lines,
    where it would be ambiguous which field owns a nucleus cut by a field edge."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    centres: list[tuple[int, int]] = []
    middle = (TILE + STEP) / 2
    while len(centres) < 25:
        y, x = (int(v) for v in rng.integers(14, SCENE - 14, 2))
        if abs(y - middle) < 14 or abs(x - middle) < 14:
            continue
        if all((y - a) ** 2 + (x - b) ** 2 > 32**2 for a, b in centres):
            centres.append((y, x))
    yy, xx = np.mgrid[:SCENE, :SCENE]
    nuclei = np.zeros((SCENE, SCENE))
    for y, x in centres:
        nuclei[(yy - y) ** 2 + (xx - x) ** 2 <= 100] = 1
    texture = ndimage.gaussian_filter(rng.uniform(0, 1, (SCENE, SCENE)), 3)
    texture = (texture - np.min(texture)) / np.ptp(texture)
    image = 100 + 300 * texture + 2000 * ndimage.gaussian_filter(nuclei, 1.5)
    return image, centres


def _stitch_plate(tmp_path: Path) -> tuple[Path, dict[int, int]]:
    from tests.builders import make_operetta

    scenes = {col: _scene(seed=col) for col in range(2)}
    origins = [(0, 0), (0, STEP), (STEP, 0), (STEP, STEP)]

    def image(row: int, col: int, field: int, channel: int, plane: int) -> np.ndarray:
        y, x = origins[field]
        tile = scenes[col][0][y : y + TILE, x : x + TILE]
        return (tile if channel == 0 else tile * 0.5).astype(np.uint16)

    def position(field: int) -> tuple[float, float]:
        y, x = origins[field]
        error = 2.0 if field == 3 else 0.0  # the stage put field 4 2 µm off
        return ((x * PX + error) * 1e-6, -y * PX * 1e-6)  # stage y points up

    root = make_operetta(
        tmp_path / "raw" / "op",
        [(0, 0), (0, 1)],
        n_fields=4,
        n_planes=1,
        shape=(TILE, TILE),
        image_fn=image,
        position_fn=position,
    )
    return root, {col: len(scene[1]) for col, scene in scenes.items()}


def _summary_row(folder: Path, well: str) -> dict[str, Any]:
    table = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    rows: dict[str, dict[str, Any]] = table.to_dict("index")  # type: ignore[assignment]
    return rows[well]


RUN_PIPELINE = """
name: stitched
steps:
  - {module: segment_classical, params: {channel: 1, min_distance_um: 4}}
  - {module: measure_objects, params: {texture: false, granularity: false}}
  - {module: qc}
  - {module: stitch_well}
  - {module: gate_objects, params: {name: big, rules: [{feature: area_um2, min: 20}]}}
  - {module: spatial_clusters, params: {simulations: 9, radii_um: [10, 20]}}
  - {module: phenotypic_clusters, params: {n_clusters: 2}}
"""


def test_run_with_stitching_and_table_steps(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.output.results import RunResults
    from hcs_studio.pipeline.executor import analyze_run, run_pipeline

    root, truth = _stitch_plate(tmp_path)
    plate = open_plate(root)
    result = run_pipeline(plate, Pipeline.from_text(RUN_PIPELINE), output_root=tmp_path / "out")
    folder = result.folder
    assert not result.failed_wells
    summary = json.loads((folder / "wells" / "A01" / "well_summary.json").read_text())
    stitching = summary["stitching"]
    assert stitching["orientation"] == [-1, 1] and stitching["refined_pairs"] >= 4
    offsets = {int(k): np.array(v) for k, v in stitching["field_offsets_um"].items()}
    for field, (y, x) in zip(
        sorted(offsets), [(0, 0), (0, STEP), (STEP, 0), (STEP, STEP)], strict=True
    ):
        expected = np.array([y, x]) * PX
        assert np.abs(offsets[field] - offsets[min(offsets)] - expected).max() < PX
    assert (folder / "wells" / "A01" / "previews" / "mosaic.png").exists()

    objects = pd.read_parquet(folder / "tables" / "per_object_all.parquet")
    nuclei = objects[objects.object_type == "nuclei"]
    for col, well in enumerate(("A01", "A02")):
        mine = nuclei[nuclei.well == well]
        assert len(mine) > truth[col]  # nuclei in overlaps are seen two or four times
        assert (~mine.in_overlap_duplicate).sum() == truth[col]
    a01 = _summary_row(folder, "A01")
    assert a01["nuclei_count"] == truth[0]
    assert a01["area_mm2"] == pytest.approx((SCENE * PX) ** 2 / 1e6, rel=0.01)
    assert a01["gate_big_count"] == truth[0]
    assert a01["spatial_objects"] == truth[0]
    fractions = a01["cluster_pheno_1_fraction"] + a01["cluster_pheno_2_fraction"]
    assert fractions == pytest.approx(1)
    assert {"gate_big", "spatial_colony", "cluster_pheno"} <= set(objects.columns)
    analysis = json.loads((folder / "tables" / "analysis_summary.json").read_text())
    assert [s["module"] for s in analysis["steps"]] == [
        "gate_objects",
        "spatial_clusters",
        "phenotypic_clusters",
    ]

    # Train a classifier on this run and apply it afterwards, without reprocessing images.
    unique = nuclei[~nuclei.in_overlap_duplicate]
    labels = np.where(unique.area_um2 > unique.area_um2.median(), "large", "small")
    model = train_classifier(
        unique,
        list(labels),
        features=["area_um2", "perimeter_um"],
        plate_stats=robust_stats(unique, ["area_um2", "perimeter_um"]),
        object_type="nuclei",
    )
    path = tmp_path / "size.joblib"
    save_classifier(model, path)
    record = analyze_run(
        folder, [PipelineStep(module="classify_objects", params={"model_path": str(path)})]
    )
    assert [s["module"] for s in record["steps"]][-1] == "classify_objects"
    a01 = _summary_row(folder, "A01")
    assert "gate_big_count" in a01  # the run's own table steps are kept
    assert a01["class_class_large_count"] + a01["class_class_small_count"] == truth[0]
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["analyses"][-1]["steps"][-1]["module"] == "classify_objects"
    results = RunResults(folder)
    assert "class_class_large_fraction" in results.readout_columns()

    # The same from the command line: train from a labels CSV, then apply only that step.
    from hcs_studio.app import main

    labels_csv = tmp_path / "labels.csv"
    unique.assign(label=labels)[["well", "field", "object_id", "label"]].to_csv(
        labels_csv, index=False
    )
    model_file = tmp_path / "cli.joblib"
    assert (
        main(
            [
                "train-classifier",
                "--run",
                str(folder),
                "--labels",
                str(labels_csv),
                "--output",
                str(model_file),
            ]
        )
        == 0
    )
    assert load_classifier(model_file).cv_accuracy is not None
    pipeline_file = tmp_path / "classify.yaml"
    pipeline_file.write_text(
        "name: cli\nsteps:\n  - module: classify_objects\n"
        f"    params: {{name: cli, model_path: '{model_file}'}}\n",
        encoding="utf-8",
    )
    assert main(["analyze", str(folder), "--pipeline", str(pipeline_file), "--only"]) == 0
    table = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert "class_cli_large_fraction" in table and "gate_big_count" not in table
    report = (folder / "index.html").read_text(encoding="utf-8")
    assert "Analysis of the object tables" in report and "CV accuracy" in report
    assert "wells/A01/previews/mosaic.png" in report
