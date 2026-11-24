"""Phase 24c: evaluating Cell Painting profiles — copairs mAP activity and consistency,
similarity, MoA kNN, batch effects, report."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("copairs")

from hcs_studio.cellpainting.evaluation import (
    HOLDOUT,
    Evaluation,
    EvaluationOptions,
    evaluate,
    read_moa,
    similar,
)
from hcs_studio.cellpainting.profiles import Profiles, ProfilingOptions, build_profiles
from hcs_studio.synthetic.cellpainting import profile_cells

FAST = EvaluationOptions(null_size=2000, umap=False)


def _run(
    cells: pd.DataFrame, options: EvaluationOptions = FAST, moa: pd.DataFrame | None = None
) -> tuple[Profiles, Evaluation]:
    profiles = build_profiles(cells, ProfilingOptions())
    result = evaluate(
        profiles.selected,
        profiles.consensus,
        profiles.log["consensus_by"],
        options,
        moa,
        profiles.aggregated,
    )
    return profiles, result


def test_seeded_phenotypes_active_vehicles_not() -> None:
    """Done-when: seeded phenotypes active (FDR < 0.05), held-out vehicle wells and an inert
    compound not."""
    cells = profile_cells(
        {"strong": dict.fromkeys(range(5), 3.0), "weak": {20: 1.0, 21: 1.0}, "inert": {}},
        seed=1,
    )
    _, result = _run(cells)
    table = result.activity.set_index("Metadata_Treatment")
    rows = table.to_dict("index")
    assert set(table.index) == {"strong", "weak", "inert", HOLDOUT}
    assert table.loc["strong", "active"] and table.loc["strong", "mAP"] == pytest.approx(1.0)
    assert table.loc["weak", "active"] and rows["weak"]["corrected_p_value"] < 0.05
    assert not table.loc["inert", "active"] and rows["inert"]["p_value"] > 0.05
    assert not table.loc[HOLDOUT, "active"] and rows[HOLDOUT]["p_value"] > 0.05
    assert (
        table.loc["strong", "n_wells"] == 6 and table.loc[HOLDOUT, "n_wells"] == 4
    )  # ⅓ of 6 per plate
    assert table.loc[HOLDOUT, "Metadata_Compound"] == "DMSO"
    assert result.log["active"] == 2 and result.log["holdout_active"] is False
    with pytest.raises(ValueError, match="vehicle"):
        evaluate(result.activity.assign(Metadata_Role="sample"), result.activity, ["x"], FAST)


MOAS = {"tubulin": range(0, 5), "hdac": range(5, 10), "mek": range(10, 15)}


def _moa_cells(seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3 MoAs × 4 compounds (shared block + one compound-specific feature) and two queries."""
    phenotypes: dict[str, dict[int, float]] = {}
    annotation = []
    for m, (moa, block) in enumerate(MOAS.items()):
        for i in range(4):
            shifts = dict.fromkeys(block, 3.0)
            shifts[15 + 4 * m + i if 15 + 4 * m + i < 30 else 29] = 2.0
            phenotypes[f"{moa}-{i}"] = shifts
            annotation.append({"compound": f"{moa}-{i}", "moa": moa})
    phenotypes["query-tubulin"] = dict.fromkeys(MOAS["tubulin"], 3.0)
    phenotypes["query-mek"] = dict.fromkeys(MOAS["mek"], 2.5)
    return profile_cells(phenotypes, seed=seed, replicates=2), pd.DataFrame(annotation)


def test_consistency_similarity_and_moa_suggestions() -> None:
    cells, annotation = _moa_cells()
    _, result = _run(cells, moa=annotation)
    assert result.consistency is not None
    consistency = result.consistency.set_index("moa")
    assert set(consistency.index) == set(MOAS)
    assert consistency["consistent"].all() and (consistency["n_compounds"] == 4).all()
    assert (consistency["mAP"] > 0.9).all()
    # similarity matrices: symmetric, unit diagonal, same MoA closest
    cosine, pearson = result.cosine, result.pearson
    assert np.allclose(np.diag(cosine), 1) and np.allclose(cosine, cosine.T)
    assert np.allclose(np.diag(pearson), 1) and np.allclose(pearson, pearson.T)
    x = np.array([1.0, 2.0, 3.0])
    assert np.corrcoef(x, 2 * x + 1)[0, 1] == pytest.approx(1.0)  # what "pearson" means
    nearest = similar(cosine, 3, "tubulin-0")
    assert nearest["similar"].str.startswith(("tubulin", "query-tubulin")).all()
    assert nearest["rank"].tolist() == [1, 2, 3] and nearest["cosine"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="No profile"):
        similar(cosine, 3, "nothing")
    # kNN: leave-one-compound-out accuracy and the unannotated queries
    assert result.moa_accuracy == 1.0
    assert result.moa_suggestions is not None
    suggestions = result.moa_suggestions.set_index("treatment")
    assert suggestions.loc["query-tubulin", "suggested_moa"] == "tubulin"
    assert suggestions.loc["query-mek", "suggested_moa"] == "mek"
    assert set(suggestions.index[suggestions["known_moa"].isna()]) == {"query-tubulin", "query-mek"}
    assert "DMSO" not in suggestions.index
    assert result.log["consistent_moas"] == 3 and result.log["moa_annotated"] == 12


def test_batch_effect_before_and_after_normalization() -> None:
    """Day 2 = gain 2, offset 5; each plate its own feature offsets (SD 1): vehicle wells
    cluster by plate and by day before normalization, not after."""
    cells = profile_cells(
        {"a": {0: 3.0}},
        seed=3,
        vehicles=8,
        plate_sd=1.0,
        plates=((1.0, 0.0), (2.0, 5.0), (1.0, 0.0), (2.0, 5.0)),
    )
    cells["Metadata_Date"] = cells["Metadata_Plate"].map(
        {"P1": "2026-01-05", "P2": "2026-01-06", "P3": "2026-01-05", "P4": "2026-01-06"}
    )
    _, result = _run(cells)
    plate, day = result.batch["plate"], result.batch["date"]
    assert plate["before"]["mAP"] > 0.9 and plate["before"]["silhouette"] > 0.5
    assert plate["after"]["mAP"] < 0.6 and plate["after"]["silhouette"] < 0.1
    assert plate["before"]["groups_clustered"] == 4 and plate["after"]["groups_clustered"] == 0
    assert day["before"]["mAP"] > 0.9 and day["after"]["mAP"] < 0.7
    assert "batch" not in result.batch


def test_read_moa(tmp_path: Path) -> None:
    path = tmp_path / "moa.csv"
    pd.DataFrame(
        {
            "Metadata_broad_sample": ["a", "b", "b"],
            "Metadata_moa": ["x", "y", "y"],
            "other": [1, 2, 3],
        }
    ).to_csv(path, index=False)
    assert read_moa(path).to_dict("list") == {"compound": ["a", "b"], "moa": ["x", "y"]}
    xlsx = tmp_path / "moa.xlsx"
    pd.DataFrame({"Compound": ["a"], "MOA": ["x"]}).to_excel(xlsx, index=False)
    assert read_moa(xlsx).to_dict("list") == {"compound": ["a"], "moa": ["x"]}
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="moa column"):
        read_moa(path)


def test_profile_and_evaluate_commands(tmp_path: Path) -> None:
    from hcs_studio.app import main

    cells, annotation = _moa_cells(4)
    source = tmp_path / "sc.parquet"
    cells.to_parquet(source)
    moa = tmp_path / "moa.csv"
    annotation.to_csv(moa, index=False)
    out = tmp_path / "profiling"
    assert (
        main(
            [
                "cellpainting",
                "profile",
                str(source),
                "--output",
                str(out),
                "--evaluate",
                "--moa",
                str(moa),
            ]
        )
        == 0
    )
    for name in (
        "activity.csv",
        "consistency.csv",
        "similarity_cosine.csv",
        "similarity_pearson.csv",
        "similar.csv",
        "moa_suggestions.csv",
        "batch.json",
        "umap.csv",
        "evaluation.html",
        "evaluation.json",
        "figures/activity.png",
        "figures/similarity_heatmap.png",
        "figures/umap_treatment.png",
        "figures/umap_plate.png",
    ):
        assert (out / name).exists(), name
    page = (out / "evaluation.html").read_text()
    assert "14 of 14 treatments active" in page and 'src="figures/activity.png"' in page
    record = json.loads((out / "evaluation.json").read_text())
    assert record["log"]["holdout_active"] is False and record["options"]["null_size"] == 10000
    assert (
        main(
            [
                "cellpainting",
                "evaluate",
                str(out),
                "--null-size",
                "500",
                "--no-umap",
                "-k",
                "3",
                "--any-plate-negatives",
                "--reason",
                "re-run",
            ]
        )
        == 0
    )
    assert json.loads((out / "evaluation.json").read_text())["options"]["k"] == 3
    assert main(["cellpainting", "similar", str(out), "--to", "hdac-0", "-n", "2"]) == 0


def test_painted_plate_end_to_end(tmp_path: Path) -> None:
    """Done-when on images: a synthetic Cell Painting plate (8 vehicle wells, 4 wells each of
    larger nuclei and fragmented mitochondria) through the preset, profiles and evaluation."""
    from hcs_studio.cellpainting.evaluation import run_evaluation
    from hcs_studio.cellpainting.preset import channel_map, preset
    from hcs_studio.cellpainting.profiles import run_profiling
    from hcs_studio.cellpainting.single_cell import write_single_cells
    from hcs_studio.io.base import open_plate
    from hcs_studio.model.annotation import PlateLayout, Treatment
    from hcs_studio.model.layout_tools import apply_fields, set_role
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.synthetic import ShotNoise, SyntheticPlate
    from hcs_studio.synthetic.cellpainting import PaintedCells

    channels = ("Hoechst", "ConA", "SYTO14", "Phalloidin", "MitoTracker")
    mapping = dict(zip(("dna", "er", "rna", "agp", "mito"), channels, strict=True))
    phenotype = {0: {}, 1: {"nucleus_size": 1.3}, 2: {"mito_spots": 0.3}}
    kind = {(r, c): 0 if r == 0 else (1 if c < 4 else 2) for r in (0, 1) for c in range(8)}
    cells = PaintedCells(spec=lambda r, c: phenotype[kind[(r, c)]])
    plate = SyntheticPlate(
        wells=list(kind),
        features=[cells, ShotNoise(sd=8)],
        channels=channels,
        size=192,
        n_fields=1,
        name="CP2",
    )
    path, _ = plate.write(tmp_path / "raw")
    layout = set_role(
        PlateLayout.for_plate_type(96),
        [w for w, k in kind.items() if k == 0],
        "negative_control",
        cell_line="U2OS",
    )
    for k, name in ((1, "Bigger"), (2, "Fragmenter")):
        layout = apply_fields(
            layout,
            [w for w, v in kind.items() if v == k],
            treatments=(Treatment(name, 1e-6, "µM"),),
            cell_line="U2OS",
        )
    folder = run_pipeline(
        open_plate(path),
        preset(mapping, channels=list(channels)),
        output_root=tmp_path / "runs",
        layout=replace(layout, barcode="CP2"),
        save_images=False,
    ).folder
    write_single_cells(folder, channel_map=channel_map(mapping))
    out = tmp_path / "profiles"
    run_profiling([folder], out, ProfilingOptions())
    result = run_evaluation(out, EvaluationOptions(null_size=2000, umap=False))
    table = result.activity.set_index("Metadata_Compound")
    dmso = table.to_dict("index")["DMSO"]
    assert table.loc["Bigger", "active"] and table.loc["Fragmenter", "active"]
    assert not table.loc["DMSO", "active"] and dmso["p_value"] > 0.05
