"""Phase 24b: the profiling pipeline (pycytominer aggregate → normalize → feature_select →
consensus), checked against direct pycytominer calls on real JUMP CellProfiler data and against
known answers on a synthetic two-plate experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pycytominer")

from hcs_studio.cellpainting.cellprofiler import read_cellprofiler
from hcs_studio.cellpainting.profiles import (
    OPERATIONS,
    STRATA,
    ProfilingOptions,
    aggregate_wells,
    build_profiles,
    consensus_profiles,
    normalize_profiles,
    select_features,
    single_cell_qc,
)

DATA = Path(__file__).parent / "data" / "cellpainting" / "BR00121431"


@pytest.fixture(scope="module")
def jump() -> pd.DataFrame:
    """120 real single cells (2 wells × 2 sites × 30) of a JUMP Cell Painting plate."""
    sites = sorted(p for p in DATA.iterdir() if p.is_dir())
    return pd.concat([read_cellprofiler(s) for s in sites], ignore_index=True)


def _features(frame: pd.DataFrame) -> list[str]:
    from pycytominer.cyto_utils import infer_cp_features

    return list(infer_cp_features(frame))


def _same(ours: pd.DataFrame, theirs: pd.DataFrame, columns: list[str]) -> None:
    a = ours[columns].astype(float).to_numpy()
    b = theirs[columns].astype(float).to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-12, atol=1e-12)


def test_reproduces_pycytominer_on_jump_data(jump: pd.DataFrame) -> None:
    """Each stage equals pycytominer's own function called directly (pycytominer's test data)."""
    from pycytominer import aggregate, consensus, feature_select, normalize

    features = _features(jump)
    assert len(jump) == 120 and len(features) > 5000
    options = ProfilingOptions(reference="all")

    ours = aggregate_wells(jump, features)
    theirs = aggregate(
        jump,
        strata=STRATA,
        features=features,
        operation="median",
        compute_object_count=True,
        object_feature="Metadata_ObjectNumber",
    )
    assert ours["Metadata_Well"].tolist() == theirs["Metadata_Well"].tolist() == ["A01", "A02"]
    assert ours["Metadata_Object_Count"].tolist() == [60, 60]
    _same(ours, theirs, features)
    assert ours["Metadata_Row"].tolist() == [1, 1]  # well-level metadata carried over

    # The cell-level stages (120 rows) are the stringent check: 2 wells are a degenerate input.
    meta = [c for c in jump.columns if c not in features]
    normalized = normalize_profiles(jump, features, options)
    expected = normalize(
        jump, features=features, meta_features=meta, samples="all", method="mad_robustize"
    )
    _same(normalized, expected, features)

    selected, removed = select_features(normalized, features, options)
    reference = feature_select(normalized, features=features, operation=list(OPERATIONS))
    assert list(selected.columns) == list(reference.columns)
    kept = [f for f in features if f in selected]
    assert 1000 < len(kept) < len(features)
    assert sum(len(v) for v in removed.values()) == len(features) - len(kept)
    assert removed["correlation_threshold"] and removed["blocklist"]
    assert all(
        f.startswith(("Cells_", "Cytoplasm_", "Nuclei_")) for v in removed.values() for f in v
    )

    by_site = ProfilingOptions(consensus_by=["Metadata_Well", "Metadata_Site"])
    ours_consensus = consensus_profiles(selected, kept, by_site)
    framed = selected.copy()
    for column in ("Metadata_Well", "Metadata_Site"):
        framed[column] = framed[column].astype(str)
    theirs_consensus = consensus(
        framed, replicate_columns=["Metadata_Well", "Metadata_Site"], features=kept
    )
    assert len(ours_consensus) == 4
    _same(ours_consensus, theirs_consensus, kept)


def test_whole_chain_on_jump_data(jump: pd.DataFrame, tmp_path: Path) -> None:
    """build_profiles = the stages chained; with a plate map (annotate) on imported data."""
    from pycytominer import annotate, normalize

    platemap = pd.DataFrame(
        {
            "well_position": ["A01", "A02"],
            "broad_sample": ["DMSO", "BRD-K0001"],
            "role": ["negative_control", "sample"],
        }
    )
    platemap = platemap.rename(columns={"role": "Role"})
    options = ProfilingOptions(
        reference="all",
        operations=["variance_threshold", "blocklist"],
        consensus_by=["Metadata_broad_sample"],
    )
    result = build_profiles(jump, options, platemap=platemap)
    assert result.single_cells == 120 and len(result.aggregated) == 2
    assert result.aggregated["Metadata_broad_sample"].tolist() == ["DMSO", "BRD-K0001"]
    features = result.features
    plain = annotate(
        aggregate_wells(jump, features),
        platemap,
        join_on=["Metadata_well_position", "Metadata_Well"],
    )
    meta = [c for c in plain.columns if c not in features]
    _same(
        result.normalized,
        normalize(
            plain, features=features, meta_features=meta, samples="all", method="mad_robustize"
        ),
        features,
    )
    assert result.consensus["Metadata_broad_sample"].tolist() == ["BRD-K0001", "DMSO"]
    assert result.log["consensus_by"] == ["Metadata_broad_sample"]
    assert result.log["removed"]["blocklist"] > 0
    # Imported data has no treatments: the default consensus falls back to one profile per well.
    fallback = build_profiles(jump, ProfilingOptions(reference="all", operations=[]))
    assert fallback.log["consensus_by"] == STRATA and len(fallback.consensus) == 2


# --- synthetic: two plates, vehicle wells, two compounds, a plate effect ---------------------

N_FEATURES = 30
SHIFTED = 5  # compound A raises the first 5 features by 3 vehicle SDs


def synthetic_cells(seed: int = 0, same_noise: bool = True) -> pd.DataFrame:
    """2 plates × 12 wells (6 DMSO, 3 cmpdA, 3 cmpdB) × 40 cells. Plate 2 = plate 1 × 2 + 5,
    cell by cell (a gain and offset that per-plate normalization must remove exactly).
    Compound B lowers features 10–14. ``same_noise=False``: independent cells on plate 2."""
    noise = np.random.default_rng(seed).normal(0, 1, (2, 12, 40, N_FEATURES))
    rows = []
    for p, (gain, offset) in enumerate(((1.0, 0.0), (2.0, 5.0)), start=1):
        for w in range(12):
            compound = "DMSO" if w < 6 else ("cmpdA" if w < 9 else "cmpdB")
            shift = np.zeros(N_FEATURES)
            if compound == "cmpdA":
                shift[:SHIFTED] = 3.0
            elif compound == "cmpdB":
                shift[10:15] = -3.0
            values = (10 + shift + noise[0 if same_noise else p - 1, w]) * gain + offset
            frame = pd.DataFrame(
                values,
                columns=[f"Cells_Intensity_MeanIntensity_F{i:02d}" for i in range(N_FEATURES)],
            )
            frame.insert(0, "Metadata_Plate", f"P{p}")
            frame.insert(1, "Metadata_Well", f"A{w + 1:02d}")
            frame.insert(2, "Metadata_ObjectNumber", np.arange(1, 41))
            frame.insert(3, "Metadata_Role", "negative_control" if compound == "DMSO" else "sample")
            frame.insert(4, "Metadata_Compound", compound)
            frame.insert(5, "Metadata_Treatment", compound)
            frame.insert(6, "Metadata_CellLine", "U2OS")
            frame.insert(7, "Metadata_QCExcluded", False)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _vehicle_medians(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    vehicle = frame[frame["Metadata_Role"] == "negative_control"]
    return vehicle.groupby("Metadata_Plate")[features].median()


def test_vehicle_normalization_per_plate() -> None:
    cells = synthetic_cells()
    features = _features(cells)
    simple = ["variance_threshold", "drop_na_columns"]
    result = build_profiles(cells, ProfilingOptions(operations=simple))
    assert len(result.aggregated) == 24
    assert set(result.aggregated["Metadata_Object_Count"]) == {40}
    # vehicle wells sit at 0 on every plate, whatever the plate's gain and offset
    assert np.abs(_vehicle_medians(result.normalized, features).to_numpy()).max() < 1e-12
    # the plate effect is gone: the same compound has the same profile on both plates
    by_plate = result.normalized.groupby(["Metadata_Compound", "Metadata_Plate"])[features].median()
    for compound in ("DMSO", "cmpdA", "cmpdB"):
        assert np.abs(by_plate.loc[compound].diff().iloc[1]).max() < 1e-9
    seeded = by_plate.xs("cmpdA").loc[["P2"], features[:SHIFTED]].to_numpy(float)
    assert seeded.min() > 5  # 3 SD ≫ well-median spread
    # consensus per treatment × cell line
    consensus = result.consensus.set_index("Metadata_Treatment")
    assert sorted(consensus.index) == ["DMSO", "cmpdA", "cmpdB"]
    assert (consensus["Metadata_CellLine"] == "U2OS").all()
    assert consensus.loc["cmpdA", features[:SHIFTED]].min() > 5
    assert consensus.loc["cmpdB", features[10:15]].max() < -5
    assert np.abs(consensus.loc["DMSO", features].to_numpy(float)).max() < 1.0
    # across plates the plate effect stays: vehicles of P2 are not at 0
    pooled = normalize_profiles(result.aggregated, features, ProfilingOptions(per_plate=False))
    pooled_vehicles = _vehicle_medians(pooled, features)
    assert pooled_vehicles.loc["P2"].min() > 0.5 and pooled_vehicles.loc["P1"].max() < -0.5


def test_reference_all_and_spherize() -> None:
    cells = synthetic_cells(1, same_noise=False)  # spherize needs full-rank vehicles
    features = _features(cells)
    wells = aggregate_wells(cells, features)
    everything = normalize_profiles(wells, features, ProfilingOptions(reference="all"))
    medians = everything.groupby("Metadata_Plate")[features].median()
    assert np.abs(medians.to_numpy()).max() < 1e-12  # all wells of each plate centred
    sphered = normalize_profiles(wells, features, ProfilingOptions(normalize="spherize"))
    vehicle = sphered[sphered["Metadata_Role"] == "negative_control"][features].to_numpy(float)
    assert np.abs(vehicle.mean(axis=0)).max() < 1e-6  # whitened against the vehicle wells
    assert sphered.shape == wells.shape
    assert normalize_profiles(wells, features, ProfilingOptions(normalize="none")).equals(wells)
    no_vehicle = wells.assign(Metadata_Role="sample")
    with pytest.raises(ValueError, match="No vehicle"):
        normalize_profiles(no_vehicle, features, ProfilingOptions())
    one_plate_without = wells[
        ~((wells["Metadata_Plate"] == "P2") & (wells["Metadata_Role"] == "negative_control"))
    ]
    with pytest.raises(ValueError, match="Plate P2"):
        normalize_profiles(one_plate_without, features, ProfilingOptions())


def test_single_cell_qc_and_feature_selection() -> None:
    cells = synthetic_cells(2)
    features = _features(cells)
    cells.loc[:9, "Metadata_QCExcluded"] = True
    cells.loc[20:24, features[:5]] = np.nan  # 5 of 30 features missing > 10 %
    cells.loc[30:31, features[0]] = np.nan  # 1 of 30: kept
    kept, counts = single_cell_qc(cells, features, ProfilingOptions())
    assert counts == {"cells": 960, "object_qc_excluded": 10, "too_many_missing": 5, "kept": 945}
    assert len(kept) == 945
    wells = aggregate_wells(kept, features)
    assert (
        wells.set_index(["Metadata_Plate", "Metadata_Well"]).loc[
            ("P1", "A01"), "Metadata_Object_Count"
        ]
        == 25
    )
    assert "Metadata_QCExcluded" not in wells and "Metadata_Compound" in wells
    # a constant feature and a copy of F00 are removed; with the reason
    wells["Cells_Intensity_MeanIntensity_Const"] = 1.0
    wells["Cells_Intensity_MeanIntensity_Copy"] = wells[features[0]] * 2 + 1
    more = [*features, "Cells_Intensity_MeanIntensity_Const", "Cells_Intensity_MeanIntensity_Copy"]
    selected, removed = select_features(wells, more, ProfilingOptions())
    assert removed["variance_threshold"] == ["Cells_Intensity_MeanIntensity_Const"]
    assert len(removed["correlation_threshold"]) >= 1
    assert not removed["drop_na_columns"] and not removed["blocklist"]
    assert "Cells_Intensity_MeanIntensity_Const" not in selected
    with pytest.raises(ValueError, match="Unknown feature-selection"):
        select_features(wells, more, ProfilingOptions(operations=["pca"]))


def test_profile_command_writes_stages(tmp_path: Path) -> None:
    from hcs_studio.app import main

    source = tmp_path / "sc.parquet"
    synthetic_cells(3).to_parquet(source)
    out = tmp_path / "profiling"
    assert (
        main(
            [
                "cellpainting",
                "profile",
                str(source),
                "--output",
                str(out),
                "--operations",
                "variance_threshold,drop_na_columns",
                "--reason",
                "test",
            ]
        )
        == 0
    )
    for name in ("aggregated", "normalized", "feature_selected", "consensus"):
        assert len(pd.read_parquet(out / f"{name}.parquet")) == (3 if name == "consensus" else 24)
    selection = json.loads((out / "feature_selection.json").read_text(encoding="utf-8"))
    assert selection["features_in"] == N_FEATURES == len(selection["kept"])
    record = json.loads((out / "profiling.json").read_text(encoding="utf-8"))
    assert record["options"]["normalize"] == "mad_robustize"
    assert record["log"]["single_cell_qc"]["kept"] == 960 and record["pycytominer"]
    assert (
        main(
            [
                "cellpainting",
                "profile",
                str(source),
                "--output",
                str(out),
                "--reference",
                "all",
                "--across-plates",
                "--consensus-by",
                "Metadata_Compound",
            ]
        )
        == 0
    )
    assert len(pd.read_parquet(out / "consensus.parquet")) == 3
