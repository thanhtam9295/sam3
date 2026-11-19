"""Phase 17b: plate normalization, the guard rail, uniformity maps, day effects, replicates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hcs_studio.normalization.plate import (
    PositionalCorrectionBlocked,
    WellInfo,
    b_score,
    mad,
    median_polish,
    normalize_plate,
    positional_guard,
    robust_z,
    wells_from_layout,
)
from hcs_studio.synthetic import screening_plate


def called_hits(scores: dict[str, float], samples: set[str], threshold: float = -3.0) -> set[str]:
    return {w for w, z in scores.items() if w in samples and math.isfinite(z) and z <= threshold}


def test_robust_z_matches_a_hand_calculation() -> None:
    values = {"A01": 10.0, "A02": 12.0, "A03": 14.0, "A04": 16.0, "A05": 100.0}
    z = robust_z(values, list(values))
    # median 14, MAD = median(|4,2,0,2,86|) = 2 → scaled 2.9652
    assert z["A03"] == 0 and z["A01"] == pytest.approx(-4 / (1.4826 * 2))
    assert z["A05"] == pytest.approx(86 / (1.4826 * 2))
    with pytest.raises(ValueError, match="no spread"):
        robust_z({"A01": 1.0, "A02": 1.0, "A03": 1.0}, ["A01", "A02", "A03"])


def test_median_polish_recovers_additive_row_and_column_effects() -> None:
    rows, cols = 6, 8
    row_fx = np.linspace(-5, 5, rows)
    col_fx = np.linspace(3, -3, cols)
    grid = 50 + row_fx[:, None] + col_fx[None, :]
    polish = median_polish(grid)
    assert np.allclose(polish.residuals, 0, atol=1e-9)
    fitted = polish.overall + polish.row_effects[:, None] + polish.col_effects[None, :]
    assert np.allclose(fitted, grid)
    assert mad(np.array([1.0, 2.0, 3.0, np.nan])) == pytest.approx(1.4826)


def test_b_score_removes_a_row_column_artifact_and_keeps_seeded_hits() -> None:
    """Done-when (17b): B-score removes a synthetic row/column artifact while keeping hits."""
    values, wells, truth = screening_plate(
        hits=16, hit_effect=-45, row_effect=-40, col_effect=25, noise=4, seed=3
    )
    samples = {w.well for w in wells if w.role == "sample"}
    hits = set(truth["hits"])
    # Without positional correction the artifact widens the plate's spread and hides hits …
    raw = called_hits(robust_z(values, samples), samples)
    assert len(raw & hits) < 0.5 * len(hits), "the artifact should hide hits without correction"
    # … with B-score every seeded hit is called and nothing else.
    scores, polish = b_score(values, samples, 16, 24)
    called = called_hits(scores, samples)
    assert len(called & hits) >= 0.95 * len(hits)
    # the artifact adds no false hits: the same plate without it calls the same wells
    # (a stray noise call at −3 SD is expected now and then: 288 wells × 0.13 %)
    clean, _, _ = screening_plate(hits=16, hit_effect=-45, noise=4, seed=3)
    assert called == called_hits(b_score(clean, samples, 16, 24)[0], samples)
    assert len(called - hits) <= 2
    assert polish.row_effects[-1] - polish.row_effects[0] == pytest.approx(-40, abs=4)
    assert all(math.isnan(scores[w.well]) for w in wells if w.role != "sample")


def test_b_score_is_blocked_on_dose_response_and_combination_layouts() -> None:
    """Done-when (17b): B-score is blocked on a dose–response layout, with an explanation."""
    from hcs_studio.pharmacology.synthetic import layout

    drc = wells_from_layout(layout())  # dilution series along rows + 6×6 combination matrices
    guard = positional_guard(drc)
    assert not guard.allowed and "uniformity" in guard.reason
    values = {w.well: 100.0 + i for i, w in enumerate(drc)}
    with pytest.raises(PositionalCorrectionBlocked, match="uniformity-plate correction"):
        normalize_plate(values, drc, 16, 24, "b_score")
    # a plain dilution series down one column, without combinations
    column = [
        WellInfo(f"{r}05", "sample", (("Drug X", 10.0 ** (-9 + i)),))
        for i, r in enumerate("ABCDEFGH")
    ]
    guard = positional_guard(column)
    assert not guard.allowed and "Drug X (column 5)" in guard.series
    # robust Z and control-based methods stay available on the same layout
    assert normalize_plate(values, drc, 16, 24, "robust_z").method == "robust_z"


def test_a_randomized_library_is_allowed() -> None:
    _, wells, _ = screening_plate(seed=1)
    assert positional_guard(wells).allowed
    # one compound at three doses in scattered wells is not a series either
    scattered = [
        WellInfo("A03", "sample", (("Drug Y", 1e-6),)),
        WellInfo("F17", "sample", (("Drug Y", 1e-7),)),
        WellInfo("K09", "sample", (("Drug Y", 1e-8),)),
    ]
    assert positional_guard(scattered).allowed


def test_control_based_normalization() -> None:
    values, wells, _ = screening_plate(noise=0, row_effect=0, col_effect=0, hits=0)
    out = normalize_plate(values, wells, 16, 24, "percent_of_control").frame.set_index("well")
    assert out.loc["A01", "normalized"] == pytest.approx(100)
    assert out.loc["A24", "normalized"] == pytest.approx(10)
    inhibition = normalize_plate(values, wells, 16, 24, "percent_inhibition").frame
    by_role = inhibition.groupby("role")["normalized"].median()
    assert by_role["negative_control"] == pytest.approx(0) and by_role[
        "positive_control"
    ] == pytest.approx(100)
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_plate(values, wells, 16, 24, "z")


# ------------------------------------------------------------------ uniformity


def test_uniformity_map_recovers_a_known_gradient_and_flattens_later_plates() -> None:
    from hcs_studio.normalization.uniformity import apply_map, build_map, edge_mask, plate_stats

    rows, cols = 16, 24
    r, c = np.mgrid[0:rows, 0:cols]
    shading = 1 + 0.15 * (c / (cols - 1) - 0.5) - 0.10 * ((r - 7.5) / 7.5) ** 2  # tilt + dome
    rng = np.random.default_rng(0)
    uniformity = 1000 * shading * (1 + rng.normal(0, 0.01, shading.shape))
    uniformity[3, 5] = 4000  # a debris well must not bend the map
    mapped = build_map(uniformity, degree=2)
    truth = shading / np.median(shading)
    assert np.max(np.abs(mapped.factors / truth - 1)) < 0.01
    assert mapped.stats["before_col_trend_pct"] == pytest.approx(15, abs=1.5)
    assert abs(mapped.stats["after_col_trend_pct"]) < 1
    later = 600 * shading * (1 + rng.normal(0, 0.01, shading.shape))
    corrected = apply_map(later, mapped)
    assert plate_stats(corrected)["cv_pct"] < 0.4 * plate_stats(later)["cv_pct"]
    assert np.median(corrected) == pytest.approx(600 * np.median(shading), rel=0.01)
    with pytest.raises(ValueError, match="16×24"):
        apply_map(np.ones((8, 12)), mapped)
    # a sharp edge step (evaporation): edge wells 10 % low, corrected by the edge-ring term
    edge = np.where(edge_mask(rows, cols), 900.0, 1000.0) * (1 + rng.normal(0, 0.005, (rows, cols)))
    assert plate_stats(edge)["edge_effect_pct"] == pytest.approx(-10, abs=0.5)
    ring_map = build_map(edge, degree=2, edge_rings=1)
    assert abs(ring_map.stats["after_edge_effect_pct"]) < 0.5
    exact = np.where(edge_mask(rows, cols), 900.0, 1000.0)  # noise-free: no division by ~0
    assert abs(build_map(exact).stats["after_edge_effect_pct"]) < 1e-6
    smooth_only = build_map(edge, degree=2, edge_rings=0)
    assert abs(smooth_only.stats["after_edge_effect_pct"]) > 2  # a polynomial cannot follow a step


def test_uniformity_correction_through_normalize_plate() -> None:
    from hcs_studio.normalization.uniformity import build_map

    values, wells, _ = screening_plate(row_effect=-30, noise=0, hits=0)
    flat = {w: 100.0 for w in values}
    rows = np.arange(16)[:, None] * np.ones((1, 24))
    shading = 1 - 0.3 * rows / 15
    mapped = build_map(100 * shading, degree=1)
    shaded = {w: v * float(shading[parse(w)]) for w, v in flat.items()}
    out = normalize_plate(shaded, wells, 16, 24, "uniformity", factors=mapped.factors).frame
    assert np.allclose(out["normalized"] / out["normalized"].median(), 1, atol=1e-6)


def parse(well: str) -> tuple[int, int]:
    from hcs_studio.model.plate import parse_well_name

    return parse_well_name(well)


# ---------------------------------------------------------------- day effects


def test_same_day_controls_remove_a_day_effect() -> None:
    import pandas as pd

    from hcs_studio.normalization.batch import batch_effects, r_squared

    rng = np.random.default_rng(2)
    rows = []
    day_level = {"2026-01-05": 1000.0, "2026-01-06": 1300.0, "2026-01-07": 800.0}
    for day, level in day_level.items():
        for p in range(3):
            plate = f"{day}-P{p}"
            for _ in range(32):
                rows.append(
                    {
                        "plate": plate,
                        "day": day,
                        "role": "negative_control",
                        "value": level * (1 + rng.normal(0, 0.03)),
                    }
                )
            for _ in range(64):
                rows.append(
                    {
                        "plate": plate,
                        "day": day,
                        "role": "sample",
                        "value": level * 0.5 * (1 + rng.normal(0, 0.05)),
                    }
                )
    frame = pd.DataFrame(rows)
    effects = batch_effects(frame)
    summary = effects.summary.set_index("normalization").to_dict("index")
    assert summary["raw"]["r2_day"] > 0.9
    assert summary["same-day controls"]["r2_day"] < 0.05
    assert summary["same-day controls"]["vehicle_cv_pct"] < 4
    assert len(effects.per_plate) == 9
    assert effects.per_plate["vehicle_median_per_day"].between(97, 103).all()
    assert r_squared(pd.Series([1.0, 1, 2, 2]), pd.Series(["a", "a", "b", "b"])) == 1.0


# ------------------------------------------------------------------ replicates


def test_replicate_concordance_statistics_and_flags() -> None:
    import pandas as pd

    from hcs_studio.normalization.concordance import replicate_concordance

    rng = np.random.default_rng(5)
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    base = {w: float(v) for w, v in zip(wells, rng.uniform(20, 120, len(wells)), strict=True)}
    conditions = {w: f"cond-{i // 2}" for i, w in enumerate(wells)}
    same = replicate_concordance(base, dict(base), conditions)
    assert same.pearson == pytest.approx(1) and same.spearman == pytest.approx(1)
    assert same.bias == 0 and same.loa_low == 0 and not same.discordant
    shifted = {w: v + 5 + rng.normal(0, 2) for w, v in base.items()}
    result = replicate_concordance(shifted, base, conditions)
    assert result.bias == pytest.approx(5, abs=0.6)
    assert result.loa_high - result.loa_low == pytest.approx(2 * 1.96 * 2, rel=0.25)
    assert (
        set(result.conditions.columns) >= {"condition", "cv_pct"} and len(result.conditions) == 48
    )
    fits_a = pd.DataFrame(
        {
            "compound": ["X", "Y"],
            "cell_line": ["L", "L"],
            "readout": ["r", "r"],
            "ec50_m": [1e-6, 1e-7],
        }
    )
    fits_b = fits_a.assign(ec50_m=[1.5e-6, 5e-7])
    ic = replicate_concordance(base, dict(base), conditions, fits_a=fits_a, fits_b=fits_b)
    assert list(ic.ic50["fold"].round(2)) == [1.5, 5.0]
    assert ic.discordant and ic.flags == ["Y in L: IC50 differs 5.0-fold"]
    noisy = {w: float(v) for w, v in zip(wells, rng.uniform(20, 120, len(wells)), strict=True)}
    assert any("Pearson" in f for f in replicate_concordance(noisy, base, conditions).flags)
