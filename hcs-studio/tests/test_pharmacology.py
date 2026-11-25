"""Phase 13: normalization, QC, dose–response fits, synergy models, outputs (known answers)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcs_studio.pharmacology.analysis import PharmOptions, analyse, run_pharmacology
from hcs_studio.pharmacology.combination import (
    find_matrices,
    interpret,
    matrix_from_values,
    score_matrix,
    summarize,
)
from hcs_studio.pharmacology.curves import (
    activity_area,
    auc_normalized,
    compare_ec50,
    fit_curve,
    gr_metrics,
    logistic,
)
from hcs_studio.pharmacology.normalize import (
    Normalization,
    exclude_outliers,
    normalize,
    plate_qc,
    wells_table,
)
from hcs_studio.pharmacology.synthetic import TRUTH, layout, summary, write_run

DATA = Path(__file__).parent / "data" / "pharmacology"
CONC = np.repeat(10.0 ** np.arange(-9, -4.4, 0.5), 3)  # 10 points, triplicates


def hill(c: Any, ic50: float, h: float = 1.0, top: float = 100.0) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.asarray(top * c**h / (c**h + ic50**h))


# ------------------------------------------------------------------- curves


def test_4pl_recovers_known_parameters_in_prism_form() -> None:
    rng = np.random.default_rng(7)
    # a falling viability curve (% of control): Prism reports a negative Hill slope
    y = logistic(np.log10(CONC), 10, 100, -7.0, -1.2) + rng.normal(0, 2, CONC.size)
    fit = fit_curve(CONC, y)
    assert fit.converged and not fit.flags
    assert fit.log_ec50 == pytest.approx(-7.0, abs=0.05)
    assert fit.hill == pytest.approx(-1.2, abs=0.1)
    assert (fit.bottom, fit.top) == pytest.approx((10, 100), abs=3)
    assert fit.log_ec50_ci[0] < -7.0 < fit.log_ec50_ci[1]
    assert fit.hill_ci[0] < -1.2 < fit.hill_ci[1]
    # absolute IC50: where the curve crosses 50 (bottom 10 → right of the midpoint)
    expected = -7.0 - math.log10((100 - 10) / (50 - 10) - 1) / -1.2
    assert fit.log_abs_ic50 == pytest.approx(expected, abs=0.06)
    assert fit.log_abs_ic50_ci[0] < fit.log_abs_ic50 < fit.log_abs_ic50_ci[1]
    assert fit.r2 > 0.99 and fit.n_concentrations == 10 and fit.n_replicates == 3

    # the same data rising (% inhibition) gives a positive slope, same EC50
    rising = fit_curve(CONC, 110 - y)
    assert rising.hill == pytest.approx(-fit.hill, abs=1e-3)
    assert rising.log_ec50 == pytest.approx(fit.log_ec50, abs=1e-3)


def test_confidence_intervals_cover_the_truth() -> None:
    """95 % intervals from the covariance matrix contain the true values ~95 % of the time."""
    hits = np.zeros(2)
    n = 200
    for seed in range(n):
        rng = np.random.default_rng(seed)
        y = logistic(np.log10(CONC), 10, 100, -7.0, -1.2) + rng.normal(0, 2, CONC.size)
        fit = fit_curve(CONC, y)
        hits += [
            fit.log_ec50_ci[0] < -7 < fit.log_ec50_ci[1],
            fit.hill_ci[0] < -1.2 < fit.hill_ci[1],
        ]
    assert np.all((hits / n > 0.9) & (hits / n < 0.99))


def test_3pl_5pl_robust_and_bootstrap() -> None:
    rng = np.random.default_rng(1)
    x = np.log10(CONC)
    y = logistic(x, 0, 100, -6.5, 1.0) + rng.normal(0, 2, CONC.size)
    fixed = fit_curve(CONC, y, fix_bottom=0.0)
    assert fixed.model == "3pl" and fixed.bottom == 0.0
    assert fixed.log_ec50 == pytest.approx(-6.5, abs=0.05)

    asym = logistic(x, 0, 100, -6.5, 1.5, 3.0)  # 5PL, S = 3
    five = fit_curve(CONC, asym, model="5pl")
    expected = -6.5 + math.log10(2 ** (1 / 3) - 1) / 1.5  # Prism's LogEC50 for 5PL
    assert five.log_ec50 == pytest.approx(expected, abs=0.02)
    assert five.asymmetry == pytest.approx(3.0, rel=0.05)

    spiked = y.copy()
    spiked[5] += 80  # one wild well
    plain = fit_curve(CONC, spiked)
    robust = fit_curve(CONC, spiked, robust=True)
    assert abs(robust.log_ec50 + 6.5) < abs(plain.log_ec50 + 6.5)

    boot = fit_curve(CONC, y, bootstrap=60, seed=3)
    assert boot.ci_method == "bootstrap (60)"
    assert boot.log_ec50_ci[0] < -6.5 < boot.log_ec50_ci[1]


def test_curve_flags_auc_activity_and_f_test() -> None:
    rng = np.random.default_rng(2)
    flat = fit_curve(CONC, rng.normal(100, 3, CONC.size))
    assert any("flat response" in f for f in flat.flags)
    partial = fit_curve(CONC, logistic(np.log10(CONC), 60, 100, -4.0, -1.0))
    assert any("not reached" in f for f in partial.flags)
    assert any("extrapolated" in f for f in partial.flags)
    assert any("too few" in f for f in fit_curve([1e-6, 1e-5], [10, 90]).flags)

    levels = np.log10([1e-8, 1e-7, 1e-6])
    assert auc_normalized(levels, np.array([100.0, 100.0, 100.0])) == pytest.approx(100)
    assert auc_normalized(levels, np.array([100.0, 50.0, 0.0])) == pytest.approx(50)
    full_kill = fit_curve(
        CONC, logistic(np.log10(CONC), 0, 100, -12.0, -5.0) + 0.01 * np.arange(30)
    )
    span = np.ptp(np.log10(CONC))
    assert activity_area(full_kill, 100.0, True) == pytest.approx(span, rel=0.02)

    y = logistic(np.log10(CONC), 0, 100, -7, 1) + rng.normal(0, 2, CONC.size)
    shifted = logistic(np.log10(CONC), 0, 100, -6.5, 1) + rng.normal(0, 2, CONC.size)
    same = logistic(np.log10(CONC), 0, 100, -7, 1) + rng.normal(0, 2, CONC.size)
    differ = compare_ec50(CONC, y, CONC, shifted)
    assert differ["delta_log_ec50"] == pytest.approx(0.5, abs=0.05) and differ["p"] < 1e-6
    assert compare_ec50(CONC, y, CONC, same)["p"] > 0.05


def test_gr_metrics_from_doubling_time() -> None:
    """GR from a known GR curve: x = x0 · 2^(log2(xctrl/x0) · log2(GR + 1))."""
    plate = layout()
    wells = wells_table(summary(plate, noise=0), plate, "P")
    gr_inf, gec50, h = -0.4, 1e-7, 1.0
    x_ctrl, duration, doubling = 1000.0, 72.0, 24.0
    x0 = x_ctrl / 2 ** (duration / doubling)

    def count(c: float) -> float:
        if c <= 0:
            return x_ctrl
        gr = gr_inf + (1 - gr_inf) / (1 + (c / gec50) ** h)
        return x0 * 2 ** (math.log2(x_ctrl / x0) * math.log2(gr + 1))

    wells["nuclei_count"] = [
        count(ts[0][1])
        if ts and ts[0][0] == "Drug D"
        else (x_ctrl if role == "negative_control" else 50.0)
        for ts, role in zip(wells["treatments"], wells["role"], strict=True)
    ]
    data = normalize(
        wells,
        "nuclei_count",
        Normalization("gr", doubling_time_h=doubling, assay_duration_h=duration),
    )
    assert data.controls["time_zero"].iloc[0] == pytest.approx(x0)
    assert any("doubling time" in n for n in data.notes)
    d = data.wells[
        (data.wells["cell_line"] == "MCF7") & (data.wells["treatment"].str.startswith("Drug D"))
    ]
    conc = d["treatments"].map(lambda ts: ts[0][1])
    metrics = gr_metrics(conc, d["normalized"])
    assert metrics["GEC50"] == pytest.approx(gec50, rel=0.02)
    assert metrics["GRinf"] == pytest.approx(gr_inf, abs=0.02)
    gr50 = gec50 * ((1 - gr_inf) / (0.5 - gr_inf) - 1) ** (1 / h)
    assert metrics["GR50"] == pytest.approx(gr50, rel=0.02)
    assert -0.4 < metrics["GRmax"] < 0
    with pytest.raises(ValueError, match="time-zero"):
        normalize(wells, "nuclei_count", Normalization("gr"))


# --------------------------------------------------------- normalization, QC


def _wells(noise: float = 15.0) -> pd.DataFrame:
    plate = layout()
    return wells_table(summary(plate, noise=noise), plate, "P")


def test_normalization_controls_and_qc() -> None:
    wells = _wells(noise=0)
    inhibition = normalize(wells, "nuclei_count", Normalization("percent_inhibition"))
    frame = inhibition.wells
    vehicle = frame[frame["role"] == "negative_control"]["normalized"]
    positive = frame[frame["role"] == "positive_control"]["normalized"]
    assert np.allclose(vehicle, 0) and np.allclose(positive, 100)
    assert set(inhibition.controls["cell_line"]) == {"MCF7", "HeLa"}
    single = frame[
        frame["treatment"].str.startswith("Drug D 1e-05") & (frame["cell_line"] == "MCF7")
    ]
    expected = 100 * 1e-5 / (1e-5 + TRUTH[("MCF7", "Drug D")][0])
    assert single["normalized"].iloc[0] == pytest.approx(expected, rel=1e-6)
    control = normalize(wells, "nuclei_count", Normalization("percent_of_control"))
    assert np.allclose(control.wells["effect"], 100 - control.wells["normalized"])
    # vehicles are not drugs: no DMSO "curves", no DMSO combinations
    assert not frame["treatment"].str.contains("DMSO").any()

    rng = np.random.default_rng(4)
    noisy = wells.copy()
    negative, positive_wells = (
        noisy["role"] == "negative_control",
        noisy["role"] == "positive_control",
    )
    noisy.loc[negative, "nuclei_count"] = rng.normal(1000, 30, int(negative.sum()))
    noisy.loc[positive_wells, "nuclei_count"] = rng.normal(50, 10, int(positive_wells.sum()))
    qc = plate_qc(normalize(noisy, "nuclei_count", Normalization()), z_threshold=0.5)
    n = noisy[noisy["role"] == "negative_control"]["nuclei_count"]
    p = noisy[noisy["role"] == "positive_control"]["nuclei_count"]
    z = 1 - 3 * (n.std() + p.std()) / abs(n.mean() - p.mean())
    assert qc["z_prime"].iloc[0] == pytest.approx(z)
    assert qc["ssmd"].iloc[0] == pytest.approx((p.mean() - n.mean()) / math.hypot(p.std(), n.std()))
    assert qc["flag"].iloc[0] == ""
    bad = plate_qc(normalize(noisy, "nuclei_count", Normalization()), z_threshold=0.95)
    assert bad["flag"].iloc[0] == "Z′ below 0.95"

    no_positive = wells[wells["role"] != "positive_control"]
    with pytest.raises(ValueError, match="positive-control"):
        normalize(no_positive, "nuclei_count", Normalization("percent_inhibition"))


def test_outliers_and_manual_exclusion() -> None:
    wells = _wells(noise=5)
    combination = wells["treatment"] == wells.loc[wells["n_active"] == 2, "treatment"].iloc[7]
    target = wells.index[combination][1]  # one of three replicate wells
    wells.loc[target, "nuclei_count"] = 5000.0  # a planted outlier
    # k = 3 flags ~2 % of noise wells in triplicates; k = 5 only the gross outlier
    data = exclude_outliers(normalize(wells, "nuclei_count", Normalization()), 5.0)
    flagged = data.wells[data.wells["excluded"]]
    assert target in flagged.index and len(flagged) <= 3  # the outlier, ≤ 1 % of noise wells
    assert "MAD" in flagged.loc[target, "exclusion_reason"]
    plate, well = data.wells.loc[target, ["plate", "well"]]
    data = exclude_outliers(data, 5.0, manual=[("P", "B05")], manual_include=[(plate, well)])
    assert not data.wells.loc[target, "excluded"]
    assert data.wells.set_index("well").loc["B05", "exclusion_reason"] == "excluded by hand"


# ------------------------------------------------------------------ synergy

CA = np.array([0, 0.1, 0.3, 1, 3, 10]) * 1e-6
CB = np.array([0, 0.03, 0.1, 0.3, 1, 3]) * 1e-6


def test_bliss_additive_matrix_scores_zero() -> None:
    ya, yb = hill(CA, 1e-6, 1.2), hill(CB, 0.3e-6, 1.5)
    additive = ya[:, None] + yb[None, :] - ya[:, None] * yb[None, :] / 100
    result = score_matrix(matrix_from_values(CA, CB, additive))
    assert abs(result.models["bliss"].mean) < 1e-9
    assert abs(result.models["zip"].mean) < 0.5
    hsa = result.models["hsa"]
    np.testing.assert_allclose(hsa.expected, np.maximum(ya[:, None], yb[None, :]))
    assert hsa.mean > 0  # Bliss-additive is above the best single agent
    assert result.models["bliss"].label == "likely additive"


def test_sham_combination_scores_zero_with_loewe() -> None:
    """A drug combined with itself: response = f(a + b) → Loewe 0, combination index 1."""
    cs = CA
    sham = hill(cs[:, None] + cs[None, :], 1e-6, 1.2)
    result = score_matrix(matrix_from_values(cs, cs, sham))
    assert abs(result.models["loewe"].mean) < 0.05
    assert np.nanmedian(result.combination_index[1:, 1:]) == pytest.approx(1.0, abs=1e-3)
    assert result.models["bliss"].mean < 0  # Bliss calls a sham combination antagonistic


def test_summary_window_labels_and_replicates() -> None:
    synergy = np.zeros((6, 6))
    synergy[2:5, 3:6] = 30.0
    synergy[1, 1] = 50.0
    mean, window, where, top, cell = summarize(synergy)
    assert mean == pytest.approx((9 * 30 + 50) / 25)
    assert window == pytest.approx(30.0) and where == (2, 3)
    assert top == 50.0 and cell == (1, 1)
    assert [interpret(v) for v in (-15, 0, 15, math.nan)] == [
        "likely antagonistic",
        "likely additive",
        "likely synergistic",
        "not computed",
    ]

    rng = np.random.default_rng(5)
    ya, yb = hill(CA, 1e-6, 1.2), hill(CB, 0.3e-6, 1.5)
    bump = np.zeros((6, 6))
    bump[1:, 1:] = 15.0
    truth = np.clip(ya[:, None] + yb[None, :] - ya[:, None] * yb[None, :] / 100 + bump, 0, 100)
    reps = [truth + rng.normal(0, 2, truth.shape) for _ in range(3)]
    for r in reps:
        r[0, :], r[:, 0] = truth[0, :], truth[:, 0]
    result = score_matrix(matrix_from_values(CA, CB, np.mean(reps, axis=0), replicates=reps))
    bliss = result.models["bliss"]
    assert len(bliss.replicate_means) == 3 and bliss.p_value < 0.01
    assert bliss.ci95[0] < bliss.mean < bliss.ci95[1]

    holes = truth.copy()
    holes[3, 2] = np.nan  # a missing well stays missing
    hole_result = score_matrix(matrix_from_values(CA, CB, holes))
    assert math.isnan(hole_result.models["bliss"].synergy[3, 2])
    assert hole_result.matrix.missing == 1


def test_matrices_found_from_the_annotation() -> None:
    data = normalize(_wells(noise=0), "nuclei_count", Normalization())
    matrices = {m.drug_b: m for m in find_matrices(data)}
    assert set(matrices) == {"Drug B", "Drug C"}
    m = matrices["Drug B"]
    assert m.response.shape == (6, 6) and m.missing == 0 and len(m.replicates) == 3
    assert m.conc_a[0] == 0 and m.conc_b[0] == 0 and np.all(np.diff(m.conc_a) > 0)
    assert m.response[0, 0] == pytest.approx(0, abs=1e-9)  # vehicle corner
    assert len(m.wells[(3, 3)]) == 3  # one well per replicate matrix
    synergistic = score_matrix(m)
    additive = score_matrix(matrices["Drug C"])
    for model in ("bliss", "loewe", "zip", "hsa"):
        assert synergistic.models[model].mean > 10, model
        assert synergistic.models[model].label == "likely synergistic"
    assert abs(additive.models["bliss"].mean) < 1e-6 and abs(additive.models["zip"].mean) < 0.5


# --------------------------------------------- reference data (sample files)


def _cases(kind: str) -> list[Path]:
    folder = DATA / kind
    return sorted(folder.glob("*.json")) if folder.exists() else []


@pytest.mark.skipif(
    not _cases("prism"), reason="no Prism reference curves in tests/data/pharmacology/prism"
)
@pytest.mark.parametrize("case", _cases("prism"), ids=lambda p: p.stem)
def test_matches_graphpad_prism(case: Path) -> None:
    """Prism reference: <name>.csv (X = log10 concentration or concentration, Y1…Yn) and
    <name>.json {"log_ec50" | "ec50", "hill", "x_is_log": bool}. Done when: EC50 within
    5 %, Hill within 0.1."""
    expected = json.loads(case.read_text(encoding="utf-8"))
    table = pd.read_csv(case.with_suffix(".csv"))
    x = table.iloc[:, 0].to_numpy(float)
    conc = 10**x if expected.get("x_is_log", True) else x
    ys = table.iloc[:, 1:]
    conc_all = np.repeat(conc, ys.shape[1])
    fit = fit_curve(conc_all, ys.to_numpy(float).ravel())
    ec50 = expected.get("ec50", 10 ** expected["log_ec50"] if "log_ec50" in expected else None)
    assert fit.ec50 == pytest.approx(ec50, rel=0.05)
    assert fit.hill == pytest.approx(expected["hill"], abs=0.1)


@pytest.mark.skipif(
    not _cases("synergyfinder"),
    reason="no SynergyFinder reference matrix in tests/data/pharmacology/synergyfinder",
)
@pytest.mark.parametrize("case", _cases("synergyfinder"), ids=lambda p: p.stem)
def test_matches_synergyfinder(case: Path) -> None:
    """SynergyFinder reference: <name>.csv (rows drug A conc, columns drug B conc, values %
    inhibition, 0 row/column included) and <name>.json {"bliss": score, "hsa": …,
    "loewe": …, "zip": …, "tolerance": 1.0}."""
    expected = json.loads(case.read_text(encoding="utf-8"))
    table = pd.read_csv(case.with_suffix(".csv"), index_col=0)
    result = score_matrix(
        matrix_from_values(
            table.index.to_numpy(float), table.columns.to_numpy(float), table.to_numpy(float)
        )
    )
    tolerance = float(expected.get("tolerance", 1.0))
    for model in ("bliss", "hsa", "loewe", "zip"):
        if model in expected:
            assert result.models[model].mean == pytest.approx(expected[model], abs=tolerance), model


# --------------------------------------------------------------- end to end


def test_full_report_from_a_384_well_combo_plate_in_under_a_minute(tmp_path: Path) -> None:
    """Done when: the full report generates from a 384-well combo plate in < 1 min."""
    run = write_run(tmp_path / "run")
    started = time.perf_counter()
    result = run_pharmacology(
        [run], PharmOptions(readouts=["nuclei_count", "pct_caspase_positive"], outlier_mad=3.0)
    )
    assert time.perf_counter() - started < 60
    out = run / "pharmacology"
    for name in (
        "normalized_well_data.csv",
        "plate_qc.csv",
        "dose_response_fits.csv",
        "combination_scores.csv",
        "ic50_comparisons.csv",
        "excluded_wells.csv",
        "pharmacology_report.html",
        "pharmacology_report.pdf",
        "pharmacology.json",
    ):
        assert (out / name).exists(), name
    assert (out / "pharmacology_report.pdf").read_bytes()[:4] == b"%PDF"
    matrix = out / "combination_matrices" / "nuclei_count" / "MCF7_Drug-A_x_Drug-B_zip.csv"
    frame = pd.read_csv(matrix, index_col=0)
    assert frame.shape == (6, 6) and frame.index.name == "Drug A (µM)"
    prism = pd.read_csv(out / "prism" / "nuclei_count" / "MCF7_Drug-D.csv")
    assert list(prism.columns) == ["log10_conc_M", "Y1", "Y2"] and len(prism) == 10
    fits = pd.read_csv(out / "dose_response_fits.csv")
    hela = fits[
        (fits.cell_line == "HeLa") & (fits.compound == "Drug D") & (fits.readout == "nuclei_count")
    ]
    assert hela["ec50"].iloc[0] == pytest.approx(1e-6, rel=0.15)
    wells = pd.read_csv(out / "normalized_well_data.csv")
    assert {"nuclei_count__percent_inhibition", "pct_caspase_positive__excluded"} <= set(
        wells.columns
    )
    report = (out / "pharmacology_report.html").read_text(encoding="utf-8")
    for text in ("Plate QC", "Dose–response", "Combinations", "Methods", "data:image/png"):
        assert text in report
    info = json.loads((out / "pharmacology.json").read_text(encoding="utf-8"))
    assert info["options"]["readouts"] == ["nuclei_count", "pct_caspase_positive"]
    assert "figures/combination_nuclei_count_MCF7_Drug-A_x_Drug-B.svg" in info["files"]
    assert result.folder == out


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from hcs_studio.app import main

    run = write_run(tmp_path / "run")
    code = main(["pharm", str(run), "--readout", "nuclei_count", "--models", "bliss",
                 "--no-figures", "--no-pdf", "--exclude", "SynthPlate:B03",
                 "--reason", "tip clog in B03", "--output", str(tmp_path / "out")])  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0, text
    assert "MCF7 · Drug A × Drug B: bliss" in text and "EC50 1" in text
    scores = pd.read_csv(tmp_path / "out" / "combination_scores.csv")
    assert list(scores["drug_b"]) == ["Drug B", "Drug C"]
    excluded = pd.read_csv(tmp_path / "out" / "excluded_wells.csv")
    assert list(excluded["well"]) == ["B03"]
    assert main(["pharm", str(run), "--readout", "nope"]) == 2


def test_analyse_without_files() -> None:
    result = analyse(_wells(), PharmOptions(readouts=["nuclei_count"], models=("bliss",)))
    rr = result.readouts["nuclei_count"]
    assert {c.compound for c in rr.curves} == {"Drug A", "Drug B", "Drug C", "Drug D"}
    assert set(rr.combinations[0].models) == {"bliss"}
    with pytest.raises(ValueError, match="Unknown synergy"):
        analyse(_wells(), PharmOptions(readouts=["nuclei_count"], models=("musyc",)))


def test_percent_of_control_needs_a_positive_vehicle_level() -> None:
    wells = _wells(noise=0)
    zero = wells.copy()
    zero.loc[zero["role"] == "negative_control", "nuclei_count"] = 0.0
    with pytest.raises(ValueError, match="vehicle median"):
        normalize(zero, "nuclei_count", Normalization("percent_of_control"))


def test_trapezoid_fallback_matches_numpy() -> None:
    from hcs_studio.pharmacology import curves

    x = np.linspace(0, 1, 11)
    assert float(curves._trapezoid(x**2, x)) == pytest.approx(0.335)
