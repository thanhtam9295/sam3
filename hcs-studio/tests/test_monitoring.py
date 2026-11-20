"""Phase 17c: control charts (rules, baseline limits, events), MSR, instrument bridging."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hcs_studio.monitoring.charts import DEFAULT_RULES, Rule, control_chart, evaluate, events_for


def series(values: list[float], start: date = date(2026, 1, 1)) -> pd.DataFrame:
    return pd.DataFrame(
        {"when": [start + timedelta(days=i) for i in range(len(values))], "value": values}
    )


def test_rules_parse_and_fire_where_expected() -> None:
    assert [r.label for r in DEFAULT_RULES] == ["1_3s", "2of3_2s", "7x"]
    assert Rule.parse("2of3_2s") == Rule("beyond", 2, 3, 2.0, "2of3_2s")
    assert Rule.parse("6t").kind == "trend"
    with pytest.raises(ValueError, match="Unknown rule"):
        Rule.parse("3sigma")
    one_3s, two_of_3, seven = DEFAULT_RULES
    z = [0.1, -0.2, 3.5, 0.0, 2.2, 0.3, 2.4, -2.1, 2.5]
    fired = {(i, r.label) for i, r in evaluate(z, [one_3s, two_of_3])}
    # 1_3s: index 2 (3.5). 2of3_2s fires on the newest point of a window of three holding two
    # points beyond 2 SD on the same side: index 4 (3.5, 2.2), 6 (2.2, 2.4), 8 (2.4, 2.5);
    # not 7 (−2.1 is alone on its side) and not 2 (3.5 is the first such point).
    assert fired == {(2, "1_3s"), (4, "2of3_2s"), (6, "2of3_2s"), (8, "2of3_2s")}
    run = [0.5, 0.2, 0.9, 0.1, 0.3, 0.4, 0.2, -0.1]
    assert [i for i, _ in evaluate(run, [seven])] == [6]
    trend = [0.0, 0.1, 0.3, 0.6, 0.8, 1.1, 0.9]
    assert [i for i, _ in evaluate(trend, [Rule.parse("6t")])] == [5]


def test_control_chart_flags_a_seeded_drift_and_not_the_baseline() -> None:
    """Done-when (17c): control charts flag a seeded drift."""
    rng = np.random.default_rng(0)
    stable = list(0.75 + rng.normal(0, 0.03, 20))  # Z′ of 20 good runs
    drift = [0.75 - 0.012 * (i + 1) + rng.normal(0, 0.01) for i in range(12)]  # slow decline
    chart = control_chart(series(stable + drift), name="Z′", baseline_n=20)
    assert chart.centre == pytest.approx(np.mean(stable)) and chart.sd == pytest.approx(
        np.std(stable, ddof=1)
    )
    assert all(i >= 20 for i in chart.flagged)  # no false alarm in the baseline
    assert chart.flagged and min(chart.flagged) <= 27  # caught within 8 runs of the start
    assert {v.rule for v in chart.violations} >= {"7x"}
    low, high = chart.limits(3)
    assert low == pytest.approx(chart.centre - 3 * chart.sd) and high > low
    fixed = control_chart(series(stable), centre=0.7, sd=0.01)
    assert fixed.flagged  # fixed limits that are too tight flag the stable runs


def test_charts_need_enough_points_and_report_short_baselines() -> None:
    short = control_chart(series([1.0, 1.1, 0.9, 1.05, 0.95]), baseline_n=20)
    assert "5 points" in short.notes[0]
    with pytest.raises(ValueError, match="at least 3"):
        control_chart(series([1.0, 1.1]))
    with pytest.raises(ValueError, match="no spread"):
        control_chart(series([1.0, 1.0, 1.0, 1.0]))


def test_event_annotations_match_on_assay_instrument_and_cell_line() -> None:
    events = pd.DataFrame(
        {
            "on": [date(2026, 1, 5), date(2026, 1, 9), date(2026, 1, 12)],
            "kind": ["reagent_lot", "instrument_service", "cell_vial"],
            "assay": ["viability", "", "viability"],
            "instrument": ["", "Operetta CLS", ""],
            "cell_line": ["", "", "HeLa"],
        }
    )
    picked = events_for(events, assay="viability", instrument="Operetta CLS", cell_line="MCF7")
    assert list(picked.kind) == ["reagent_lot", "instrument_service"]
    assert len(events_for(events)) == 3


# ------------------------------------------------------------------------- MSR


def test_msr_matches_a_hand_calculation() -> None:
    """Done-when (17c): MSR matches a hand calculation."""
    from hcs_studio.monitoring.msr import msr_from_runs, retest_statistics, rolling_msr

    potencies = [1.0e-7, 2.0e-7, 1.0e-7, 0.5e-7]  # log10: −7, −6.699, −7, −7.301
    logs = np.log10(potencies)
    s = np.sqrt(((logs - logs.mean()) ** 2).sum() / 3)  # sample SD by hand
    assert s == pytest.approx(0.24578, abs=1e-5)
    assert msr_from_runs(potencies) == pytest.approx(10 ** (2 * np.sqrt(2) * s))
    assert msr_from_runs(potencies) == pytest.approx(4.96, abs=0.01)
    assert msr_from_runs([3e-6] * 5) == pytest.approx(1.0)  # perfect reproducibility
    with pytest.raises(ValueError, match="at least 2"):
        msr_from_runs([1e-6])

    # test–retest: run 2 is run 1 shifted by ×2 with a ±10 % wobble
    run1 = [1e-8, 5e-8, 2e-7, 1e-6, 4e-6, 2e-5]
    wobble = [1.1, 0.9, 1.0, 1.1, 0.9, 1.0]
    run2 = [p * 2 * w for p, w in zip(run1, wobble, strict=True)]
    d = np.log10(run1) - np.log10(run2)
    stats = retest_statistics(run1, run2)
    assert stats.mr == pytest.approx(10 ** d.mean())
    assert stats.mr == pytest.approx(0.5, rel=0.02)  # run 1 is 2× more potent
    assert stats.msr == pytest.approx(10 ** (2 * d.std(ddof=1)))
    assert stats.lsa_low == pytest.approx(10 ** (d.mean() - 2 * d.std(ddof=1)))
    assert stats.lsa_high == pytest.approx(10 ** (d.mean() + 2 * d.std(ddof=1)))
    assert stats.msr < 3 and stats.acceptable()  # 2-fold shift: limits 0.42–0.60, inside 1/3–3
    shifted = retest_statistics(run1, [p * 4 * w for p, w in zip(run1, wobble, strict=True)])
    assert shifted.msr == pytest.approx(stats.msr) and not shifted.acceptable()  # LsA < 1/3
    rolling = rolling_msr([1e-7, 1.2e-7, 0.9e-7, 1.1e-7, 3e-7, 4e-7], window=4)
    assert rolling.isna().sum() == 3 and rolling.iloc[-1] > rolling.iloc[3]


# -------------------------------------------------------------------- bridging


def test_bridging_ratios_and_equivalence() -> None:
    from hcs_studio.monitoring.bridging import bridge, log_se_from_ci

    keys = {"cell_line": "L", "readout": "nuclei_count"}
    compounds = ["A", "B", "C", "D"]
    ec50 = np.array([1e-8, 1e-7, 1e-6, 1e-5])

    def fits(scale: float) -> pd.DataFrame:
        values = ec50 * scale
        return pd.DataFrame(
            {
                "compound": compounds,
                **keys,
                "ec50_m": values,
                "ec50_ci_low_m": values / 1.5,
                "ec50_ci_high_m": values * 1.5,
            }
        )

    se = log_se_from_ci(1 / 1.5, 1.5)
    assert se == pytest.approx(np.log10(1.5) / 1.959964)
    rng = np.random.default_rng(0)
    plate = {f"A{i:02d}": float(v) for i, v in enumerate(rng.uniform(50, 150, 40), start=1)}
    same = bridge(
        "Operetta", "Ji", "nuclei_count", [("P1", plate, dict(plate))], fits(1), fits(1.2)
    )
    assert same.equivalent and same.agreement["pearson"].iloc[0] == pytest.approx(1)
    row = same.ratios.iloc[0]
    assert row["ratio"] == pytest.approx(1 / 1.2)
    half_width = 10 ** (1.959964 * np.hypot(se, se))
    assert row["ratio_ci_high"] / row["ratio"] == pytest.approx(half_width)
    assert same.retest is not None and same.retest.mr == pytest.approx(1 / 1.2)
    shifted = bridge("Operetta", "Ji", "nuclei_count", [], fits(1), fits(5), msr=3)
    assert not shifted.equivalent and not shifted.ratios["within_msr"].any()
    assert "No plate" in shifted.notes[0]
