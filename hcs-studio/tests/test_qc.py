"""Phase 18a: field QC metrics, rules and decisions on labeled synthetic artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from hcs_studio.qc.decide import apply_rules, robust_z
from hcs_studio.qc.metrics import (
    MetricSettings,
    field_metrics,
    laplacian_variance,
    plls,
    saturation_level,
)
from hcs_studio.qc.rules import FieldRule, QcRules, default_rules, load_rules, save_rules


@pytest.fixture(scope="module")
def labeled(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, dict[str, object]]:
    from hcs_studio.io.base import open_plate
    from hcs_studio.qc.run import run_qc
    from hcs_studio.synthetic.artifacts import qc_plate

    root = tmp_path_factory.mktemp("qc")
    path, truth = qc_plate(root / "raw", per_kind=20, clean=100)
    plate = open_plate(path)
    rules = default_rules(nucleus_diameter_um=truth["nucleus_diameter_um"])
    result = run_qc(plate, rules, root / "out", workers=4)
    fields = result.fields
    fields["truth"] = [
        truth["labels"][f"{w}/{f}"] for w, f in zip(fields.well, fields.field, strict=True)
    ]
    return fields, {"root": root, "truth": truth, "result": result}


def test_plls_follows_the_cellprofiler_definition_and_falls_with_blur() -> None:
    rng = np.random.default_rng(1)
    sharp = rng.normal(100, 10, (256, 256))
    sharp[100:140, 60:200] += 400
    blurred = ndimage.gaussian_filter(sharp, 3)
    assert plls(blurred) < plls(sharp) - 1
    assert laplacian_variance(blurred) < 0.2 * laplacian_variance(sharp)
    assert np.isnan(plls(np.full((64, 64), 5.0)))


def test_saturation_level_uses_metadata_or_a_camera_maximum() -> None:
    data = np.array([[0, 100, 4095, 4095]], dtype=np.uint16)
    assert saturation_level(data, 12) == 4095
    assert saturation_level(data, 16) == 4095  # 12-bit camera in a 16-bit container
    assert saturation_level(np.array([[0, 3000]], dtype=np.uint16), 16) == 65535


def test_rules_round_trip_and_validation(tmp_path: Path) -> None:
    rules = default_rules(content_channel=2)
    path = save_rules(rules, tmp_path / "rules.yaml")
    again = load_rules(path)
    assert again == rules and again.sha256() == rules.sha256()
    assert "schema: hcs-studio-qc-rules/1" in path.read_text()
    with pytest.raises(ValueError, match="Unknown QC metric"):
        FieldRule(flag="x", metric="sharpness", above=1)
    with pytest.raises(ValueError, match="no condition"):
        FieldRule(flag="x", metric="plls")
    bumped = rules.model_copy(update={"version": 2})
    assert bumped.sha256() != rules.sha256()


def test_each_artifact_class_is_detected_with_few_false_positives(
    labeled: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    """Done-when (18): ≥ 95 % sensitivity per artifact class, ≤ 5 % false positives on clean
    fields."""
    from hcs_studio.synthetic.artifacts import KINDS

    fields, _ = labeled
    clean = fields[fields.truth.isna()]
    report = {}
    for kind in KINDS:
        hits = fields[fields.truth == kind]
        sensitivity = float(hits[f"flag_{kind}"].mean())
        false_positive = float(clean[f"flag_{kind}"].mean())
        report[kind] = (round(sensitivity, 3), round(false_positive, 3))
        assert len(hits) == 20
        assert sensitivity >= 0.95, report
        assert false_positive <= 0.05, report
    any_flag = clean["flags"].astype(bool).mean()
    assert any_flag <= 0.05, f"clean fields with any flag: {any_flag:.1%}"


def test_decisions_and_files(labeled: tuple[pd.DataFrame, dict[str, object]]) -> None:
    from hcs_studio.qc.run import load_qc

    fields, info = labeled
    result = info["result"]
    excluded = fields[fields.excluded]
    exclude_kinds = {"blur", "saturation", "bubble", "out_of_well"}
    assert set(excluded.truth.dropna()) >= exclude_kinds
    assert fields[fields.truth == "gradient"].excluded.mean() < 0.1  # flagged, not excluded
    assert all(excluded.reasons.str.len() > 0)
    assert (fields.loc[fields.excluded, "severity"] >= 1).all()
    stored = load_qc(Path(str(info["root"])) / "out")
    assert len(stored.fields) == len(fields) == 282
    assert (
        stored.plate["rules"]["sha256"]
        == default_rules(
            nucleus_diameter_um=info["truth"]["nucleus_diameter_um"]  # type: ignore[index]
        ).sha256()
    )
    assert stored.plate["fields"] == 282 and "seconds" in stored.plate
    assert set(stored.wells.columns) >= {"well", "n_good_fields", "failed", "reasons"}
    assert result.plate["passed"] is not None  # type: ignore[attr-defined]


def _table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_well_and_plate_decisions() -> None:
    rows = []
    for w in range(10):
        for f in range(4):
            bad = (w == 0) or (w == 1 and f < 3)
            rows.append(
                {"well": f"A{w + 1:02d}", "row": 0, "col": w, "field": f + 1,
                 "c1_saturated_fraction": 0.05 if bad else 0.0, "object_count": 50.0}
            )  # fmt: skip
    rules = QcRules(
        field=[FieldRule(flag="saturation", metric="saturated_fraction", above=0.002)],
    )
    result = apply_rules(_table(rows), rules)
    wells = result.wells.set_index("well")
    assert wells.loc["A01", "failed"] and wells.loc["A01", "n_good_fields"] == 0
    assert wells.loc["A02", "failed"]  # 1 of 4 good < 50 %
    assert not wells.loc["A03", "failed"]
    assert not result.plate["passed"] and "2 of 10 wells failed" in result.plate["reasons"][0]
    roles = {"A01": "negative_control", "A03": "negative_control", "A04": "positive_control"}
    strict = rules.model_copy(
        update={"plate": rules.plate.model_copy(update={"max_failed_well_fraction": 0.5})}
    )
    decided = apply_rules(_table(rows), strict, roles=roles, z_prime=0.3)
    assert decided.plate["controls"]["negative_control"] == {"wells": 2, "failed": 1}
    assert any("Z′ 0.30" in r for r in decided.plate["reasons"])
    assert not any("negative_control" in r for r in decided.plate["reasons"])  # 1 of 2 = 50 %


def test_plate_relative_conditions_and_autofocus_trend() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for r in range(8):
        for c in range(12):
            best = 5 + 0.25 * c + (4 if (r, c) in {(2, 3), (6, 9)} else 0)
            sharpness = 0.3 if (r, c) == (0, 0) else rng.uniform(0.95, 1.05)
            rows.append(
                {
                    "well": f"{chr(65 + r)}{c + 1:02d}",
                    "row": r,
                    "col": c,
                    "field": 1,
                    "c1_laplacian_variance": 0.08 * sharpness,
                    "c1_best_z": round(best),
                    "c1_background": 100 + rng.normal(0, 2),
                }
            )
    rules = default_rules()
    result = apply_rules(_table(rows), rules)
    flagged = result.fields.set_index("well")
    assert set(flagged.index[flagged.flag_autofocus]) == {"C04", "G10"}  # the trend is not flagged
    assert set(flagged.index[flagged.flag_blur]) == {"A01"}
    z = robust_z(pd.Series([1.0, 1.0, 1.0, 5.0]))
    assert np.isfinite(z).all() and z.iloc[-1] > 100


def test_field_metrics_shapes_and_errors() -> None:
    stack = np.full((2, 3, 64, 64), 100, dtype=np.uint16)
    stack[0, 1, 20:30, 20:30] = 2000
    metrics = field_metrics(stack, MetricSettings(content_channel=1, pixel_size_um=0.65))
    assert metrics["c1_best_z"] == 2 and metrics["object_count"] == 1
    assert {"c2_plls", "c2_saturated_fraction", "bubble_fraction"} <= set(metrics)
    with pytest.raises(ValueError, match="Content channel 3"):
        field_metrics(stack, MetricSettings(content_channel=3))
    with pytest.raises(ValueError, match="C, Z, Y, X"):
        field_metrics(stack[0], MetricSettings())


def test_command_line_writes_qc_files_and_records_the_run(tmp_path: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.services import recent_events
    from hcs_studio.synthetic.artifacts import qc_plate

    path, truth = qc_plate(tmp_path / "raw", per_kind=2, clean=8, kinds=("blur", "bubble"))
    rules_file = tmp_path / "rules.yaml"
    assert main(["qc", "--write-rules", str(rules_file)]) == 0
    text = rules_file.read_text()
    assert "null" not in text
    text = text.replace(
        "nucleus_diameter_um: 10.0", f"nucleus_diameter_um: {truth['nucleus_diameter_um']}"
    )
    rules_file.write_text(text)
    code = main(["qc", str(path), "--output", str(tmp_path / "out"), "--rules", str(rules_file),
                 "--reason", "new plate"])  # fmt: skip
    assert code == 0
    files = {p.name for p in (tmp_path / "out" / "qc").iterdir()}
    assert files == {"field_qc.parquet", "well_qc.csv", "plate_qc.json", "rules.yaml"}
    event = recent_events(1, "qc.run")[0]
    assert event["reason"] == "new plate" and event["after"]["excluded_fields"] == 4
    assert main(["qc", str(path), "--output", str(path / "inside")]) == 2  # never into raw data
