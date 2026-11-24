"""Phase 25: segmentation and classifier validation metrics, known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcs_studio.training.metrics import (
    calibration,
    classifier_metrics,
    field_metrics,
    overlap,
    segmentation_report,
    summarize,
    true_positives,
)


def squares(*boxes: tuple[int, int, int, int], shape: tuple[int, int] = (40, 40)) -> np.ndarray:
    out = np.zeros(shape, np.int32)
    for i, (y0, x0, y1, x1) in enumerate(boxes, start=1):
        out[y0:y1, x0:x1] = i
    return out


def test_identical_masks_are_perfect() -> None:
    gt = squares((2, 2, 12, 12), (20, 20, 30, 30))
    row = field_metrics(gt, gt)
    assert row["n_true"] == row["n_pred"] == 2 and row["merges"] == row["splits"] == 0
    assert all(row[f"tp@{t:.2f}"] == 2 for t in np.arange(0.5, 0.96, 0.05))
    summary = summarize(pd.DataFrame([row]))
    assert summary["ap50"] == summary["map"] == summary["f1_50"] == 1.0
    assert summary["mean_abs_count_error"] == 0


def test_iou_thresholds_shifted_square() -> None:
    """A 10×10 square shifted by 2 px: IoU = 80/120 = 0.667 → TP up to 0.65, not from 0.70."""
    gt = squares((10, 10, 20, 20))
    pred = squares((12, 10, 22, 20))
    assert overlap(gt, pred).iou[0, 0] == pytest.approx(80 / 120)
    row = field_metrics(gt, pred)
    assert row["tp@0.65"] == 1 and row["tp@0.70"] == 0
    summary = summarize(pd.DataFrame([row]))
    assert summary["ap"]["0.65"] == 1.0 and summary["ap"]["0.70"] == 0.0  # 0 / (0 + 1 + 1)
    assert summary["map"] == pytest.approx(0.4)  # 4 of 10 thresholds


def test_merges_splits_misses_and_count_error() -> None:
    gt = squares((5, 5, 15, 15), (5, 15, 15, 25), (30, 30, 36, 36))
    merged = squares((5, 5, 15, 25))  # one object over the first two; the third missed
    row = field_metrics(gt, merged)
    assert row["merges"] == 1 and row["splits"] == 0
    assert row["tp@0.50"] == 1 and row["tp@0.55"] == 0  # IoU exactly 0.5 with both: one match
    assert row["count_error"] == -2 and row["count_error_pct"] == pytest.approx(-200 / 3)
    split = field_metrics(squares((5, 5, 15, 25)), squares((5, 5, 15, 15), (5, 15, 15, 25)))
    assert split["splits"] == 1 and split["merges"] == 0
    empty = summarize(pd.DataFrame([field_metrics(gt, np.zeros_like(gt))]))
    assert empty["recall50"] == 0 and np.isnan(empty["precision50"]) and empty["ap50"] == 0


def test_counts_are_summed_over_fields_and_grouped() -> None:
    gt = squares((2, 2, 12, 12), (20, 20, 30, 30))
    half = squares((2, 2, 12, 12))
    rows = pd.DataFrame([{**field_metrics(gt, gt), "cell_line": "A"},
                         {**field_metrics(gt, half), "cell_line": "B"}])  # fmt: skip
    report = segmentation_report(rows)
    assert report["overall"]["ap50"] == pytest.approx(3 / 4)  # TP 3, FP 0, FN 1
    assert report["overall"]["recall50"] == pytest.approx(3 / 4)
    assert report["by_cell_line"]["A"]["ap50"] == 1.0
    assert report["by_cell_line"]["B"]["ap50"] == pytest.approx(0.5)


def test_low_threshold_matching_is_one_to_one() -> None:
    """At IoU 0.2 one prediction overlapping two truths matches only one of them."""
    iou = np.array([[0.3], [0.25]])
    assert true_positives(iou, 0.2) == 1
    assert true_positives(np.array([[0.3, 0.0], [0.25, 0.4]]), 0.2) == 2


def test_classifier_metrics_known_answer() -> None:
    truth = ["a"] * 8 + ["b"] * 2
    pred = ["a"] * 6 + ["b"] * 2 + ["b"] * 2  # a: 6/8 right, b: 2/2 right
    result = classifier_metrics(truth, pred)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx((0.75 + 1.0) / 2)
    assert result["per_class"]["b"]["precision"] == pytest.approx(0.5)
    assert result["confusion"] == [[6, 2], [0, 2]]


def test_calibration_known_answers() -> None:
    truth = np.array(["a", "b", "a", "b"])
    perfect = calibration(truth, np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float), ["a", "b"])
    assert perfect["ece"] == 0 and perfect["brier"] == 0
    # always 90 % sure of "a", right half the time: ECE = |0.5 − 0.9| = 0.4
    over = calibration(truth, np.tile([0.9, 0.1], (4, 1)), ["a", "b"])
    assert over["ece"] == pytest.approx(0.4)
    assert over["brier"] == pytest.approx((0.01 + 0.01 + 0.81 + 0.81) / 2 * 2 / 2)
    assert over["bins"][0]["n"] == 4
    with pytest.raises(ValueError, match="probabilities"):
        calibration(truth, np.ones((3, 2)), ["a", "b"])
