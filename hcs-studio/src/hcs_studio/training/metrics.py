"""Validation metrics for segmentation and classification models (Phase 25).

Segmentation (instance masks, ground truth vs prediction, per field):

- **matching**: IoU of every ground-truth/predicted pair from one joint histogram; pairs are
  matched one-to-one by maximum total IoU (Hungarian), a pair counts when IoU ≥ threshold.
  Above 0.5 the match is unique anyway (two objects cannot both have IoU > 0.5 with a
  third); at exactly 0.5 they can tie, so the assignment decides.
- **AP at threshold t** = TP / (TP + FP + FN) — the definition of Cellpose and StarDist (not
  the COCO area under a precision–recall curve: segmentation outputs have no scores). Over a set
  of fields the counts are summed before dividing; **mAP** = mean AP over IoU 0.50:0.05:0.95.
- **precision, recall, F1 at IoU 0.5**: TP/(TP+FP), TP/(TP+FN), 2TP/(2TP+FP+FN).
- **merges and splits** (majority overlap): each object is assigned to the object of the other
  set holding more than half of its pixels. A predicted object that is the majority holder of
  ≥ 2 ground-truth objects is a merge (counted once per extra object); a ground-truth object
  that holds the majority of ≥ 2 predicted objects is split (counted once per extra piece).
- **count error** per field: predicted − true number of objects (and relative to the truth).

Classification: balanced accuracy (mean per-class recall), per-class precision/recall/support,
confusion matrix (rows = truth), and calibration of the predicted probability of the chosen
class: reliability bins, expected calibration error (ECE, weighted by bin size) and the
multi-class Brier score (mean squared distance between the probability vector and one-hot truth).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class Overlap:
    """Pixel counts of every (ground truth, prediction) pair; index 0 = background."""

    counts: NDArray[np.int64]  # (n_gt + 1, n_pred + 1)
    gt_ids: NDArray[np.int64]
    pred_ids: NDArray[np.int64]

    @property
    def iou(self) -> NDArray[np.float64]:
        inter = self.counts[1:, 1:].astype(np.float64)
        gt_area = self.counts[1:, :].sum(axis=1).astype(np.float64)
        pred_area = self.counts[:, 1:].sum(axis=0).astype(np.float64)
        union = gt_area[:, None] + pred_area[None, :] - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def overlap(gt: NDArray[Any], pred: NDArray[Any]) -> Overlap:
    """Joint pixel counts of two label images of the same shape (any dimensionality)."""
    if gt.shape != pred.shape:
        raise ValueError(f"Shapes differ: ground truth {gt.shape}, prediction {pred.shape}")
    gt_ids, gt_index = np.unique(np.asarray(gt).ravel(), return_inverse=True)
    pred_ids, pred_index = np.unique(np.asarray(pred).ravel(), return_inverse=True)
    # make sure label 0 (background) has index 0 in both
    if gt_ids.size == 0 or gt_ids[0] != 0:
        gt_ids = np.concatenate([[0], gt_ids])
        gt_index = gt_index + 1
    if pred_ids.size == 0 or pred_ids[0] != 0:
        pred_ids = np.concatenate([[0], pred_ids])
        pred_index = pred_index + 1
    n_pred = pred_ids.size
    counts = np.bincount(gt_index * n_pred + pred_index, minlength=gt_ids.size * n_pred)
    return Overlap(
        counts.reshape(gt_ids.size, n_pred).astype(np.int64),
        gt_ids[1:].astype(np.int64),
        pred_ids[1:].astype(np.int64),
    )


def true_positives(iou: NDArray[np.float64], threshold: float) -> int:
    """One-to-one matches with IoU ≥ threshold (maximum total IoU among eligible pairs)."""
    if iou.size == 0:
        return 0
    eligible = iou >= threshold - 1e-12
    if not eligible.any():
        return 0
    if threshold > 0.5:  # unique matches (IoU > 0.5 with two objects is impossible)
        return int(eligible.sum())
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(np.where(eligible, iou, 0.0), maximize=True)
    return int(eligible[rows, cols].sum())


def merges_and_splits(o: Overlap) -> tuple[int, int]:
    inter = o.counts[1:, 1:]
    if inter.size == 0:
        return 0, 0
    gt_area = o.counts[1:, :].sum(axis=1)
    pred_area = o.counts[:, 1:].sum(axis=0)
    # ground-truth object → the predicted object holding most of it (> half)
    gt_holder = np.where(inter.max(axis=1) * 2 > gt_area, inter.argmax(axis=1), -1)
    pred_holder = np.where(inter.max(axis=0) * 2 > pred_area, inter.argmax(axis=0), -1)
    merged = np.bincount(gt_holder[gt_holder >= 0], minlength=inter.shape[1])
    split = np.bincount(pred_holder[pred_holder >= 0], minlength=inter.shape[0])
    return int(np.clip(merged - 1, 0, None).sum()), int(np.clip(split - 1, 0, None).sum())


def field_metrics(
    gt: NDArray[Any], pred: NDArray[Any], thresholds: Sequence[float] = THRESHOLDS
) -> dict[str, Any]:
    """Counts for one field: objects, TP per threshold, merges, splits, count error."""
    o = overlap(gt, pred)
    iou = o.iou
    n_gt, n_pred = len(o.gt_ids), len(o.pred_ids)
    merges, splits = merges_and_splits(o)
    row: dict[str, Any] = {
        "n_true": n_gt,
        "n_pred": n_pred,
        "merges": merges,
        "splits": splits,
        "count_error": n_pred - n_gt,
        "count_error_pct": 100.0 * (n_pred - n_gt) / n_gt if n_gt else float("nan"),
        "mean_matched_iou": float(iou.max(axis=1)[iou.max(axis=1) >= 0.5].mean())
        if n_gt and n_pred and (iou.max(axis=1) >= 0.5).any()
        else float("nan"),
    }
    for t in thresholds:
        row[f"tp@{t:.2f}"] = true_positives(iou, t)
    return row


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarize(rows: pd.DataFrame, thresholds: Sequence[float] = THRESHOLDS) -> dict[str, Any]:
    """Set-level metrics from per-field rows (counts summed over fields, then divided)."""
    n_true, n_pred = float(rows["n_true"].sum()), float(rows["n_pred"].sum())
    ap = {}
    for t in thresholds:
        tp = float(rows[f"tp@{t:.2f}"].sum())
        ap[f"{t:.2f}"] = _ratio(tp, n_true + n_pred - tp)  # TP + FP + FN
    tp50 = float(rows["tp@0.50"].sum()) if "tp@0.50" in rows else float("nan")
    precision, recall = _ratio(tp50, n_pred), _ratio(tp50, n_true)
    return {
        "fields": len(rows),
        "objects_true": int(n_true),
        "objects_pred": int(n_pred),
        "ap": ap,
        "ap50": ap.get("0.50", float("nan")),
        "map": float(np.nanmean(list(ap.values()))) if ap else float("nan"),
        "precision50": precision,
        "recall50": recall,
        "f1_50": _ratio(2 * tp50, n_true + n_pred),
        "merges": int(rows["merges"].sum()),
        "splits": int(rows["splits"].sum()),
        "mean_abs_count_error": float(rows["count_error"].abs().mean()),
        "mean_count_error_pct": float(rows["count_error_pct"].mean()),
    }


def segmentation_report(
    rows: pd.DataFrame, by: str = "cell_line", thresholds: Sequence[float] = THRESHOLDS
) -> dict[str, Any]:
    """Overall metrics plus the same per group (e.g. per cell line) when the column exists."""
    out: dict[str, Any] = {"overall": summarize(rows, thresholds)}
    if by in rows and rows[by].nunique(dropna=True) >= 1:
        out[f"by_{by}"] = {
            str(name): summarize(group, thresholds)
            for name, group in rows.groupby(rows[by].fillna(""), sort=True)
        }
    return out


# --- classification ------------------------------------------------------------------------


def classifier_metrics(
    truth: Iterable[Any],
    predicted: Iterable[Any],
    probabilities: NDArray[Any] | None = None,
    classes: Sequence[str] | None = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Balanced accuracy, per-class precision/recall, confusion matrix, calibration.

    ``probabilities``: (n, n_classes) in the order of ``classes`` (required with it)."""
    y = np.asarray([str(v) for v in truth])
    p = np.asarray([str(v) for v in predicted])
    if y.shape != p.shape:
        raise ValueError("truth and predicted differ in length")
    names = list(classes) if classes is not None else sorted(set(y) | set(p))
    index = {c: i for i, c in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for t, q in zip(y, p, strict=True):
        if t in index and q in index:
            matrix[index[t], index[q]] += 1
    per_class = {}
    recalls = []
    for i, name in enumerate(names):
        support = int(matrix[i].sum())
        recall = _ratio(float(matrix[i, i]), support)
        per_class[name] = {
            "precision": _ratio(float(matrix[i, i]), float(matrix[:, i].sum())),
            "recall": recall,
            "support": support,
        }
        if support:
            recalls.append(recall)
    out: dict[str, Any] = {
        "n": len(y),
        "accuracy": _ratio(float(np.trace(matrix)), float(matrix.sum())),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "per_class": per_class,
        "classes": names,
        "confusion": matrix.tolist(),
    }
    if probabilities is not None:
        out["calibration"] = calibration(y, np.asarray(probabilities, float), names, bins)
    return out


def calibration(
    truth: NDArray[Any], probabilities: NDArray[np.float64], classes: Sequence[str], bins: int = 10
) -> dict[str, Any]:
    """Reliability of the top-class probability; ECE and multi-class Brier score."""
    if probabilities.shape != (len(truth), len(classes)):
        raise ValueError("probabilities must be (n samples, n classes)")
    onehot = (np.asarray(truth)[:, None] == np.asarray(classes)[None, :]).astype(float)
    brier = float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))) if len(truth) else 0.0
    confidence = probabilities.max(axis=1)
    correct = onehot[np.arange(len(truth)), probabilities.argmax(axis=1)] == 1
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    table = []
    ece = 0.0
    for b in range(bins):
        members = which == b
        n = int(members.sum())
        if not n:
            continue
        conf, acc = float(confidence[members].mean()), float(correct[members].mean())
        ece += n / len(truth) * abs(acc - conf)
        table.append(
            {
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n": n,
                "confidence": conf,
                "accuracy": acc,
            }
        )
    return {"ece": ece, "brier": brier, "bins": table}
