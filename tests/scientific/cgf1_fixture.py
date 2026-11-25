"""A tiny synthetic cgF1 problem whose answer can be derived by hand.

Five (image, noun-phrase) pairs, bbox IoU, score threshold 0.5:

  id 1  GT box A,   prediction == A (score 0.9)      -> TP; image-level TP
  id 2  no GT,      one prediction (score 0.9)       -> FP; image-level FP
  id 3  no GT,      no prediction                    -> image-level TN
  id 4  GT box,     only a low-score pred (0.3)      -> FN; image-level FN
  id 5  no GT,      no prediction                    -> image-level TN

Box level (all IoU thresholds): TP=1, FP=1, FN=1; positive-micro FP=0
  positive_micro_precision = 1 / (1 + 0 + 1e-4)
  recall                   = 1 / (1 + 1 + 1e-4)
Image level: TP=1, FP=1, TN=2, FN=1
  IL_MCC = (1*2 - 1*1) / sqrt(2 * 2 * 3 * 3) = 1/6
  cgF1   = positive_micro_F1 * IL_MCC  ~= 0.6666 * 0.1667 ~= 0.1111
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path
from typing import Dict


def build_fixture(tmpdir: Path) -> tuple[Path, Path]:
    images = [
        {
            "id": i,
            "width": 100,
            "height": 100,
            "file_name": f"{i}.jpg",
            "is_instance_exhaustive": True,
        }
        for i in range(1, 6)
    ]
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 30, 30],
            "area": 900,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 4,
            "category_id": 1,
            "bbox": [50, 50, 20, 20],
            "area": 400,
            "iscrowd": 0,
        },
    ]
    gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "x"}]}
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 4, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.3},
    ]
    gt_path, pred_path = tmpdir / "gt.json", tmpdir / "pred.json"
    gt_path.write_text(json.dumps(gt))
    pred_path.write_text(json.dumps(preds))
    return gt_path, pred_path


def run_cgf1_fixture() -> Dict[str, float]:
    from sam3.eval.cgf1_eval import CGF1Evaluator

    with tempfile.TemporaryDirectory() as d:
        gt_path, pred_path = build_fixture(Path(d))
        with contextlib.redirect_stdout(io.StringIO()):
            ev = CGF1Evaluator(gt_path=str(gt_path), iou_type="bbox")
            return ev.evaluate(str(pred_path))


def hand_computed() -> Dict[str, float]:
    pmp = 1 / (1 + 0 + 1e-4)
    rec = 1 / (1 + 1 + 1e-4)
    pmf1 = 2 * pmp * rec / (pmp + rec + 1e-4)
    mcc = (1 * 2 - 1 * 1) / (math.sqrt(2 * 2 * 3 * 3) + 1e-6)
    return {"positive_micro_F1": pmf1, "IL_MCC": mcc, "cgF1": pmf1 * mcc}
