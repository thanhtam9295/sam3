"""Integration tests: multiple modules working together on synthetic data."""

import contextlib
import io
import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.integration


def _disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_masks_to_rle_to_cgf1_segm_perfect_predictions(tmp_path):
    """Masks -> GPU-style RLE encoder -> COCO json -> cgF1 (segm). Perfect = 1."""
    from sam3.agent.helpers.rle import rle_encode
    from sam3.eval.cgf1_eval import CGF1Evaluator

    h = w = 64
    images, anns, preds = [], [], []
    for img_id in range(1, 7):
        images.append(
            {
                "id": img_id,
                "height": h,
                "width": w,
                "file_name": f"{img_id}.png",
                "is_instance_exhaustive": True,
            }
        )
        if img_id > 4:  # negatives: no GT, no predictions -> image-level TN
            continue
        m = torch.from_numpy(
            np.stack(
                [_disk(h, w, 16 + img_id, 20, 8), _disk(h, w, 44, 40 + img_id, 10)]
            )
        )
        for k, rle in enumerate(rle_encode(m, return_areas=True)):
            seg = {"counts": rle["counts"], "size": rle["size"]}
            ys, xs = np.nonzero(m[k].numpy())
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(np.ptp(xs) + 1),
                int(np.ptp(ys) + 1),
            ]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": seg,
                    "area": rle["area"],
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": seg,
                    "score": 0.95,
                }
            )
    (tmp_path / "gt.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": anns,
                "categories": [{"id": 1, "name": "disk"}],
            }
        )
    )
    (tmp_path / "pred.json").write_text(json.dumps(preds))

    with contextlib.redirect_stdout(io.StringIO()):
        res = CGF1Evaluator(str(tmp_path / "gt.json"), iou_type="segm").evaluate(
            str(tmp_path / "pred.json")
        )
    assert res["cgF1_eval_segm_cgF1"] == pytest.approx(1.0, abs=1e-3)
    assert res["cgF1_eval_segm_IL_MCC"] == pytest.approx(1.0, abs=1e-3)
    assert res["cgF1_eval_segm_recall"] == pytest.approx(1.0, abs=1e-3)


def test_rle_strings_feed_agent_overlap_removal():
    """Agent path: RLE-encoded predictions are decoded and de-duplicated by IoM."""
    from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
    from sam3.agent.helpers.rle import rle_encode

    h, w = 40, 40
    m = torch.from_numpy(
        np.stack(
            [
                _disk(h, w, 20, 20, 10),
                _disk(h, w, 20, 21, 9),  # near-duplicate of 0
                _disk(h, w, 5, 5, 3),
            ]
        )
    )
    sample = {
        "orig_img_h": h,
        "orig_img_w": w,
        "pred_masks": [r["counts"] for r in rle_encode(m)],
        "pred_scores": [0.7, 0.9, 0.5],
        "pred_boxes": [[0, 0, 1, 1]] * 3,
    }
    out = remove_overlapping_masks(sample, iom_thresh=0.5)
    assert out["kept_indices"] == [1, 2]
    assert len(out["pred_boxes"]) == 2


def test_mask_nms_then_boxes_pipeline():
    """Detector post-processing: score filter + mask NMS + mask->box conversion."""
    from sam3.perflib.masks_ops import masks_to_boxes
    from sam3.perflib.nms import nms_masks

    h = w = 32
    masks = torch.from_numpy(
        np.stack(
            [
                _disk(h, w, 10, 10, 5),
                _disk(h, w, 10, 11, 5),
                _disk(h, w, 24, 24, 4),
            ]
        )
    ).float()
    probs = torch.tensor([0.6, 0.9, 0.8])
    keep = nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.tolist() == [False, True, True]
    boxes = masks_to_boxes(masks[keep] > 0, [0, 1])
    assert boxes.shape == (2, 4) and (boxes[:, 2] >= boxes[:, 0]).all()


@pytest.mark.gpu
@pytest.mark.slow
def test_image_model_builds_and_runs_with_random_weights(repo_root):
    """Architecture smoke test. Random weights: checks wiring/shapes, not accuracy.

    Requires CUDA (model construction hard-codes CUDA, KI-009). Downloads nothing.
    """
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(load_from_HF=False, checkpoint_path=None)
    proc = Sam3Processor(model)
    with torch.inference_mode():
        state = proc.set_image(Image.open(repo_root / "assets/images/truck.jpg"))
        out = proc.set_text_prompt(state=state, prompt="truck")
    n = out["scores"].shape[0]
    assert out["boxes"].shape == (n, 4)
    assert out["masks"].shape[0] == n
