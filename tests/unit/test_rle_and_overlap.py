"""Unit tests for RLE encoding and agent-side overlapping-mask removal."""

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util
from sam3.agent.helpers import mask_overlap_removal as mor
from sam3.agent.helpers.rle import rle_encode, robust_rle_encode

pytestmark = pytest.mark.unit


def _random_masks(n=4, h=13, w=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, h, w, generator=g) > 0.6


def test_rle_round_trip_and_area():
    masks = _random_masks()
    rles = rle_encode(masks, return_areas=True)
    assert len(rles) == masks.shape[0]
    for m, rle in zip(masks, rles):
        decoded = mask_util.decode({"counts": rle["counts"], "size": rle["size"]})
        assert np.array_equal(decoded.astype(bool), m.numpy())
        assert rle["area"] == int(m.sum())


def test_rle_empty_batch():
    assert rle_encode(torch.zeros(0, 3, 3, dtype=torch.bool)) == []


def test_robust_rle_matches_fast_path():
    masks = _random_masks(seed=3)
    assert [r["counts"] for r in robust_rle_encode(masks)] == [
        r["counts"] for r in rle_encode(masks)
    ]


def _sample(masks, scores):
    return {
        "orig_img_h": masks.shape[1],
        "orig_img_w": masks.shape[2],
        "pred_masks": [m.astype(np.uint8).tolist() for m in masks],
        "pred_scores": scores,
    }


def test_remove_overlapping_masks_keeps_higher_score():
    m = np.zeros((3, 8, 8), bool)
    m[0, :4, :4] = True
    m[1, :2, :2] = True  # fully inside 0 -> IoM == 1
    m[2, 6:, 6:] = True
    out = mor.remove_overlapping_masks(_sample(m, [0.5, 0.9, 0.4]), iom_thresh=0.3)
    assert out["kept_indices"] == [1, 2]
    assert out["removed_indices"] == [0]
    assert out["pred_scores"] == [0.9, 0.4]


def test_remove_overlapping_masks_single_mask_is_untouched():
    sample = _sample(np.ones((1, 4, 4), bool), [0.5])
    assert mor.remove_overlapping_masks(sample) is sample
