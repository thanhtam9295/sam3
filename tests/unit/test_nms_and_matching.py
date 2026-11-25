"""Unit tests for NMS variants, Hungarian matching and det<->track association."""

import numpy as np
import pytest
import torch
from sam3.perflib.nms import generic_nms_cpu, nms_masks
from sam3.train import nms_helper
from sam3.train.matcher import _do_matching

pytestmark = pytest.mark.unit


def test_generic_nms_cpu_suppresses_strictly_above_threshold():
    ious = torch.tensor([[1.0, 0.5, 0.9], [0.5, 1.0, 0.1], [0.9, 0.1, 1.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    # IoU(0,1)=0.5 is NOT > 0.5 -> kept; IoU(0,2)=0.9 -> suppressed
    assert generic_nms_cpu(ious, scores, 0.5).tolist() == [0, 1]
    # A stricter threshold suppresses box 1 too
    assert generic_nms_cpu(ious, scores, 0.4).tolist() == [0]


def test_generic_nms_cpu_returns_score_order():
    ious = torch.eye(3)
    scores = torch.tensor([0.1, 0.9, 0.5])
    assert generic_nms_cpu(ious, scores, 0.5).tolist() == [1, 2, 0]


def test_nms_masks_prefilters_by_probability():
    masks = torch.zeros(3, 8, 8)
    masks[0, :4, :4] = 1
    masks[1, :4, :4] = 1  # duplicate of 0
    masks[2, 4:, 4:] = 1
    probs = torch.tensor([0.9, 0.8, 0.2])
    keep = nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.tolist() == [True, False, False]


def test_nms_masks_no_valid_detections():
    keep = nms_masks(torch.tensor([0.1, 0.2]), torch.ones(2, 4, 4), 0.5, 0.5)
    assert keep.tolist() == [False, False]


def test_frame_nms_suppresses_at_or_above_threshold():
    """Frame-level NMS uses >= (unlike generic_nms_cpu's >). See KI-006."""
    bboxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = [int(k) for k in nms_helper.apply_frame_nms(bboxes, scores, 1.0)]
    assert keep == [0, 2], "identical box (IoU == threshold) is suppressed"


def test_is_zero_box_and_format():
    assert nms_helper.is_zero_box(None)
    assert nms_helper.is_zero_box([0, 0, 0, 0])
    assert not nms_helper.is_zero_box([1, 1, 2, 2])
    assert nms_helper.convert_bbox_format([1, 2, 3, 4]) == [1, 2, 4, 6]


def test_do_matching_finds_minimum_cost_assignment():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    i, j = _do_matching(cost, return_tgt_indices=True)
    assert cost[i, j].sum() == pytest.approx(5.0)


def test_do_matching_filtering_drops_invalid_pairs():
    cost = np.array([[1e9, 1.0], [1e9, 1e9]])
    i, j = _do_matching(cost, return_tgt_indices=True, do_filtering=True)
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1)]


def test_associate_det_trk_basic():
    from sam3.perflib.associate_det_trk import associate_det_trk

    det = torch.zeros(2, 8, 8, dtype=torch.bool)
    det[0, :4, :4] = True  # matches track 0
    det[1, 4:, 4:] = True  # new object
    trk = torch.zeros(2, 8, 8, dtype=torch.bool)
    trk[0, :4, :4] = True
    trk[1, :2, 6:] = True  # nothing overlaps -> unmatched
    new_det, unmatched_trk, det_to_trk, _ = associate_det_trk(
        det, trk, det_scores=torch.tensor([0.9, 0.8])
    )
    assert new_det == [1]
    assert unmatched_trk == [1]
    assert dict(det_to_trk) == {0: [0]}


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="KI-007: associate_det_trk indexes det_scores even when it is None",
)
def test_associate_det_trk_without_scores():
    from sam3.perflib.associate_det_trk import associate_det_trk

    m = torch.zeros(1, 4, 4, dtype=torch.bool)
    m[0, :2, :2] = True
    associate_det_trk(m, m.clone(), det_scores=None)
