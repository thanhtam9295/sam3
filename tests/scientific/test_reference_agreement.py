"""Scientific validity: SAM 3 implementations vs independent references.

Golden baselines only prove that outputs did not *change*. These tests prove
the outputs are *correct* by comparing against well-established third-party
implementations (torchvision, pycocotools, scipy) or brute force.
"""

import itertools

import numpy as np
import pytest
import torch
import torchvision.ops as tvo
from pycocotools import mask as mu
from tests.scientific.cases import _blob_masks, _boxes_xyxy, _rng

pytestmark = pytest.mark.scientific


def test_box_iou_matches_torchvision():
    from sam3.model.box_ops import box_iou

    rs = _rng(1)
    a, b = _boxes_xyxy(rs, 20), _boxes_xyxy(rs, 15)
    torch.testing.assert_close(box_iou(a, b)[0], tvo.box_iou(a, b))


def test_giou_matches_torchvision():
    from sam3.model.box_ops import generalized_box_iou

    rs = _rng(2)
    a, b = _boxes_xyxy(rs, 20), _boxes_xyxy(rs, 15)
    torch.testing.assert_close(generalized_box_iou(a, b), tvo.generalized_box_iou(a, b))


def test_mask_iou_matches_pycocotools():
    from sam3.perflib.masks_ops import mask_iou

    m = _blob_masks(_rng(3), 12, 37, 53)
    rles = mu.encode(np.asfortranarray(m.permute(1, 2, 0).numpy().astype(np.uint8)))
    ref = mu.iou(rles, rles, [0] * len(rles))
    np.testing.assert_allclose(mask_iou(m, m).numpy(), ref, atol=1e-6)


def test_reference_iom_matches_bruteforce():
    from sam3.train.masks_ops import mask_iom

    m = _blob_masks(_rng(4), 7, 20, 20).numpy()
    ref = np.zeros((7, 7))
    for i, j in itertools.product(range(7), range(7)):
        inter = (m[i] & m[j]).sum()
        ref[i, j] = inter / (min(m[i].sum(), m[j].sum()) + 1e-8)
    np.testing.assert_allclose(
        mask_iom(torch.from_numpy(m), torch.from_numpy(m)), ref, atol=1e-6
    )


def test_perflib_masks_to_boxes_matches_torchvision():
    from sam3.perflib.masks_ops import masks_to_boxes

    m = _blob_masks(_rng(5), 9, 31, 45)
    m = m[m.flatten(1).any(1)]  # torchvision is undefined for empty masks
    torch.testing.assert_close(
        masks_to_boxes(m, list(range(len(m)))), tvo.masks_to_boxes(m)
    )


def test_generic_nms_cpu_matches_torchvision_nms():
    from sam3.model.box_ops import box_iou
    from sam3.perflib.nms import generic_nms_cpu

    rs = _rng(6)
    boxes = _boxes_xyxy(rs, 40).float()
    scores = torch.tensor(rs.permutation(40) / 40.0, dtype=torch.float32)  # no ties
    for thr in (0.1, 0.3, 0.5, 0.7):
        ours = generic_nms_cpu(box_iou(boxes, boxes)[0], scores, thr)
        assert ours.tolist() == tvo.nms(boxes, scores, thr).tolist(), thr


def test_frame_nms_matches_greedy_reference():
    from sam3.train.nms_helper import apply_frame_nms

    rs = _rng(7)
    boxes = _boxes_xyxy(rs, 30).float()
    scores = rs.permutation(30).astype(np.float32)
    iou = tvo.box_iou(boxes, boxes).numpy()
    for thr in (0.3, 0.5):
        keep, alive = [], np.ones(30, bool)
        for i in np.argsort(-scores):
            if alive[i]:
                keep.append(int(i))
                alive &= ~(iou[i] >= thr)  # frame NMS convention: >= (KI-006)
        ours = [int(k) for k in apply_frame_nms(boxes.numpy(), scores, thr)]
        assert ours == keep


def test_hungarian_is_globally_optimal():
    from sam3.train.matcher import _do_matching

    rs = _rng(8)
    cost = rs.uniform(size=(5, 4))
    i, j = _do_matching(cost, return_tgt_indices=True)
    best = min(
        sum(cost[r, c] for r, c in zip(rows, range(4)))
        for rows in itertools.permutations(range(5), 4)
    )
    assert cost[i, j].sum() == pytest.approx(best)


def test_rle_matches_pycocotools_encode():
    from sam3.agent.helpers.rle import rle_encode

    m = _blob_masks(_rng(9), 6, 29, 43)
    ref = [mu.encode(np.asfortranarray(x.numpy().astype(np.uint8))) for x in m]
    assert [r["counts"] for r in rle_encode(m)] == [r["counts"].decode() for r in ref]


def test_connected_components_matches_scipy_8_connectivity():
    from sam3.perflib.connected_components import connected_components
    from scipy import ndimage

    x = torch.from_numpy(np.random.RandomState(10).uniform(size=(3, 30, 30)) > 0.55)
    labels, counts = connected_components(x.to(torch.uint8))
    for b in range(3):
        ref_labels, n = ndimage.label(x[b].numpy(), structure=np.ones((3, 3)))
        assert int(labels[b].max()) == n
        ref_sizes = np.bincount(ref_labels.ravel())[ref_labels]
        ref_sizes[ref_labels == 0] = 0
        np.testing.assert_array_equal(counts[b, 0].numpy(), ref_sizes)


def test_cgf1_matches_hand_derivation():
    from tests.scientific.cgf1_fixture import hand_computed, run_cgf1_fixture

    got, exp = run_cgf1_fixture(), hand_computed()
    assert got["cgF1_eval_bbox_positive_micro_F1"] == pytest.approx(
        exp["positive_micro_F1"], abs=1e-6
    )
    assert got["cgF1_eval_bbox_IL_MCC"] == pytest.approx(exp["IL_MCC"], abs=1e-6)
    assert got["cgF1_eval_bbox_cgF1"] == pytest.approx(exp["cgF1"], abs=1e-6)
    assert got["cgF1_eval_bbox_cgF1@0.5"] == pytest.approx(exp["cgF1"], abs=1e-6)
