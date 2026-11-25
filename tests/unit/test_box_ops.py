"""Unit tests for sam3.model.box_ops (box conversions, IoU, GIoU)."""

import pytest
import torch
from sam3.model import box_ops

pytestmark = pytest.mark.unit


def _random_xyxy(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 100
    wh = torch.rand(n, 2, generator=g) * 50 + 1
    return torch.cat([xy, xy + wh], dim=-1)


@pytest.mark.parametrize(
    "fwd,inv",
    [
        (box_ops.box_xyxy_to_cxcywh, box_ops.box_cxcywh_to_xyxy),
        (box_ops.box_xyxy_to_xywh, box_ops.box_xywh_to_xyxy),
    ],
)
def test_conversion_round_trip(fwd, inv):
    boxes = _random_xyxy(32)
    torch.testing.assert_close(inv(fwd(boxes)), boxes, rtol=0, atol=1e-4)


def test_xywh_cxcywh_consistency():
    boxes = _random_xyxy(16)
    xywh = box_ops.box_xyxy_to_xywh(boxes)
    via_xywh = box_ops.box_xywh_to_cxcywh(xywh)
    direct = box_ops.box_xyxy_to_cxcywh(boxes)
    torch.testing.assert_close(via_xywh, direct, rtol=0, atol=1e-4)
    torch.testing.assert_close(
        box_ops.box_cxcywh_to_xywh(direct), xywh, rtol=0, atol=1e-4
    )


def test_box_area_known_values():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 5.0]])
    torch.testing.assert_close(box_ops.box_area(boxes), torch.tensor([6.0, 0.0]))


def test_box_iou_identity_disjoint_and_half():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [5.0, 0.0, 15.0, 10.0]]
    )
    iou, union = box_ops.box_iou(a, b)
    torch.testing.assert_close(iou, torch.tensor([[1.0, 0.0, 1.0 / 3.0]]))
    torch.testing.assert_close(union, torch.tensor([[100.0, 200.0, 150.0]]))


def test_box_iou_is_symmetric_and_bounded():
    a, b = _random_xyxy(8, 1), _random_xyxy(5, 2)
    iou_ab, _ = box_ops.box_iou(a, b)
    iou_ba, _ = box_ops.box_iou(b, a)
    torch.testing.assert_close(iou_ab, iou_ba.T)
    assert (iou_ab >= 0).all() and (iou_ab <= 1 + 1e-6).all()


def test_giou_bounds_and_relation_to_iou():
    a, b = _random_xyxy(8, 3), _random_xyxy(6, 4)
    giou = box_ops.generalized_box_iou(a, b)
    iou, _ = box_ops.box_iou(a, b)
    assert (giou >= -1 - 1e-6).all() and (giou <= 1 + 1e-6).all()
    assert (giou <= iou + 1e-6).all(), "GIoU must never exceed IoU"


def test_giou_far_apart_tends_to_minus_one():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[1000.0, 1000.0, 1001.0, 1001.0]])
    assert box_ops.generalized_box_iou(a, b).item() < -0.99


def test_fast_diag_variants_match_full_matrix_diagonal():
    a, b = _random_xyxy(10, 5), _random_xyxy(10, 6)
    iou, _ = box_ops.box_iou(a, b)
    giou = box_ops.generalized_box_iou(a, b)
    torch.testing.assert_close(box_ops.fast_diag_box_iou(a, b), iou.diagonal())
    torch.testing.assert_close(
        box_ops.fast_diag_generalized_box_iou(a, b), giou.diagonal()
    )


def test_box_xywh_inter_union():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
    inter, union = box_ops.box_xywh_inter_union(a, b)
    torch.testing.assert_close(inter, torch.tensor([25.0]))
    torch.testing.assert_close(union, torch.tensor([175.0]))


def test_model_masks_to_boxes_uses_exclusive_max_corner():
    """box_ops.masks_to_boxes returns x_max/y_max as (last pixel index + 1).

    This is a convention difference from sam3.perflib.masks_ops (inclusive).
    See docs/KNOWN_ISSUES.md KI-003. Uses a batch of 1, the only size other
    than 4 that does not hit KI-005.
    """
    m = torch.zeros(1, 8, 8)
    m[0, 2:5, 1:4] = 1  # rows 2..4, cols 1..3
    boxes = box_ops.masks_to_boxes(m)
    torch.testing.assert_close(boxes[0], torch.tensor([1.0, 2.0, 4.0, 5.0]))


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="KI-005: box_ops.masks_to_boxes final broadcast fails unless N in {1, 4}",
)
def test_model_masks_to_boxes_batch_of_two():
    m = torch.zeros(2, 8, 8)
    m[0, 2:5, 1:4] = 1
    boxes = box_ops.masks_to_boxes(m)
    torch.testing.assert_close(boxes[1], torch.zeros(4))  # empty mask -> zero box


def test_model_masks_to_boxes_empty_input():
    assert box_ops.masks_to_boxes(torch.zeros(0, 4, 4)).shape == (0, 4)
