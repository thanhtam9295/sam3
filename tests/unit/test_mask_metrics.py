"""Unit tests for mask-level metrics: IoU, IoM, intersection, boxes, boundaries."""

import pytest
import torch
from sam3.perflib import iou as perf_iou, masks_ops as perf_masks
from sam3.train import masks_ops as train_masks

pytestmark = pytest.mark.unit


def _square(h, w, y0, y1, x0, x1):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def masks():
    a = _square(16, 16, 0, 8, 0, 8)  # area 64
    b = _square(16, 16, 0, 8, 4, 12)  # area 64, overlap 32 with a
    c = _square(16, 16, 12, 16, 12, 16)  # area 16, disjoint
    d = _square(16, 16, 0, 4, 0, 4)  # area 16, inside a
    return torch.stack([a, b, c, d])


def test_mask_iou_known_values(masks):
    iou = perf_masks.mask_iou(masks, masks)
    assert iou.shape == (4, 4)
    torch.testing.assert_close(iou.diagonal(), torch.ones(4))
    assert iou[0, 1].item() == pytest.approx(32 / 96)
    assert iou[0, 2].item() == 0.0
    assert iou[0, 3].item() == pytest.approx(16 / 64)
    torch.testing.assert_close(iou, iou.T)


def test_mask_iou_empty_masks_do_not_divide_by_zero():
    empty = torch.zeros(2, 4, 4, dtype=torch.bool)
    iou = perf_masks.mask_iou(empty, empty)
    assert torch.isfinite(iou).all() and (iou == 0).all()


def test_mask_iou_rejects_non_bool():
    with pytest.raises(AssertionError):
        perf_masks.mask_iou(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))


def test_pairwise_iou_matches_mask_iou(masks):
    torch.testing.assert_close(
        perf_iou.pairwise_iou(masks, masks, eps=None),
        perf_masks.mask_iou(masks, masks),
    )


def test_reference_iom_contained_mask_is_one(masks):
    iom = train_masks.mask_iom(masks, masks)
    # d is fully inside a -> intersection over min area == 1
    assert iom[0, 3].item() == pytest.approx(1.0, abs=1e-6)
    assert iom[3, 0].item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(iom, iom.T)


@pytest.mark.xfail(
    strict=True,
    reason="KI-004: perflib.iou.pairwise_iom takes a row-wise, not pairwise, min area",
)
def test_perflib_iom_matches_reference_iom(masks):
    torch.testing.assert_close(
        perf_iou.pairwise_iom(masks, masks),
        train_masks.mask_iom(masks, masks).float(),
        atol=1e-6,
        rtol=0,
    )


def test_blocked_and_vectorized_intersections_agree(masks):
    a = train_masks.mask_intersection(masks, masks, block_size=3)
    b = train_masks.mask_intersection_vectorized(masks, masks)
    assert torch.equal(a, b)
    assert a[0, 1].item() == 32


def test_perflib_masks_to_boxes_inclusive_convention(masks):
    boxes = perf_masks.masks_to_boxes(masks, list(range(4)))
    # a spans rows 0..7, cols 0..7 -> inclusive max = 7
    torch.testing.assert_close(boxes[0], torch.tensor([0.0, 0.0, 7.0, 7.0]))
    torch.testing.assert_close(boxes[2], torch.tensor([12.0, 12.0, 15.0, 15.0]))


def test_instance_to_semantic_masks():
    inst = torch.zeros(3, 4, 4, dtype=torch.bool)
    inst[0, 0, 0] = True
    inst[1, 1, 1] = True
    inst[2, 3, 3] = True
    sem = train_masks.instance_masks_to_semantic_masks(inst, torch.tensor([2, 0, 1]))
    assert sem.shape == (3, 4, 4)
    assert sem[0].sum() == 2 and sem[1].sum() == 0 and sem[2].sum() == 1


def test_compute_boundary_of_square():
    seg = _square(8, 8, 2, 6, 2, 6)
    b = train_masks.compute_boundary(seg)
    assert b.dtype == torch.bool
    assert b[4, 4].item() is False, "interior pixel is not boundary"
    assert b.sum().item() > 0
