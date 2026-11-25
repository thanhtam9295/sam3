"""Deterministic scientific regression cases.

Each case is a pure function that builds seeded synthetic inputs, runs one
quantitative algorithm from the ``sam3`` package and returns JSON-serialisable
outputs. The frozen outputs live in ``tests/scientific/baselines/`` and are
compared by ``test_golden_baselines.py``.

Rules (see docs/validation/SCIENTIFIC_VALIDATION.md):
  * Never edit a baseline by hand.
  * Regenerate only with ``python tools/validation/regenerate_baselines.py``
    and only together with an approved validation report.
  * Adding a new case is fine; changing an existing case's inputs is a
    baseline change and needs a report.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

# Bump only when the *case definitions* change (not the algorithms).
CASES_VERSION = 1


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _boxes_xyxy(rs: np.random.RandomState, n: int) -> torch.Tensor:
    xy = rs.uniform(0, 100, size=(n, 2))
    wh = rs.uniform(1, 60, size=(n, 2))
    return torch.tensor(np.concatenate([xy, xy + wh], axis=1), dtype=torch.float64)


def _blob_masks(rs: np.random.RandomState, n: int, h: int, w: int) -> torch.Tensor:
    """Random elliptical blobs: realistic-looking, overlapping instance masks."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((n, h, w), dtype=bool)
    for i in range(n):
        cy, cx = rs.uniform(0, h), rs.uniform(0, w)
        ry, rx = rs.uniform(2, h / 3), rs.uniform(2, w / 3)
        out[i] = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return torch.from_numpy(out)


def _r(t: torch.Tensor | np.ndarray, nd: int = 10) -> Any:
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return np.round(a.astype(np.float64), nd).tolist()


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


def case_box_iou_giou() -> Dict[str, Any]:
    from sam3.model.box_ops import box_iou, generalized_box_iou

    rs = _rng(101)
    a, b = _boxes_xyxy(rs, 12), _boxes_xyxy(rs, 9)
    iou, union = box_iou(a, b)
    return {"iou": _r(iou), "union": _r(union), "giou": _r(generalized_box_iou(a, b))}


def case_mask_iou_iom() -> Dict[str, Any]:
    from sam3.perflib.masks_ops import mask_iou
    from sam3.train.masks_ops import mask_iom

    rs = _rng(202)
    m = _blob_masks(rs, 10, 48, 64)
    return {"iou": _r(mask_iou(m, m)), "iom_reference": _r(mask_iom(m, m))}


def case_perflib_pairwise_iom() -> Dict[str, Any]:
    """Freezes the *current* behaviour of the perflib IoM (KI-004).

    This baseline intentionally records the known-defective output so that a
    fix is visible as a deliberate, reviewed baseline change.
    """
    from sam3.perflib.iou import pairwise_iom

    rs = _rng(203)
    m = _blob_masks(rs, 8, 32, 32)
    return {"iom_perflib": _r(pairwise_iom(m, m), 6)}


def case_masks_to_boxes() -> Dict[str, Any]:
    from sam3.perflib.masks_ops import masks_to_boxes

    rs = _rng(303)
    m = _blob_masks(rs, 10, 40, 56)
    return {"boxes_inclusive": _r(masks_to_boxes(m, list(range(10))))}


def case_mask_nms() -> Dict[str, Any]:
    from sam3.perflib.nms import nms_masks

    rs = _rng(404)
    base = _blob_masks(rs, 6, 40, 40)
    # duplicate + jitter some masks to create genuine NMS competition
    dup = torch.roll(base, shifts=1, dims=2)
    masks = torch.cat([base, dup]).float()
    probs = torch.tensor(rs.uniform(0.0, 1.0, size=12), dtype=torch.float32)
    out = {}
    for iou_thr in (0.3, 0.5, 0.7):
        keep = nms_masks(probs, masks, prob_threshold=0.2, iou_threshold=iou_thr)
        out[f"keep@{iou_thr}"] = keep.tolist()
    return out


def case_frame_nms() -> Dict[str, Any]:
    from sam3.train.nms_helper import apply_frame_nms

    rs = _rng(505)
    b = _boxes_xyxy(rs, 15).numpy().astype(np.float32)
    s = rs.uniform(0, 1, size=15).astype(np.float32)
    return {
        f"keep@{t}": sorted(int(k) for k in apply_frame_nms(b, s, t))
        for t in (0.3, 0.5)
    }


def case_hungarian() -> Dict[str, Any]:
    from sam3.train.matcher import _do_matching

    rs = _rng(606)
    cost = rs.uniform(0, 1, size=(9, 6))
    i, j = _do_matching(cost, return_tgt_indices=True)
    return {
        "pairs": sorted([int(a), int(b)] for a, b in zip(i, j)),
        "cost": round(float(cost[i, j].sum()), 10),
    }


def case_rle() -> Dict[str, Any]:
    from sam3.agent.helpers.rle import rle_encode

    rs = _rng(707)
    m = _blob_masks(rs, 5, 30, 41)
    return {"rles": [r["counts"] for r in rle_encode(m)]}


def case_boundary_f_measure() -> Dict[str, Any]:
    from pycocotools import mask as mu
    from sam3.train.masks_ops import compute_boundary, compute_F_measure, dilation

    rs = _rng(808)
    gt = _blob_masks(rs, 3, 48, 48)
    dt = torch.roll(gt, shifts=(1, 2), dims=(1, 2))

    def enc(x):
        return mu.encode(np.asfortranarray(x.numpy().astype(np.uint8)))

    scores = []
    for g, d in zip(gt, dt):
        gb, db = compute_boundary(g), compute_boundary(d)
        gbd = dilation(gb[None], 5)[0]
        dbd = dilation(db[None], 5)[0]
        scores.append(compute_F_measure(enc(gb), enc(gbd), enc(db), enc(dbd)))
    return {"f_measure": _r(np.array(scores))}


def case_connected_components() -> Dict[str, Any]:
    from sam3.perflib.connected_components import connected_components

    rs = _rng(909)
    x = torch.from_numpy(rs.uniform(size=(2, 24, 24)) > 0.6).to(torch.uint8)
    labels, counts = connected_components(x)
    return {
        "num_components": [int(labels[b].max()) for b in range(2)],
        "sorted_sizes": [
            sorted({int(c) for c in counts[b].flatten().tolist() if c > 0})
            for b in range(2)
        ],
    }


def case_cgf1_synthetic() -> Dict[str, Any]:
    from tests.scientific.cgf1_fixture import run_cgf1_fixture

    return {"metrics": {k: round(v, 10) for k, v in run_cgf1_fixture().items()}}


CASES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "box_iou_giou": case_box_iou_giou,
    "mask_iou_iom": case_mask_iou_iom,
    "perflib_pairwise_iom": case_perflib_pairwise_iom,
    "masks_to_boxes": case_masks_to_boxes,
    "mask_nms": case_mask_nms,
    "frame_nms": case_frame_nms,
    "hungarian": case_hungarian,
    "rle": case_rle,
    "boundary_f_measure": case_boundary_f_measure,
    "connected_components": case_connected_components,
    "cgf1_synthetic": case_cgf1_synthetic,
}
