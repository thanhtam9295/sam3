# Quantitative analysis algorithms

Specification of every algorithm in this repository that turns model outputs
into numbers (geometry, matching, suppression, encoding, metrics). Each entry
lists the implementation, the exact definition **as implemented**, conventions
that affect results, and the tests that pin it down.

All entries are **protected** (`tools/validation/protected_paths.txt`):
changing behaviour requires [SCIENTIFIC_VALIDATION.md](validation/SCIENTIFIC_VALIDATION.md).
Defects are tracked as `KI-###` in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and are
documented here as they behave today, not as they ought to behave.

Test legend: **U** unit (`tests/unit/`), **R** reference agreement
(`tests/scientific/test_reference_agreement.py`), **G** golden baseline
(`tests/scientific/baselines/<case>.json`), **E** end-to-end toy evaluation.

---

## 1. Box geometry — `sam3/model/box_ops.py`

Formats: `xyxy` (x0, y0, x1, y1), `xywh` (x0, y0, w, h), `cxcywh`
(centre, size). Conversions are exact affine maps; the model predicts
normalised `cxcywh`, post-processing emits pixel `xyxy`, COCO json uses `xywh`.

* Area `A = (x1 − x0)(y1 − y0)` (no +1; continuous coordinates).
* IoU `= |B₁∩B₂| / (A₁ + A₂ − |B₁∩B₂|)`, intersection clamped at 0. No
  epsilon: two zero-area boxes give NaN.
* GIoU (Rezatofighi et al., 2019) `= IoU − (C − U)/C`, `C` = area of the
  smallest enclosing box. Range (−1, 1]; GIoU ≤ IoU.
* `fast_diag_*`: TorchScript element-wise (paired) versions.

Tests: U `test_box_ops.py`; R vs `torchvision.ops.box_iou` / `generalized_box_iou`; G `box_iou_giou`.

## 2. Mask overlap — IoU and IoM

| Function | Definition | Empty-mask behaviour |
|---|---|---|
| `perflib.masks_ops.mask_iou(a, b)` | `I / max(U, 1)`, `I` via float matmul | 0 |
| `perflib.iou.pairwise_iou(a, b, eps)` | `I / (U + eps)`, or `I / max(U,1)` if `eps=None` | ≈0 |
| `train.masks_ops.mask_iom(a, b)` | `I / (min(Aᵢ, Aⱼ) + 1e-8)` — **reference IoM** | 0 |
| `agent.helpers.mask_overlap_removal.mask_iom` | `I / (max(min(Aᵢ,Aⱼ),1) + 1e-8)` | 0 |
| `perflib.iou.pairwise_iom(a, b)` | intended IoM; **actually** `I / (min(Aᵢ, A'ᵢ) + 1e-8)` row-wise (KI-004) | — |

IoM (intersection over minimum) is 1 when one mask is inside the other; it
is used to suppress nested duplicates that IoU would keep.

Tests: U `test_mask_metrics.py`; R vs `pycocotools.mask.iou` and brute force;
G `mask_iou_iom`, `perflib_pairwise_iom` (freezes current KI-004 output);
strict xfail `test_perflib_iom_matches_reference_iom`.

## 3. Mask → box

| Function | Max-corner convention |
|---|---|
| `perflib.masks_ops.masks_to_boxes` (used in video inference) | **inclusive**: x1 = last foreground column (= `torchvision.ops.masks_to_boxes`); empty mask → (W, H, 0, 0) |
| `model.box_ops.masks_to_boxes` (no in-repo callers) | **exclusive**: x1 = last column + 1; empty → zeros; crashes unless N ∈ {1, 4} (KI-005) |

The 1-pixel difference (KI-003) matters when comparing box areas across code
paths. Tests: U; R vs torchvision; G `masks_to_boxes`.

## 4. Non-maximum suppression

* **Mask NMS** `perflib.nms.nms_masks`: keep `p > prob_threshold`, compute
  `mask_iou` on `mask > 0`, then greedy NMS in descending score; a candidate is
  suppressed when IoU **>** `iou_threshold` (same as `torchvision.ops.nms`).
  Backends: `torch_generic_nms` (CUDA) → Triton → `generic_nms_cpu`; all must
  agree. Ties in score: order from `numpy.argsort` (not stable across backends).
* **Detector NMS with IoM** `sam3_multiplex_detector_utils` (SAM 3.1): IoU or IoM
  matrix → generic NMS. With perflib enabled, IoM comes from `pairwise_iom` (KI-004).
* **Frame-level / track-level NMS for video predictions** `train.nms_helper`:
  boxes `xywh → xyxy`; suppression when IoU **≥** threshold (KI-006, note the
  ≥ vs > difference). Track IoU is pooled, not averaged: Σ intersections /
  Σ unions over the frames where both tracks have a valid box (0 if none).

Tests: U; R `generic_nms_cpu` vs torchvision NMS, frame NMS vs a greedy reference;
G `mask_nms`, `frame_nms`.

## 5. Matching and association

* **Hungarian matching** `train.matcher` (`scipy.optimize.linear_sum_assignment`)
  on a weighted cost of classification (focal or CE), L1 box and −GIoU; optional
  filtering of pairs with cost ≥ 1e8; `repeats` tiles targets for one-to-many.
  Training-only: affects learned weights, not evaluation.
* **Detection ↔ track association** `perflib.associate_det_trk`: masks are
  resized to a common size (bilinear, `> 0`), IoU matrix, Hungarian on `1 − IoU`.
  A track is *matched* if its Hungarian partner has IoU ≥ `iou_threshold_trk`;
  a detection is *new* if it has IoU < `iou_threshold` with **every** track and
  score ≥ `new_det_thresh` (many-to-one allowed). Caveats: KI-007, KI-012.

Tests: U; R Hungarian vs brute-force optimum; G `hungarian`.

## 6. cgF1 — concept-grounded F1 (image) — `sam3/eval/cgf1_eval.py`

Each "image" is an (image, noun-phrase) pair. Only `is_instance_exhaustive`
pairs are evaluated. Predictions with `score < 0.5` are discarded. IoU
thresholds `τ ∈ {0.50, 0.55, …, 0.95}`; reported metrics average over τ unless
suffixed `@0.5` / `@0.75`.

Per pair, Hungarian matching on −IoU (one-to-one), then at each τ:
`TP = #matches with IoU ≥ τ`, `FP = |dt| − TP`, `FN = |gt| − TP`.

Image-level (IL) classification: `IL_TP` gt>0 & dt>0; `IL_FP` gt=0 & dt>0;
`IL_FN` gt>0 & dt=0; `IL_TN` gt=0 & dt=0.

Accumulated (ε = 1e-4 for box level, 1e-6 for IL):

```
precision      = ΣTP / (ΣTP + ΣFP + ε)
recall         = ΣTP / (ΣTP + ΣFN + ε)
pm_precision   = ΣTP / (ΣTP + ΣFP_pos + ε)      FP_pos: FPs only from pairs with gt>0 & dt>0
pmF1           = 2·pm_precision·recall / (pm_precision + recall + ε)   "positive_micro_F1"
IL_MCC         = (TP·TN − FP·FN) / (√((TP+FP)(TP+FN)(TN+FP)(TN+FN)) + 1e-6)
cgF1           = pmF1 × IL_MCC
```

Interpretation: pmF1 measures localisation quality on images where the concept
is present; IL_MCC measures whether the model correctly decides presence.
Also reported: F1, positive_macro_F1 (mean per-pair F1), IL precision/recall/F1/FPR.
Oracle mode (several GT files): per pair, the GT giving the best mean local F1 is used.

Note: the ε terms bias metrics slightly downwards (e.g. perfect precision is
0.9999…); results are comparable only with identical ε (they are frozen).

Tests: R hand-derived 5-pair fixture (`tests/scientific/cgf1_fixture.py`);
integration perfect-prediction segm case = 1.0; G `cgf1_synthetic`; E toy VEval.

## 7. Video metrics — `sam3/eval/saco_veval_*`, `ytvis_eval.py`, toolkits

* **Video mAP** (`YTVISPredFileEvaluator`): YT-VIS COCO-style AP with
  spatio-temporal mask/box IoU (sum of per-frame intersections / sum of unions).
* **Phrase AP** (`VideoPhraseApEvaluator`): AP where each (video, phrase) is a class.
* **HOTA** (`hota_eval_toolkit`, fork of TrackEval): `HOTA = √(DetA · AssA)`
  averaged over localisation thresholds α ∈ [0.05, 0.95]; reports DetRe/DetPr,
  AssRe/AssPr, LocA, OWTA.
* **TETA** (`teta_eval_toolkit`): mean of localisation (LocA), association
  (AssocA) and classification (ClsA) accuracies.
* **Video cgF1** (`VideoCGF1Evaluator`): §6 applied with spatio-temporal IoU per (video, phrase).
* Before scoring, predictions pass frame/track NMS (§4) where configured.

Tests: E `test_toy_video_eval_regression.py` reproduces all 54 shipped
reference metrics exactly (KI-008: 34 legacy `*_demo_*` keys no longer emitted).

## 8. Boundary F-measure — `train.masks_ops`

`compute_boundary`: 1-px boundary via XOR with right/down/diagonal shifts
(TrackEval J&F). `dilation`: square kernel (OpenCV on CPU, separable conv on
GPU). `compute_F_measure`: boundary precision/recall with tolerance given by
the dilated boundaries; empty cases follow TrackEval (both empty → 1).
Tests: U; G `boundary_f_measure`.

## 9. Encodings and pixel operations

* **RLE** `agent.helpers.rle.rle_encode` / `train.masks_ops.rle_encode`:
  column-major (Fortran) run lengths computed on GPU, compressed with
  pycocotools; byte-identical to `pycocotools.mask.encode` (R, G `rle`).
* **Connected components** `perflib.connected_components`: CUDA `cc_torch` →
  Triton → CPU `skimage.measure.label` (8-connectivity). Returns labels and
  per-pixel component size. R vs `scipy.ndimage.label`; G `connected_components`.
* **Euclidean distance transform** `model/edt.py`: Triton implementation of
  Felzenszwalb & Huttenlocher (2012), mimicking `cv2.distanceTransform(DIST_L2, 0)`.
  GPU-only; not covered by CPU CI (KI-011, task T-005).

## 10. Thresholds that change reported numbers

| Parameter | Where | Default |
|---|---|---|
| detection confidence | `Sam3Processor(confidence_threshold=)` | 0.5 |
| cgF1 score filter | `CGF1Eval(threshold=)` | 0.5 |
| mask binarisation | post-processing, `masks_logits > 0` / prob `> 0.5` | — |
| NMS IoU / IoM | detector / video configs | per config |
| association IoU | `associate_det_trk(iou_threshold, iou_threshold_trk)` | 0.5 / 0.5 |

Always report these with results.
