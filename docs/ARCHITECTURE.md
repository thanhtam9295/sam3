# Software architecture

This document describes how the `sam3` package is organised and how data flows
through it. Algorithms and metrics are specified in [ALGORITHMS.md](ALGORITHMS.md).
Model-level design rationale is in the SAM 3 paper (arXiv:2511.16719) and
[`RELEASE_SAM3p1.md`](../RELEASE_SAM3p1.md).

## 1. System context

```
             text prompt / exemplar boxes / points
                           │
 image(s) or video ──► ┌──────────────────────────┐ ──► masks, boxes, scores, object ids
                       │   sam3 (PyTorch, CUDA)   │
 checkpoint (HF, gated)►└──────────────────────────┘
                           │
        ┌──────────────────┼────────────────────┬─────────────────────────┐
  sam3.eval (metrics)  sam3.train (training)  sam3.agent (LLM tool use)  front ends added here:
  cgF1, HOTA, TETA, AP  Hydra configs, losses  calls an OpenAI-compatible  packaging/windows (CLI)
                                               LLM endpoint (§7)          deploy/service (REST)
                                                                           deploy/m365-copilot
```

Runtime requirements: Python 3.12, PyTorch 2.10, **CUDA GPU** (model
construction allocates on `cuda`, KI-009). Checkpoints are gated on Hugging
Face (`facebook/sam3`, `facebook/sam3.1`) and are never stored in the repo.

## 2. Package map

| Package | Role | Key entry points |
|---|---|---|
| `sam3/model_builder.py` | Assembles every model variant from components; downloads checkpoints | `build_sam3_image_model`, `build_sam3_video_model`, `build_sam3_multiplex_video_model`, `build_sam3_predictor(version="sam3"\|"sam3.1")` |
| `sam3/model/` | Architecture and inference logic | see §3–§5 |
| `sam3/sam/` | SAM-1/2 lineage prompt encoder, mask decoder, two-way transformer, RoPE | `PromptEncoder`, `MaskDecoder` |
| `sam3/perflib/` | Performance kernels with CPU fallbacks (IoU, NMS, connected components, mask→box, det↔track association, Triton kernels, FA3, compile helpers) | `mask_iou`, `nms_masks`, `connected_components`, `associate_det_trk` |
| `sam3/eval/` | Evaluation: cgF1, YT-VIS/COCO AP, phrase AP, HOTA (TrackEval fork), TETA, demo F1; post-processors | `CGF1Evaluator`, `saco_veval_eval.VEvalEvaluator` |
| `sam3/train/` | Training: Hydra configs, trainer, datasets, transforms, Hungarian matchers, losses, NMS helpers | `train.py`, `matcher.py`, `loss/` |
| `sam3/agent/` | "SAM 3 Agent": an LLM plans, calls SAM 3 as a tool, inspects results | `agent_core.py`, `client_llm.py`, `client_sam3.py` |
| `sam3/visualization_utils.py` | Plotting helpers for notebooks | |
| `scripts/` | Benchmarks (`measure_speed.py`, `bench_compile.py`), dataset prep for SA-Co Gold/Silver/VEval | |
| `examples/` | Jupyter notebooks for every workflow | |

## 3. Image model (concept detection + segmentation)

Built by `build_sam3_image_model` → `Sam3Image` (`sam3/model/sam3_image.py`).

```
image ─► ViT backbone (vitdet.py, RoPE, windowed+global attn, 1008 px)
        └► neck (necks.py) ─► multi-scale features ─┐
text ─► BPE tokenizer (tokenizer_ve.py) ─► text encoder (text_encoder_ve.py)
        ─► VL combiner (vl_combiner.py) ─────────────┤
boxes/points ─► geometry encoder (geometry_encoders.py) ─┤
                                                     ▼
          fusion transformer encoder (encoder.py) ─► DETR-style decoder (decoder.py)
          with learned object queries + **presence token**
                                                     ▼
          dot-product scoring (model_misc.py) ─► per-query logits × presence score
          segmentation head (maskformer_segmentation.py) ─► per-query masks
          optional SAM-style interactive head (sam1_task_predictor.py) for point/box refinement
```

`Sam3Processor` (`sam3_image_processor.py`) is the user-facing wrapper:
`set_image` → `set_text_prompt` / `add_geometric_prompt` → thresholded
`masks`, `boxes` (XYXY pixels), `scores`. The presence token decouples
"is the concept in the image" from "where are its instances", which is what
the image-level part of cgF1 measures (ALGORITHMS.md §6).

## 4. Video model (SAM 3): decoupled detector + tracker

`Sam3VideoInference*` (`sam3_video_base.py`, `sam3_video_inference.py`):

1. **Detector** = the image model, run per frame on the text prompt.
2. **Tracker** = SAM 2–style memory tracker (`sam3_tracker_base.py`,
   `memory.py`) propagating each object with a memory bank of past masks.
3. **Association / heuristics** per frame (`sam3_video_base.py`,
   `perflib/associate_det_trk.py`): match detections to tracklets by mask IoU
   (Hungarian), spawn new tracks from unmatched confident detections, suppress
   duplicates with IoM, re-condition drifting tracks, remove stale ones.
4. **Predictors** (`sam3_video_predictor.py`, `sam3_base_predictor.py`): a
   session/request API — `start_session`, `add_prompt`, `propagate_in_video`
   (streaming, optional cache eviction), `remove_object`, `close_session`.

## 5. Video model (SAM 3.1): Object Multiplex

`build_sam3_multiplex_video_model` (`video_tracking_multiplex.py`,
`sam3_multiplex_base.py`, `sam3_multiplex_tracking.py`,
`sam3_multiplex_detector*.py`, `multiplex_mask_decoder.py`). Objects are packed
into fixed-capacity buckets (default 16) that share one memory read/write, so
cost scales with buckets, not objects. Adds batched post-processing, fewer
CPU↔GPU syncs and `torch.compile` paths (`perflib/compile.py`). Detection NMS
here can use IoM (see KI-004).

## 6. Evaluation stack

```
predictions (COCO/YT-VIS json) ─► sam3/eval
   image:  cgf1_eval.CGF1Evaluator        (cgF1, IL_MCC, pmF1, … at IoU .5:.95)
           coco_eval / coco_eval_offline  (standard COCO AP)
   video:  saco_veval_eval.VEvalEvaluator ─► YTVISPredFileEvaluator (video mAP)
                                           ─► VideoPhraseApEvaluator (phrase AP)
                                           ─► VideoTetaEvaluator     (TETA)
                                           ─► VideoPhraseHotaEvaluator (HOTA)
                                           ─► VideoCGF1Evaluator     (video cgF1)
```

Frame/track-level NMS for video predictions is in `train/nms_helper.py`.

## 7. SAM 3 Agent

`sam3/agent/` wraps SAM 3 as a tool for a multimodal LLM reached through an
OpenAI-compatible endpoint (`client_llm.py`, `server_url`/`api_key`
parameters). **Images are base64-encoded and sent to that endpoint**, so it
must be a locally hosted or enterprise-approved model when images are
proprietary (KI-013, [SECURITY_AND_DATA.md](SECURITY_AND_DATA.md)).

## 8. Added for collaborative development (this fork)

| Path | Purpose |
|---|---|
| `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.agent/rules/`, `.github/copilot-instructions.md`, `.github/instructions/` | Agent instructions (AGENTS.md is canonical) |
| `.agents/skills/` (+ mirrors) | Agent Skills (SKILL.md standard) |
| `tools/agents/` | Worktrees, file claims, PR-overlap check, skill sync |
| `tools/validation/` | Protected-path guard, baseline regeneration |
| `tools/security/` | Secret / data / weights scanner |
| `tests/` | Unit, integration, scientific, tooling tests |
| `requirements/` | Inputs + hash-pinned lock files |
| `packaging/windows/` | PyInstaller CLI bundle |
| `deploy/service/`, `deploy/m365-copilot/` | REST service and M365 Copilot app package |

None of these change the behaviour of the `sam3` package. They are kept out of
`sam3/` so upstream merges stay conflict-free.

## 9. Extension guidelines

* New functionality goes in new modules; keep edits to upstream files minimal.
* Anything producing numbers someone may publish belongs in
  `tools/validation/protected_paths.txt` and needs reference tests.
* Prefer pure functions with explicit tensor shapes/dtypes in docstrings;
  keep CPU fallbacks for every GPU kernel so it can be tested in CI.
