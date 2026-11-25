---
name: sam3-segmentation-analysis
description: How to run SAM 3 / SAM 3.1 to segment, count and measure objects in images or videos from a text or box prompt, and how to report results with appropriate scientific caveats. Use when asked to analyse an image or video, count objects, extract masks/boxes/areas, or evaluate predictions against ground truth with cgF1, HOTA or AP.
---

# SAM 3 segmentation analysis

## Requirements

* NVIDIA GPU with CUDA (model construction is CUDA-only today, KI-009).
* Checkpoint access: gated on Hugging Face (`facebook/sam3`, `facebook/sam3.1`).
  The human authenticates (`hf auth login`); never handle their token yourself.
* Data stays local. Do not upload images or results anywhere.

## Image: text prompt → masks, boxes, scores

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

processor = Sam3Processor(build_sam3_image_model())
state = processor.set_image(Image.open(path))
out = processor.set_text_prompt(state=state, prompt="cell nucleus")
masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
```

Box convention: `boxes` are XYXY in pixels. Mask area in pixels: `masks.flatten(1).sum(1)`.
Convert to physical units only with the instrument's calibrated pixel size.

## Video

Use `build_sam3_predictor(...)` and the request API (`start_session`,
`add_prompt`, `propagate_in_video`, `close_session`); see
`examples/sam3_video_predictor_example.ipynb` and `examples/sam3.1_video_predictor_example.ipynb`.

## Evaluation against ground truth

* Image, open-vocabulary: `sam3.eval.cgf1_eval.CGF1Evaluator(gt_json, iou_type="segm").evaluate(pred_json)`.
* Video: `python -m sam3.eval.saco_veval_eval one --gt_annot_file ... --pred_file ... --eval_res_file ...`.
* Metric definitions and caveats: `docs/ALGORITHMS.md`.

## Reporting results (always include)

* Model + checkpoint version, prompt text, score threshold, NMS/IoU thresholds.
* Counts are threshold-dependent; report the threshold and, ideally, a sweep.
* SAM 3 is a general-domain model: performance on specialised imagery
  (microscopy, medical) is unvalidated unless a domain validation report exists.
* Known caveats that may affect numbers: `docs/KNOWN_ISSUES.md` (e.g. KI-004 on IoM NMS).
