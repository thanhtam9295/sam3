---
applyTo: "sam3/eval/**,sam3/perflib/**,sam3/model/box_ops.py,sam3/model/edt.py,sam3/model/sam3_multiplex_detector_utils.py,sam3/model/sam3_tracker_utils.py,sam3/model/sam3_video_base.py,sam3/model/sam3_video_inference.py,sam3/model/sam3_multiplex_tracking.py,sam3/train/matcher.py,sam3/train/masks_ops.py,sam3/train/nms_helper.py,sam3/train/loss/**,sam3/agent/helpers/mask_overlap_removal.py,sam3/agent/helpers/rle.py,sam3/agent/helpers/masks.py,sam3/agent/helpers/boxes.py,scripts/eval/**,tests/scientific/**,assets/veval/**"
---

This file is a **protected quantitative-analysis path** (the authoritative list
is `tools/validation/protected_paths.txt`).

* Do not change numerical behaviour unless the task explicitly asks for it and a
  human has approved. Follow `.github/skills/scientific-validation/SKILL.md`.
* Any change must ship with `docs/validation/reports/<date>-<slug>.md`
  (from `TEMPLATE.md`) left as `Status: PROPOSED` for a human to approve.
* Never edit `tests/scientific/baselines/*.json` by hand or loosen tolerances.
* Known defects in these files are listed in `docs/KNOWN_ISSUES.md`; reference
  their KI ids instead of silently "fixing" them.
