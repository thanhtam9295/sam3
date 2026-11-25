# Known issues

Defects, caveats and environment problems. **Algorithm defects are recorded,
not silently fixed**: a fix changes numbers and needs the
[scientific validation procedure](validation/SCIENTIFIC_VALIDATION.md).
Many are pinned by strict `xfail` tests that will fail loudly once fixed, so
the fix is reviewed deliberately.

Format: id · severity (High/Medium/Low/Info) · area · status. Next free id: **KI-018**.

---

### KI-001 · Medium · environment · Mitigated
**`import sam3` fails with setuptools ≥ 81.** `sam3/model_builder.py:8` imports
`pkg_resources` (removed from modern setuptools) to locate the BPE vocabulary.
*Mitigation:* lock files pin `setuptools<81`. *Fix proposal:* switch to
`importlib.resources` (T-001; not a protected path, behaviour-neutral).

### KI-002 · Medium · packaging · Mitigated
**Runtime imports not declared in `pyproject.toml`:** `psutil`
(`sam3_video_predictor.py`), `pycocotools`, `scipy`, `torchvision`, `torch`,
plus `scikit-image` / `opencv-python` for CPU fallbacks. A plain `pip install -e .`
cannot import the video predictor. *Mitigation:* `requirements/runtime.in` and
the lock files. `pyproject.toml` left unchanged to preserve upstream metadata (T-002).

### KI-003 · Info · algorithms · Documented
**Two `masks_to_boxes` with different max-corner conventions.**
`perflib.masks_ops` is inclusive (= torchvision), `model.box_ops` is exclusive (+1).
Box areas differ by one row/column. Tests: `test_perflib_masks_to_boxes_inclusive_convention`,
`test_model_masks_to_boxes_uses_exclusive_max_corner`.

### KI-004 · **High** · algorithms (SAM 3.1 detection NMS) · Open — needs validation
**`sam3/perflib/iou.py::pairwise_iom` is not a pairwise IoM.** `torch.min(area_pred (N,1), area_gt (M,1))`
broadcasts row-wise, giving `I_ij / min(A_i, A'_i)`; with identical inputs this is
`I_ij / A_i` — asymmetric, and for N≠M it can exceed 1 (observed 4.0) or fail.
*Reach:* used by `sam3_multiplex_detector_utils.perf_mask_iom`, selected when
perflib is enabled (default, `USE_PERFLIB=1`) **and** `det_nms_use_iom=True`,
which `build_sam3_multiplex_video_predictor` sets (`model_builder.py:1172`), i.e.
the **default SAM 3.1 video path**. The non-perflib path uses the correct
`train.masks_ops.mask_iom`, so results depend on `USE_PERFLIB`.
*Likely effect:* nested smaller duplicates inside a larger, higher-scoring mask
are under-suppressed (needs confirmation, T-003).
*Caution:* published SAM 3.1 numbers were presumably produced with this
behaviour; a fix must quantify the change on SA-Co/VEval before adoption.
Tests: strict xfail `test_perflib_iom_matches_reference_iom`; baseline `perflib_pairwise_iom`.

### KI-005 · Low · algorithms · Open
**`model.box_ops.masks_to_boxes` crashes unless N ∈ {1, 4}**: final
`boxes * masks.flatten(-2).any(-1)` broadcasts (N,4)×(N,). No in-repo callers.
Test: strict xfail `test_model_masks_to_boxes_batch_of_two`.

### KI-006 · Info · algorithms · Documented
**NMS threshold comparison differs between implementations:** `perflib.nms`
suppresses IoU **>** t (torchvision convention); `train.nms_helper` frame/track
NMS suppresses IoU **≥** t. Thresholds are not interchangeable at the boundary.

### KI-007 · Low · algorithms (video association) · Open
**`associate_det_trk(..., det_scores=None)` raises `TypeError`** (indexes
`None` when building `matched_det_scores`), although the signature allows it.
Also, the second empty-input early return yields 3 values vs 4 elsewhere
(unreachable today). Test: strict xfail `test_associate_det_trk_without_scores`.

### KI-008 · Info · evaluation assets · Documented
**`assets/veval/.../toy_saco_veval_sav_test_eval_res.json` contains 34
`video_*_demo_*` metrics the current `VEvalEvaluator` does not emit** (the
demo-F1 evaluator is imported but not in its list). All 54 shared metrics match
exactly; the regression test compares only shared keys and requires ≥ 54.

### KI-009 · Medium · portability · Open
**Models cannot be constructed without CUDA.** `PositionEmbeddingSine.__init__`
precomputes on `device="cuda"` (`sam3/model/position_encoding.py:55`), even when
`device="cpu"` is requested. Blocks CPU inference, CPU model tests and the
Windows executable on machines without an NVIDIA GPU (T-004). Model smoke test
is `gpu`-marked.

### KI-010 · Low · packaging · Open
**Python version metadata is inconsistent:** `requires-python >=3.8` and
classifiers 3.8–3.12, README requires 3.12+, and `numpy<2` has no wheels for
Python 3.13. Reference interpreter here: 3.12.

### KI-011 · Medium · test coverage · Open
**GPU kernels are not exercised in CI:** Triton EDT (`model/edt.py`), Triton
NMS / connected components, `cc_torch`, `torch_generic_nms`, FA3. Their
agreement with the CPU reference paths is assumed, not tested (T-005).

### KI-012 · Low · algorithms (video association) · Open
**Resize heuristic in `associate_det_trk` compares the wrong sizes:**
`torch.numel(det_masks[-2:])` is the element count of the *last two masks*, not
the spatial size, so which side is resized depends on the number of detections.
Affects IoU only when det and track masks differ in resolution.

### KI-013 · Medium · security / data egress · Documented
**SAM 3 Agent sends images to the configured LLM endpoint** (base64 in
`sam3/agent/client_llm.py`). With a cloud endpoint, proprietary images leave the
organisation. Use only a locally hosted or enterprise-approved endpoint
(SECURITY_AND_DATA.md §4).

### KI-014 · Info · process · Open
**Local clone is shallow (50 commits)**, so the history secret scan covered only
those commits. CI runs `scan_sensitive.py --history` with full history; run it on
a full mirror before any migration (ENTERPRISE_MIGRATION.md).

### KI-015 · Low · reproducibility · Open
**Windows CUDA torch wheels are not hash-locked.** The lock files resolve
`torch==2.10.0` from PyPI (CUDA 12.8 on Linux, CPU-only on Windows); the Windows
build reinstalls the same version from the PyTorch cu128 index without hashes (T-008).

### KI-016 · Medium · deployment · Open
**M365 Copilot file hand-off is unverified.** The API plugin declares a
`multipart/form-data` image upload; whether Copilot passes a chat attachment to
an API plugin depends on tenant features and must be tested in an approved
tenant. A by-reference (SharePoint/OneDrive URL) operation is planned (T-010).

### KI-017 · Low · formatting · Open
**58 upstream files in `sam3/` and `scripts/` fail `ufmt check`** with the
pinned toolchain (black 24.2.0, usort 1.0.2, ruff-api 0.1.0 from `[dev]`), so
`.github/workflows/format.yml` will likely fail on PRs. Not reformatted here to
keep upstream files byte-identical (easier upstream merges). Decide: reformat
in one dedicated commit, or pin the formatter versions that upstream used (T-014).

---

## Resolved

### KI-R01 · Low · tests · Resolved 2026-11-25
`pyproject.toml` `testpaths = ["tests"]` pointed at a non-existent directory;
upstream tests live in `test/` and `sam3/perflib/tests/`. Now collects all three.
