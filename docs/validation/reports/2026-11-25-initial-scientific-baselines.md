# Validation report: initial capture of the scientific reference

Status: PROPOSED
Approved-by: <pending — human reviewer name; AI agents must not fill this in>
Date: 2026-11-25
Task: T-000
Author / agent: claude
Branch / commit: claude/beautiful-einstein-h50l6a

## 1. Change
Creates the scientific regression reference. **No algorithm code was changed**
(no file under `sam3/` or `scripts/` is modified). The baselines freeze the
*current* behaviour of the validated algorithms, including known defects
(KI-004 is frozen deliberately in `perflib_pairwise_iom`).

## 2. Protected files changed
All new; nothing pre-existing was modified.
- tests/scientific/cases.py — 11 deterministic, seeded synthetic cases
- tests/scientific/compare.py — tolerance-aware comparator (atol = rtol = 1e-6)
- tests/scientific/cgf1_fixture.py — hand-derivable 5-pair cgF1 problem
- tests/scientific/baselines/*.json — 11 baselines captured with `--init`

## 3. Expected numerical effect
None. This records existing outputs.

## 4. Environment
- Lock file: requirements/lock/py312-dev.txt (installed with `--require-hashes`)
- Python 3.12.3, torch 2.10.0 (CPU execution), numpy 1.26.4, Linux x86_64
- No checkpoint used

## 5. Synthetic evidence
- `pytest -m scientific`: 23 passed
- Reference agreement (independent implementations): all 11 pass —
  box IoU and GIoU = torchvision; mask IoU = pycocotools; reference IoM =
  brute force; perflib masks→boxes = torchvision; CPU mask NMS = torchvision NMS
  (4 thresholds); frame NMS = greedy reference; Hungarian = brute-force optimum;
  RLE = pycocotools (byte-identical); connected components = scipy (8-conn);
  cgF1 / IL_MCC / pmF1 = hand derivation (abs 1e-6).
- Baselines stable across two independent runs and in a fresh env from the lock.
- Toy SA-Co/VEval end-to-end: 54/54 shared reference metrics reproduced
  exactly (rel 1e-6); 34 legacy keys absent (KI-008).
- Strict xfails documenting defects: KI-004, KI-005, KI-007.

## 6. Benchmark evidence
Not applicable (no numerical change). GPU kernels not covered (KI-011, T-005).

## 7. Performance
Not applicable.

## 8. Risks and limitations
- Baselines are CPU-only; GPU backends (Triton, cc_torch, generic NMS CUDA)
  may differ and are not yet compared (T-005).
- Synthetic inputs exercise the algorithms, not model accuracy on real data;
  domain validation is a separate task (T-012).
- Golden baselines may need platform-specific tolerance review if CI runs on
  other OSes; only Linux was verified.

## 9. Reviewer decision
<Reviewer: confirm the reference is acceptable, then set "Status: APPROVED"
and your name on the Approved-by line.>
