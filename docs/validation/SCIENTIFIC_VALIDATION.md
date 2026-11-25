# Scientific validation policy

Quantitative algorithms in this repository (geometry, IoU/IoM, NMS, matching,
association, encodings, metrics, post-processing) are treated as **validated
scientific instruments**. Their outputs may underpin published or regulated
results, so they change only with evidence and explicit human approval.

## 1. Scope

The authoritative list is [`tools/validation/protected_paths.txt`](../../tools/validation/protected_paths.txt);
the algorithms are specified in [ALGORITHMS.md](../ALGORITHMS.md). It includes
the scientific reference itself (`tests/scientific/baselines/`, case
definitions, comparison tolerances, `assets/veval/`).

Changes that are **not** algorithm changes (still need tests to pass):
docs, comments, type annotations, logging, pure refactors proven
output-identical by the full scientific tier. A refactor of a protected file
still needs a (short) report stating "no numerical change" with evidence.

## 2. Evidence layers

| Layer | Proves | Where |
|---|---|---|
| Reference agreement | outputs are **correct** vs independent implementations (torchvision, pycocotools, scipy, brute force, hand derivation) | `tests/scientific/test_reference_agreement.py` |
| Golden baselines | outputs **did not change** on seeded synthetic cases (tolerance 1e-6) | `tests/scientific/test_golden_baselines.py`, `baselines/*.json` |
| End-to-end toy evaluation | the full video evaluation stack reproduces the shipped reference metrics | `tests/scientific/test_toy_video_eval_regression.py` |
| Known-defect pins | documented defects are still present (strict xfail) | `tests/unit/*` (KI-004/005/007) |
| Benchmark evaluation (manual, GPU) | effect on real benchmarks (SA-Co Gold/Silver/VEval, domain data) | report, §4 step 5 |

## 3. Enforcement

* `.githooks/pre-commit` warns locally; the `guards` CI job **fails** a PR that
  changes protected paths without a report under `docs/validation/reports/` that
  says `Status: APPROVED`, names a human in `Approved-by:` and mentions every
  changed protected file (`tools/validation/check_protected_changes.py`).
* `regenerate_baselines.py` refuses to overwrite baselines without
  `--approved-by` and an existing report.
* `.github/CODEOWNERS` + branch protection require the scientific owner's review.
* Agents are instructed (AGENTS.md §2) never to approve reports, regenerate
  baselines on their own initiative, or loosen tolerances.

## 4. Procedure

1. **Propose** (before editing): the problem, intended change, affected
   outputs, expected magnitude; get the human's go-ahead. Record a `T-###` task.
2. **Before state:** scientific tier green; `regenerate_baselines.py --diff` all `[same]`.
3. **Change** in a dedicated branch; no unrelated edits.
4. **Synthetic evidence:** reference-agreement tests pass; capture every
   `[CHANGED]` baseline diff; toy evaluation result; flipped xfails explained.
5. **Benchmark evidence** (required if any number changes): run the relevant
   evaluation on the same GPU/software before and after, e.g.
   `python -m sam3.eval.saco_veval_eval one ...` or the `scripts/eval/*` flows;
   report every headline metric with the delta. Use only data you are allowed to
   use; do not commit it.
6. **Performance** (if motivated by speed): before/after timings, same hardware.
7. **Report:** copy [`reports/TEMPLATE.md`](reports/TEMPLATE.md) → `reports/YYYY-MM-DD-<slug>.md`, `Status: PROPOSED`.
8. **Human review** → reviewer sets `Status: APPROVED` and `Approved-by:`; only
   then regenerate affected baselines with `--approved-by/--report` in the same PR.
9. **Log:** DEVLOG entry; KNOWN_ISSUES status; TASKS status.

## 5. Acceptance criteria

* Reference agreement must not regress (a deliberate departure from a reference
  must be justified scientifically in the report).
* Every changed baseline is explained; unexplained diffs block approval.
* Benchmark changes are within the tolerance the reviewer states in the report,
  or are the intended improvement.
* Tolerances (`tests/scientific/compare.py`: atol = rtol = 1e-6; toy eval rel 1e-6)
  change only through this procedure.

## 6. Reproducibility requirements for any reported number

Commit SHA, lock file used, Python/torch/CUDA versions, GPU model, checkpoint
name + SHA-256, prompts, thresholds (ALGORITHMS.md §10), dataset version/split,
and the exact command. `packaging/windows/sam3_cli.py` writes this provenance
into `results.json` automatically.
