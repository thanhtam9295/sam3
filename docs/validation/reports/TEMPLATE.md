# Validation report: <short title>

Status: PROPOSED
Approved-by: <pending — human reviewer name; AI agents must not fill this in>
Date: YYYY-MM-DD
Task: T-###
Author / agent: <name or agent id>
Branch / commit: <branch> @ <sha>

## 1. Change
<What changed and why. Link KI-### if fixing a known issue.>

## 2. Protected files changed
<List every path, exactly as in the diff, e.g. sam3/perflib/iou.py>

## 3. Expected numerical effect
<Which outputs/metrics should change, in which direction, roughly how much.>

## 4. Environment
- Lock file: requirements/lock/py312-dev.txt @ <sha>
- Python / torch / CUDA / GPU:
- Checkpoint (name + SHA-256), if used:

## 5. Synthetic evidence
- `pytest -m scientific`: <result>
- Reference agreement: <pass / justified departures>
- `regenerate_baselines.py --diff`: <paste [CHANGED] lines, or "all same">
- Toy VEval: <result>
- Strict xfails flipped: <none / which, and why>

## 6. Benchmark evidence (required if any number changes)
| Dataset / split | Metric | Before | After | Δ |
|---|---|---|---|---|

Command(s):

## 7. Performance (if relevant)

## 8. Risks and limitations

## 9. Reviewer decision
<Reviewer notes, accepted tolerance, conditions. Then set Status: APPROVED.>
