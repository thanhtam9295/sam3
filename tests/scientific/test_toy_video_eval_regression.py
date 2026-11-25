"""End-to-end scientific regression on the public toy SA-Co/VEval sample.

Runs the full video evaluation stack (mAP, phrase AP, TETA, HOTA, cgF1) on the
GT/prediction pair shipped in assets/veval/toy_gt_and_pred/ and compares to the
reference results shipped alongside it. Only metrics the current pipeline
produces are compared; the 34 legacy *_demo_* keys are tracked as KI-008.
"""

import contextlib
import io
import json
import math

import pytest

pytestmark = [pytest.mark.scientific, pytest.mark.integration, pytest.mark.slow]

TOY = "assets/veval/toy_gt_and_pred/toy_saco_veval_sav_test"
# All 54 overlapping metrics matched exactly on 2026-11-25; keep this tight.
REL_TOL, ABS_TOL = 1e-6, 1e-9
MIN_EXPECTED_SHARED_METRICS = 54


def test_toy_veval_reproduces_reference(repo_root, tmp_path):
    from sam3.eval.saco_veval_eval import VEvalEvaluator

    ref = json.loads((repo_root / f"{TOY}_eval_res.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        got = VEvalEvaluator(
            gt_annot_file=str(repo_root / f"{TOY}_gt.json"),
            eval_res_file=str(tmp_path / "res.json"),
        ).run_eval(pred_file=str(repo_root / f"{TOY}_pred.json"))

    exp_d, got_d = ref["dataset_results"], got["dataset_results"]
    shared = sorted(set(exp_d) & set(got_d))
    assert (
        len(shared) >= MIN_EXPECTED_SHARED_METRICS
    ), "evaluator silently dropped metrics"
    bad = [
        (k, exp_d[k], got_d[k])
        for k in shared
        if not math.isclose(exp_d[k], got_d[k], rel_tol=REL_TOL, abs_tol=ABS_TOL)
    ]
    assert not bad, f"dataset metrics drifted: {bad[:10]}"
    assert len(got["video_np_results"]) == len(ref["video_np_results"])
