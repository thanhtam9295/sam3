"""Tolerance-aware comparison of nested JSON-like outputs."""

from __future__ import annotations

import math
from typing import Any, List

# Absolute tolerance for floats. Tight on purpose: baselines are produced on
# CPU in float64/float32 and deterministic. Loosen only via a validation report.
ATOL = 1e-6
RTOL = 1e-6


def diff_outputs(expected: Any, actual: Any, path: str = "$") -> List[str]:
    diffs: List[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                diffs.append(f"{path}.{k}: missing in actual")
            elif k not in expected:
                diffs.append(f"{path}.{k}: unexpected new key")
            else:
                diffs += diff_outputs(expected[k], actual[k], f"{path}.{k}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs += diff_outputs(e, a, f"{path}[{i}]")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            diffs.append(f"{path}: {expected!r} != {actual!r}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
            diffs.append(f"{path}: {expected!r} != {actual!r}")
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs
