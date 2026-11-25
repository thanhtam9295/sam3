"""Scientific regression: current outputs must match frozen golden baselines.

A failure here means a quantitative result changed. Do NOT regenerate the
baseline to make it pass. Follow docs/validation/SCIENTIFIC_VALIDATION.md.
"""

import json
from pathlib import Path

import pytest
from tests.scientific.cases import CASES
from tests.scientific.compare import diff_outputs

pytestmark = pytest.mark.scientific
BASE_DIR = Path(__file__).parent / "baselines"


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden_baseline(name):
    path = BASE_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"No baseline for case '{name}'. Create it with "
            "`python tools/validation/regenerate_baselines.py --init --case "
            f"{name}` and commit it in a PR that adds the case."
        )
    expected = json.loads(path.read_text())["outputs"]
    diffs = diff_outputs(expected, CASES[name]())
    assert not diffs, (
        f"Scientific regression in '{name}' ({len(diffs)} differences):\n  "
        + "\n  ".join(diffs[:25])
        + "\nSee docs/validation/SCIENTIFIC_VALIDATION.md before changing anything."
    )
