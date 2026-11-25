#!/usr/bin/env python3
"""Regenerate scientific golden baselines (tests/scientific/baselines/).

THIS CHANGES THE SCIENTIFIC REFERENCE. It refuses to run unless you pass
--approved-by and --report pointing at a validation report that exists in
docs/validation/reports/. See docs/validation/SCIENTIFIC_VALIDATION.md.

Usage:
  python tools/validation/regenerate_baselines.py --list
  python tools/validation/regenerate_baselines.py --diff            # dry run
  python tools/validation/regenerate_baselines.py --case mask_nms \
      --approved-by "<human reviewer>" \
      --report docs/validation/reports/2026-11-25-mask-nms.md
  python tools/validation/regenerate_baselines.py --init           # only for cases with no baseline yet
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BASE_DIR = REPO / "tests" / "scientific" / "baselines"


def _env() -> dict:
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
    }


def _write(name: str, outputs: dict, meta: dict) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASE_DIR / f"{name}.json"
    path.write_text(
        json.dumps({"_meta": meta, "outputs": outputs}, indent=1, sort_keys=True) + "\n"
    )
    print(f"wrote {path.relative_to(REPO)}")


def main() -> int:
    from tests.scientific.cases import CASES, CASES_VERSION
    from tests.scientific.compare import diff_outputs

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--case", action="append", help="case name (repeatable); default all"
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--diff", action="store_true", help="show differences, write nothing"
    )
    ap.add_argument(
        "--init", action="store_true", help="create baselines that do not exist yet"
    )
    ap.add_argument("--approved-by", help="human who approved the baseline change")
    ap.add_argument(
        "--report", help="path to validation report (docs/validation/reports/...)"
    )
    args = ap.parse_args()

    names = args.case or sorted(CASES)
    unknown = set(names) - set(CASES)
    if unknown:
        ap.error(f"unknown case(s): {sorted(unknown)}")
    if args.list:
        for n in sorted(CASES):
            print(
                n,
                "(baseline exists)"
                if (BASE_DIR / f"{n}.json").exists()
                else "(NO baseline)",
            )
        return 0

    changed = 0
    results = {}
    for n in names:
        out = CASES[n]()
        results[n] = out
        p = BASE_DIR / f"{n}.json"
        if p.exists():
            d = diff_outputs(json.loads(p.read_text())["outputs"], out)
            if d:
                changed += 1
                print(f"[CHANGED] {n}:\n  " + "\n  ".join(d[:20]))
            else:
                print(f"[same]    {n}")
        else:
            print(f"[new]     {n}")

    if args.diff:
        return 0

    meta = {
        "generated": _dt.date.today().isoformat(),
        "cases_version": CASES_VERSION,
        "env": _env(),
    }
    if args.init:
        for n in names:
            if not (BASE_DIR / f"{n}.json").exists():
                _write(
                    n,
                    results[n],
                    {**meta, "approved_by": "initial-capture", "report": None},
                )
        return 0

    if not args.approved_by or not args.report:
        print(
            "\nRefusing to overwrite baselines without --approved-by and --report.",
            file=sys.stderr,
        )
        return 2
    report = (REPO / args.report).resolve()
    if not report.is_file() or "docs/validation/reports" not in report.as_posix():
        print(
            f"Report not found under docs/validation/reports/: {args.report}",
            file=sys.stderr,
        )
        return 2
    for n in names:
        _write(
            n,
            results[n],
            {**meta, "approved_by": args.approved_by, "report": args.report},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
