#!/usr/bin/env python3
"""Block changes to quantitative algorithms that lack a validation report.

  python tools/validation/check_protected_changes.py --base origin/main
  python tools/validation/check_protected_changes.py --staged
  python tools/validation/check_protected_changes.py --files a.py b.py

Passes when no protected path changed, or when the same change set adds or
edits a report in docs/validation/reports/ that
  * contains the line  "Status: APPROVED",
  * contains a non-empty "Approved-by:" line (a human, never an AI agent), and
  * mentions every changed protected path.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTECTED = REPO / "tools" / "validation" / "protected_paths.txt"
REPORTS = "docs/validation/reports/"
AI_NAMES = re.compile(r"\b(claude|codex|copilot|antigravity|gemini|gpt|bot)\b", re.I)


def patterns() -> list[str]:
    lines = PROTECTED.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def is_protected(path: str, pats: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in pats)


def changed(base: str | None, staged: bool) -> list[str]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only"]
    else:
        cmd = ["git", "diff", "--name-only", f"{base}...HEAD"]
    return [
        l for l in subprocess.check_output(cmd, cwd=REPO, text=True).splitlines() if l
    ]


def evaluate(files: list[str]) -> tuple[bool, list[str]]:
    pats = patterns()
    prot = sorted(f for f in files if is_protected(f, pats))
    if not prot:
        return True, ["No protected quantitative-analysis paths changed."]
    reports = [
        f
        for f in files
        if f.startswith(REPORTS)
        and f.endswith(".md")
        and not f.endswith("TEMPLATE.md")
        and (REPO / f).is_file()
    ]
    msgs = ["Protected paths changed:"] + [f"  - {p}" for p in prot]
    if not reports:
        msgs.append(
            f"FAIL: no validation report added/edited under {REPORTS}. "
            "Copy docs/validation/reports/TEMPLATE.md and follow "
            "docs/validation/SCIENTIFIC_VALIDATION.md."
        )
        return False, msgs
    text = "\n".join((REPO / r).read_text(encoding="utf-8") for r in reports)
    ok = True
    if not re.search(r"^Status:\s*APPROVED\s*$", text, re.M):
        msgs.append("FAIL: report is not 'Status: APPROVED' (needs human sign-off).")
        ok = False
    m = re.search(r"^Approved-by:\s*(.+)$", text, re.M)
    if not m or not m.group(1).strip() or m.group(1).strip().startswith("<"):
        msgs.append("FAIL: report has no 'Approved-by:' human reviewer.")
        ok = False
    elif AI_NAMES.search(m.group(1)):
        msgs.append("FAIL: 'Approved-by:' must be a human, not an AI agent.")
        ok = False
    missing = [p for p in prot if p not in text and not p.endswith(".json")]
    if missing:
        msgs.append("FAIL: report does not mention: " + ", ".join(missing))
        ok = False
    if ok:
        msgs.append(f"OK: covered by {', '.join(reports)}")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base")
    g.add_argument("--staged", action="store_true")
    g.add_argument("--files", nargs="+")
    ap.add_argument("--warn-only", action="store_true")
    a = ap.parse_args()
    files = a.files or changed(a.base, a.staged)
    ok, msgs = evaluate(files)
    print("\n".join(msgs), file=sys.stdout if ok else sys.stderr)
    return 0 if ok or a.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
