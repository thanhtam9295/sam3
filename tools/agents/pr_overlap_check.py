#!/usr/bin/env python3
"""Detect open pull requests that touch the same files (cross-machine guard).

Local claims (claims.py) only protect agents sharing one clone. Cloud agents
(Codex cloud tasks, Copilot coding agent, remote Claude Code sessions) each get
their own clone, so CI compares the files of the current PR with every other
open PR in the repository.

  GITHUB_TOKEN=... python tools/agents/pr_overlap_check.py --repo OWNER/NAME --pr 42

Exit codes: 0 no overlap (or only overlaps on non-protected files with
--warn-only-unprotected), 1 overlap found, 2 configuration/API error.
Only reads the GitHub API; it never comments, labels or modifies anything.
Works with GitHub Enterprise via --api-url (or $GITHUB_API_URL).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHARED_OK = {  # append-only logs: overlaps are expected and resolved by merge
    "docs/DEVLOG.md",
    "docs/TASKS.md",
    "docs/KNOWN_ISSUES.md",
}


def api(url: str, token: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pr-overlap-check",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def paged(url: str, token: str):
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        batch = api(f"{url}{sep}per_page=100&page={page}", token)
        if not batch:
            return
        yield from batch
        if len(batch) < 100:
            return
        page += 1


def protected_patterns() -> list[str]:
    p = REPO / "tools" / "validation" / "protected_paths.txt"
    return [
        l.strip()
        for l in p.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    ap.add_argument(
        "--warn-only-unprotected",
        action="store_true",
        help="fail only when the overlap includes a protected algorithm path",
    )
    a = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token or not a.repo:
        print("GITHUB_TOKEN and --repo are required", file=sys.stderr)
        return 2

    base = f"{a.api_url}/repos/{a.repo}"
    try:
        mine = {
            f["filename"] for f in paged(f"{base}/pulls/{a.pr}/files", token)
        } - SHARED_OK
        others = [
            p for p in paged(f"{base}/pulls?state=open", token) if p["number"] != a.pr
        ]
        overlaps = {}
        for p in others:
            if p.get("draft") and "wip" in p["title"].lower():
                continue
            files = {
                f["filename"] for f in paged(f"{base}/pulls/{p['number']}/files", token)
            }
            common = sorted(mine & files)
            if common:
                overlaps[p["number"]] = (p["head"]["ref"], p["title"], common)
    except urllib.error.HTTPError as e:
        print(f"GitHub API error: {e.code} {e.reason}", file=sys.stderr)
        return 2

    if not overlaps:
        print(f"PR #{a.pr}: no file overlap with {len(others)} other open PR(s).")
        return 0

    pats = protected_patterns()
    fail = False
    for num, (ref, title, files) in overlaps.items():
        prot = [f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)]
        level = "error" if (prot or not a.warn_only_unprotected) else "warning"
        fail |= level == "error"
        print(
            f"::{level}::PR #{a.pr} overlaps PR #{num} ({ref}: {title}) on {len(files)} file(s)"
            + (f", {len(prot)} PROTECTED" if prot else "")
        )
        for f in files:
            print(f"    {'[protected] ' if f in prot else ''}{f}")
    print(
        "\nResolve by sequencing the PRs (merge one, rebase the other) or by splitting "
        "the work. See docs/agents/COORDINATION.md."
    )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
