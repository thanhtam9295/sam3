#!/usr/bin/env python3
"""File-claim registry that stops two AI agents editing the same files at once.

Claims live in ``<git-common-dir>/agent-claims/``. Every ``git worktree`` of a
clone shares that directory, so Claude Code, Codex, Antigravity and Copilot
sessions working in separate worktrees on one machine all see the same claims.
Claims are never committed. Cross-machine / cloud agents are covered by the
PR-overlap CI check (tools/agents/pr_overlap_check.py); see
docs/agents/COORDINATION.md.

  python tools/agents/claims.py claim sam3/eval/cgf1_eval.py tests/unit/ --task T-012
  python tools/agents/claims.py status
  python tools/agents/claims.py check --staged         # used by .githooks/pre-commit
  python tools/agents/claims.py release --all
  python tools/agents/claims.py prune                  # drop expired claims

Agent identity: --agent, else $AGENT_ID, else derived from a branch named
``agent/<agent>/<task>``, else "human".
A claimed directory (trailing "/") covers every file beneath it.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DEFAULT_TTL_HOURS = 8.0


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def repo_root() -> Path:
    return Path(_git("rev-parse", "--show-toplevel"))


def registry_dir() -> Path:
    override = os.environ.get("SAM3_CLAIMS_DIR")
    if override:
        d = Path(override)
    else:
        common = Path(_git("rev-parse", "--git-common-dir"))
        if not common.is_absolute():
            common = (Path.cwd() / common).resolve()
        d = common / "agent-claims"
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_branch() -> str:
    try:
        return _git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return "unknown"


def agent_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("AGENT_ID"):
        return os.environ["AGENT_ID"]
    parts = current_branch().split("/")
    if len(parts) >= 3 and parts[0] == "agent":
        return parts[1]
    return "human"


def normalize(path: str, root: Path) -> str:
    """Repo-relative POSIX path; directories keep a trailing slash."""
    p = Path(path)
    absolute = p if p.is_absolute() else (Path.cwd() / p)
    try:
        rel = absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = p.as_posix()
    is_dir = path.endswith(("/", "\\")) or absolute.is_dir()
    rel = rel.rstrip("/")
    return rel + "/" if is_dir and rel not in ("", ".") else rel


def overlaps(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.endswith("/") and b.startswith(a):
        return True
    if b.endswith("/") and a.startswith(b):
        return True
    return False


@contextlib.contextmanager
def registry_lock(d: Path, timeout: float = 10.0) -> Iterator[None]:
    """Global mutex so overlapping directory/file claims cannot race."""
    lock = d / ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            # Break locks left behind by a crashed process (> 60 s old).
            with contextlib.suppress(FileNotFoundError):
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink()
                    continue
            if time.monotonic() > deadline:
                raise SystemExit(f"Timed out waiting for claim registry lock {lock}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _claim_file(d: Path, path: str) -> Path:
    return d / (hashlib.sha1(path.encode()).hexdigest()[:16] + ".json")


def load_claims(d: Path, include_expired: bool = False) -> List[Dict]:
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            c = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        c["_file"] = str(f)
        c["expired"] = dt.datetime.fromisoformat(c["expires"]) < now
        if include_expired or not c["expired"]:
            out.append(c)
    return out


def conflicts_for(paths: List[str], me: str, claims: List[Dict]) -> List[tuple]:
    return [
        (p, c)
        for p in paths
        for c in claims
        if c["agent"] != me and overlaps(p, c["path"])
    ]


# ----------------------------------------------------------------- commands


def cmd_claim(args) -> int:
    root, d, me = repo_root(), registry_dir(), agent_id(args.agent)
    paths = [normalize(p, root) for p in args.paths]
    now = dt.datetime.now(dt.timezone.utc)
    with registry_lock(d):
        claims = load_claims(d)
        bad = conflicts_for(paths, me, claims)
        if bad:
            for p, c in bad:
                print(
                    f"CONFLICT {p} is claimed by {c['agent']} (branch {c['branch']}, "
                    f"task {c.get('task') or '-'}, path {c['path']}, until {c['expires']})",
                    file=sys.stderr,
                )
            print(
                "Nothing claimed. Coordinate, pick other files, or wait.",
                file=sys.stderr,
            )
            return 1
        for p in paths:
            rec = {
                "path": p,
                "agent": me,
                "branch": current_branch(),
                "worktree": str(root),
                "task": args.task,
                "note": args.note,
                "created": now.isoformat(timespec="seconds"),
                "expires": (now + dt.timedelta(hours=args.ttl_hours)).isoformat(
                    timespec="seconds"
                ),
            }
            _claim_file(d, p).write_text(json.dumps(rec, indent=1))
            print(f"claimed {p} for {me} until {rec['expires']}")
    return 0


def cmd_release(args) -> int:
    root, d, me = repo_root(), registry_dir(), agent_id(args.agent)
    targets = None if args.all else {normalize(p, root) for p in args.paths}
    n = 0
    with registry_lock(d):
        for c in load_claims(d, include_expired=True):
            if c["agent"] == me and (targets is None or c["path"] in targets):
                Path(c["_file"]).unlink(missing_ok=True)
                print(f"released {c['path']}")
                n += 1
    if n == 0:
        print("nothing to release")
    return 0


def cmd_status(args) -> int:
    claims = load_claims(registry_dir(), include_expired=args.all)
    if args.json:
        print(
            json.dumps(
                [{k: v for k, v in c.items() if k != "_file"} for c in claims], indent=1
            )
        )
        return 0
    if not claims:
        print("no active claims")
    for c in claims:
        flag = " (EXPIRED)" if c["expired"] else ""
        print(
            f"{c['agent']:<12} {c['path']:<50} {c['branch']:<40} {c.get('task') or '-'} "
            f"until {c['expires']}{flag}"
        )
    return 0


def _staged_files() -> List[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMRD")
    return [l for l in out.splitlines() if l]


def cmd_check(args) -> int:
    root, me = repo_root(), agent_id(args.agent)
    files = _staged_files() if args.staged else [normalize(p, root) for p in args.paths]
    if not files:
        return 0
    claims = load_claims(registry_dir())
    bad = conflicts_for(files, me, claims)
    for p, c in bad:
        print(
            f"BLOCKED {p}: claimed by {c['agent']} on {c['branch']} "
            f"(task {c.get('task') or '-'}) until {c['expires']}",
            file=sys.stderr,
        )
    if bad:
        print(
            "Another agent holds these files. Unstage them or coordinate first "
            "(docs/agents/COORDINATION.md). Override only with human approval: "
            "SAM3_CLAIMS_OVERRIDE=1.",
            file=sys.stderr,
        )
        return 0 if os.environ.get("SAM3_CLAIMS_OVERRIDE") == "1" else 1
    if args.require_own:
        mine = [c for c in claims if c["agent"] == me]
        unclaimed = [f for f in files if not any(overlaps(f, c["path"]) for c in mine)]
        for f in unclaimed:
            print(
                f"UNCLAIMED {f}: claim it first with tools/agents/claims.py claim {f}",
                file=sys.stderr,
            )
        if unclaimed:
            return 1
    return 0


def cmd_prune(args) -> int:
    d = registry_dir()
    with registry_lock(d):
        for c in load_claims(d, include_expired=True):
            if c["expired"]:
                Path(c["_file"]).unlink(missing_ok=True)
                print(f"pruned {c['path']} ({c['agent']})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--agent", help="agent id (default: $AGENT_ID or agent/<id>/... branch)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("claim", help="claim files/directories for exclusive editing")
    p.add_argument("paths", nargs="+")
    p.add_argument("--task", help="task id from docs/TASKS.md, e.g. T-012")
    p.add_argument("--note")
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("release", help="release your claims")
    p.add_argument("paths", nargs="*")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("status", help="list claims")
    p.add_argument("--all", action="store_true", help="include expired")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("check", help="fail if files are claimed by another agent")
    p.add_argument("paths", nargs="*")
    p.add_argument(
        "--staged", action="store_true", help="check files staged for commit"
    )
    p.add_argument(
        "--require-own",
        action="store_true",
        help="also fail if a file is not claimed by you",
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("prune", help="delete expired claims")
    p.set_defaults(func=cmd_prune)

    args = ap.parse_args(argv)
    if args.cmd == "release" and not args.all and not args.paths:
        ap.error("release needs paths or --all")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
