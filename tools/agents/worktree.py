#!/usr/bin/env python3
"""Create / list / remove isolated git worktrees for AI agents.

One agent = one task = one branch = one worktree. Branch naming:

    agent/<agent-id>/<task-id>-<short-slug>     e.g. agent/codex/T-012-cgf1-docs

  python tools/agents/worktree.py new --agent codex --task T-012 --slug cgf1-docs \
         --claim sam3/eval/cgf1_eval.py --claim tests/unit/
  python tools/agents/worktree.py list
  python tools/agents/worktree.py remove --agent codex --task T-012 --slug cgf1-docs

Worktrees are created next to the main checkout in ``../<repo>-worktrees/``,
so they never appear inside the repository tree. Nothing is pushed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = {"claude", "codex", "antigravity", "copilot", "human"}
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
TASK = re.compile(r"^T-\d{3,}$")


def git(*a: str, cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *a], cwd=cwd, text=True, capture_output=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r.stdout.strip()


def main_checkout() -> Path:
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent


def names(args) -> tuple[str, Path]:
    if args.agent not in AGENTS:
        sys.exit(f"--agent must be one of {sorted(AGENTS)}")
    if not TASK.match(args.task):
        sys.exit("--task must look like T-012 (see docs/TASKS.md)")
    if not SLUG.match(args.slug):
        sys.exit("--slug must be lowercase letters, digits and dashes")
    branch = f"agent/{args.agent}/{args.task}-{args.slug}"
    root = main_checkout()
    path = (
        root.parent / f"{root.name}-worktrees" / f"{args.agent}-{args.task}-{args.slug}"
    )
    return branch, path


def cmd_new(args) -> int:
    branch, path = names(args)
    base = args.base
    if not args.no_fetch:
        git("fetch", "origin", base.split("/", 1)[-1], check=False)
    ref = (
        base
        if git("rev-parse", "--verify", "--quiet", base, check=False)
        else base.split("/", 1)[-1]
    )
    if git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False):
        git("worktree", "add", str(path), branch)
    else:
        git("worktree", "add", "-b", branch, str(path), ref)
    print(f"worktree: {path}\nbranch:   {branch} (from {ref})")
    if args.claim:
        env = {**os.environ, "AGENT_ID": args.agent}
        r = subprocess.run(
            [
                sys.executable,
                str(HERE / "claims.py"),
                "claim",
                *args.claim,
                "--task",
                args.task,
            ],
            cwd=path,
            env=env,
        )
        if r.returncode:
            print(
                "Claim failed: the worktree exists but you hold no claims. "
                "Resolve the conflict before editing.",
                file=sys.stderr,
            )
            return r.returncode
    print(
        f"\nNext:\n  cd {path}\n  export AGENT_ID={args.agent}   # PowerShell: $env:AGENT_ID='{args.agent}'"
    )
    return 0


def cmd_list(_args) -> int:
    print(git("worktree", "list"))
    return 0


def cmd_remove(args) -> int:
    branch, path = names(args)
    env = {**os.environ, "AGENT_ID": args.agent}
    if path.exists():
        subprocess.run(
            [sys.executable, str(HERE / "claims.py"), "release", "--all"],
            cwd=path,
            env=env,
        )
        git("worktree", "remove", *(["--force"] if args.force else []), str(path))
        print(f"removed worktree {path}")
    if args.delete_branch:
        git("branch", "-d" if not args.force else "-D", branch)
        print(f"deleted local branch {branch}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("new", "remove"):
        p = sub.add_parser(name)
        p.add_argument("--agent", required=True)
        p.add_argument("--task", required=True)
        p.add_argument("--slug", required=True)
        if name == "new":
            p.add_argument("--base", default="origin/main")
            p.add_argument("--no-fetch", action="store_true")
            p.add_argument(
                "--claim", action="append", help="path to claim (repeatable)"
            )
            p.set_defaults(func=cmd_new)
        else:
            p.add_argument("--force", action="store_true")
            p.add_argument("--delete-branch", action="store_true")
            p.set_defaults(func=cmd_remove)
    sub.add_parser("list").set_defaults(func=cmd_list)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
