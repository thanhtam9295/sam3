#!/usr/bin/env python3
"""Mirror the canonical Agent Skills into each platform's discovery path.

Canonical source (edit here only):   .agents/skills/<skill-name>/SKILL.md

Mirrors (generated, byte-identical copies; do not edit):
  .claude/skills/    Claude Code project skills
  .github/skills/    GitHub Copilot (coding agent, CLI, VS Code agent mode)
  .agent/skills/     Google Antigravity workspace skills

OpenAI Codex reads .agents/skills/ directly, so it needs no mirror.

Copies (not symlinks) are used because symlinks are unreliable on Windows
checkouts and in some cloud agent sandboxes.

  python tools/agents/sync_skills.py            # write mirrors
  python tools/agents/sync_skills.py --check    # CI: fail if mirrors drift

Skills present only in a mirror are reported, never deleted.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / ".agents" / "skills"
MIRRORS = [
    REPO / ".claude" / "skills",
    REPO / ".github" / "skills",
    REPO / ".agent" / "skills",
]
NAME_RX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def skills() -> list[Path]:
    return sorted(
        p for p in CANONICAL.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()
    )


def validate(skill: Path) -> list[str]:
    """Checks from the Agent Skills spec that every target platform relies on."""
    errs = []
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return [f"{skill.name}: SKILL.md must start with YAML front matter"]
    front = dict(
        (k.strip(), v.strip())
        for k, v in (
            l.split(":", 1)
            for l in m.group(1).splitlines()
            if ":" in l and not l.startswith(" ")
        )
    )
    name, desc = front.get("name", ""), front.get("description", "")
    if name != skill.name:
        errs.append(
            f"{skill.name}: front-matter name '{name}' must equal directory name"
        )
    if not NAME_RX.match(name) or len(name) > 64:
        errs.append(f"{skill.name}: name must be lowercase-hyphenated, <= 64 chars")
    if not desc or len(desc) > 1024:
        errs.append(f"{skill.name}: description required, <= 1024 chars")
    if len(text.splitlines()) > 500:
        errs.append(
            f"{skill.name}: keep SKILL.md under 500 lines; move detail to references/"
        )
    return errs


def tree_equal(a: Path, b: Path) -> bool:
    if not b.is_dir():
        return False
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(tree_equal(a / d, b / d) for d in cmp.common_dirs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    found = skills()
    errs = [e for s in found for e in validate(s)]
    drift = []
    for mirror in MIRRORS:
        for s in found:
            dst = mirror / s.name
            if tree_equal(s, dst):
                continue
            if a.check:
                drift.append(dst.relative_to(REPO).as_posix())
            else:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(s, dst)
                print(f"synced {dst.relative_to(REPO).as_posix()}")
        if mirror.is_dir():
            extra = {p.name for p in mirror.iterdir() if p.is_dir()} - {
                s.name for s in found
            }
            for e in sorted(extra):
                print(
                    f"note: {mirror.relative_to(REPO).as_posix()}/{e} is not in .agents/skills "
                    "(platform-specific skill; left untouched)"
                )

    for e in errs:
        print(f"INVALID {e}", file=sys.stderr)
    for d in drift:
        print(
            f"OUT OF SYNC {d}  -> run: python tools/agents/sync_skills.py",
            file=sys.stderr,
        )
    if not errs and not drift:
        print(
            f"{len(found)} skills valid and mirrored to {len(MIRRORS)} platform paths."
        )
    return 1 if errs or drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
