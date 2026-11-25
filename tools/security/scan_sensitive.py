#!/usr/bin/env python3
"""Refuse commits that contain credentials, API keys or experimental data.

  python tools/security/scan_sensitive.py --staged     # pre-commit hook
  python tools/security/scan_sensitive.py --all        # every tracked file (CI)
  python tools/security/scan_sensitive.py --history    # every blob ever committed (slow)
  python tools/security/scan_sensitive.py path ...     # explicit files

Three rule families:
  1. Secret patterns (API keys, tokens, private keys, connection strings).
  2. Forbidden file types: microscopy / imaging / instrument raw data, model
     weights, key stores, .env files.
  3. Size: any file > MAX_BYTES unless listed in tools/security/allowlist.txt.

A single line can be exempted with the marker  ``sensitive-scan: allow``  in a
comment (requires review). This is a guard rail, not a substitute for running
a dedicated scanner (e.g. gitleaks) in CI — see docs/SECURITY_AND_DATA.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALLOWLIST = REPO / "tools" / "security" / "allowlist.txt"
MAX_BYTES = 5 * 1024 * 1024
ALLOW_MARK = "sensitive-scan: allow"

SECRET_PATTERNS = {
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "OpenAI key": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b",
    "Anthropic key": r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
    "AWS access key id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "Azure storage key": r"AccountKey=[A-Za-z0-9+/=]{40,}",
    "Azure/AAD client secret": r"(?i)client[_-]?secret['\"]?\s*[:=]\s*['\"][A-Za-z0-9~._-]{30,}['\"]",
    "Private key block": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "Generic assigned secret": (
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd)"
        r"\b['\"]?\s*[:=]\s*['\"][^'\"\s<>{}$]{12,}['\"]"
    ),
}
COMPILED = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}

FORBIDDEN_GLOBS = [
    # credentials / key stores
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_ed25519*",
    "credentials.json",
    "*service-account*.json",
    "*.publishsettings",
    ".npmrc",
    ".pypirc",
    ".netrc",
    # microscopy / imaging / instrument raw data
    "*.nd2",
    "*.czi",
    "*.lif",
    "*.lsm",
    "*.oib",
    "*.oif",
    "*.ims",
    "*.vsi",
    "*.svs",
    "*.ndpi",
    "*.mrxs",
    "*.scn",
    "*.dv",
    "*.r3d",
    "*.sld",
    "*.ome.tif",
    "*.ome.tiff",
    "*.dcm",
    "*.nii",
    "*.nii.gz",
    "*.fcs",
    "*.mrc",
    "*.zarr/*",
    # tabular / array dumps that usually hold experiment data
    "*.h5",
    "*.hdf5",
    "*.mat",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.pickle",
    "*.parquet",
    "*.feather",
    "*.sav",
    "*.xlsx",
    "*.xls",
    # model weights
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    "*.bin",
    "*.engine",
]
# Never flag the example template or this scanner's own rule table.
ALLOWED_ALWAYS = {".env.example", "tools/security/scan_sensitive.py"}


def allowlist() -> list[str]:
    if not ALLOWLIST.exists():
        return []
    return [
        l.split("#")[0].strip()
        for l in ALLOWLIST.read_text().splitlines()
        if l.split("#")[0].strip()
    ]


def git_lines(*args: str) -> list[str]:
    return [
        l
        for l in subprocess.check_output(
            ["git", *args], cwd=REPO, text=True
        ).splitlines()
        if l
    ]


def read_blob(path: str, staged: bool) -> bytes:
    if staged:
        return subprocess.run(
            ["git", "show", f":{path}"], cwd=REPO, capture_output=True
        ).stdout
    return (REPO / path).read_bytes()


def scan_content(path: str, data: bytes) -> list[str]:
    if b"\x00" in data[:8192]:
        return []  # binary: covered by type/size rules
    text = data.decode("utf-8", errors="replace")
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line or len(line) > 20000:
            continue
        for name, rx in COMPILED.items():
            if rx.search(line):
                hits.append(f"{path}:{lineno}: possible {name}")
    return hits


def check_file(path: str, data: bytes, allow: list[str]) -> list[str]:
    if path in ALLOWED_ALWAYS:
        return []
    allowed = any(fnmatch.fnmatch(path, a) for a in allow)
    problems = []
    name = Path(path).name
    if not allowed and any(
        fnmatch.fnmatch(name, g) or fnmatch.fnmatch(path, g) for g in FORBIDDEN_GLOBS
    ):
        problems.append(f"{path}: forbidden file type (data, weights or credentials)")
    if not allowed and len(data) > MAX_BYTES:
        problems.append(
            f"{path}: {len(data) / 1e6:.1f} MB exceeds {MAX_BYTES / 1e6:.0f} MB limit"
        )
    problems += scan_content(path, data)
    return problems


def scan_history() -> list[str]:
    problems, seen = [], set()
    allow = allowlist()
    for line in git_lines("rev-list", "--objects", "--all"):
        parts = line.split(" ", 1)
        if len(parts) != 2 or parts[0] in seen:
            continue
        sha, path = parts
        seen.add(sha)
        if (
            subprocess.run(
                ["git", "cat-file", "-t", sha], cwd=REPO, capture_output=True, text=True
            ).stdout.strip()
            != "blob"
        ):
            continue
        data = subprocess.run(
            ["git", "cat-file", "-p", sha], cwd=REPO, capture_output=True
        ).stdout
        problems += [f"[history {sha[:8]}] {p}" for p in check_file(path, data, allow)]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--history", action="store_true")
    a = ap.parse_args()

    if a.history:
        problems = scan_history()
    else:
        if a.staged:
            files = git_lines("diff", "--cached", "--name-only", "--diff-filter=ACMR")
        elif a.all:
            files = git_lines("ls-files")
        else:
            files = a.paths
        allow = allowlist()
        problems = []
        for f in files:
            try:
                problems += check_file(f, read_blob(f, a.staged), allow)
            except (FileNotFoundError, IsADirectoryError):
                continue

    if problems:
        print("Sensitive-content scan FAILED:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(
            "\nRemove the file/secret (rotate any real credential!), or, if it is a "
            "false positive, add the path to tools/security/allowlist.txt or mark the "
            f"line with '{ALLOW_MARK}' and get it reviewed.",
            file=sys.stderr,
        )
        return 1
    print("Sensitive-content scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
