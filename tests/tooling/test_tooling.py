"""Tests for the multi-agent coordination and safety tooling in tools/."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def sandbox_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "b.py").write_text("y = 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    return repo


def _claims(repo, agent, *args):
    env = {**os.environ, "AGENT_ID": agent}
    return subprocess.run(
        [sys.executable, str(REPO / "tools/agents/claims.py"), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_claims_block_other_agents_and_directory_prefixes(sandbox_repo):
    assert (
        _claims(sandbox_repo, "codex", "claim", "pkg/", "--task", "T-001").returncode
        == 0
    )
    r = _claims(sandbox_repo, "claude", "claim", "pkg/b.py")
    assert r.returncode == 1 and "CONFLICT" in r.stderr
    assert _claims(sandbox_repo, "claude", "claim", "a.py").returncode == 0
    # same agent may re-claim its own files
    assert _claims(sandbox_repo, "codex", "claim", "pkg/b.py").returncode == 0


def test_claims_check_staged_blocks_commit(sandbox_repo):
    _claims(sandbox_repo, "codex", "claim", "a.py")
    (sandbox_repo / "a.py").write_text("x = 3\n")
    _git(sandbox_repo, "add", "a.py")
    assert _claims(sandbox_repo, "claude", "check", "--staged").returncode == 1
    assert _claims(sandbox_repo, "codex", "check", "--staged").returncode == 0


def test_claims_release_and_status(sandbox_repo):
    _claims(sandbox_repo, "codex", "claim", "a.py")
    st = json.loads(_claims(sandbox_repo, "x", "status", "--json").stdout)
    assert [c["path"] for c in st] == ["a.py"]
    _claims(sandbox_repo, "codex", "release", "--all")
    assert json.loads(_claims(sandbox_repo, "x", "status", "--json").stdout) == []


def test_claims_shared_across_worktrees(sandbox_repo, tmp_path):
    wt = tmp_path / "wt"
    _git(sandbox_repo, "worktree", "add", "-q", "-b", "agent/claude/T-002-x", str(wt))
    assert _claims(sandbox_repo, "codex", "claim", "a.py").returncode == 0
    # agent id derived from the worktree's branch name
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/agents/claims.py"), "claim", "a.py"],
        cwd=wt,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "AGENT_ID"},
    )
    assert r.returncode == 1 and "codex" in r.stderr


def test_claims_expired_are_ignored(sandbox_repo):
    _claims(sandbox_repo, "codex", "claim", "a.py", "--ttl-hours", "-1")
    assert _claims(sandbox_repo, "claude", "claim", "a.py").returncode == 0


def _scan(tmp_path, name, content):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    return subprocess.run(
        [sys.executable, str(REPO / "tools/security/scan_sensitive.py"), str(p)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "secret",
    [
        "token = 'hf_" + "a" * 34 + "'",
        "OPENAI_API_KEY=sk-proj-" + "b" * 40,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "-----BEGIN RSA " + "PRIVATE KEY-----",
        'password = "' + "correct-horse-battery" + '"',
    ],
)
def test_scanner_flags_secrets(tmp_path, secret):
    assert _scan(tmp_path, "cfg.py", secret + "\n").returncode == 1


@pytest.mark.parametrize(
    "name", ["exp/run1.nd2", "w/model.safetensors", ".env", "d/cells.czi"]
)
def test_scanner_flags_data_and_credentials_files(tmp_path, name):
    assert _scan(tmp_path, name, b"x").returncode == 1


def test_scanner_allows_clean_code_and_marker(tmp_path):
    assert _scan(tmp_path, "ok.py", "def f():\n    return 1\n").returncode == 0
    line = 'password = "' + "example-not-real-1" + '"  # sensitive-scan: allow\n'
    assert _scan(tmp_path, "doc.py", line).returncode == 0


def test_current_tree_is_clean():
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/security/scan_sensitive.py"), "--all"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def _guard(files):
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/validation/check_protected_changes.py"),
            "--files",
            *files,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_protected_change_without_report_fails():
    r = _guard(["sam3/eval/cgf1_eval.py"])
    assert r.returncode == 1 and "no validation report" in r.stderr


def test_unprotected_change_passes():
    assert _guard(["README.md", "docs/ARCHITECTURE.md"]).returncode == 0


def test_template_report_does_not_count():
    r = _guard(["sam3/perflib/nms.py", "docs/validation/reports/TEMPLATE.md"])
    assert r.returncode == 1


def test_skill_mirrors_are_in_sync():
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/agents/sync_skills.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
