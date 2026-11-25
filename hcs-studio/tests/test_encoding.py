"""Text files are read and written as UTF-8 everywhere: without ``encoding=``, Python on Windows
uses the ANSI code page (cp1252), so µ, ′, ×, – and ⚠ in reports, figures and JSON come out
garbled or fail to encode (found by the first Windows test run)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT.parent / "lab_events" / "src"]


def _binary_mode(call: ast.Call, position: int) -> bool:
    mode = call.args[position] if len(call.args) > position else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def unspecified_encodings(path: Path) -> list[str]:
    """``Path.read_text``/``write_text`` and text-mode ``open()`` calls without an encoding.
    (``<object>.open(...)`` is not checked: Qt dialogs and many libraries have ``open``.)"""
    found = []
    label = path.relative_to(ROOT.parent) if path.is_relative_to(ROOT.parent) else path.name
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            if len(node.args) <= (1 if func.attr == "write_text" else 0):
                found.append(f"{label}:{node.lineno} {func.attr}")
        elif (
            isinstance(func, ast.Name)
            and func.id == "open"
            and len(node.args) <= 3
            and not _binary_mode(node, 1)
        ):
            found.append(f"{label}:{node.lineno} open")
    return found


def test_text_io_names_its_encoding() -> None:
    found = [hit for folder in FOLDERS for p in sorted(folder.rglob("*.py")) for hit in
             unspecified_encodings(p)]  # fmt: skip
    assert not found, "add encoding='utf-8':\n" + "\n".join(found)


def test_the_check_finds_a_missing_encoding(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from pathlib import Path\nPath('a').read_text()\nopen('b', 'rb')\nopen('c')\n",
        encoding="utf-8",
    )
    assert [hit.rsplit(" ", 1)[1] for hit in unspecified_encodings(sample)] == ["read_text", "open"]
