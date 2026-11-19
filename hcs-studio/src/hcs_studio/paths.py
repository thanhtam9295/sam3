"""Paths relative to named data roots, so the same registry works on every computer.

A root such as ``SCREENING`` is ``\\\\labserver\\screening`` on Windows and
``/mnt/screening`` on Linux. Paths are stored as ``RootedPath(root, rel)`` with ``rel`` in
POSIX form; :func:`resolve` turns one back into a path on this computer. Paths under no
root are kept as root ``ABSOLUTE`` (with the absolute path as ``rel``): they still work on
this computer, and :func:`to_rooted` reports them so the user can add a root.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from hcs_studio.config import DataRootConfig, load_settings

ABSOLUTE = "ABSOLUTE"
_ROOT_NAME = re.compile(r"[A-Z0-9][A-Z0-9_-]+")


def platform_key(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform.startswith("win"):
        return "windows"
    return "macos" if platform == "darwin" else "linux"


@dataclass(frozen=True)
class RootedPath:
    root: str
    rel: str

    def __str__(self) -> str:
        return f"{self.root}:{self.rel}"

    @classmethod
    def parse(cls, text: str) -> RootedPath:
        root, sep, rel = text.partition(":")
        if not sep or not _ROOT_NAME.fullmatch(root):  # also rejects drive letters (C:)
            raise ValueError(f"Not a rooted path: {text!r} (expected ROOT:relative/path)")
        return cls(root, rel)

    def to_dict(self) -> dict[str, str]:
        return {"root": self.root, "rel": self.rel}

    @property
    def is_absolute_fallback(self) -> bool:
        return self.root == ABSOLUTE


def _location(root: DataRootConfig, platform: str) -> str:
    return str(getattr(root, platform_key(platform)))


def _parts(path: str, platform: str) -> tuple[str, ...]:
    if platform_key(platform) == "windows":
        pure = PureWindowsPath(path)
        return tuple(p.lower() for p in pure.parts)
    return PurePosixPath(path).parts


def to_rooted(
    path: str | Path,
    roots: Sequence[DataRootConfig] | None = None,
    platform: str | None = None,
) -> RootedPath:
    """The rooted form of an absolute path (longest matching root wins)."""
    platform = platform or sys.platform
    roots = load_settings().data_roots if roots is None else roots
    text = str(path)
    if platform_key(platform) != "windows":
        text = str(Path(text).absolute()) if not text.startswith("/") else text
    parts = _parts(text, platform)
    best: tuple[int, DataRootConfig] | None = None
    for root in roots:
        location = _location(root, platform)
        if not location:
            continue
        prefix = _parts(location, platform)
        if parts[: len(prefix)] == prefix and (best is None or len(prefix) > best[0]):
            best = (len(prefix), root)
    if best is None:
        posix = text.replace("\\", "/") if platform_key(platform) == "windows" else text
        return RootedPath(ABSOLUTE, posix)
    original = PureWindowsPath(text).parts if platform_key(platform) == "windows" else parts
    rel = "/".join(original[best[0] :])
    return RootedPath(best[1].name, rel)


def resolve(
    rooted: RootedPath | str,
    roots: Sequence[DataRootConfig] | None = None,
    platform: str | None = None,
) -> Path:
    """The path of ``rooted`` on this computer (raises when the root has no location here)."""
    if isinstance(rooted, str):
        rooted = RootedPath.parse(rooted)
    if rooted.is_absolute_fallback:
        return Path(rooted.rel)
    platform = platform or sys.platform
    roots = load_settings().data_roots if roots is None else roots
    root = next((r for r in roots if r.name == rooted.root), None)
    if root is None:
        raise KeyError(
            f"Data root {rooted.root!r} is not defined on this computer (Settings → Data roots)"
        )
    location = _location(root, platform)
    if not location:
        raise KeyError(f"Data root {rooted.root!r} has no {platform_key(platform)} location")
    if platform_key(platform) == "windows":
        return Path(str(PureWindowsPath(location).joinpath(*rooted.rel.split("/"))))
    return Path(location).joinpath(*[p for p in rooted.rel.split("/") if p])
