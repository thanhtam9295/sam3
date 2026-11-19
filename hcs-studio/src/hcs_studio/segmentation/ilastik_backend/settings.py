"""Where ilastik and Fiji are, and how ilastik may run (threads, RAM, parallel processes).

Settings live in ``<settings>/ilastik.json``. Detection looks in the usual install folders
and in ``HCS_ILASTIK`` / ``HCS_FIJI``; it reads ilastik's version from ``--version`` (or the
install folder name) and blocks the versions the ilastik4ij plugin is incompatible with.

The ilastik "executable" may be the program itself (``ilastik.exe``, ``run_ilastik.sh``,
``run_ilastik.bat``) or the Python of an ilastik conda environment (then ``-m ilastik``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from hcs_studio.io.sidecar import config_dir

#: Incompatible with the ilastik4ij plugin (its README); HCS Studio blocks them everywhere.
INCOMPATIBLE = ("1.3.3b2", "1.3.3rc2", "1.3.3", "1.3.3post1", "1.3.3post2", "1.4.0b27")
TESTED = "1.4.2"
MIN_RAM_MB = 500  # ilastik refuses LAZYFLOW_TOTAL_RAM_MB below this
VERSION_TIMEOUT_S = 180  # ilastik imports a lot before printing its version

Runner = Literal["headless", "fiji"]
Level = Literal["ok", "warning", "error"]

_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.?(a|b|rc|post|dev)(\d*))?", re.IGNORECASE)
_STAGES = {"dev": 0, "a": 1, "b": 2, "rc": 3, "": 4, "post": 5}


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Sort key for ilastik versions (1.3.3 < 1.3.3post3 < 1.4.0b27 < 1.4.0rc1 < 1.4.0)."""
    match = _VERSION.search(version)
    if not match:
        raise ValueError(f"Not an ilastik version: {version!r}")
    major, minor, patch, stage, number = match.groups()
    return (
        int(major),
        int(minor),
        int(patch or 0),
        _STAGES[(stage or "").lower()],
        int(number or 0),
    )


def find_version(text: str) -> str | None:
    """The first version number in ``text`` (e.g. ``ilastik-1.4.2-Linux`` → ``1.4.2``)."""
    match = _VERSION.search(text)
    return match.group(0) if match else None


@dataclass(frozen=True)
class VersionCheck:
    level: Level
    message: str

    @property
    def ok(self) -> bool:
        return self.level != "error"


def check_version(version: str | None) -> VersionCheck:
    if not version:
        return VersionCheck("warning", "ilastik version unknown; tested with " + TESTED)
    try:
        key = version_key(version)
    except ValueError:
        return VersionCheck("warning", f"Unrecognised ilastik version {version!r}")
    if any(key == version_key(v) for v in INCOMPATIBLE):
        return VersionCheck(
            "error",
            f"ilastik {version} is not compatible (also not with Fiji's ilastik4ij plugin). "
            f"Install the latest stable release ({TESTED} or newer), or 1.3.3post3 if you "
            "must stay on 1.3.3",
        )
    if key < version_key("1.3.3post3"):
        return VersionCheck("error", f"ilastik {version} is too old; install {TESTED} or newer")
    if key[3] < _STAGES[""]:
        return VersionCheck(
            "warning", f"ilastik {version} is a pre-release; {TESTED} is the tested release"
        )
    if key < version_key("1.4.0"):
        return VersionCheck("warning", f"ilastik {version} is old; tested with {TESTED}")
    return VersionCheck("ok", f"ilastik {version}")


# ------------------------------------------------------------------- settings


@dataclass(frozen=True)
class IlastikSettings:
    """ilastik settings. ``threads`` -1 = ilastik decides; ``ram_mb`` 0 = no limit."""

    ilastik: str = ""
    fiji: str = ""
    threads: int = -1
    ram_mb: int = 0
    max_processes: int = 1
    runner: Runner = "headless"

    @classmethod
    def path(cls) -> Path:
        return config_dir() / "ilastik.json"

    @classmethod
    def load(cls) -> IlastikSettings:
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: data[k] for k in asdict(cls()) if k in data}
        try:
            settings = replace(cls(), **known)
            settings.validate()
        except (TypeError, ValueError):
            return cls()
        return settings

    def save(self) -> Path:
        self.validate()
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def validate(self) -> None:
        if not isinstance(self.threads, int) or self.threads < -1:
            raise ValueError("Threads must be -1 (no limit) or a positive number")
        if self.threads == 0:
            raise ValueError("Threads must be -1 (no limit) or at least 1")
        if not isinstance(self.ram_mb, int) or (self.ram_mb != 0 and self.ram_mb < MIN_RAM_MB):
            raise ValueError(f"RAM limit must be 0 (no limit) or at least {MIN_RAM_MB} MB")
        if not isinstance(self.max_processes, int) or self.max_processes < 1:
            raise ValueError("At least one ilastik process must be allowed")
        if self.runner not in ("headless", "fiji"):
            raise ValueError(f"Unknown runner {self.runner!r} (headless or fiji)")

    def environment(self) -> dict[str, str]:
        """Environment variables for a headless ilastik process (empty = ilastik's default)."""
        self.validate()
        env: dict[str, str] = {}
        if self.threads != -1:
            env["LAZYFLOW_THREADS"] = str(self.threads)
        if self.ram_mb:
            env["LAZYFLOW_TOTAL_RAM_MB"] = str(self.ram_mb)
        return env


# ------------------------------------------------------------------ detection


def ilastik_command(executable: str | Path) -> list[str]:
    """The command that starts ilastik (a Python interpreter gets ``-m ilastik``)."""
    path = Path(executable)
    if path.name.lower().startswith("python"):
        return [str(path), "-m", "ilastik"]
    return [str(path)]


def _platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    return "macos" if sys.platform == "darwin" else "linux"


def ilastik_candidates(platform: str | None = None, home: Path | None = None) -> list[Path]:
    """Executables in the usual install locations, newest version first."""
    platform = platform or _platform()
    home = home or Path.home()
    patterns: list[tuple[Path, str]] = []
    if platform == "windows":
        folders = {
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMW6432", r"C:\Program Files"),
            os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")),
            str(home),
        }
        patterns += [
            (Path(folder), f"ilastik-*/{name}")
            for folder in folders
            for name in ("ilastik.exe", "run_ilastik.bat", "run-ilastik.bat")
        ]
    elif platform == "macos":
        for base in (Path("/Applications"), home / "Applications", home):
            patterns.append((base, "ilastik-*.app/Contents/ilastik-release/run_ilastik.sh"))
    else:
        for base in (home, Path("/opt"), Path("/usr/local")):
            patterns.append((base, "ilastik-*/run_ilastik.sh"))
    found: list[Path] = []
    for base, pattern in patterns:
        try:
            found += [p for p in base.glob(pattern) if p.is_file()]
        except OSError:
            continue
    unique = sorted(set(found), key=lambda p: _sort_key(str(p)), reverse=True)
    return unique


def _sort_key(text: str) -> tuple[int, int, int, int, int]:
    version = find_version(Path(text).parent.name) or find_version(text)
    try:
        return version_key(version) if version else (0, 0, 0, 0, 0)
    except ValueError:
        return (0, 0, 0, 0, 0)


def ilastik_version(executable: str | Path, timeout: float = VERSION_TIMEOUT_S) -> str | None:
    """ilastik's version from ``--version``; falls back to the install folder's name."""
    try:
        result = subprocess.run(
            [*ilastik_command(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        for line in reversed(lines):
            if re.fullmatch(r"\d+\.\d+[\w.]*", line):
                return line
    except (OSError, subprocess.TimeoutExpired):
        pass
    for part in reversed(Path(executable).parts):
        if part.lower().startswith("ilastik-"):
            return find_version(part)
    return None


# Fiji ------------------------------------------------------------------------

_LAUNCHERS = {
    "windows": ("ImageJ-win64.exe", "fiji-windows-x64.exe", "fiji-windows-arm64.exe"),
    "linux": ("ImageJ-linux64", "fiji-linux-x64", "fiji-linux-arm64"),
    "macos": (
        "Contents/MacOS/ImageJ-macosx",
        "Contents/MacOS/fiji-macos-universal",
        "Contents/MacOS/fiji-macos-arm64",
        "Contents/MacOS/fiji-macos-x64",
    ),
}


def fiji_root(path: str | Path) -> Path:
    """The Fiji folder (``Fiji.app`` / ``Fiji``) for a folder or a launcher inside it."""
    path = Path(path)
    if path.is_file():
        path = path.parent
        if path.name == "MacOS" and path.parent.name == "Contents":
            path = path.parent.parent
    return path


def fiji_launcher(root: Path, platform: str | None = None) -> Path | None:
    platform = platform or _platform()
    for name in _LAUNCHERS[platform]:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def fiji_candidates(platform: str | None = None, home: Path | None = None) -> list[Path]:
    platform = platform or _platform()
    home = home or Path.home()
    roots = [home, home / "Desktop", home / "Applications"]
    if platform == "windows":
        roots += [Path(r"C:\\"), Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))]
        roots.append(Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))))
    elif platform == "macos":
        roots.insert(0, Path("/Applications"))
    else:
        roots += [Path("/opt"), Path("/usr/local")]
    found = []
    for root in roots:
        for name in ("Fiji.app", "Fiji"):
            candidate = root / name
            if candidate.is_dir() and fiji_launcher(candidate, platform):
                found.append(candidate)
    return found


def ilastik4ij_jar(root: Path) -> Path | None:
    """The ilastik4ij plugin of a Fiji folder (from the "ilastik" update site)."""
    for folder in ("jars", "plugins"):
        jars = sorted((root / folder).glob("ilastik4ij*.jar"))
        if jars:
            return jars[-1]
    return None


# Report ----------------------------------------------------------------------


@dataclass
class Detection:
    """What is installed, for the settings dialog, ``hcs-studio ilastik --detect`` and the
    manifest."""

    ilastik: str = ""
    ilastik_version: str | None = None
    version_check: VersionCheck = field(default_factory=lambda: check_version(None))
    ilastik_source: str = ""  # settings | HCS_ILASTIK | detected | none
    fiji: str = ""
    fiji_launcher: str = ""
    ilastik4ij: str = ""
    ilastik4ij_version: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ilastik_ok(self) -> bool:
        return bool(self.ilastik) and self.version_check.ok

    @property
    def fiji_ok(self) -> bool:
        return bool(self.fiji_launcher) and bool(self.ilastik4ij)

    def runners(self) -> list[Runner]:
        out: list[Runner] = []
        if self.ilastik_ok:
            out.append("headless")
            if self.fiji_ok:
                out.append("fiji")
        return out

    def text(self) -> str:
        lines = []
        if self.ilastik:
            lines.append(f"ilastik: {self.ilastik} ({self.ilastik_source})")
            level = self.version_check.level.upper()
            lines.append(f"  {level}: {self.version_check.message}")
        else:
            lines.append(
                "ilastik: not found (set its path in Tools → ilastik settings, or HCS_ILASTIK)"
            )
        if self.fiji:
            lines.append(f"Fiji: {self.fiji}")
            lines.append(f"  launcher: {self.fiji_launcher or 'not found'}")
            if self.ilastik4ij:
                lines.append(f"  ilastik4ij: {Path(self.ilastik4ij).name}")
            else:
                lines.append(
                    "  ilastik4ij: not installed (Help → Update → Manage update sites → ilastik)"
                )
        else:
            lines.append("Fiji: not found (optional: only for the Fiji runner)")
        lines.append("Runners available: " + (", ".join(self.runners()) or "none"))
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def detect(
    settings: IlastikSettings | None = None,
    *,
    query_version: bool = True,
    platform: str | None = None,
    home: Path | None = None,
) -> Detection:
    """Find ilastik and Fiji: the settings first, then ``HCS_ILASTIK`` / ``HCS_FIJI``, then
    the usual install folders."""
    settings = settings or IlastikSettings.load()
    out = Detection()
    candidates: list[tuple[str, str]] = []
    if settings.ilastik:
        candidates.append((settings.ilastik, "settings"))
    if os.environ.get("HCS_ILASTIK"):
        candidates.append((os.environ["HCS_ILASTIK"], "HCS_ILASTIK"))
    candidates += [(str(p), "detected") for p in ilastik_candidates(platform, home)]
    for path, source in candidates:
        if Path(path).is_file():
            out.ilastik, out.ilastik_source = path, source
            break
        if source != "detected":
            out.notes.append(f"{source}: {path} does not exist")
    if out.ilastik:
        out.ilastik_version = (
            ilastik_version(out.ilastik) if query_version else find_version(out.ilastik)
        )
        out.version_check = check_version(out.ilastik_version)

    fiji_paths = [p for p in (settings.fiji, os.environ.get("HCS_FIJI", "")) if p]
    roots = [fiji_root(p) for p in fiji_paths if Path(p).exists()]
    roots += fiji_candidates(platform, home)
    for root in roots:
        launcher = fiji_launcher(root, platform)
        out.fiji = str(root)
        out.fiji_launcher = str(launcher) if launcher else ""
        jar = ilastik4ij_jar(root)
        if jar:
            out.ilastik4ij = str(jar)
            out.ilastik4ij_version = find_version(jar.stem)
        if launcher:
            break
    return out
