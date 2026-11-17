"""Command-line entry point: launches the GUI or prints environment information."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hcs_studio import __version__
from hcs_studio.utils.gpu import detect_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcs-studio",
        description="Image analysis for high-content cellular imaging.",
    )
    parser.add_argument("--version", action="version", version=f"hcs-studio {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="launch the desktop application (default)")
    sub.add_parser("info", help="print version and GPU status, then exit")
    inspect = sub.add_parser(
        "inspect",
        help="describe how a plate folder or image file is read (opens nothing in the GUI)",
    )
    inspect.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print(f"hcs-studio {__version__}")
        print(detect_gpu().summary)
        return 0
    if args.command == "inspect":
        from hcs_studio.inspect import describe_path

        print(describe_path(args.path))
        return 0
    try:
        from hcs_studio.gui.launch import run_gui
    except ImportError as exc:
        print(
            f"The GUI dependencies are not installed ({exc}).\n"
            'Install them with: pip install "hcs-studio[gui]"'
        )
        return 1
    return run_gui()
