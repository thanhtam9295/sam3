"""Command-line entry point: launches the GUI or prints environment information."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    sub.add_parser("modules", help="list processing modules and their parameters")
    train = sub.add_parser("train-cellpose", help="fine-tune Cellpose on edited masks")
    train.add_argument("--data", type=Path, required=True, help="training folder")
    train.add_argument("--base", default="cpsam_v2", help="model to start from")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--name", default="hcs_custom", help="name of the new model")
    train.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    bench = sub.add_parser("benchmark", help="time deconvolution of a synthetic stack")
    bench.add_argument(
        "--shape", type=int, nargs=3, default=[30, 2048, 2048], metavar=("Z", "Y", "X")
    )
    bench.add_argument("--iterations", type=int, default=40)
    bench.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    bench.add_argument("--tile", type=int, default=0, help="tile size in pixels; 0 = automatic")
    bench.add_argument("--max-memory-gb", type=float, help="memory limit (default: half free RAM)")
    run = sub.add_parser("run", help="run a pipeline on a plate, unattended")
    run.add_argument("--pipeline", type=Path, help="pipeline YAML or JSON file")
    run.add_argument(
        "--input",
        type=Path,
        action="append",
        help="plate folder or image file (repeat for several)",
    )
    run.add_argument(
        "--workers", default="1", help="worker processes: a number, or 'auto' (default: 1)"
    )
    run.add_argument(
        "--resume", type=Path, help="continue an interrupted run folder (skips completed wells)"
    )
    run.add_argument(
        "--retry-failed", type=Path, help="run folder: redo only the wells that failed"
    )
    run.add_argument("--wells", default="all", help="e.g. all, A01-H12, or B02,B03 (default: all)")
    run.add_argument("--output", type=Path, help="output root (default: next to the raw data)")
    run.add_argument("--layout", type=Path, help="plate layout CSV")
    run.add_argument(
        "--no-checksums",
        action="store_true",
        help="record input file size and time only, instead of SHA-256 checksums",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # no-op except in frozen (installer) builds
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print(f"hcs-studio {__version__}")
        print(detect_gpu().summary)
        return 0
    if args.command == "inspect":
        from hcs_studio.inspect import describe_path

        print(describe_path(args.path))
        return 0
    if args.command == "modules":
        print(_describe_modules())
        return 0
    if args.command == "run":
        return _run(args)
    if args.command == "train-cellpose":
        from hcs_studio.processing.base import ModuleError
        from hcs_studio.segmentation.cellpose_backend import fine_tune

        try:
            path, history = fine_tune(
                args.data,
                base_model=args.base,
                model_name=args.name,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=args.device,
            )
        except ModuleError as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Trained on {history['images_train']} image(s) for {args.epochs} epochs")
        print(f"Model: {path}")
        return 0
    if args.command == "benchmark":
        from hcs_studio.benchmark import run_benchmark
        from hcs_studio.processing.base import ModuleError

        try:
            result = run_benchmark(
                tuple(args.shape),
                args.iterations,
                args.device,
                args.tile,
                args.max_memory_gb,
            )
        except (ModuleError, MemoryError) as exc:
            print(f"Error: {exc}")
            return 2
        print(result.summary())
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


def _describe_modules() -> str:
    from hcs_studio.processing.base import MODULES, load_builtin_modules

    load_builtin_modules()
    lines: list[str] = []
    for name in MODULES:
        module = MODULES.get(name)
        note = "" if module.preserves_linearity else "  [display only: not for measurement]"
        lines.append(f"{name}: {module.title}{note}")
        for key, info in module.Params.model_fields.items():
            if info.exclude:
                continue
            default = (
                "required"
                if info.is_required()
                else f"default {info.get_default(call_default_factory=True)!r}"
            )
            lines.append(f"    {key} ({default}): {info.description or ''}")
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> int:
    import json

    from pydantic import ValidationError

    from hcs_studio.io.base import ReaderError, open_plate, reopen_plate
    from hcs_studio.io.sidecar import apply_overrides
    from hcs_studio.model.layout import read_plate_layout
    from hcs_studio.pipeline.executor import parse_wells, run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.processing.base import ModuleError

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}", flush=True)

    folder = args.resume or args.retry_failed
    try:
        workers = None if args.workers == "auto" else int(args.workers)
        if workers is not None and workers < 1:
            raise ValueError("--workers must be at least 1")
        jobs: list[tuple[Any, Pipeline, Any, Any, Path | None]] = []
        if folder is not None:
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            pipeline = Pipeline.load(folder / "pipeline.yaml")
            source = manifest["plate"].get("source") or {"path": manifest["plate"]["path"]}
            plate = apply_overrides(reopen_plate(source))
            layout_file = folder / "plate_layout.csv"
            layout = read_plate_layout(layout_file) if layout_file.exists() else None
            if args.retry_failed:
                failed = list(manifest.get("failed_wells", {}))
                if not failed:
                    print("No failed wells in this run.")
                    return 0
                wells = parse_wells(",".join(failed), plate)
            else:
                wells = parse_wells(",".join(manifest["wells"]), plate)
            jobs.append((plate, pipeline, wells, layout, None))
        else:
            if args.pipeline is None or not args.input:
                raise ValueError("--pipeline and --input are required (or use --resume)")
            pipeline = Pipeline.load(args.pipeline)
            layout = read_plate_layout(args.layout) if args.layout else None
            for path in args.input:
                plate = apply_overrides(open_plate(path))
                jobs.append((plate, pipeline, parse_wells(args.wells, plate), layout, args.layout))
    except (OSError, ValueError, KeyError, ValidationError, ReaderError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2

    code = 0
    for plate, pipeline, wells, layout, layout_path in jobs:
        if len(jobs) > 1:
            print(f"== {plate.name} ==", flush=True)
        try:
            result = run_pipeline(
                plate,
                pipeline,
                output_root=args.output,
                wells=wells,
                layout=layout,
                layout_path=layout_path,
                checksums=not args.no_checksums,
                progress=progress,
                workers=workers,
                resume=folder,
            )
        except (OSError, ValueError, ValidationError, ReaderError, ModuleError) as exc:
            print(f"Error: {exc}")
            code = 2
            continue
        print(f"Output folder: {result.folder}")
        if result.skipped_wells:
            print(f"Already complete (skipped): {len(result.skipped_wells)} well(s)")
        print(f"Fields written: {result.fields_written}")
        for well, error in result.failed_wells.items():
            print(f"FAILED {well}: {error}")
        if result.failed_wells:
            code = max(code, 1)
    return code
