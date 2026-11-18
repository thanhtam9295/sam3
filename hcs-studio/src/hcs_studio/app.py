"""Command-line entry point: launches the GUI or prints environment information."""

from __future__ import annotations

import argparse
import math
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
    gui = sub.add_parser("gui", help="launch the desktop application (default)")
    gui.add_argument(
        "--smoke-test",
        action="store_true",
        help="open the window, show the test image and close (checks an installation)",
    )
    sub.add_parser("info", help="print version and GPU status, then exit")
    inspect = sub.add_parser(
        "inspect",
        help="describe how a plate folder or image file is read (opens nothing in the GUI)",
    )
    inspect.add_argument("path", type=Path)
    sub.add_parser("modules", help="list processing modules and their parameters")
    report = sub.add_parser("report", help="(re)write index.html of a run folder")
    report.add_argument("folder", type=Path)
    train = sub.add_parser("train-cellpose", help="fine-tune Cellpose on edited masks")
    train.add_argument("--data", type=Path, required=True, help="training folder")
    train.add_argument("--base", default="cpsam_v2", help="model to start from")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--name", default="hcs_custom", help="name of the new model")
    train.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    analyze = sub.add_parser(
        "analyze",
        help="apply table steps (gates, classifiers, clustering, spatial) to a finished run",
    )
    analyze.add_argument("folder", type=Path, help="run folder")
    analyze.add_argument(
        "--pipeline", type=Path, required=True, help="pipeline file whose table steps to apply"
    )
    analyze.add_argument(
        "--only",
        action="store_true",
        help="apply only these steps (default: the run's own table steps first)",
    )
    annotate = sub.add_parser(
        "annotate", help="apply a plate layout to a finished run (tables and report rebuilt)"
    )
    annotate.add_argument("folder", type=Path, help="run folder")
    annotate.add_argument(
        "--layout", type=Path, required=True, help="layout: JSON, long CSV/XLSX or plate-map XLSX"
    )
    layout_cmd = sub.add_parser("layout", help="check or convert a plate layout file")
    layout_cmd.add_argument("path", type=Path)
    layout_cmd.add_argument("--validate", action="store_true", help="list problems")
    layout_cmd.add_argument(
        "--export", type=Path, help="write as .json, .csv, .xlsx, or a figure (.png/.pdf)"
    )
    layout_cmd.add_argument("--plate-map", action="store_true", help="export as a plate-map XLSX")
    layout_cmd.add_argument("--layer", default="compound", help="figure colouring")
    selftest = sub.add_parser(
        "selftest", help="check this installation on a small synthetic plate (about a minute)"
    )
    selftest.add_argument("--workers", type=int, default=2)
    selftest.add_argument("--keep", type=Path, help="keep the test plate and run in this folder")
    acceptance = sub.add_parser(
        "acceptance", help="acceptance tests on real data (PROMPT.md §10), PASS/FAIL report"
    )
    checks = acceptance.add_subparsers(dest="check", required=True)
    counts = checks.add_parser("counts", help="counts per field against manual counts (±5 %%)")
    counts.add_argument("--run", type=Path, required=True)
    counts.add_argument("--manual", type=Path, required=True, help="CSV: well, field, count")
    counts.add_argument("--object-type", default="nuclei")
    counts.add_argument("--tolerance", type=float, default=0.05)
    coloc = checks.add_parser("coloc", help="Pearson/Manders against Fiji Coloc 2 (±0.02)")
    coloc.add_argument("folders", type=Path, nargs="+", help="folders from Export for Fiji")
    coloc.add_argument("--tolerance", type=float, default=0.02)
    run_check = checks.add_parser("run", help="a run folder is complete; report works offline")
    run_check.add_argument("folder", type=Path)
    kill = checks.add_parser("kill-resume", help="kill an unattended run part-way, then resume")
    kill.add_argument("--pipeline", type=Path, required=True)
    kill.add_argument("--input", type=Path, required=True)
    kill.add_argument("--output", type=Path, required=True)
    kill.add_argument("--layout", type=Path)
    kill.add_argument("--kill-after", type=int, default=10, help="complete wells before the kill")
    kill.add_argument("--workers", default="auto")
    beads = checks.add_parser("beads", help="deconvolution narrows the FWHM of beads")
    beads.add_argument("path", type=Path, help="bead Z stack (.nd2, OME-TIFF, …)")
    beads.add_argument("--channel", type=int, default=1)
    beads.add_argument("--psf", type=Path, help="measured PSF file (default: theoretical)")
    beads.add_argument("--iterations", type=int, default=40)
    acceptance.add_argument(
        "--report", type=Path, help="write the report as JSON (default: acceptance_<check>.json)"
    )
    pharm = sub.add_parser(
        "pharm",
        help="dose-response curves, combination matrices and synergy scores of finished runs",
    )
    pharm.add_argument("runs", nargs="*", type=Path, help="run folder(s); plates are merged")
    pharm.add_argument(
        "--run",
        dest="extra_runs",
        action="append",
        type=Path,
        default=[],
        help="run folder (may be repeated)",
    )
    pharm.add_argument(
        "--readout", required=True, help="per-well readout(s), comma separated, e.g. nuclei_count"
    )
    pharm.add_argument(
        "--normalize",
        default="percent_inhibition",
        choices=["percent_of_control", "percent_inhibition", "gr"],
    )
    pharm.add_argument(
        "--models",
        default="bliss,hsa,loewe,zip",
        help="synergy models: bliss, hsa, loewe, zip (comma separated; empty = none)",
    )
    pharm.add_argument("--curve", default="4pl", choices=["4pl", "5pl", "3pl_bottom", "3pl_top"])
    pharm.add_argument("--robust", action="store_true", help="robust (soft L1) curve fitting")
    pharm.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="bootstrap intervals with this many resamples (default: covariance)",
    )
    pharm.add_argument(
        "--outliers",
        type=float,
        default=None,
        metavar="K",
        help="exclude wells beyond median ± K·MAD of their replicates",
    )
    pharm.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PLATE:WELL",
        help="exclude a well (repeatable)",
    )
    pharm.add_argument("--z-threshold", type=float, default=0.5, help="flag plates below this Z′")
    pharm.add_argument("--doubling-time", type=float, help="GR: doubling time (h)")
    pharm.add_argument("--duration", type=float, help="GR: assay duration (h)")
    pharm.add_argument(
        "--time-zero",
        action="append",
        type=Path,
        default=[],
        help="GR: run folder of a time-zero plate",
    )
    pharm.add_argument("--surfaces", action="store_true", help="also draw 3D synergy surfaces")
    pharm.add_argument("--no-figures", action="store_true")
    pharm.add_argument("--no-pdf", action="store_true")
    pharm.add_argument(
        "--output", type=Path, help="output folder (default: <first run>/pharmacology)"
    )

    profile = sub.add_parser(
        "profile",
        help="PCA/UMAP phenotypic profiles, distance to control and dose-response of runs",
    )
    profile.add_argument("runs", type=Path, nargs="+", help="run folder(s), one per plate")
    profile.add_argument("--object-type", default="nuclei")
    profile.add_argument("--level", choices=["cell", "well"], default="cell")
    profile.add_argument(
        "--normalisation",
        choices=["negative_controls", "all_wells", "none"],
        default="negative_controls",
    )
    profile.add_argument("--plate-correction", action="store_true", help="centre each plate")
    profile.add_argument("--max-cells", type=int, default=50_000)
    profile.add_argument("--correlation", type=float, default=0.95, help="drop features above")
    profile.add_argument("--pca-components", type=int, default=10)
    profile.add_argument("--no-umap", action="store_true")
    profile.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    profile.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    profile.add_argument("--metric", default="euclidean", help="UMAP metric")
    profile.add_argument("--seed", type=int, default=0)
    profile.add_argument(
        "--dose-readouts",
        default="profile_distance_to_control",
        help="comma-separated per-well readouts for dose-response fits",
    )
    profile.add_argument("--output", type=Path, help="default: advanced/ of the first run")
    classify = sub.add_parser(
        "train-classifier", help="train an object classifier from labelled objects of a run"
    )
    classify.add_argument("--run", type=Path, required=True, help="run folder")
    classify.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV with well, field, object_id, label (and optionally timepoint)",
    )
    classify.add_argument("--object-type", default="nuclei")
    classify.add_argument(
        "--algorithm", choices=["random_forest", "hist_gradient_boosting"], default="random_forest"
    )
    classify.add_argument(
        "--normalisation", choices=["plate_robust_z", "none"], default="plate_robust_z"
    )
    classify.add_argument("--output", type=Path, required=True, help="classifier file (.joblib)")
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
    run.add_argument("--layout", type=Path, help="plate layout (CSV, XLSX, plate-map XLSX or JSON)")
    run.add_argument(
        "--no-images",
        action="store_true",
        help="do not save corrected images (masks, previews and tables are still written)",
    )
    run.add_argument(
        "--image-dtype",
        choices=["float32", "original"],
        default="float32",
        help="corrected images as float32 (default) or in the camera's bit depth (rounded, "
        "clipped; smaller files)",
    )
    run.add_argument(
        "--max-memory-gb", type=float, help="RAM the whole run may use (all workers together)"
    )
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
    if args.command == "report":
        from hcs_studio.output.report import write_report

        try:
            path = write_report(args.folder)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Report: {path}")
        return 0
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "annotate":
        return _annotate(args)
    if args.command == "layout":
        return _layout(args)
    if args.command == "selftest":
        from hcs_studio.selftest import run_selftest

        checks = run_selftest(workers=args.workers, keep=args.keep)
        failed = [c.name for c in checks if not c.ok]
        print(
            f"\n{len(checks) - len(failed)} of {len(checks)} checks passed"
            + (f"; failed: {', '.join(failed)}" if failed else "")
        )
        return 1 if failed else 0
    if args.command == "acceptance":
        return _acceptance(args)
    if args.command == "profile":
        return _profile(args)
    if args.command == "pharm":
        return _pharm(args)
    if args.command == "train-classifier":
        return _train_classifier(args)
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
    return run_gui(smoke_test=bool(getattr(args, "smoke_test", False)))


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
    from hcs_studio.model.layout import load_plate_layout, run_layout_file
    from hcs_studio.pipeline.executor import parse_wells
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
            layout_file = run_layout_file(folder)
            layout = load_plate_layout(layout_file) if layout_file else None
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
            layout = load_plate_layout(args.layout) if args.layout else None
            for path in args.input:
                plate = apply_overrides(open_plate(path))
                jobs.append((plate, pipeline, parse_wells(args.wells, plate), layout, args.layout))
    except (OSError, ValueError, KeyError, ValidationError, ReaderError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2

    # Ctrl+C stops after the current field; the run folder can then be resumed.
    import signal
    import threading

    cancel = threading.Event()

    def interrupt(signum: int, frame: Any) -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        print("\nStopping after the current field (Ctrl+C again to quit at once)…", flush=True)
        cancel.set()

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        return _run_jobs(args, jobs, workers, folder, progress, cancel)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _run_jobs(
    args: argparse.Namespace,
    jobs: list[Any],
    workers: int | None,
    folder: Path | None,
    progress: Any,
    cancel: Any,
) -> int:
    from pydantic import ValidationError

    from hcs_studio.io.base import ReaderError
    from hcs_studio.pipeline.executor import run_pipeline
    from hcs_studio.processing.base import ModuleError

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
                save_images=not args.no_images,
                max_memory_gb=args.max_memory_gb,
                image_dtype=args.image_dtype,
                cancel=cancel,
            )
        except (OSError, ValueError, ValidationError, ReaderError, ModuleError) as exc:
            print(f"Error: {exc}")
            code = 2
            continue
        print(f"Output folder: {result.folder}")
        if result.cancelled:
            print(f'Interrupted. Continue with: hcs-studio run --resume "{result.folder}"')
            return 130
        if result.skipped_wells:
            print(f"Already complete (skipped): {len(result.skipped_wells)} well(s)")
        print(f"Fields written: {result.fields_written}")
        for well, error in result.failed_wells.items():
            print(f"FAILED {well}: {error}")
        if result.failed_wells:
            code = max(code, 1)
    return code


def _analyze(args: argparse.Namespace) -> int:
    from hcs_studio.pipeline.executor import analyze_run
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.processing.base import ModuleError

    try:
        pipeline = Pipeline.load(args.pipeline)
        if not pipeline.table_steps() and args.only:
            print("Error: the pipeline has no table steps")
            return 2
        record = analyze_run(args.folder, pipeline, include_run_steps=not args.only)
    except (OSError, ValueError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(f"Applied {len(record['steps'])} table step(s); tables and report updated")
    return 0


def _train_classifier(args: argparse.Namespace) -> int:
    import pandas as pd

    from hcs_studio.analysis.classifier import save_classifier, train_from_run
    from hcs_studio.processing.base import ModuleError

    try:
        model = train_from_run(
            args.run,
            pd.read_csv(args.labels),
            args.object_type,
            algorithm=args.algorithm,
            normalisation=args.normalisation,
        )
    except (OSError, ValueError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    save_classifier(model, args.output)
    examples = ", ".join(f"{k}: {v}" for k, v in model.n_examples.items())
    print(f"Trained on {examples} ({len(model.features)} features)")
    if model.cv_accuracy is not None:
        print(
            f"{model.cv_folds}-fold cross-validated accuracy {model.cv_accuracy:.3f} "
            f"(balanced {model.cv_balanced_accuracy:.3f})"
        )
    print(f"Classifier: {args.output}")
    return 0


def _conc(value: float, unit: str) -> str:
    from hcs_studio.model import units

    if not math.isfinite(value):
        return "—"
    try:
        return units.format_concentration(value, unit)
    except (ValueError, KeyError):
        return f"{value:.3g}"


def _pharm(args: argparse.Namespace) -> int:
    from hcs_studio.pharmacology.analysis import PharmOptions, run_pharmacology

    runs = [*args.runs, *args.extra_runs]
    if not runs:
        print("Error: give at least one run folder")
        return 2
    excluded = []
    for item in args.exclude:
        plate, _, well = item.rpartition(":")
        if not plate or not well:
            print(f"Error: --exclude needs PLATE:WELL, got {item!r}")
            return 2
        excluded.append((plate, well))
    options = PharmOptions(
        readouts=[r.strip() for r in args.readout.split(",") if r.strip()],
        normalization=args.normalize,
        models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
        curve_model=args.curve,
        robust=args.robust,
        bootstrap=args.bootstrap,
        outlier_mad=args.outliers,
        z_threshold=args.z_threshold,
        doubling_time_h=args.doubling_time,
        assay_duration_h=args.duration,
        time_zero_runs=[str(p) for p in args.time_zero],
        excluded_wells=excluded,
        figures=not args.no_figures,
        surfaces=args.surfaces,
        pdf=not args.no_pdf,
    )
    try:
        result = run_pharmacology(runs, options, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    for readout, rr in result.readouts.items():
        print(f"== {readout} ==")
        for row in rr.qc.itertuples():
            print(f"plate {row.plate}: Z′ {row.z_prime:.2f}, SSMD {row.ssmd:.1f}  {row.flag}")
        for c in rr.curves:
            f = c.fit
            print(
                f"{c.cell_line} · {c.compound}: EC50 {_conc(f.ec50, c.unit)}, Hill "
                f"{f.hill:.2f}, R² {f.r2:.3f}" + (f"  [{'; '.join(f.flags)}]" if f.flags else "")
            )
        for combo in rr.combinations:
            scores = ", ".join(
                f"{name} {res.mean:+.1f} ({res.label})" for name, res in combo.models.items()
            )
            print(f"{combo.matrix.name}: {scores}")
        for note in rr.notes:
            print(f"Note: {note}")
    print(f"Wrote {result.folder} in {result.seconds:.1f} s")
    return 0


def _profile(args: argparse.Namespace) -> int:
    from hcs_studio.analysis.profiling import ProfileOptions, run_profiling
    from hcs_studio.output.report import write_report

    options = ProfileOptions(
        object_type=args.object_type,
        level=args.level,
        normalisation=args.normalisation,
        plate_correction=args.plate_correction,
        max_cells=args.max_cells,
        correlation_threshold=args.correlation,
        pca_components=args.pca_components,
        umap=not args.no_umap,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
        dose_readouts=[r.strip() for r in args.dose_readouts.split(",") if r.strip()],
    )
    try:
        result = run_profiling(args.runs, options, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    variance = result.variance["explained_variance_ratio"].head(3).tolist()
    print(
        f"{len(result.cells)} points, {len(result.wells)} wells, {len(result.features)} features "
        f"({len(result.dropped)} dropped); PC1-3 explain "
        + ", ".join(f"{100 * v:.0f} %" for v in variance)
    )
    for note in result.notes:
        print(f"Note: {note}")
    if args.output is None:
        write_report(args.runs[0])  # the report shows the profiles of the first run
    print(f"Profiles: {result.folder}")
    return 0


def _acceptance(args: argparse.Namespace) -> int:
    from hcs_studio import acceptance
    from hcs_studio.processing.base import ModuleError

    try:
        if args.check == "counts":
            report = acceptance.check_counts(
                args.run, args.manual, args.object_type, args.tolerance
            )
        elif args.check == "coloc":
            report = acceptance.check_coloc(args.folders, args.tolerance)
        elif args.check == "run":
            report = acceptance.check_run(args.folder)
        elif args.check == "kill-resume":
            report = acceptance.check_kill_resume(
                args.pipeline,
                args.input,
                args.output,
                kill_after_wells=args.kill_after,
                workers=args.workers,
                layout=args.layout,
            )
        else:
            report = acceptance.check_beads(
                args.path, args.channel, psf_file=args.psf, iterations=args.iterations
            )
    except (OSError, ValueError, KeyError, RuntimeError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(report.text())
    target = args.report or Path(f"acceptance_{args.check.replace('-', '_')}.json")
    report.save(target)
    print(f"Report: {target}")
    return 0 if report.passed else 1


def _annotate(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import load_layout
    from hcs_studio.pipeline.executor import annotate_run
    from hcs_studio.processing.base import ModuleError

    try:
        imported = load_layout(args.layout)
        for note in imported.notes:
            print(f"Note: {note}")
        record = annotate_run(args.folder, imported.layout, args.layout)
    except (OSError, ValueError, KeyError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    print(
        f"Layout applied ({record['wells']} wells, sha256 {record['sha256'][:12]}…); "
        "tables and report rebuilt"
    )
    return 0


def _layout(args: argparse.Namespace) -> int:
    from hcs_studio.io.layout_io import (
        load_layout,
        save_layout,
        write_layout_figure,
        write_platemap_xlsx,
    )
    from hcs_studio.model.layout_tools import validate

    try:
        imported = load_layout(args.path)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 2
    layout = imported.layout
    print(
        f"{len(layout.wells)} annotated wells on a {layout.rows}×{layout.cols} plate; "
        f"cell lines: {', '.join(layout.cell_lines()) or '-'}; compounds: "
        f"{', '.join(layout.compounds()) or '-'}"
    )
    for note in imported.notes:
        print(f"Note: {note}")
    code = 0
    if args.validate:
        issues = validate(layout)
        for issue in issues:
            print(f"{issue.level.upper()}: {issue.text()}")
        print("No problems found." if not issues else f"{len(issues)} problem(s)")
        code = 1 if any(i.level == "error" for i in issues) else 0
    if args.export:
        suffix = args.export.suffix.lower()
        if suffix in (".png", ".pdf", ".svg"):
            write_layout_figure(layout, args.export, args.layer)
        elif args.plate_map:
            write_platemap_xlsx(layout, args.export)
        else:
            save_layout(layout, args.export)
        print(f"Wrote {args.export}")
    return code
