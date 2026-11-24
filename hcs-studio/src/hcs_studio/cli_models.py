"""Command-line interface for ground truth, validation and models (Phase 25)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_parsers(sub: Any) -> None:
    gt = sub.add_parser("gt", help="ground-truth sets for training and validating models")
    gt_sub = gt.add_subparsers(dest="gt_command", required=True)
    gt_sub.add_parser("list", help="list the sets (latest versions)")
    show = gt_sub.add_parser("show", help="one set: items per split and cell line")
    show.add_argument("ref", help="NAME or NAME@VERSION")
    imp = gt_sub.add_parser("import", help="a new version from *_img.tif / *_masks.tif pairs")
    imp.add_argument("folder", type=Path)
    imp.add_argument("--name", required=True)
    imp.add_argument("--provenance", type=Path, help="CSV: id, plate, well, field, cell_line…")
    for key in ("plate", "cell-line", "stain", "annotator", "source", "run"):
        imp.add_argument(f"--{key}", default="", help=f"default {key} of every item")
    imp.add_argument("--pixel-size", type=float, help="µm per pixel of every item")
    imp.add_argument("--channels", default="", help="channel names, comma-separated")
    imp.add_argument("--based-on", help="NAME@VERSION whose items are kept")
    imp.add_argument("--notes", default="")
    imp.add_argument("--reason", default="")
    split = gt_sub.add_parser("split", help="a new version split into train/validation/test")
    split.add_argument("ref")
    split.add_argument("--by", default="well", choices=["plate", "well", "cell_line"])
    split.add_argument("--fractions", default="0.6,0.2,0.2", help="train,validation,test")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument(
        "--test-group",
        action="append",
        default=[],
        help="a plate / well (PLATE|WELL) / cell line that goes to test",
    )
    split.add_argument("--reason", default="")
    export = gt_sub.add_parser("export", help="Cellpose training pairs of one split")
    export.add_argument("ref")
    export.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    export.add_argument("--output", type=Path, required=True)

    val = sub.add_parser("validate-seg", help="segmentation metrics against a ground-truth set")
    val.add_argument("--set", dest="gt", required=True, help="NAME or NAME@VERSION")
    val.add_argument(
        "--pipeline",
        type=Path,
        required=True,
        help="pipeline whose image steps make the segmentation",
    )
    val.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    val.add_argument("--labels", help="label set to compare (default: the last one made)")
    val.add_argument("--pixel-size", type=float, help="µm per pixel when items lack it")
    val.add_argument("--output", type=Path, help="folder for fields.csv and metrics.json")

    models = sub.add_parser("models", help="the model registry: train, register, promote, pin")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    listing = models_sub.add_parser("list", help="registered model versions")
    listing.add_argument("--name")
    listing.add_argument("--kind")
    show = models_sub.add_parser("show", help="one version with its metrics")
    show.add_argument("ref", help="NAME, NAME@VERSION or NAME@production")
    reg = models_sub.add_parser("register", help="register a model file as a new candidate")
    reg.add_argument("file", type=Path)
    reg.add_argument("--name", required=True)
    reg.add_argument(
        "--kind",
        required=True,
        choices=["cellpose", "steps", "classifier", "qc_classifier", "ilastik", "micro_sam"],
    )
    reg.add_argument("--base-model", default="")
    reg.add_argument("--training-set", default="", help="ground-truth NAME@VERSION")
    reg.add_argument("--notes", default="")
    reg.add_argument("--reason", default="")
    status = models_sub.add_parser("status", help="candidate → validated → production → retired")
    status.add_argument("ref", help="NAME@VERSION")
    status.add_argument("status", choices=["candidate", "validated", "production", "retired"])
    status.add_argument("--reason", required=True)
    status.add_argument("--yes", action="store_true", help="confirm a promotion to production")
    train = models_sub.add_parser("train", help="train on a ground-truth set and register")
    train.add_argument("--kind", required=True, choices=["steps", "cellpose", "classifier"])
    train.add_argument("--name", required=True)
    train.add_argument("--set", dest="gt", required=True, help="ground-truth NAME[@VERSION]")
    train.add_argument("--channel", type=int, default=1)
    train.add_argument("--second-channel", type=int)
    train.add_argument("--base-model", default="")
    train.add_argument("--pixel-size", type=float)
    train.add_argument("--epochs", type=int, help="Cellpose")
    train.add_argument("--learning-rate", type=float, help="Cellpose")
    train.add_argument("--grid", help='steps: JSON, e.g. {"method": ["otsu", "li"]}')
    train.add_argument("--algorithm", choices=["random_forest", "logistic_regression"])
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    train.add_argument("--notes", default="")
    train.add_argument("--reason", default="")
    compare = models_sub.add_parser(
        "compare", help="candidate vs baseline on held-out ground truth"
    )
    compare.add_argument("candidate", help="registry:NAME@VERSION or NAME@VERSION")
    compare.add_argument("--baseline", help="default: the production version, else the base model")
    compare.add_argument("--set", dest="gt", help="default: the candidate's training set")
    compare.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    compare.add_argument("--output", type=Path, help="default: the candidate's reports folder")
    compare.add_argument("--reason", default="")
    impact = models_sub.add_parser("impact", help="re-run a reference plate with a candidate")
    impact.add_argument("reference_run", type=Path)
    impact.add_argument("candidate")
    impact.add_argument("--output", type=Path, required=True)
    impact.add_argument("--step", type=int, help="step to replace (from 1)")
    impact.add_argument("--readout", help="default: <labels>_count")
    impact.add_argument("--msr", type=float, default=3.0, help="when the registry has no MSR")
    impact.add_argument("--count-tolerance", type=float, default=10.0, help="% mean difference")
    impact.add_argument("--workers", type=int, default=1)
    impact.add_argument("--reason", default="")
    drift = models_sub.add_parser("drift", help="a run's objects vs the model's training data")
    drift.add_argument("run", type=Path)
    drift.add_argument("--model", help="default: the registered model the run used")
    drift.add_argument("--threshold", type=float, default=1.0, help="IQR")
    spot = models_sub.add_parser("spot-check", help="queue, list and review field spot checks")
    spot_sub = spot.add_subparsers(dest="spot_command", required=True)
    queue = spot_sub.add_parser("queue", help="queue random fields of a run")
    queue.add_argument("run", type=Path)
    queue.add_argument("-n", type=int, default=3)
    queue.add_argument("--seed", type=int)
    queue.add_argument("--reason", default="")
    spot_list = spot_sub.add_parser("list", help="spot checks (pending by default)")
    spot_list.add_argument(
        "--status", default="pending", choices=["pending", "accepted", "flagged", "all"]
    )
    review = spot_sub.add_parser("review", help="accept or flag a spot check")
    review.add_argument("id", type=int)
    review.add_argument("status", choices=["accepted", "flagged", "pending"])
    review.add_argument("--reason", default="")
    spot_sub.add_parser("summary", help="accept rate per model")
    pin = models_sub.add_parser("pin", help="replace @production references by exact versions")
    pin.add_argument("pipeline", type=Path)
    pin.add_argument("--output", type=Path, help="default: overwrite the pipeline file")


def _context(args: argparse.Namespace) -> Any:
    from hcs_studio.services import context

    return context(getattr(args, "reason", "") or "")


def _print_set(ann: Any) -> None:
    counts = ann.counts()
    lines = sorted({i.cell_line or "–" for i in ann.items})
    print(
        f"{ann.ref}  {ann.kind}  {len(ann.items)} items  "
        f"train {counts['train']} / validation {counts['validation']} / test {counts['test']}"
        f" / unassigned {counts['unassigned']}  cell lines: {', '.join(lines)}  "
        f"sha256 {ann.content_sha256[:12]}"
    )


def run_gt(args: argparse.Namespace) -> int:
    import pandas as pd

    from hcs_studio.services import training as services
    from hcs_studio.training import groundtruth as gt

    try:
        if args.gt_command == "list":
            for ann in gt.list_sets():
                _print_set(ann)
        elif args.gt_command == "show":
            ann = gt.load_set(args.ref)
            _print_set(ann)
            table = pd.DataFrame([i.model_dump() for i in ann.items])
            print(table.groupby(["cell_line", "split"]).size().to_string())
        elif args.gt_command == "import":
            defaults = {
                k.replace("-", "_"): getattr(args, k.replace("-", "_"))
                for k in ("plate", "cell-line", "stain", "annotator", "source", "run")
                if getattr(args, k.replace("-", "_"))
            }
            if args.pixel_size:
                defaults["pixel_size_um"] = args.pixel_size
            provenance = pd.read_csv(args.provenance) if args.provenance else None
            ann = services.import_ground_truth(
                _context(args),
                args.folder,
                args.name,
                provenance=provenance,
                defaults=defaults,
                based_on=args.based_on,
                notes=args.notes,
                channels=[c for c in args.channels.split(",") if c],
            )
            _print_set(ann)
        elif args.gt_command == "split":
            fractions = [float(v) for v in args.fractions.split(",")]
            ann = services.split_ground_truth(
                _context(args),
                args.ref,
                by=args.by,
                fractions=fractions,
                seed=args.seed,
                test_groups=args.test_group,
            )
            _print_set(ann)
        else:
            split = None if args.split == "all" else args.split
            written = gt.export_pairs(args.ref, split, args.output)
            print(f"{len(written)} training pairs → {args.output}")
    except (ValueError, FileExistsError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def run_validate(args: argparse.Namespace) -> int:
    from hcs_studio.pipeline.schema import Pipeline
    from hcs_studio.processing.base import ModuleError
    from hcs_studio.training.validate import (
        segmentation_steps,
        validate_segmentation,
        write_validation,
    )

    try:
        steps = segmentation_steps(Pipeline.load(args.pipeline))
        result = validate_segmentation(
            args.gt,
            steps,
            split=None if args.split == "all" else args.split,
            labels=args.labels,
            pixel_size_um=args.pixel_size,
            model=args.pipeline.stem,
        )
    except (ValueError, ModuleError) as exc:
        print(f"Error: {exc}")
        return 2
    o = result.overall
    print(
        f"{result.set_ref} ({result.split}, {o['fields']} fields, {o['objects_true']} objects): "
        f"AP@0.5 {o['ap50']:.3f}  mAP@0.5:0.95 {o['map']:.3f}  precision {o['precision50']:.3f}"
        f"  recall {o['recall50']:.3f}  F1 {o['f1_50']:.3f}  merges {o['merges']}  splits "
        f"{o['splits']}  mean |count error| {o['mean_abs_count_error']:.1f}"
    )
    for line, values in result.report.get("by_cell_line", {}).items():
        print(f"  {line or '–'}: AP@0.5 {values['ap50']:.3f}  F1 {values['f1_50']:.3f}")
    if args.output:
        write_validation(result, args.output)
        print(json.dumps({"written": str(args.output)}))
    return 0


def _model_line(m: Any) -> str:
    validation = m.metrics.get("validation", {})
    score = validation.get("overall", {}).get("ap50", validation.get("balanced_accuracy"))
    shown = f"{score:.3f}" if isinstance(score, float) else "–"
    return (
        f"{m.name}@{m.version}  {m.kind:<12} {m.status:<10} validation {shown:<6} "
        f"set {m.training_set or '–'}  sha256 {m.sha256[:12]}  {m.created[:10]}"
    )


def run_models(args: argparse.Namespace) -> int:
    from hcs_studio.services import training as services
    from hcs_studio.training import registry

    try:
        if args.models_command == "list":
            for model in registry.list_models(args.name, args.kind):
                print(_model_line(model))
        elif args.models_command == "show":
            name, _, version = args.ref.removeprefix(registry.PREFIX).partition("@")
            model = registry.get_model(name, version or None)
            print(_model_line(model))
            print(
                json.dumps(
                    {
                        "file": str(model.path),
                        "base_model": model.base_model,
                        "hyperparameters": model.hyperparameters,
                        "metrics": model.metrics,
                        "notes": model.notes,
                    },
                    indent=1,
                    default=str,
                )
            )
        elif args.models_command == "register":
            model = services.register_model(
                _context(args),
                args.name,
                args.kind,
                args.file,
                base_model=args.base_model,
                training_set=args.training_set,
                notes=args.notes,
            )
            print(_model_line(model))
        elif args.models_command == "status":
            name, _, version = args.ref.removeprefix(registry.PREFIX).partition("@")
            if not version.isdigit():
                raise ValueError("give an exact version: NAME@VERSION")
            if args.status == "production" and not args.yes:
                print(
                    f"Promoting {name}@{version} to production changes what new pipelines use; "
                    "repeat with --yes to confirm"
                )
                return 2
            model = services.set_model_status(
                _context(args), name, int(version), args.status, approved=args.yes
            )
            print(_model_line(model))
        elif args.models_command == "train":
            from hcs_studio.training.train import TrainingSpec

            hyper: dict[str, Any] = {}
            if args.epochs:
                hyper["epochs"] = args.epochs
            if args.learning_rate:
                hyper["learning_rate"] = args.learning_rate
            if args.grid:
                hyper["grid"] = json.loads(args.grid)
            if args.algorithm:
                hyper["algorithm"] = args.algorithm
            spec = TrainingSpec(
                kind=args.kind,
                name=args.name,
                set=args.gt,
                channel=args.channel,
                second_channel=args.second_channel,
                base_model=args.base_model,
                pixel_size_um=args.pixel_size,
                hyperparameters=hyper,
                device=args.device,
                notes=args.notes,
            )
            result = services.train_model(_context(args), spec, print)
            print(_model_line(result.model))
            if result.baseline is not None:
                print(
                    f"Validation AP@0.5: {result.validation['overall']['ap50']:.3f} "
                    f"(base model {result.baseline['overall']['ap50']:.3f})"
                )
        elif args.models_command == "compare":
            ref = (
                args.candidate
                if registry.is_ref(args.candidate)
                else registry.PREFIX + args.candidate
            )
            baseline = args.baseline
            if baseline and not registry.is_ref(baseline):
                baseline = registry.PREFIX + baseline
            comparison, report = services.compare_models(
                _context(args),
                ref,
                baseline=baseline,
                set_ref=args.gt,
                split="all" if args.split == "all" else args.split,
                out=args.output,
            )
            print(comparison.table().to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print(f"Report: {report}")
        elif args.models_command == "impact":
            ref = (
                args.candidate
                if registry.is_ref(args.candidate)
                else registry.PREFIX + args.candidate
            )
            result, report = services.model_impact(
                _context(args),
                args.reference_run,
                ref,
                args.output,
                step=args.step,
                readout=args.readout,
                msr=args.msr,
                count_tolerance_pct=args.count_tolerance,
                workers=args.workers,
            )
            s = result.counts_summary
            verdict = (
                "equivalent"
                if result.verdict["equivalent"]
                else "; ".join(result.verdict["reasons"])
            )
            print(
                f"{result.replaced}\n{result.readout}: mean difference "
                f"{s['mean_difference_pct']:+.1f} %"
                f" (limits of agreement {s['loa_low_pct']:+.1f} to {s['loa_high_pct']:+.1f} %)"
            )
            print(f"Verdict: {verdict}\nReport: {report}")
        elif args.models_command == "drift":
            from hcs_studio.training.qa import drift_report

            report_data = drift_report(args.run, args.model, args.threshold)
            print(json.dumps(report_data, indent=1, default=float))
        elif args.models_command == "spot-check":
            return _spot_check(args)
        else:
            from hcs_studio.pipeline.schema import Pipeline

            pipeline, changes = registry.pin_pipeline(Pipeline.load(args.pipeline))
            pipeline.save(args.output or args.pipeline)
            print("\n".join(changes) or "Nothing to pin: every model reference names a version")
    except (ValueError, PermissionError, FileExistsError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0


def _spot_check(args: argparse.Namespace) -> int:
    from hcs_studio.services import training as services
    from hcs_studio.training import qa

    if args.spot_command == "queue":
        ids = services.queue_spot_checks(_context(args), args.run, args.n, args.seed)
        print(f"{len(ids)} field(s) queued for a spot check: ids {ids}")
    elif args.spot_command == "list":
        table = qa.spot_checks(None if args.status == "all" else args.status)
        columns = ["id", "barcode", "well", "field", "model", "status", "reviewer", "reason"]
        print(table[columns].to_string(index=False) if len(table) else "No spot checks")
    elif args.spot_command == "review":
        record = services.review_spot_check(_context(args), args.id, args.status)
        print(
            f"Spot check {record['id']} ({record['well']} field {record['field']}): "
            f"{record['status']}"
        )
    else:
        summary = qa.spot_check_summary()
        print(summary.to_string(index=False) if len(summary) else "No spot checks")
    return 0
