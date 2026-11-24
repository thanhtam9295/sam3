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
