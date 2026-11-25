"""A stand-in for ilastik's headless command line (tests only).

Accepts the real flags (argparse rejects unknown ones with exit code 2, as ilastik does) and
writes known-answer exports: the class-2 ("foreground") probability is a linear ramp of the
first input channel between the project's ``fake/lo`` and ``fake/hi`` attributes; class 1 is
1 − that. ``Simple Segmentation`` gives 2 where the probability is ≥ 0.5, else 1.

Environment: ``FAKE_ILASTIK_LOG`` (append one JSON line per call), ``FAKE_ILASTIK_FAIL_ON``
(exit 1 when an input path contains this text), ``FAKE_ILASTIK_TOUCH_PROJECT`` (modify the
project, which a headless ilastik must never do).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def axistags(axes: str) -> str:
    flags = {"c": 1, "t": 8, "z": 2, "y": 2, "x": 2}
    return json.dumps(
        {
            "axes": [
                {"key": a, "typeFlags": flags[a], "resolution": 0, "description": ""} for a in axes
            ]
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="ilastik")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--export_source")
    parser.add_argument("--output_format", default="hdf5")
    parser.add_argument("--output_filename_format")
    parser.add_argument("--raw_data", nargs="+", default=[])
    parser.add_argument("--prediction_mask", nargs="+", default=[])
    parser.add_argument("--label_image", nargs="+", default=[])
    parser.add_argument("--prediction_maps", nargs="+", default=[])
    parser.add_argument("--segmentation_image", nargs="+", default=[])
    parser.add_argument("--probabilities", nargs="+", default=[])
    parser.add_argument("--export_plugin")
    # flags the ilastik4ij plugin passes (real ilastik flags)
    parser.add_argument("--readonly")
    parser.add_argument("--input_axes")
    parser.add_argument("--output_axis_order")
    parser.add_argument("--output_internal_path", default="exported_data")
    args = parser.parse_args()
    global ARGS
    ARGS = args
    if args.version:
        print("1.4.2")
        return 0
    if not args.headless or not args.project:
        print("ERROR: --headless and --project are required", file=sys.stderr)
        return 2
    log = os.environ.get("FAKE_ILASTIK_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "argv": sys.argv[1:],
                        "threads": os.environ.get("LAZYFLOW_THREADS"),
                        "ram": os.environ.get("LAZYFLOW_TOTAL_RAM_MB"),
                    }
                )
                + "\n"
            )
    fail_on = os.environ.get("FAKE_ILASTIK_FAIL_ON")
    if fail_on and any(fail_on in spec for spec in args.raw_data):
        print(
            "Traceback (most recent call last):\nRuntimeError: simulated ilastik failure",
            file=sys.stderr,
        )
        return 1
    with h5py.File(args.project, "r") as project:
        lo = float(project["fake"].attrs.get("lo", 0))
        hi = float(project["fake"].attrs.get("hi", 1))
        min_area = int(project["fake"].attrs.get("min_area", 0))
        workflow = project["workflowName"][()].decode()
    if "ObjectClassification" in workflow:
        return objects(args, min_area)
    if "Multicut" in workflow:
        return multicut(args)
    if "Tracking" in workflow:
        return tracking(args)
    if os.environ.get("FAKE_ILASTIK_TOUCH_PROJECT"):
        with h5py.File(args.project, "a") as project:
            project["fake"].attrs["touched"] = 1
    sources = {
        "PixelClassificationWorkflow": ("Probabilities", "Simple Segmentation"),
        "NewAutocontextWorkflowBase": ("Probabilities Stage 2", "Simple Segmentation Stage 2"),
    }
    if args.export_source not in sources.get(workflow, ()):
        print(f"ERROR: unknown export source {args.export_source!r}", file=sys.stderr)
        return 1
    print(f"Starting ilastik 1.4.2 headless ({len(args.raw_data)} inputs)")
    for spec in args.raw_data:
        data, axes = read(spec)
        c = axes.index("c")
        first = np.take(data, 0, axis=c).astype(np.float64)
        fg = np.clip((first - lo) / (hi - lo), 0, 1)
        out: np.ndarray
        if args.export_source.startswith("Probabilities"):
            out = np.stack([1 - fg, fg], axis=-1).astype(np.float32)
        else:
            out = np.where(fg >= 0.5, 2, 1).astype(np.uint8)[..., np.newaxis]
        out_axes = "".join(a for a in axes if a != "c") + "c"
        write(target_for(args, spec), out, out_axes)
    print("Finished")
    return 0


ARGS: argparse.Namespace | None = None


def split(spec: str) -> tuple[str, str | None]:
    """``<file>.h5/<dataset>``, or a bare ``<file>.h5`` (its only dataset), as ilastik takes."""
    if spec.endswith(".h5"):
        return spec, None
    file, _, dataset = spec.rpartition(".h5/")
    return file + ".h5", dataset


def read(spec: str) -> tuple[np.ndarray, list[str]]:
    file, dataset = split(spec)
    with h5py.File(file, "r") as f:
        name = dataset or next(iter(f.keys()))
        data = f[name][()]
        if ARGS is not None and ARGS.input_axes:
            return data, list(ARGS.input_axes)
        return data, [a["key"] for a in json.loads(f[name].attrs["axistags"])["axes"]]


def objects(args: argparse.Namespace, min_area: int) -> int:
    """Object Classification: class 2 for objects of at least ``fake/min_area`` pixels."""
    from scipy import ndimage

    roles = [
        r for r in ("label_image", "prediction_maps", "segmentation_image") if getattr(args, r)
    ]
    if len(roles) != 1 or len(getattr(args, roles[0])) != len(args.raw_data):
        print("ERROR: give one object input per raw image", file=sys.stderr)
        return 2
    if args.export_source not in ("Object Predictions", "Object Identities"):
        print(f"ERROR: unknown export source {args.export_source!r}", file=sys.stderr)
        return 1
    for raw_spec, spec in zip(args.raw_data, getattr(args, roles[0]), strict=True):
        data, axes = read(spec)
        c = axes.index("c")
        if roles[0] == "label_image":
            ids = np.take(data, 0, axis=c).astype(np.int64)
        else:
            channel = np.take(data, data.shape[c] - 1, axis=c)
            ids, _ = ndimage.label(channel >= 0.5 if roles[0] == "prediction_maps" else channel > 0)
        if args.export_source == "Object Identities":
            out = ids.astype(np.uint32)
        else:
            sizes = np.bincount(ids.ravel())
            classes = np.where(sizes >= min_area, 2, 1).astype(np.uint8)
            classes[0] = 0
            out = classes[ids]
        out_axes = "".join(a for a in axes if a != "c") + "c"
        write(target_for(args, raw_spec), out[..., np.newaxis], out_axes)
    return 0


def write(target: str, out: np.ndarray, axes: str) -> None:
    """Write an export; ``--output_axis_order`` reorders it (missing axes get length 1)."""
    order = ARGS.output_axis_order if ARGS is not None else None
    if order:
        for a in order:
            if a not in axes:
                out, axes = out[..., np.newaxis], axes + a
        out = np.transpose(out, [axes.index(a) for a in order])
        axes = order
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    name = ARGS.output_internal_path if ARGS is not None else "exported_data"
    with h5py.File(target, "w") as f:
        ds = f.create_dataset(name, data=out)
        ds.attrs["axistags"] = axistags(axes)


def target_for(args: argparse.Namespace, raw_spec: str) -> str:
    file, _ = split(raw_spec)
    return str(args.output_filename_format).replace("{nickname}", Path(file).stem)


def multicut(args: argparse.Namespace) -> int:
    """Segments = connected regions where the boundary probability (last channel) < 0.5."""
    from scipy import ndimage

    if args.export_source != "Multicut Segmentation" or not args.probabilities:
        print(
            "ERROR: multicut needs --probabilities and export Multicut Segmentation",
            file=sys.stderr,
        )
        return 1
    for raw_spec, spec in zip(args.raw_data, args.probabilities, strict=True):
        data, axes = read(spec)
        c = axes.index("c")
        boundary = np.take(data, data.shape[c] - 1, axis=c)
        labels, _ = ndimage.label(boundary < 0.5)
        write(
            target_for(args, raw_spec),
            labels.astype(np.uint32)[..., np.newaxis],
            "".join(a for a in axes if a != "c") + "c",
        )
    return 0


def tracking(args: argparse.Namespace) -> int:
    """Objects per frame; each linked to the nearest object of the previous frame within 20 px;
    two objects linked to one → a division (new tracks, parent = the mother's track)."""
    from scipy import ndimage

    role = "segmentation_image" if args.segmentation_image else "prediction_maps"
    for raw_spec, spec in zip(args.raw_data, getattr(args, role), strict=True):
        data, axes = read(spec)
        if axes[0] != "t":
            print("ERROR: tracking needs a time axis", file=sys.stderr)
            return 1
        c = axes.index("c")
        channel = np.take(data, data.shape[c] - 1, axis=c)
        mask = channel >= 0.5 if role == "prediction_maps" else channel > 0
        frames = [ndimage.label(m)[0] for m in mask]
        if args.export_source == "Object-Identities":
            out = np.stack(frames).astype(np.uint32)[..., np.newaxis]
            write(target_for(args, raw_spec), out, "".join(a for a in axes if a != "c") + "c")
            continue
        if args.export_source != "Plugin" or args.export_plugin != "CSV-Table":
            print("ERROR: unknown export", file=sys.stderr)
            return 1
        rows: list[tuple[int, int, int, int, int]] = []
        previous: dict[int, tuple[Any, int]] = {}
        next_track = 1
        for t, labels in enumerate(frames):
            ids = [i for i in np.unique(labels) if i > 0]
            centres = dict(zip(ids, ndimage.center_of_mass(labels > 0, labels, ids), strict=True))
            claims: dict[int, list[int]] = {}
            for i in ids:
                near = [
                    (np.hypot(*np.subtract(centres[i], p)), j) for j, (p, _) in previous.items()
                ]
                near = [n for n in near if n[0] <= 20]
                if near:
                    claims.setdefault(min(near)[1], []).append(i)
            current = {}
            for i in ids:
                mother = next((m for m, kids in claims.items() if i in kids), None)
                if mother is not None and len(claims[mother]) == 1:
                    track, parent = previous[mother][1], 0
                else:
                    track, next_track = next_track, next_track + 1
                    parent = previous[mother][1] if mother is not None else 0
                current[i] = (centres[i], track)
                rows.append((t, int(i), track, parent, 1))
            previous = current
        target = target_for(args, raw_spec)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("frame,labelimage_oid,track_id1,parent_track_id,lineage_id\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
