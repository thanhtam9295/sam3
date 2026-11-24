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
    args = parser.parse_args()
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
        lo = float(project["fake"].attrs["lo"])
        hi = float(project["fake"].attrs["hi"])
        workflow = project["workflowName"][()].decode()
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
        file, _, dataset = spec.rpartition(".h5/")
        file += ".h5"
        with h5py.File(file, "r") as f:
            data = f[dataset][()]
            axes = [a["key"] for a in json.loads(f[dataset].attrs["axistags"])["axes"]]
        c = axes.index("c")
        first = np.take(data, 0, axis=c).astype(np.float64)
        fg = np.clip((first - lo) / (hi - lo), 0, 1)
        out: np.ndarray
        if args.export_source.startswith("Probabilities"):
            out = np.stack([1 - fg, fg], axis=-1).astype(np.float32)
        else:
            out = np.where(fg >= 0.5, 2, 1).astype(np.uint8)[..., np.newaxis]
        out_axes = "".join(a for a in axes if a != "c") + "c"
        target = args.output_filename_format.replace("{nickname}", Path(file).stem)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, "w") as f:
            ds = f.create_dataset("exported_data", data=out)
            ds.attrs["axistags"] = axistags(out_axes)
    print("Finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
