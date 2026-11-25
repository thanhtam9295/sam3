"""A stand-in for ``Fiji --headless --console -macro <file>`` with the ilastik4ij plugin (tests
only).

It interprets the statements HCS Studio's FijiRunner writes (literal ``run("…", "…")``,
``rename``, ``close``, ``print``, ``File.saveString``) and does what the plugin does, as read in
its source: Import HDF5 reads a dataset with the given row-major ``axisorder``; a ``Run …``
command writes its inputs to a temporary folder as ``tzcyx`` HDF5 and starts the configured
ilastik with the plugin's own flags (``--readonly=1 --input_axes --output_axis_order
--output_internal_path=exported_data``, ``LAZYFLOW_*`` from Configure); Export HDF5 writes
``tzyxc`` with vigra axistags. Like Fiji, it exits 0 even when the macro fails (it prints the
error); FijiRunner must notice by the missing marker file.

``FAKE_FIJI_LOG``: append one JSON line per plugin command (name, options, ilastik argv).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

import h5py
import numpy as np

COMMANDS: dict[str, tuple[str, dict[str, str]]] = {
    "Run Pixel Classification Prediction": ("pixelclassificationtype", {}),
    "Run Autocontext Prediction": ("autocontextpredictiontype", {}),
    "Run Object Classification Prediction": ("objectexportsource", {}),
    "Run Multicut": ("", {}),
    "Run Tracking": ("", {}),
}
STATEMENT = re.compile(r"^(run|rename|close|print|File\.saveString)\((.*)\);$")


def axistags(axes: str) -> str:
    flags = {"c": 1, "t": 8, "z": 2, "y": 2, "x": 2}
    return json.dumps(
        {
            "axes": [
                {"key": a, "typeFlags": flags[a], "resolution": 0, "description": ""} for a in axes
            ]
        }
    )


def options(text: str) -> dict[str, str]:
    out = {}
    for match in re.finditer(r"(\w+)=(\[[^\]]*\]|\S+)", text):
        value = match.group(2)
        out[match.group(1)] = value[1:-1] if value.startswith("[") else value
    return out


def strings(args: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', args)]


def conform(data: np.ndarray, axes: str, order: str) -> np.ndarray:
    """Transpose to ``order``, adding the missing axes with length 1."""
    for a in order:
        if a not in axes:
            data, axes = data[..., np.newaxis], axes + a
    return np.transpose(data, [axes.index(a) for a in order])


class Fiji:
    def __init__(self) -> None:
        self.images: dict[str, tuple[np.ndarray, str]] = {}
        self.current = ""
        self.config: dict[str, str] = {}

    def error(self, text: str) -> NoReturn:
        raise RuntimeError(text)

    def run(self, command: str, opts: dict[str, str]) -> None:
        if command == "Configure ilastik executable location":
            self.config = opts
            return
        if command == "Import HDF5":
            with h5py.File(opts["select"], "r") as f:
                data = f[opts["datasetname"].lstrip("/")][()]
            axes = opts["axisorder"]
            if len(axes) != data.ndim:
                self.error(f"axis order {axes} does not match {data.ndim} dimensions")
            title = f"{opts['select']}{opts['datasetname']}"
            self.images[title] = (data, axes)
            self.current = title
            return
        if command == "Export HDF5":
            data, axes = self.images[opts["input"]]
            with h5py.File(opts["exportpath"], "w") as f:
                ds = f.create_dataset(
                    opts["datasetname"].lstrip("/"), data=conform(data, axes, "tzyxc")
                )
                ds.attrs["axistags"] = axistags("tzyxc")
            return
        if command in COMMANDS:
            self.predict(command, opts)
            return
        self.error(f'Unrecognized command: "{command}"')

    def predict(self, command: str, opts: dict[str, str]) -> None:
        if not self.config.get("executablefile"):
            self.error("ilastik executable location is not configured")
        for key in ("projectfilename", "inputimage"):
            if key not in opts:
                self.error(f"{command}: missing parameter {key}")
        source = {
            ("Run Pixel Classification Prediction", "Probabilities"): "Probabilities",
            ("Run Pixel Classification Prediction", "Segmentation"): "Simple Segmentation",
            ("Run Autocontext Prediction", "Probabilities"): "Probabilities Stage 2",
            ("Run Autocontext Prediction", "Segmentation"): "Simple Segmentation Stage 2",
        }
        key = COMMANDS[command][0]
        inputs = {"raw_data": opts["inputimage"]}
        if command == "Run Object Classification Prediction":
            export = opts.get("objectexportsource", "Object Predictions")
            role = {"Probabilities": "prediction_maps", "Segmentation": "segmentation_image"}
            inputs[role[opts.get("secondinputtype", "Probabilities")]] = opts["inputproborsegimage"]
        elif command == "Run Multicut":
            export = "Multicut Segmentation"
            inputs["probabilities"] = opts["boundarypredictionimage"]
        elif command == "Run Tracking":
            export = "Tracking-Result"
            role = {"Probabilities": "prediction_maps", "Segmentation": "binary_image"}
            inputs[role[opts.get("secondinputtype", "Probabilities")]] = opts["inputproborsegimage"]
        else:
            export = source[(command, opts.get(key, ""))]
        with tempfile.TemporaryDirectory(prefix="ilastik4ij_") as tmp:
            argv = [
                self.config["executablefile"],
                "--headless",
                f"--project={opts['projectfilename']}",
                "--output_format=hdf5",
                "--output_axis_order=tzcyx",
                "--input_axes=tzcyx",
                "--readonly=1",
                "--output_internal_path=exported_data",
                f"--export_source={export}",
            ]
            for flag, title in inputs.items():
                if title not in self.images:
                    self.error(f"no image titled {title!r}")
                data, axes = self.images[title]
                path = Path(tmp) / f"{flag}.h5"
                with h5py.File(path, "w") as f:
                    f.create_dataset("data", data=conform(data, axes, "tzcyx"))
                argv.append(f"--{flag}={path}")
            out = Path(tmp) / "predictions.h5"
            argv.append(f"--output_filename_format={out}")
            env = dict(os.environ)
            threads = int(self.config.get("numthreads", "-1"))
            if threads >= 0:
                env["LAZYFLOW_THREADS"] = str(threads)
            env["LAZYFLOW_TOTAL_RAM_MB"] = self.config.get("maxrammb", "4096")
            log = os.environ.get("FAKE_FIJI_LOG")
            if log:
                with open(log, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"command": command, "options": opts, "argv": argv}) + "\n"
                    )
            done = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
            print(done.stdout + done.stderr, end="")
            if done.returncode != 0:
                self.error(f"Subprocess failed with exit code {done.returncode}")
            with h5py.File(out, "r") as f:
                data = f["exported_data"][()]
        title = f"{command} result"
        self.images[title] = (data, "tzcyx")
        self.current = title

    def execute(self, line: str) -> None:
        match = STATEMENT.match(line)
        if not match:
            if line in ("setBatchMode(true);",) or line.startswith("//") or not line:
                return
            self.error(f"fake Fiji cannot interpret: {line}")
        name, args = match.groups()
        values = strings(args)
        if name == "run":
            self.run(values[0], options(values[1]) if len(values) > 1 else {})
        elif name == "rename":
            self.images[values[0]] = self.images.pop(self.current)
            self.current = values[0]
        elif name == "close":
            self.images.clear()
        elif name == "print":
            print(values[0])
        else:
            Path(values[1]).write_text(values[0], encoding="utf-8")


def main() -> int:
    argv = sys.argv[1:]
    if "--headless" not in argv or "-macro" not in argv:
        print("usage: fake Fiji --headless --console -macro <file>", file=sys.stderr)
        return 2
    macro = Path(argv[argv.index("-macro") + 1])
    fiji = Fiji()
    for number, line in enumerate(macro.read_text(encoding="utf-8").splitlines(), 1):
        try:
            fiji.execute(line.strip())
        except Exception as exc:  # like Fiji: report, stop the macro, exit 0
            print(f"Macro Error: {exc} in line {number}\n{line}")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
