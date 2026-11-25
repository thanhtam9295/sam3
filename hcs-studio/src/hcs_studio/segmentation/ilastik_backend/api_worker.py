"""ilastik Python-API worker (PROMPT_ILASTIK.md §3.2). Runs inside **ilastik's own Python
environment**, never in HCS Studio's (ilastik pins its dependencies):

    <ilastik env>/bin/python api_worker.py <job.json>

Imports only the standard library, numpy, h5py, xarray and ilastik, so the ilastik environment
does not need HCS Studio installed. The job file lists the project, the workflow and pairs of
(input spec ``<file>.h5/<dataset>``, output file). The project is loaded **once** and applied to
every input; each result is written as ilastik writes exports (dataset ``exported_data`` with
vigra ``axistags``).

API (ilastik ``ilastik.experimental.api``, read in its source, main branch Nov 2026):
``PixelClassificationPipeline.from_ilp_file(path)`` (older releases: ``from_project_file``),
``.get_probabilities(xarray.DataArray)``; ``AutocontextPipeline.from_ilp_file(path)``,
``.get_probabilities_stage_2(...)``. Both return an ``xarray.DataArray`` whose ``dims`` name
the axes.

Exit codes: 0 done; 1 error (traceback on stderr); 3 the installed ilastik has no Python API.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

FLAGS = {"c": 1, "t": 8, "z": 2, "y": 2, "x": 2}
NO_API = 3


def axistags(axes: str) -> str:
    return json.dumps(
        {
            "axes": [
                {"key": a, "typeFlags": FLAGS[a], "resolution": 0, "description": ""} for a in axes
            ]
        }
    )


def read_input(spec: str) -> tuple[Any, str]:
    import h5py

    file, sep, dataset = spec.rpartition(".h5/")
    file, dataset = (file + ".h5", dataset) if sep else (spec, "data")
    with h5py.File(file, "r") as f:
        ds = f[dataset]
        data = ds[()]
        axes = "".join(a["key"] for a in json.loads(ds.attrs["axistags"])["axes"])
    return data, axes


def load_pipeline(project: str, workflow: str) -> Any:
    try:
        from ilastik.experimental import api
    except ImportError:
        print("ilastik.experimental.api is not available in this ilastik", file=sys.stderr)
        raise SystemExit(NO_API) from None
    if workflow == "autocontext":
        if not hasattr(api, "AutocontextPipeline"):
            print("this ilastik's Python API has no AutocontextPipeline", file=sys.stderr)
            raise SystemExit(NO_API)
        return api.AutocontextPipeline.from_ilp_file(project)
    cls = api.PixelClassificationPipeline
    if hasattr(cls, "from_ilp_file"):
        return cls.from_ilp_file(project)
    return api.from_project_file(project)  # ilastik ≤ 1.4.0


def predict(pipeline: Any, workflow: str, data: Any, axes: str) -> tuple[Any, str]:
    import numpy as np
    import xarray

    image = xarray.DataArray(data, dims=tuple(axes))
    if workflow == "autocontext":
        result = pipeline.get_probabilities_stage_2(image)
    else:
        result = pipeline.get_probabilities(image)
    return np.asarray(getattr(result, "values", result)), "".join(result.dims)


def main(argv: list[str]) -> int:
    import h5py

    with open(argv[1], encoding="utf-8") as handle:
        job = json.load(handle)
    started = time.monotonic()
    pipeline = load_pipeline(job["project"], job["workflow"])
    print(f"project loaded in {time.monotonic() - started:.1f} s", flush=True)
    for k, (spec, target) in enumerate(job["inputs"], start=1):
        data, axes = read_input(spec)
        out, out_axes = predict(pipeline, job["workflow"], data, axes)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        partial = target + ".partial"
        with h5py.File(partial, "w") as f:
            ds = f.create_dataset("exported_data", data=out.astype("float32"))
            ds.attrs["axistags"] = axistags(out_axes)
        os.replace(partial, target)
        print(f"image {k} of {len(job['inputs'])} done", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
