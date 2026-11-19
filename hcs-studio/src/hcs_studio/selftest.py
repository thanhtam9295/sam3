"""``hcs-studio selftest``: check an installation end to end on a small synthetic plate.

Writes an ImageXpress-style plate to a temporary folder, runs a pipeline with worker
processes, resumes it, writes the report, applies a gate, computes PCA/UMAP profiles and a
tiny deconvolution, and reports each check. Used by the installer build and by users who
want to know whether their installation works.
"""

from __future__ import annotations

import importlib
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE = """
name: selftest
steps:
  - {module: flatfield, params: {min_fields: 8}}
  - {module: segment_classical, params: {channel: 1}}
  - {module: expand_cells, params: {distance_um: 4}}
  - {module: measure_objects, params: {texture: false}}
  - {module: colocalization, params: {channel_a: 1, channel_b: 2, scope: field}}
  - {module: qc}
  - {module: gate_objects, params: {name: big, rules: [{feature: area_um2, min: 20}]}}
"""
LAYOUT = "well,compound,concentration,treatment,control_type\n" + "\n".join(
    f"{w},{'DMSO' if w.endswith(('01', '02')) else 'Drug'},,,"
    f"{'negative' if w.endswith(('01', '02')) else ''}"
    for w in [f"{r}{c:02d}" for r in "ABC" for c in range(1, 5)]
)
REQUIRED = (
    "numpy",
    "scipy",
    "skimage",
    "cv2",
    "tifffile",
    "imagecodecs",
    "nd2",
    "dask",
    "zarr",
    "pandas",
    "pyarrow",
    "sklearn",
    "openpyxl",
    "reportlab",
    "imageio_ffmpeg",
    "h5py",
    "sqlalchemy",
    "alembic",
    "jsonschema",
    "keyring",
    "lab_events",
    "umap",
    "numba",
    "mahotas",
    "psfmodels",
    "jinja2",
    "matplotlib",
)
OPTIONAL = ("PySide6", "napari", "pyqtgraph", "torch", "cellpose")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    seconds: float


def _field(row: int, col: int, site: int, wave: int, z: int) -> np.ndarray:
    """Separated nuclei (channel 1) with a cytoplasm ring (channel 2), so the number of
    cells drawn is the number a correct segmentation finds."""
    from scipy import ndimage

    rng = np.random.default_rng(row * 10 + col)
    n, size = 5 + 3 * ((row * 4 + col) % 5), 192
    centres: list[tuple[float, float]] = []
    while len(centres) < n:
        y, x = rng.uniform(14, size - 14, 2)
        if all((y - a) ** 2 + (x - b) ** 2 > 26**2 for a, b in centres):
            centres.append((y, x))
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size))
    for y, x in centres:
        d = np.hypot(yy - y, xx - x)
        image += 1500 * (d <= 6) if wave == 0 else 600 * ((d > 6) & (d <= 11))
    image = ndimage.gaussian_filter(image, 1.0) + 150
    noisy = np.random.default_rng(row * 10 + col + 100 * wave).poisson(image) + 100
    return np.asarray(np.clip(noisy, 0, 65535), dtype=np.uint16)


def run_selftest(workers: int = 2, keep: Path | None = None) -> list[Check]:
    checks: list[Check] = []

    def check(name: str, work: Callable[[], str]) -> bool:
        started = time.monotonic()
        try:
            detail = work()
            ok = True
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
            traceback.print_exc()
        checks.append(Check(name, ok, detail, time.monotonic() - started))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        return ok

    def imports() -> str:
        missing = []
        for name in REQUIRED:
            try:
                importlib.import_module(name)
            except Exception as exc:
                missing.append(f"{name} ({exc})")
        if missing:
            raise ImportError("; ".join(missing))
        present = []
        for name in OPTIONAL:
            try:
                importlib.import_module(name)
                present.append(name)
            except Exception:
                present.append(f"{name} (not installed)")
        return f"{len(REQUIRED)} required packages; optional: {', '.join(present)}"

    def gpu() -> str:
        from hcs_studio.utils.gpu import detect_gpu

        return detect_gpu().summary

    def registry() -> str:
        import tempfile

        from lab_events import make_event

        from hcs_studio.db.engine import current_revision, head_revision, make_engine
        from hcs_studio.db.models import EventRecord

        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(f"sqlite:///{Path(tmp) / 'selftest.sqlite'}")
            try:
                revision = current_revision(engine)
                if revision != head_revision():
                    raise RuntimeError(f"schema {revision}, expected {head_revision()}")
                from sqlalchemy.orm import Session

                with Session(engine) as session:
                    session.add(
                        EventRecord(type="selftest", actor_id="selftest", actor_kind="system")
                    )
                    session.commit()
            finally:
                engine.dispose()
        make_event(
            "analysis.failed",
            {
                "barcode": "SELFTEST",
                "stage": "qc",
                "error": "none",
                "finished_at": "2026-01-01T00:00:00Z",
            },
            source={"system": "hcs-studio selftest"},
        )
        return f"migrations to {revision}; lab event schemas valid"

    check("Python packages", imports)
    check("GPU detection", gpu)
    check("Registry database", registry)

    temporary = tempfile.TemporaryDirectory(prefix="hcs-selftest-")
    root = Path(keep) if keep is not None else Path(temporary.name)
    state: dict[str, Any] = {}
    try:

        def plate() -> str:
            from hcs_studio.io.base import open_plate
            from hcs_studio.utils.synthetic import write_imagexpress_plate

            wells = [(r, c) for r in range(3) for c in range(4)]
            folder = write_imagexpress_plate(root / "raw" / "plate", wells, _field, n_sites=1)
            (root / "layout.csv").write_text(LAYOUT + "\n", encoding="utf-8")
            state["plate"] = open_plate(folder)
            return f"{len(state['plate'].wells)} wells written and read back"

        def run() -> str:
            from hcs_studio.model.layout import read_plate_layout
            from hcs_studio.pipeline.executor import run_pipeline
            from hcs_studio.pipeline.schema import Pipeline

            result = run_pipeline(
                state["plate"],
                Pipeline.from_text(PIPELINE),
                output_root=root / "out",
                layout=read_plate_layout(root / "layout.csv"),
                layout_path=root / "layout.csv",
                checksums=False,
                workers=workers,
            )
            if result.failed_wells:
                raise RuntimeError(f"failed wells: {result.failed_wells}")
            state["run"] = result.folder
            return f"{result.fields_written} fields with {workers} worker process(es)"

        def counts() -> str:
            import pandas as pd

            summary = pd.read_csv(state["run"] / "tables" / "per_well_summary.csv")
            found = dict(zip(summary["well"], summary["nuclei_count"], strict=True))
            expected = {
                f"{'ABC'[r]}{c + 1:02d}": 5 + 3 * ((r * 4 + c) % 5)
                for r in range(3)
                for c in range(4)
            }
            off = {
                w: (found.get(w), n) for w, n in expected.items() if abs(found.get(w, 0) - n) > 1
            }
            if off:
                raise ValueError(f"nuclei counts (found, drawn) differ: {off}")
            return "nuclei counts match the drawn cells (±1) in all 12 wells"

        def resume() -> str:
            from hcs_studio.model.layout import read_plate_layout
            from hcs_studio.pipeline.executor import run_pipeline
            from hcs_studio.pipeline.schema import Pipeline

            result = run_pipeline(
                state["plate"],
                Pipeline.from_text(PIPELINE),
                layout=read_plate_layout(root / "layout.csv"),
                checksums=False,
                workers=workers,
                resume=state["run"],
            )
            if len(result.skipped_wells) != 12:
                raise RuntimeError(f"resume redid wells: skipped {result.skipped_wells}")
            return "all 12 completed wells skipped"

        def report() -> str:
            html = (state["run"] / "index.html").read_text(encoding="utf-8")
            if '"wells":{' not in html:
                raise ValueError("index.html has no well data")
            return f"index.html written ({len(html) // 1024} kB)"

        def profiles() -> str:
            from hcs_studio.analysis.profiling import ProfileOptions, run_profiling

            result = run_profiling(
                [state["run"]], ProfileOptions(max_cells=500, n_neighbors=5, figures=True)
            )
            if "UMAP1" not in result.cells:
                raise ValueError("no UMAP coordinates")
            return f"PCA + UMAP of {len(result.cells)} objects, figures written"

        def deconvolution() -> str:
            from hcs_studio.benchmark import run_benchmark

            result = run_benchmark((8, 64, 64), 5, "auto", 0, None)
            return f"{result.backend}, {result.seconds:.1f} s"

        if check("Write and read a plate", plate) and check("Run a pipeline", run):
            check("Segmentation counts", counts)
            check("Resume", resume)
            check("HTML report", report)
            check("PCA / UMAP profiles", profiles)
        check("Deconvolution", deconvolution)
    finally:
        if keep is None:
            temporary.cleanup()
    return checks
