"""Jobs: one pipeline run on one plate, executed in its own process.

The GUI (or any caller) starts ``run_job`` in a separate process and listens to its event
queue; it never blocks. Each job may itself use a pool of worker processes for wells.
"""

from __future__ import annotations

import multiprocessing
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hcs_studio.model.plate import ImageMetadata


@dataclass(frozen=True)
class Job:
    """Everything needed to run a pipeline on a plate; picklable."""

    plate_source: dict[str, Any]
    pipeline: dict[str, Any]
    #: Metadata with the user's overrides; None = reopen and apply the saved overrides.
    metadata: ImageMetadata | None = None
    wells: list[tuple[int, int]] | None = None
    fields: dict[tuple[int, int], list[int]] | None = None
    layout_path: str | None = None
    output_root: str | None = None
    workers: int | None = 1
    checksums: bool = True
    save_images: bool = True
    max_memory_gb: float | None = None
    image_dtype: str = "float32"
    resume: str | None = None
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def run_job(job: Job, events: Any, cancel: Any, pause: Any) -> None:
    """Process entry point: run the job and report through ``events`` (a queue)."""
    try:
        from hcs_studio.io.base import reopen_plate
        from hcs_studio.io.sidecar import apply_overrides
        from hcs_studio.model.layout import load_plate_layout, run_layout_file
        from hcs_studio.pipeline.executor import run_pipeline
        from hcs_studio.pipeline.schema import Pipeline

        plate = reopen_plate(job.plate_source)
        plate = replace(plate, metadata=job.metadata) if job.metadata else apply_overrides(plate)
        layout_path = Path(job.layout_path) if job.layout_path else None
        if layout_path is None and job.resume:
            layout_path = run_layout_file(Path(job.resume))
        layout = load_plate_layout(layout_path) if layout_path else None
        run_pipeline(
            plate,
            Pipeline.model_validate(job.pipeline),
            output_root=Path(job.output_root) if job.output_root else None,
            wells=job.wells,
            fields=job.fields,
            layout=layout,
            layout_path=layout_path if not job.resume else None,
            checksums=job.checksums,
            events=events.put,
            cancel=cancel,
            pause=pause,
            workers=job.workers,
            resume=Path(job.resume) if job.resume else None,
            save_images=job.save_images,
            max_memory_gb=job.max_memory_gb,
            image_dtype=job.image_dtype,
        )
    except Exception as exc:
        events.put({"type": "log", "level": "error", "text": traceback.format_exc()})
        events.put({"type": "finished", "status": "error", "error": f"{type(exc).__name__}: {exc}"})


class JobProcess:
    """A job running in its own process, with cancel and pause events and an event queue."""

    def __init__(self, job: Job) -> None:
        context = multiprocessing.get_context("spawn")
        self.job = job
        self.events = context.Queue()
        self.cancel = context.Event()
        self.pause = context.Event()
        self.process = context.Process(
            target=run_job, args=(job, self.events, self.cancel, self.pause), name="hcs-job"
        )

    def start(self) -> None:
        self.process.start()

    def alive(self) -> bool:
        return self.process.is_alive()

    def poll(self) -> list[dict[str, Any]]:
        """Events received since the last call (non-blocking)."""
        import queue

        received = []
        while True:
            try:
                received.append(self.events.get_nowait())
            except queue.Empty:
                return received
            except (EOFError, OSError):
                return received

    def kill(self) -> None:
        """Stop immediately (as a crash would); the run folder can be resumed."""
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=10)
