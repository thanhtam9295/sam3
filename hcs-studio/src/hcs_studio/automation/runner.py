"""The automatic flow (Phase 19.3), driven by :meth:`AutoRunner.step`:

``event → register acquisition → wait for complete data → route → QC + pipeline (job queue,
priority "auto") → pharmacology / hit calling / kinetics as the rule says → registry → report
→ analysis.completed + notification``.

- The image QC of the routed rule runs first inside the job (Phase 18b): a plate that fails QC
  stops there (``qc_failed``), people are told and, if configured, a ``reimage.request`` with
  the failed wells/fields goes to the scheduler.
- No layout for the barcode → the image analysis still runs, pharmacology and hit calling wait,
  and the owners are asked to annotate the plate (``needs_layout``).
- Auto runs go into the same queue as interactive jobs with the lower priority ``auto``:
  interactive jobs start first; at night (``night_hours``) auto runs may use
  ``night_workers`` processes (all GPUs/CPUs of the machine).
- Each acquisition runs once (``auto_run.key``, see :mod:`~hcs_studio.automation.intake`).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from hcs_studio.automation.notify import Notice, Notifier
from hcs_studio.automation.routing import Routing, Rule, load_routing, registry_assay
from hcs_studio.automation.settings import AutomationSettings
from hcs_studio.db import session_scope
from hcs_studio.db.automation import AutoRun
from hcs_studio.services import Actor, ServiceContext

log = logging.getLogger("hcs_studio.automation")

ACTOR = Actor("automation", "Automatic runs", "system", "admin")
FINAL = (
    "completed",
    "completed_with_failures",
    "qc_failed",
    "cancelled",
    "error",
    "interrupted",
    "failed",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _context() -> ServiceContext:
    return ServiceContext(actor=ACTOR, reason="automatic run", client="automation")


def is_night(hours: str, when: datetime | None = None) -> bool:
    """``"20-7"`` = from 20:00 to 07:00 local time."""
    if not hours:
        return False
    start, end = (int(h) for h in hours.split("-"))
    hour = (when or datetime.now()).hour
    return start <= hour or hour < end if start > end else start <= hour < end


class AutoRunner:
    def __init__(
        self,
        settings: AutomationSettings,
        queue: Any,
        *,
        notifier: Notifier | None = None,
        clock: Any = time.time,
    ) -> None:
        from hcs_studio.automation.watcher import FolderWatcher

        self.settings = settings
        self.queue = queue
        self.notifier = notifier or Notifier(settings.channels, settings.admins)
        self.clock = clock
        self.watchers = [FolderWatcher(w, settings.data_wait_minutes) for w in settings.watch]

    # --- one round --------------------------------------------------------------------------

    def step(self) -> None:
        """Take in events, check waiting acquisitions, start and finish runs."""
        from hcs_studio.automation.intake import poll_inbox

        if self.settings.inbox:
            poll_inbox(Path(self.settings.inbox))
        for watcher in self.watchers:
            try:
                watcher.scan(now=self.clock())
            except Exception:
                log.exception("Watching %s failed", watcher.watch.path)
        if hasattr(self.queue, "poll"):
            self.queue.poll()
        with session_scope() as session:
            keys = session.scalars(
                select(AutoRun.key).where(
                    AutoRun.status.in_(("pending", "waiting_data", "running"))
                )
            ).all()
        for key in keys:
            try:
                self._advance(key)
            except Exception as exc:
                log.exception("Automatic run %s failed", key)
                self._fail(key, "analysis", f"{type(exc).__name__}: {exc}")

    def _load(self, key: str) -> AutoRun:
        with session_scope() as session:
            run = session.scalars(select(AutoRun).where(AutoRun.key == key)).one()
            session.expunge(run)
            return run

    def _update(self, key: str, **values: Any) -> None:
        with session_scope() as session:
            run = session.scalars(select(AutoRun).where(AutoRun.key == key)).one()
            for name, value in values.items():
                setattr(run, name, value)
            run.updated = datetime.now(UTC)

    def _advance(self, key: str) -> None:
        run = self._load(key)
        if run.status in ("pending", "waiting_data"):
            self._start(run)
        elif run.status == "running":
            self._follow(run)

    # --- starting ---------------------------------------------------------------------------

    def _raw_path(self, run: AutoRun) -> Path:
        from hcs_studio.paths import resolve

        return resolve(run.raw) if ":" in run.raw.split("/", 1)[0] else Path(run.raw)

    def routing(self) -> Routing | None:
        if not self.settings.routing_file:
            return None
        return load_routing(self.settings.routing_file)

    def _start(self, run: AutoRun) -> None:
        from hcs_studio.automation.completeness import check

        try:
            raw = self._raw_path(run)
        except (KeyError, ValueError) as exc:
            self._fail(run.key, "registration", f"raw data location not usable here: {exc}")
            return
        completeness = check(raw)
        if not completeness.complete:
            created = run.created if run.created.tzinfo else run.created.replace(tzinfo=UTC)
            waited = (datetime.now(UTC) - created).total_seconds() / 60
            if waited > self.settings.data_wait_minutes:
                self._fail(
                    run.key,
                    "registration",
                    f"the raw data is still incomplete after {waited:.0f} min: "
                    f"{completeness.reason}",
                )
                return
            self._update(run.key, status="waiting_data", detail=completeness.reason)
            return
        plate_info = {
            "barcode": run.barcode,
            "instrument": run.instrument,
            "protocol": run.protocol,
            "assay": registry_assay(run.barcode),
        }
        routing = self.routing()
        rule = routing.route(plate_info) if routing is not None else None
        if routing is None or rule is None:
            self._update(run.key, status="unrouted", detail="no routing rule matches")
            self.notifier.notify(
                Notice(
                    "unrouted",
                    run.barcode,
                    "no routing rule matches",
                    [
                        f"Instrument {run.instrument}, protocol "
                        f"{run.protocol or '–'}, raw data {run.raw}"
                    ],
                )
            )
            return
        layout_path = self._layout(run.barcode)
        job = self._job(run, raw, routing, rule, layout_path)
        from hcs_studio.services.operations import submit_job

        job_id = submit_job(_context(), self.queue, job, priority="auto")
        self._update(
            run.key,
            status="running",
            route=rule.name,
            job_id=job_id,
            detail="" if layout_path else "needs_layout",
        )
        self._send(
            "analysis.started",
            {
                "barcode": run.barcode,
                "acquisition_id": run.acquisition_id or run.key,
                "run_id": job_id,
                "pipeline": rule.pipeline,
                "started_at": _now_iso(),
            },
            run,
        )
        if layout_path is None:
            self.notifier.notify(
                Notice(
                    "needs_layout",
                    run.barcode,
                    "please annotate this plate",
                    [
                        "No plate layout is saved for this barcode. The image analysis runs; "
                        "pharmacology and hit calling wait until the plate is annotated "
                        "(Layout tab, then 'annotate' the run)."
                    ],
                    owners=rule.notify,
                )
            )
        log.info("Automatic run of %s started (rule %s, job %s)", run.barcode, rule.name, job_id)

    def _layout(self, barcode: str) -> Path | None:
        from hcs_studio.io.layout_store import LayoutStore, plate_key

        store = LayoutStore(plate_key(barcode=barcode))
        versions = store.versions()
        return store.path(versions[-1]) if versions else None

    def _job(
        self, run: AutoRun, raw: Path, routing: Routing, rule: Rule, layout: Path | None
    ) -> Any:
        import yaml

        from hcs_studio.io.base import open_plate
        from hcs_studio.pipeline.jobs import Job
        from hcs_studio.pipeline.schema import Pipeline
        from hcs_studio.qc.rules import load_rules

        pipeline = Pipeline.model_validate(
            yaml.safe_load(routing.path(rule.pipeline).read_text(encoding="utf-8"))
        )
        if rule.qc_rules:
            pipeline = pipeline.model_copy(update={"qc": load_rules(routing.path(rule.qc_rules))})
        workers = rule.workers or self.settings.workers
        if self.settings.night_workers and is_night(self.settings.night_hours):
            workers = self.settings.night_workers
        output = self.settings.output_root
        if output and ":" in output.split("/", 1)[0] and len(output.split(":")[0]) > 1:
            from hcs_studio.paths import resolve

            output = str(resolve(output))
        return Job(
            plate_source=open_plate(raw).source,
            pipeline=pipeline.to_dict(),
            layout_path=str(layout) if layout else None,
            output_root=output or None,
            workers=workers,
            label=f"auto {run.barcode}" + (f" t{run.time_point}" if run.time_point else ""),
            extra={"auto_run": run.key},
        )

    # --- following --------------------------------------------------------------------------

    def _follow(self, run: AutoRun) -> None:
        try:
            state = self.queue.status(run.job_id)
        except KeyError:
            self._fail(run.key, "analysis", f"job {run.job_id} is unknown to the queue")
            return
        if state.status not in FINAL:
            return
        folder = Path(state.folder) if state.folder else None
        if folder is not None:
            self._update(run.key, run_folder=str(folder))
        if state.status == "qc_failed" and folder is not None:
            self._qc_failed(run, folder)
            return
        if state.status not in ("completed", "completed_with_failures") or folder is None:
            self._fail(run.key, "analysis", state.message or f"the job ended {state.status}")
            return
        self._finish(run, folder, state.status)

    def _rule(self, run: AutoRun) -> tuple[Routing | None, Rule | None]:
        routing = self.routing()
        if routing is None:
            return None, None
        return routing, next((r for r in routing.rules if r.name == run.route), None)

    def _finish(self, run: AutoRun, folder: Path, status: str) -> None:
        from hcs_studio.services import operations
        from hcs_studio.services.registry import scan_folders

        ctx = _context()
        routing, rule = self._rule(run)
        notes = []
        has_layout = (folder / "plate_layout.json").exists()
        if rule is not None and rule.pharmacology and has_layout:
            from hcs_studio.pharmacology.analysis import PharmOptions

            options = PharmOptions(**{"pdf": False, **rule.pharmacology})
            operations.run_pharmacology(ctx, [folder], options)
            notes.append("pharmacology written")
        if rule is not None and rule.screen and has_layout and routing is not None:
            from hcs_studio.screening.screen import ScreenConfig

            config = ScreenConfig.load(routing.path(rule.screen))
            output = folder / "screen"
            operations.run_screen(ctx, [folder], config, output)
            notes.append("hits called")
        if rule is not None and (rule.pharmacology or rule.screen) and not has_layout:
            notes.append("pharmacology/hit calling wait for the plate layout")
        scan_folders(ctx, [folder])
        if rule is not None and rule.kinetics and run.time_point is not None:
            self._kinetics(run.barcode)
            notes.append("kinetic dataset updated")
        report = operations.write_report(ctx, folder)
        verdict, numbers = summarise(folder)
        from hcs_studio.output.manifest import rooted_text

        self._send(
            "analysis.completed",
            {
                "barcode": run.barcode,
                "acquisition_id": run.acquisition_id or run.key,
                "run_id": folder.name,
                "qc_verdict": verdict,
                "report_path": rooted_text(report),
                "key_numbers": numbers,
                "finished_at": _now_iso(),
            },
            run,
        )
        self._update(run.key, status="completed", detail="; ".join(notes))
        title = "analysis completed" + (" with failed wells" if status != "completed" else "")
        self.notifier.notify(
            Notice(
                "completed",
                run.barcode,
                f"{title} (QC {verdict})",
                [f"Assay: {registry_assay(run.barcode) or '–'}; rule {run.route}", *notes],
                numbers,
                str(report),
                owners=rule.notify if rule else [],
            )
        )

    def _kinetics(self, barcode: str) -> None:
        """Analyse all finished time points of the plate together (Phase 22)."""
        from hcs_studio.kinetics.analysis import analyse_runs

        with session_scope() as session:
            folders = [
                Path(f)
                for f in session.scalars(
                    select(AutoRun.run_folder).where(
                        (AutoRun.barcode == barcode) & (AutoRun.run_folder != "")
                    )
                )
            ]
        if len(folders) >= 2:
            from hcs_studio.kinetics.analysis import KineticsParams

            ordered = sorted(folders)
            analyse_runs(ordered, KineticsParams(), ordered[-1] / "kinetics_plate")

    def _qc_failed(self, run: AutoRun, folder: Path) -> None:
        from hcs_studio.qc.run import load_qc

        _, rule = self._rule(run)
        try:
            qc = load_qc(folder)
            reasons = list(qc.plate.get("reasons") or []) or ["the plate failed image QC"]
            wells = qc.failed_wells
            fields = sorted(qc.excluded_fields)
        except (OSError, ValueError, KeyError):
            reasons, wells, fields = ["the plate failed image QC"], [], []
        reason = "; ".join(str(r) for r in reasons)
        if self.settings.reimage_on_qc_fail:
            payload: dict[str, Any] = {
                "barcode": run.barcode,
                "reason": reason,
                "requested_by": "hcs-studio",
            }
            if run.acquisition_id:
                payload["acquisition_id"] = run.acquisition_id
            if wells:
                payload["wells"] = wells
            if fields:
                payload["fields"] = [{"well": w, "field": int(f)} for w, f in fields]
            self._send("reimage.request", payload, run)
        self._send(
            "analysis.failed",
            {
                "barcode": run.barcode,
                "acquisition_id": run.acquisition_id or run.key,
                "run_id": folder.name,
                "stage": "qc",
                "error": reason,
                "finished_at": _now_iso(),
            },
            run,
        )
        self._update(run.key, status="qc_failed", detail=reason)
        self.notifier.notify(
            Notice(
                "qc_failed",
                run.barcode,
                "the plate failed image QC",
                [reason]
                + ([f"Failed wells: {', '.join(wells[:24])}"] if wells else [])
                + (
                    ["A re-imaging request was sent to the scheduler."]
                    if self.settings.reimage_on_qc_fail
                    else []
                ),
                report=str(folder / "qc_report.html"),
                owners=rule.notify if rule else [],
            )
        )

    def _fail(self, key: str, stage: str, error: str) -> None:
        run = self._load(key)
        _, rule = self._rule(run) if run.route else (None, None)
        self._update(key, status="failed", detail=error)
        self._send(
            "analysis.failed",
            {
                "barcode": run.barcode,
                "acquisition_id": run.acquisition_id or run.key,
                "stage": stage,
                "error": error[:2000],
                "finished_at": _now_iso(),
            },
            run,
        )
        self.notifier.notify(
            Notice(
                "failed",
                run.barcode,
                f"automatic run failed ({stage})",
                [error],
                owners=rule.notify if rule else [],
            )
        )

    def _send(self, event_type: str, payload: dict[str, Any], run: AutoRun) -> None:
        from hcs_studio.automation.intake import send

        try:
            send(
                event_type,
                payload,
                outbox=self.settings.outbox,
                scheduler_url=self.settings.scheduler_url,
                correlation_id=run.event_id or None,
            )
        except Exception:
            log.exception("Could not send %s for %s", event_type, run.barcode)

    def status(self) -> dict[str, Any]:
        return {
            "watch": [w.status() for w in self.watchers],
            "notifications": self.notifier.summary(),
        }


def summarise(folder: Path) -> tuple[str, dict[str, float | None]]:
    """The QC verdict (pass/warn/fail) and key numbers (Z′, mean counts) of a finished run."""
    import math

    import pandas as pd

    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    qc = manifest.get("image_qc") or {}
    verdict = "pass"
    if qc and qc.get("passed") is False:
        verdict = "fail"
    elif manifest.get("failed_wells") or qc.get("excluded_fields"):
        verdict = "warn"
    numbers: dict[str, float | None] = {}
    table = folder / "tables" / "per_well_summary.csv"
    if table.exists():
        frame = pd.read_csv(table)
        counts = [c for c in frame.columns if str(c).endswith("_count")]
        for column in counts[:2]:
            value = float(pd.to_numeric(frame[column], errors="coerce").mean())
            numbers[f"mean {column}"] = None if math.isnan(value) else round(value, 4)
        if counts:
            from hcs_studio.output.results import RunResults

            try:
                results = RunResults(folder)
                z = results.z_prime(results.values(counts[0]))
                if z is not None and not math.isnan(z):
                    numbers[f"Z′ {counts[0]}"] = round(float(z), 4)
                    if z < 0.5 and verdict == "pass":
                        verdict = "warn"
            except Exception:  # no controls in the layout
                pass
    return verdict, numbers


def retry(key: str) -> None:
    """Start an acquisition again (after fixing routing, data or the layout)."""
    with session_scope() as session:
        run = session.scalars(select(AutoRun).where(AutoRun.key == key)).one()
        run.status, run.job_id, run.detail = "pending", "", "retried"
        run.created = datetime.now(UTC)
