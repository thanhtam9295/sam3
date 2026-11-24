"""Read-only views of automatic runs and the lab-event log (Automation tab, ``hcs-studio auto``)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from hcs_studio.db import session_scope
from hcs_studio.db.automation import AutoRun, LabEventRecord


def auto_runs(limit: int = 200) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(select(AutoRun).order_by(AutoRun.id.desc()).limit(limit)).all()
        return [
            {
                "key": r.key,
                "barcode": r.barcode,
                "instrument": r.instrument,
                "time_point": r.time_point,
                "status": r.status,
                "route": r.route,
                "job": r.job_id,
                "run_folder": r.run_folder,
                "detail": r.detail,
                "created": r.created,
                "updated": r.updated,
            }
            for r in rows
        ]


def lab_events(
    limit: int = 500, *, status: str | None = None, direction: str | None = None
) -> list[dict[str, Any]]:
    with session_scope() as session:
        query = select(LabEventRecord).order_by(LabEventRecord.id.desc()).limit(limit)
        if status:
            query = query.where(LabEventRecord.status == status)
        if direction:
            query = query.where(LabEventRecord.direction == direction)
        return [
            {
                "id": r.id,
                "time": r.logged_at,
                "direction": r.direction,
                "channel": r.channel,
                "type": r.type,
                "status": r.status,
                "barcode": r.barcode,
                "detail": r.detail,
                "event_id": r.event_id,
                "message": r.message,
            }
            for r in session.scalars(query)
        ]
