"""Settings of automatic runs (``settings.json`` → ``automation``). Secrets (SMTP passwords,
webhook URLs) are references ``secret:<name>`` resolved through :mod:`hcs_studio.secrets`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NOTIFY_KINDS = ("completed", "qc_failed", "failed", "needs_layout", "unrouted")


class WatchFolder(BaseModel):
    """An instrument export location watched for finished plates."""

    model_config = ConfigDict(extra="forbid")

    path: str
    instrument: str = ""
    protocol: str = ""
    #: A plate is considered finished when nothing changed for this long and it is complete.
    quiet_seconds: float = Field(default=120.0, ge=0)


class Channel(BaseModel):
    """A notification channel. ``webhook`` and ``password`` must be ``secret:<name>``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["email", "teams", "slack", "desktop"]
    #: Which messages go to this channel (default: all).
    on: list[str] = Field(default_factory=lambda: list(NOTIFY_KINDS))
    # email
    to: list[str] = Field(default_factory=list)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_tls: bool = True
    username: str = ""
    password: str = ""  # secret:<name>
    sender: str = "hcs-studio@localhost"
    # teams / slack
    webhook: str = ""  # secret:<name>


class AutomationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Folder of incoming event files (JSON); processed files move to ``processed/`` or
    #: ``rejected/`` below it.
    inbox: str = ""
    #: Folder the outgoing events are written to (empty = none).
    outbox: str = ""
    #: Local HTTP API for incoming events (0 = off). Listens on ``http_host`` only.
    http_port: int = Field(default=0, ge=0, le=65535)
    http_host: str = "127.0.0.1"
    #: Outgoing events are also POSTed here (empty = files only).
    scheduler_url: str = ""
    watch: list[WatchFolder] = Field(default_factory=list)
    #: Routing rules (YAML) choosing the pipeline of each plate.
    routing_file: str = ""
    #: Where automatic runs write their run folders (empty = next to the raw data's parent).
    output_root: str = ""
    workers: int | None = None
    #: At night (local hours "20-7") automatic runs may use more worker processes.
    night_hours: str = Field(default="", pattern=r"^$|^\d{1,2}-\d{1,2}$")
    night_workers: int | None = None
    #: Send a ``reimage.request`` to the scheduler when a plate fails QC.
    reimage_on_qc_fail: bool = True
    #: How long an event's raw data may take to become complete before the plate is given up.
    data_wait_minutes: float = Field(default=120.0, gt=0)
    poll_seconds: float = Field(default=10.0, gt=0)
    #: The same raw data (and time point) reported again within this window is a duplicate,
    #: even with another acquisition ID (e.g. by the scheduler and the folder watcher).
    duplicate_window_hours: float = Field(default=24.0, ge=0)
    channels: list[Channel] = Field(default_factory=list)
    #: Addresses (emails) that also receive failures.
    admins: list[str] = Field(default_factory=list)
